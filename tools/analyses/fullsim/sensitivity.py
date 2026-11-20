"""Full-simulation CE sensitivity: S/sqrt(B) in the best momentum window.

The full-simulation counterpart of approx_ce_sensitivity, built the same way
but with the signal efficiency, shape and detector response measured from
reconstructed CE events mixed with pileup (CeMLeadingLogMix1BB EventNtuple
files) instead of from EdepAna's truth-level tree:

1. Select CE-like tracks with pyfitter's run1a_tsda cut set (cuts.py),
   with a 540-1650 ns time window, file by file.
   Each surviving event is represented by its first selected track: its
   momentum and time at the tracker front, its MC origin
   (eventntuple.origin_codes) and its true momentum at birth. Events whose
   first track has more than one tracker-front segment are skipped, as
   pyCount's Count.CheckMCTruth does. Only true CE count as signal, so a
   pileup track that passes the cuts is not.
2. Signal: the reconstructed momentum of the CE events inside the time
   window, binned as approx_ce_sensitivity's signal, and scaled as
   Production's normalizations.py has it (normalization.py): NPOT x stopped
   mu- per POT x captures per stopped mu- x R_mue x efficiency. Stopped mu-
   per POT is computed from the stop chain's datasets in SAM, as
   CreateSimEfficiency.sh computes the stage efficiencies, the chain being
   the one the input files descend from. The
   efficiency is measured: CE counted over the events generated to make the
   input files (their SAM dh.gencount, provenance.py), so it includes the
   acceptance of the digitization filter. No smearing is added, since
   the reconstruction already resolves the momentum.
3. DIO background: approx_ce_sensitivity's theoretical spectrum, scaled to a
   count per bin for NPOT x stopped mu- per POT x DIO per stopped mu-, then
   convolved with the measured response of the same CE events
   (reconstructed momentum at the tracker front minus true momentum at
   birth: energy loss and resolution together), normalized to the measured
   efficiency so DIO electrons pass the selection as CE ones do.
4. Cosmic background: measured from the official cosmic mix that matches
   the inputs (cosmics.py; CosmicSignalMix1BB for CeMLeadingLogMix1BB, same
   configuration). Its files go through the same cuts and time window; the
   selected events whose first track a cosmic made each stand for the
   on-spill time NPOT implies over the sample's cosmic livetime (S2 events
   resampled, from SAM, times the S1 livetime per event), spread flat over
   the momentum axis at the count in 100-110 MeV/c by default. With
   cosmic_source 'rate', flat at `cosmic_rate_per_s_per_mev` instead, as
   approx_ce_sensitivity has it. (cut_mode="optimize" tunes against its
   own cosmic reference, normalized by that reference's CRY livetime the
   same way; the loose fast-sim rate only for a sidecar without one.)
5. Scan every momentum window with approx_ce_sensitivity's scan_signal_box
   and keep the one maximizing S/sqrt(B).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from ... import catalog
from ...catalog import CatalogError, dataset_file_names, resolve_inputs
from ...spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from ...spectrum import Kernel, Spectrum
from ..approx_ce_sensitivity import (COSMIC_RATE_PER_SECOND_PER_MEV,
                                     MEAN_POT_PER_EVENT, NPOT,
                                     ONSPILL_SECONDS_PER_EVENT, SIGNAL_REBIN,
                                     SensitivityError, load_dio_spectrum,
                                     recommended_cosmic_rate, scan_signal_box)
from .cuts import (CUT_ORDER, CUT_SET, CUT_SETS, CutError, active_cuts,
                   apply_cuts, cut_descriptions, cut_masks, default_cuts,
                   get_cut_set)
from .eventntuple import (DEFAULT_TRIGGERS, ORIGIN_NAMES, SID_TT_FRONT,
                          TREE_PATH, EventNtupleError, origin_codes,
                          parse_trigger_paths, read_eventntuple)
from .normalization import (CAPTURES_PER_STOPPED_MUON, DIO_PER_STOPPED_MUON,
                            STOP_CHAIN, STOP_CHAIN_DATASETS,
                            STOPPED_MUONS_PER_POT, NormalizationError,
                            sim_efficiencies, stop_chain_of_inputs,
                            stopped_muons_per_pot)
from .provenance import (ProvenanceError, generated_events,
                         generated_events_bulk, root_ancestor)
from . import cosmic_reference
from . import cosmics
from . import tune

# cut_mode values: the cut set as it stands, or tuned per input (tune.py).
CUT_MODES = ("fixed", "optimize")
# The opt_* metrics' value outside cut_mode="optimize".
NOT_OPTIMIZED = -1.0
# An opt_cut_* edge the search left open (no cut there).
UNPLACED_EDGE = float("nan")
OPT_METRICS = (
    "opt_punzi_train", "opt_punzi_test", "opt_seed_punzi_train",
    "opt_seed_punzi_test", "opt_gain_test",
    "opt_signal_test", "opt_background_test",
    "opt_seed_signal_test", "opt_seed_background_test",
    "opt_window_low_mevc", "opt_window_high_mevc",
    "opt_seed_window_low_mevc", "opt_seed_window_high_mevc",
    "opt_n_ce_rows", "opt_n_cosmic_rows",
    # the cosmic reference's CRY livetime; -1 for a reference normalized
    # to the loose rate (a sidecar without livetime_s)
    "opt_cosmic_livetime_s",
    *(f"opt_cut_{v}_{edge}" for v in tune.VARIABLE_NAMES for edge in ("low", "high")),
)

SIGN = "minus"
# stop_datasets' value for tracing the stop chain from the inputs.
STOP_DATASETS_AUTO = "auto"
# The origin label of a true CE- (eventntuple.ORIGIN_NAMES).
CE_ORIGIN = 168

# The mu- -> e- conversion rate relative to capture, R_mue, assumed.
RMUE = 1.0e-13

# The time window at the tracker front (ns) a CE must fall in: pyCount's
# fixed window, which is also pyfitter's signal-region time window.
# The default time window is the cut set's (CutSet.time_window_ns).
TIME_WINDOW_NS = CUT_SETS[CUT_SET].time_window_ns

# Reconstructed momentum, binned as approx_ce_sensitivity's signal
# (EdepAna's trk_front_energy, then rebinned by SIGNAL_REBIN).
SIGNAL_BINNING = (1500, 0.0, 150.0)
# Reconstructed minus true momentum (MeV/c): energy loss, never more than
# this, and resolution, which can be a gain.
RESPONSE_BINNING = (1100, -100.0, 10.0)
# The momentum range drawn in the plots.
PLOT_MOM_RANGE = (95.0, 110.0)


class FullsimError(RuntimeError):
    """Raised for a problem the caller should see verbatim."""


# --- per-file reduction ------------------------------------------------------

def reduce_file(path: Path, active: list[str], trkqual_min: float,
                trkpid_min: float, triggers=DEFAULT_TRIGGERS,
                cut_set: str = CUT_SET) -> dict:
    """One file's selected events, reduced to flat per-event arrays, and its
    cut flow. Done per file so the jagged arrays never all sit in memory."""
    return reduce_data(read_eventntuple(path, triggers), active, trkqual_min,
                       trkpid_min, triggers, cut_set)


def reduce_data(data: dict, active: list[str], trkqual_min: float,
                trkpid_min: float, triggers=DEFAULT_TRIGGERS,
                cut_set: str = CUT_SET) -> dict:
    """reduce_file for EventNtuple groups already in memory."""
    import awkward as ak

    masks = cut_masks(data, SIGN, trkqual_min, trkpid_min, triggers, cut_set)
    track_mask, flow = apply_cuts(masks, active, data["trk"]["trk.pdg"])
    keep = ak.to_numpy(ak.any(track_mask, axis=-1))

    segs = data["trkfit"]["trksegs"][track_mask][keep]
    sims = data["trkmc"]["trkmcsim"][track_mask][keep]
    evt = data["evt"][keep]

    front = segs[segs["sid"] == SID_TT_FRONT]
    mom = front["mom"]["fCoordinates"]
    p = np.sqrt(mom["fX"] ** 2 + mom["fY"] ** 2 + mom["fZ"] ** 2)
    t = front["time"]

    # Count the first track's front segments before dropping non-finite
    # ones, as pyCount does: a NaN segment still makes a two-segment track.
    # A lone NaN segment is kept and then fails the time window.
    first_p = ak.firsts(p, axis=1)                  # events x front segments
    first_t = ak.firsts(t, axis=1)
    nfront = ak.to_numpy(ak.fill_none(ak.num(first_p, axis=1), 0))
    single = nfront == 1

    # The first track's own particle, at birth.
    birth = ak.firsts(ak.firsts(sims, axis=1), axis=1)["mom"]["fCoordinates"]
    p_true = np.sqrt(birth["fX"] ** 2 + birth["fY"] ** 2 + birth["fZ"] ** 2)

    def one(values):
        return ak.to_numpy(ak.fill_none(ak.firsts(values, axis=1), np.nan)
                           ).astype(np.float64)

    def flat(values):
        return ak.to_numpy(ak.fill_none(values, np.nan)).astype(np.float64)

    return {
        "n_events": len(data["evt"]),
        "n_processed": int(data.get("n_processed", len(data["evt"]))),
        "flow": flow,
        "n_selected": int(keep.sum()),
        "p": one(first_p)[single] if single.any() else np.zeros(0),
        "t": one(first_t)[single] if single.any() else np.zeros(0),
        "p_true": flat(p_true)[single] if single.any() else np.zeros(0),
        "origin": origin_codes(sims)[single],
        "run": ak.to_numpy(evt["run"])[single],
        "subrun": ak.to_numpy(evt["subrun"])[single],
        "event": ak.to_numpy(evt["event"])[single],
    }


def combine(parts: list[dict]) -> dict:
    out = {key: np.concatenate([part[key] for part in parts])
           for key in ("p", "t", "p_true", "origin", "run", "subrun", "event")}
    for key in ("n_events", "n_processed", "n_selected"):
        out[key] = sum(part[key] for part in parts)
    out["flow"] = list(np.sum([part["flow"] for part in parts], axis=0)
                       ) if parts[0]["flow"] else []
    return out


def parse_time_window(text: str) -> tuple[float, float]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != 2 or not values[0] < values[1]:
        raise FullsimError(
            f"time_window {text!r} must be two comma-separated numbers "
            "'t_low,t_high' (ns) with the low below the high"
        )
    return values[0], values[1]


# --- the rates ---------------------------------------------------------------

def build_spectra(p_reco: np.ndarray, p_true: np.ndarray, n_generated: int,
                  upstream_eff: float, npot: float, mean_pot_per_event: float,
                  cosmic_rate_per_s_per_mev: float,
                  stopped_muons_per_pot: float = STOPPED_MUONS_PER_POT,
                  rmue: float = RMUE, cosmic_values=None) -> dict:
    """Signal, DIO and cosmic rates on one momentum binning, from the
    signal events' reconstructed and true momenta.

    Every count is per event generated, times upstream_eff, so the signal
    and the response carry the measured efficiency. Signal and DIO scale
    with the stopped mu- NPOT gives; cosmics with the on-spill time, flat
    at cosmic_rate_per_s_per_mev unless `cosmic_values` (a count per bin of
    the signal's binning, measured) is given.
    """
    per_event = upstream_eff / n_generated
    stopped_muons = npot * stopped_muons_per_pot

    signal = Spectrum.from_values(p_reco, *SIGNAL_BINNING, name="signal",
                                  title="Signal (CE)")
    signal = signal.rebin(SIGNAL_REBIN).scaled(
        stopped_muons * CAPTURES_PER_STOPPED_MUON * rmue * per_event)

    # A density in momentum offset, integrating to the efficiency.
    response = Spectrum.from_values(p_reco - p_true, *RESPONSE_BINNING,
                                    name="response")
    response = response.scaled(per_event / response.width)

    # load_dio_spectrum is a density per MeV; times its bin width it is a
    # count per bin, which is what regrid sums onto the signal's bins.
    dio_density = load_dio_spectrum()
    dio_true = dio_density.scaled(DIO_PER_STOPPED_MUON * stopped_muons
                                  * dio_density.width)
    dio = (dio_true.smear(Kernel.from_density(response, dio_true.width))
           .regrid(signal))

    onspill_seconds = npot / mean_pot_per_event * ONSPILL_SECONDS_PER_EVENT
    cosmic_rate = cosmic_rate_per_s_per_mev * onspill_seconds
    if cosmic_values is None:
        cosmic_values = np.full(signal.nbins, cosmic_rate * signal.width)
    cosmic = replace(signal, name="cosmic", title="Cosmics",
                     values=np.asarray(cosmic_values, dtype=np.float64))
    return {"signal": signal, "dio": dio, "cosmic": cosmic,
            "response": response, "dio_true": dio_true,
            "efficiency": float(p_reco.size * per_event),
            "stopped_muons": stopped_muons,
            "conversions": stopped_muons * CAPTURES_PER_STOPPED_MUON * rmue,
            "cosmic_rate_per_mev": cosmic_rate,
            "onspill_seconds": onspill_seconds}


# --- cut_mode="optimize" -----------------------------------------------------

def _optimize_selection(candidates, cut_set, presel, time_window, n_generated,
                        upstream_eff, spectra, rmue, reference_path):
    """Tune the selection on the inputs' CE and the cosmic reference
    (tune.py). Returns (tune.Tuned, opt_* metrics, extra)."""
    ce = tune.combine_tracks(candidates)
    keep = (ce["origin"] == CE_ORIGIN) & np.isfinite(ce["p"]) & np.isfinite(ce["p_true"])
    for name in tune.VARIABLE_NAMES:
        keep &= np.isfinite(ce[name])
    ce = tune.in_time({k: v[keep] for k, v in ce.items()}, time_window)
    if ce["p"].size == 0:
        raise tune.TuneError("no true CE passes the preselection in the time window")
    rows, meta = cosmic_reference.load(reference_path)
    flags = tuple(f for f in tune.EXTRA_PRESELECTION if f in presel)
    rate = recommended_cosmic_rate("box")
    livetime = float(meta.get("livetime_s") or 0.0)
    cosmic, per_mev = cosmic_reference.rows_for(rows, flags, rate,
                                                spectra["onspill_seconds"], livetime)
    # the time window cuts the cosmics too; their weight is per track of the
    # loose (untimed) selection, so dropping rows is the whole effect
    cosmic = tune.in_time(cosmic, time_window)
    per_event = upstream_eff / n_generated
    stopped = spectra["stopped_muons"]
    built = tune.build_rows(
        ce, cosmic,
        signal_per_event=stopped * CAPTURES_PER_STOPPED_MUON * rmue * per_event,
        dio_per_event=stopped * DIO_PER_STOPPED_MUON * per_event,
        cosmic_per_mev_per_row=per_mev, dio_density=load_dio_spectrum())
    seed = tune.seed_cuts(cut_set, time_window)
    result = tune.tune(built, seed)
    metrics = {
        "opt_punzi_train": result.best_train.fom,
        "opt_punzi_test": result.best_test.fom,
        "opt_seed_punzi_train": result.seed_train.fom,
        "opt_seed_punzi_test": result.seed_test.fom,
        "opt_gain_test": result.gain_test,
        "opt_signal_test": result.best_test.signal,
        "opt_background_test": result.best_test.background,
        "opt_seed_signal_test": result.seed_test.signal,
        "opt_seed_background_test": result.seed_test.background,
        "opt_n_ce_rows": float(ce["p"].size),
        "opt_n_cosmic_rows": float(cosmic["p"].size),
        "opt_cosmic_livetime_s": livetime if livetime > 0 else NOT_OPTIMIZED,
    }
    for key, window in (("opt_window", result.window), ("opt_seed_window", result.seed_window)):
        lo, hi = result.window_mev(window)
        metrics[f"{key}_low_mevc"], metrics[f"{key}_high_mevc"] = lo, hi
    for name, (lo, hi) in result.cuts.items():
        # an edge never placed (no cut) is NaN: a range end can equal
        # NOT_OPTIMIZED (-1, the low end of trkqual, trkpid and t0err)
        metrics[f"opt_cut_{name}_low"] = float(lo) if np.isfinite(lo) else UNPLACED_EDGE
        metrics[f"opt_cut_{name}_high"] = float(hi) if np.isfinite(hi) else UNPLACED_EDGE
    extra = {"cuts": {k: [float(a), float(b)] for k, (a, b) in result.cuts.items()},
             "seed_cuts": {k: [float(a), float(b)] for k, (a, b) in seed.items()},
             "preselection": list(presel),
             "time_window": [float(t) for t in time_window],
             "cosmic_reference": meta,
             # the loose rate only normalizes a reference without a livetime
             "cosmic_rate_per_s_per_mev": None if livetime > 0 else rate,
             "rounds": [{"window": list(result.window_mev(r["window"])),
                         "fom_train": r["fom_train"]} for r in result.rounds]}
    return result, metrics, extra


def _optimize_lines(result, extra) -> list[str]:
    def cut_text(cuts):
        return " && ".join(f"{lo:.5g} < {k} < {hi:.5g}" if np.isfinite(lo) and np.isfinite(hi)
                           else (f"{k} > {lo:.5g}" if np.isfinite(lo) else f"{k} < {hi:.5g}")
                           for k, (lo, hi) in cuts.items()
                           if np.isfinite(lo) or np.isfinite(hi))
    ref = extra["cosmic_reference"]
    return [
        "cut_mode = optimize: the preselection (" + ", ".join(extra["preselection"])
        + "), the time window {:g}-{:g} ns (fixed)".format(*extra["time_window"])
        + " and cuts tuned on the Punzi FoM, chosen on even events, reported on odd",
        f"  cosmic reference {ref.get('path')} ({ref.get('dataset', '?')}; "
        f"reco {ref.get('reco', '?')}; "
        + (f"CRY livetime {ref['livetime_s']:.4g} s from {ref.get('s2_events', 0):.4g} S2 events)"
           if ref.get("livetime_s") else
           f"{ref.get('n_normalizing', '?')} tracks normalizing to the loose rate)"),
        "  seed:  [{:.1f}, {:.1f}] MeV/c, ".format(*result.window_mev(result.seed_window))
        + cut_text(result.seed_cuts),
        f"         Punzi train {result.seed_train.fom:.4g}, test {result.seed_test.fom:.4g} "
        f"(S = {result.seed_test.signal:.3g}, B = {result.seed_test.background:.3g})",
        "  tuned: [{:.1f}, {:.1f}] MeV/c, ".format(*result.window_mev(result.window))
        + cut_text(result.cuts),
        f"         Punzi train {result.best_train.fom:.4g}, test {result.best_test.fom:.4g} "
        f"(S = {result.best_test.signal:.3g}, B = {result.best_test.background:.3g}); "
        f"gain on test {result.gain_test:.3f}",
    ]


# --- the measured cosmics ----------------------------------------------------

def find_cosmic_dataset(source: str, input_names: list[str]) -> str:
    """The cosmic mix dataset cosmic_source names, or for 'auto' the first
    of the ones matching the inputs that the catalogs know."""
    if source.lower() != cosmics.COSMIC_AUTO:
        return source
    candidates = cosmics.cosmic_datasets_for(input_names)
    for candidate in candidates:
        try:
            if dataset_file_names(candidate, catalog.CLIENT)[0]:
                return candidate
        except CatalogError:
            continue
    raise cosmics.CosmicError(
        f"no cosmic mix matching the inputs in SAM or metacat (tried "
        f"{', '.join(candidates)}); name one in cosmic_source, or 'rate' "
        "for the flat rate")


def measure_cosmics(params: dict, input_names: list[str], active: list[str],
                    triggers, time_window, onspill_seconds: float,
                    trkqual_min: float, trkpid_min: float,
                    cut_set: str) -> dict | None:
    """The cosmic background from the cosmic mix: its files reduced with the
    same cuts and thresholds as the inputs (trkqual_min/trkpid_min as
    resolved from the cut set), the cosmic tracks in the time window, and
    the weight each carries (on-spill time over the sample's livetime).
    None for 'rate'."""
    source = params["cosmic_source"].strip()
    if source.lower() == cosmics.COSMIC_RATE:
        return None
    shape = params["cosmic_shape"].strip().lower()
    if shape not in (cosmics.SHAPE_FLAT, cosmics.SHAPE_MEASURED):
        # Before reading the sample, which can take most of an hour.
        raise cosmics.CosmicError(
            f"cosmic_shape {shape!r} must be '{cosmics.SHAPE_FLAT}' or "
            f"'{cosmics.SHAPE_MEASURED}'")
    dataset = find_cosmic_dataset(source, input_names)
    max_files = int(params["cosmic_max_files"]) or None
    paths = resolve_inputs(dataset=dataset, max_files=max_files).paths
    names = [path.name for path in paths]
    if set(names) & set(input_names):
        raise cosmics.CosmicError(
            f"the cosmic sample {dataset} shares files with the inputs; the "
            "inputs must be the CE mix")
    s1_file = root_ancestor(names[0])
    tau = params["cosmic_livetime_per_event"] or cosmics.livetime_per_event(s1_file)
    s2_events = sum(generated_events_bulk(names).values())
    livetime = s2_events * tau
    ev = combine([reduce_file(path, active, trkqual_min, trkpid_min, triggers,
                              cut_set) for path in paths])
    counted = cosmics.cosmic_counts(ev, time_window)
    like = Spectrum.from_values(np.zeros(0), *SIGNAL_BINNING).rebin(SIGNAL_REBIN)
    weight = onspill_seconds / livetime
    spectrum = cosmics.cosmic_spectrum(ev["p"][counted], weight, like, shape)
    return {"dataset": dataset, "n_files": len(paths), "s1_file": s1_file,
            "livetime_per_event": tau, "s2_events": s2_events,
            "livetime": livetime, "weight": weight, "ev": ev,
            "counted": counted, "n_counted": int(counted.sum()),
            "shape": shape, **spectrum}


COSMIC_METRICS = ("cosmic_livetime_s", "cosmic_weight", "n_cosmic_files",
                  "n_cosmic_events", "n_cosmic_counted",
                  "n_cosmic_flat_range", "cosmic_upper_limit")


def cosmic_metrics(cosmic: dict | None) -> dict[str, float]:
    if cosmic is None:
        return {name: -1.0 for name in COSMIC_METRICS}
    return {
        "cosmic_livetime_s": float(cosmic["livetime"]),
        "cosmic_weight": float(cosmic["weight"]),
        "n_cosmic_files": float(cosmic["n_files"]),
        "n_cosmic_events": float(cosmic["ev"]["n_events"]),
        "n_cosmic_counted": float(cosmic["n_counted"]),
        "n_cosmic_flat_range": float(cosmic["n_flat_range"]),
        "cosmic_upper_limit": float(cosmic["upper_limit"]),
    }


def cosmic_log_lines(cosmic: dict | None, rate: float, spectra: dict) -> list[str]:
    onspill = f"({spectra['onspill_seconds']:.4g} s on-spill)"
    if cosmic is None:
        return [f"  cosmic rate      {rate:.4g} per s per MeV/c (cosmic_source "
                f"'rate') -> {spectra['cosmic_rate_per_mev']:.4g} per MeV/c {onspill}"]
    lo, hi = cosmics.COSMIC_FLAT_RANGE
    seen = (f"{cosmics.ZERO_COUNT_UPPER_LIMIT:g} (90% CL limit: none seen)"
            if cosmic["upper_limit"] else str(cosmic["n_flat_range"]))
    return [
        f"  cosmic sample    {cosmic['dataset']}, {cosmic['n_files']} files, "
        f"{cosmic['ev']['n_events']} events",
        f"    livetime       {cosmic['s2_events']:.6g} S2 events resampled x "
        f"{cosmic['livetime_per_event']:.5g} s per S1 event (S1 "
        f"{cosmic['s1_file']}) = {cosmic['livetime']:.5g} s",
        f"    weight         {cosmic['weight']:.4g} per cosmic {onspill}",
        f"    counted        {cosmic['n_counted']} cosmic tracks pass the cuts "
        f"in the time window, {cosmic['n_flat_range']} in {lo:g}-{hi:g} MeV/c",
        f"    rate           {seen} x weight / {hi - lo:g} MeV/c = "
        f"{cosmic['rate_per_mev']:.4g} per MeV/c ({rate:.4g} per s per MeV/c); "
        f"shape {cosmic['shape']}",
    ]


# --- plots -------------------------------------------------------------------

def _write_plots(outdir: Path, spectra: dict, best: dict, ev: dict,
                 is_signal: np.ndarray, time_window, cosmic_sample=None) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []

    def save(fig, name):
        fig.tight_layout()
        path = figdir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    signal, dio, cosmic = spectra["signal"], spectra["dio"], spectra["cosmic"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for spectrum, color, label in ((signal, "tab:blue", "Signal (CE)"),
                                   (dio, "tab:red", "DIO"),
                                   (cosmic, "tab:green", "Cosmic")):
        ax.stairs(spectrum.values, spectrum.edges(), color=color, lw=1.8, label=label)
    ax.axvspan(best["low_mev"], best["high_mev"], color="0.85", zorder=0,
               label=f"window [{best['low_mev']:.1f}, {best['high_mev']:.1f}]")
    ax.set_yscale("log")
    ax.set_xlim(*PLOT_MOM_RANGE)
    ax.set_ylim(1e-6, 1e2)
    ax.set_xlabel("Reconstructed momentum at tracker front (MeV/c)")
    ax.set_ylabel(f"Rate / {signal.width:.1f} MeV/c")
    ax.set_title(f"Signal vs. background — S/sqrt(B) = {best['sensitivity']:.3g}")
    ax.legend(ncols=2, fontsize="small")
    save(fig, "sig_vs_bkg.png")

    response = spectra["response"]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.stairs(response.values, response.edges(), color="tab:blue", lw=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("Reconstructed minus true momentum (MeV/c)")
    ax.set_ylabel("Density (per event processed, per MeV/c)")
    ax.set_title("Measured CE response")
    save(fig, "response.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ev["p"][~is_signal], ev["t"][~is_signal], s=8, color="tab:red",
               alpha=0.5, label="Not CE (pileup)")
    ax.scatter(ev["p"][is_signal], ev["t"][is_signal], s=6, color="tab:blue",
               alpha=0.5, label="True CE")
    if cosmic_sample is not None:
        from .eventntuple import ORIGIN_COSMIC
        cev = cosmic_sample["ev"]
        mine = cev["origin"] == ORIGIN_COSMIC
        ax.scatter(cev["p"][mine], cev["t"][mine], s=14, marker="x",
                   color="tab:green", label="Cosmic (cosmic mix)")
    t_lo, t_hi = time_window
    ax.add_patch(matplotlib.patches.Rectangle(
        (best["low_mev"], t_lo), best["high_mev"] - best["low_mev"], t_hi - t_lo,
        fill=False, color="k", lw=1.5))
    ax.set_xlim(*PLOT_MOM_RANGE)
    ax.set_ylim(400.0, 1700.0)
    ax.set_xlabel("Reconstructed momentum at tracker front (MeV/c)")
    ax.set_ylabel("Time at tracker front (ns)")
    ax.legend(fontsize="small")
    save(fig, "mom_vs_time.png")
    return written


# --- the runner --------------------------------------------------------------

def run(context: RunContext) -> RunOutcome:
    params = context.params
    outdir = context.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    npot = params["npot"]
    upstream_eff = params["upstream_eff"]
    cosmic_rate_per_s_per_mev = params["cosmic_rate_per_s_per_mev"]
    stops_per_pot = params["stopped_muons_per_pot"]
    n_generated = int(params["n_generated"])
    rmue = params["rmue"]
    extra: dict = {}

    try:
        cut_set = params["cut_set"].strip()
        cs = get_cut_set(cut_set)
        # A negative threshold means the cut set's own.
        trkqual_min = (cs.trkqual_min if params["trkqual_min"] < 0
                       else params["trkqual_min"])
        trkpid_min = (cs.trkpid_min if params["trkpid_min"] < 0
                      else params["trkpid_min"])
        active = active_cuts(params["enable_cuts"], params["disable_cuts"], cut_set)
        extra["cut_set"] = cut_set
        extra["trkqual_min"] = trkqual_min
        extra["trkpid_min"] = trkpid_min
        extra["cuts"] = active
        triggers = parse_trigger_paths(params["trigger_paths"])
        extra["trigger_paths"] = triggers
        t_lo, t_hi = (parse_time_window(params["time_window"])
                      if params["time_window"].strip() else cs.time_window_ns)
        extra["time_window"] = [t_lo, t_hi]
        cut_mode = params["cut_mode"].strip()
        if cut_mode not in CUT_MODES:
            raise FullsimError(f"cut_mode must be one of {', '.join(CUT_MODES)}, "
                               f"got {cut_mode!r}")
        extra["cut_mode"] = cut_mode
        # optimize: the active cuts less the tuned and target ones, checked
        # against what the cosmic reference can follow before any file is read
        presel = tune.optimize_preselection(active) if cut_mode == "optimize" else ()
        parts, candidates = [], []
        for path in context.input_paths:
            data = read_eventntuple(path, triggers)
            parts.append(reduce_data(data, active, trkqual_min, trkpid_min,
                                     triggers, cut_set))
            if cut_mode == "optimize":
                candidates.append(tune.candidate_tracks(
                    data, presel, triggers, cut_set))
            del data
        ev = combine(parts)
        if n_generated <= 0:
            n_generated = sum(generated_events(path.name)
                              for path in context.input_paths)
        extra["n_generated_from"] = ("parameter" if params["n_generated"] > 0
                                     else "SAM dh.gencount")
        if stops_per_pot <= 0:
            if params["stop_datasets"].strip().lower() == STOP_DATASETS_AUTO:
                datasets = stop_chain_of_inputs(
                    [path.name for path in context.input_paths])
                extra["stop_chain_from"] = "SAM ancestry of the inputs"
            else:
                datasets = [d.strip() for d in params["stop_datasets"].split(",")
                            if d.strip()]
                extra["stop_chain_from"] = "stop_datasets"
            stages = sim_efficiencies(datasets)
            stops_per_pot = stopped_muons_per_pot(stages)
            extra["sim_efficiencies"] = {
                stage: {"dataset": e.dataset, "n_files": e.n_files,
                        "passed": e.passed, "generated": e.generated,
                        "efficiency": e.efficiency}
                for stage, e in stages.items()}
        extra["stopped_muons_per_pot_from"] = (
            "parameter" if params["stopped_muons_per_pot"] > 0
            else f"SAM, chain from {extra['stop_chain_from']}")
        if n_generated < ev["n_processed"]:
            raise FullsimError(
                f"n_generated {n_generated} is fewer than the {ev['n_processed']} "
                "events the ntuple job processed: it must count every event "
                "generated to make the input files"
            )
        is_signal = ev["origin"] == CE_ORIGIN
        in_time = (ev["t"] > t_lo) & (ev["t"] < t_hi)
        counted = is_signal & in_time & np.isfinite(ev["p"]) & np.isfinite(ev["p_true"])
        if not counted.any():
            raise FullsimError(
                f"no true CE passes the cuts ({', '.join(active) or 'none'}) "
                f"inside {t_lo:g}-{t_hi:g} ns in {ev['n_events']} events. The "
                "input must be reconstructed CE- signal mixed with pileup, e.g. "
                "CeMLeadingLogMix1BB EventNtuple (nts.*.root) files; or loosen "
                "the cuts with disable_cuts."
            )
        onspill_seconds = (npot / params["mean_pot_per_event"]
                           * ONSPILL_SECONDS_PER_EVENT)
        cosmic = measure_cosmics(params, [path.name for path in context.input_paths],
                                 active, triggers, (t_lo, t_hi), onspill_seconds,
                                 trkqual_min, trkpid_min, cut_set)
        if cosmic is not None:
            # The rate the measurement amounts to, per second per MeV/c of
            # on-spill time, so the metric means what it did with 'rate'.
            cosmic_rate_per_s_per_mev = cosmic["rate_per_mev"] / onspill_seconds
            extra["cosmic_sample"] = {
                key: cosmic[key] for key in
                ("dataset", "n_files", "s1_file", "livetime_per_event",
                 "s2_events", "livetime", "weight", "n_counted",
                 "n_flat_range", "upper_limit")}
        spectra = build_spectra(ev["p"][counted], ev["p_true"][counted],
                                n_generated, upstream_eff, npot,
                                params["mean_pot_per_event"],
                                cosmic_rate_per_s_per_mev,
                                stops_per_pot, rmue,
                                None if cosmic is None else cosmic["values"])
        best, top = scan_signal_box(spectra["signal"], spectra["dio"],
                                    spectra["cosmic"])
    except (FullsimError, EventNtupleError, CutError, SensitivityError,
            cosmics.CosmicError) as exc:
        return RunOutcome(error=str(exc), extra=extra)
    except CatalogError as exc:
        return RunOutcome(error=(
            f"cosmic sample: {exc}. Or set cosmic_source to 'rate' for the "
            "flat cosmic rate"), extra=extra)
    except tune.TuneError as exc:
        return RunOutcome(error=f"cut_mode='optimize': {exc}", extra=extra)
    except NormalizationError as exc:
        return RunOutcome(error=(
            f"{exc}. Stopped mu- per POT is computed from the stop chain's "
            f"SAM datasets ({', '.join(STOP_CHAIN)}), traced from the inputs' "
            "ancestry unless stop_datasets names them (e.g. "
            f"'{', '.join(STOP_CHAIN_DATASETS)}'); without SAM, "
            "pass stopped_muons_per_pot instead (MuBeamCat x MuminusStopsCat "
            f"efficiency x 1000; {STOPPED_MUONS_PER_POT:.4g} for the MDC2025 chain)"
        ), extra=extra)
    except ProvenanceError as exc:
        return RunOutcome(error=(
            f"{exc}. The efficiency counts against the events generated, and "
            "the cosmic livetime against the cosmic sample's, both from SAM; "
            "without SAM, pass n_generated (the sum of dh.gencount over the "
            "input files' parent mcs files) and cosmic_source 'rate'"
        ), extra=extra)

    in_window = counted & (ev["p"] >= best["low_mev"]) & (ev["p"] <= best["high_mev"])
    efficiency_window = best["signal"] / spectra["conversions"]

    metrics = {
        "sensitivity": float(best["sensitivity"]),
        "signal_mom_low_mevc": float(best["low_mev"]),
        "signal_mom_high_mevc": float(best["high_mev"]),
        "signal_time_low_ns": float(t_lo),
        "signal_time_high_ns": float(t_hi),
        "signal_rate": float(best["signal"]),
        "dio_background": float(best["dio"]),
        "cosmic_background": float(best["cosmic"]),
        "total_background": float(best["background"]),
        "signal_efficiency": spectra["efficiency"],
        "signal_efficiency_window": float(efficiency_window),
        "n_signal_window": float(in_window.sum()),
        "n_events": float(ev["n_events"]),
        "n_events_processed": float(ev["n_processed"]),
        "n_events_generated": float(n_generated),
        "acceptance": float(ev["n_processed"] / n_generated),
        "n_events_selected": float(ev["n_selected"]),
        "n_events_counted": float(counted.sum()),
        "n_files": float(len(context.input_paths)),
        # The assumptions the rates are built on, reported with them so a
        # number never travels without the normalization behind it.
        "npot": float(npot),
        "upstream_eff": float(upstream_eff),
        "cosmic_rate_per_s_per_mev": float(cosmic_rate_per_s_per_mev),
        "stopped_muons_per_pot": float(stops_per_pot),
        "n_stopped_muons": float(spectra["stopped_muons"]),
        "rmue": float(rmue),
        "n_conversions": float(spectra["conversions"]),
        # The cosmic sample behind cosmic_background; -1 with cosmic_source
        # 'rate', which measures none.
        **cosmic_metrics(cosmic),
    }

    metrics.update({name: NOT_OPTIMIZED for name in OPT_METRICS})
    tuned = None
    if cut_mode == "optimize":
        try:
            tuned, opt_metrics, opt_extra = _optimize_selection(
                candidates, cut_set, presel, (t_lo, t_hi), n_generated, upstream_eff,
                spectra, rmue, params["cosmic_reference"])
        except (tune.TuneError, cosmic_reference.ReferenceError) as exc:
            return RunOutcome(error=f"cut_mode='optimize': {exc}", extra=extra)
        metrics.update(opt_metrics)
        extra["optimize"] = opt_extra

    flow_rows = [("No cuts", ev["n_events"], "no selection")] + [
        (name, n, cut_descriptions(cut_set)[name])
        for name, n in zip(active, ev["flow"])]
    flow_path = outdir / "cut_flow.csv"
    flow_path.write_text(
        "cut,events_passing,description\n"
        + "".join(f"{name},{n},\"{desc}\"\n" for name, n, desc in flow_rows),
        encoding="utf-8")

    def by_origin(mask):
        codes, counts = np.unique(ev["origin"][mask], return_counts=True)
        return {ORIGIN_NAMES.get(int(c), str(int(c))): int(n)
                for c, n in zip(codes, counts)}

    selected_by_origin = by_origin(np.ones(ev["origin"].size, dtype=bool))
    log_path = outdir / "fullsim_sensitivity.log"
    lines = [
        "fullsim_sensitivity",
        *(f"  input            {path}" for path in context.input_paths),
        f"  cut set          pyfitter {cut_set}",
        f"  trigger paths    any of {', '.join(triggers)}",
        f"  time window      {t_lo:g}-{t_hi:g} ns at the tracker front",
        f"  NPOT             {npot:g}",
        f"  stopped mu-/POT  {stops_per_pot:.4g} -> "
        f"{spectra['stopped_muons']:.4g} stopped mu- "
        f"({extra['stopped_muons_per_pot_from']})",
        *(f"    {stage:<14s} {e['passed']} / {e['generated']} = "
          f"{e['efficiency']:.6g} ({e['dataset']}, {e['n_files']} files)"
          for stage, e in extra.get("sim_efficiencies", {}).items()),
        f"  R_mue            {rmue:.4g} -> {spectra['conversions']:.4g} CE "
        f"({CAPTURES_PER_STOPPED_MUON:g} captures, {DIO_PER_STOPPED_MUON:g} "
        "DIO per stopped mu-)",
        f"  upstream_eff     {upstream_eff:g}",
        *cosmic_log_lines(cosmic, cosmic_rate_per_s_per_mev, spectra),
        "",
        "Cut flow (events with a track passing every cut so far):",
        *(f"  {name:<28s} {n:>9d}   {desc}" for name, n, desc in flow_rows),
        "",
        f"Selected events with one tracker-front segment, by origin: {selected_by_origin}",
        f"{int(counted.sum())} true CE in the time window, of "
        f"{n_generated} generated ({extra['n_generated_from']}; "
        f"{ev['n_processed']} reached the ntuple, acceptance "
        f"{ev['n_processed'] / n_generated:.4g}): efficiency "
        f"{spectra['efficiency']:.4g} (with upstream_eff).",
        "",
        f"Best window [{best['low_mev']:.1f}, {best['high_mev']:.1f}] MeV/c: "
        f"S = {best['signal']:.3g}, DIO = {best['dio']:.3g}, "
        f"cosmic = {best['cosmic']:.3g} -> B = {best['background']:.3g}, "
        f"S/sqrt(B) = {best['sensitivity']:.4g}",
        "",
        f"Top {len(top)} windows scanned:",
        *(f"  [{e['low_mev']:6.1f}, {e['high_mev']:6.1f}] MeV/c  S = {e['signal']:9.3g}  "
          f"DIO = {e['dio']:9.3g}  cosmic = {e['cosmic']:9.3g}  "
          f"S/sqrt(B) = {e['sensitivity']:.4g}" for e in top),
    ]
    if tuned is not None:
        lines += ["", *_optimize_lines(tuned, extra["optimize"])]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = [str(flow_path)]
    files += _write_plots(outdir, spectra, best, ev, is_signal, (t_lo, t_hi),
                          cosmic)

    extra.update({
        "cut_flow": {name: int(n) for name, n, _ in flow_rows},
        "selected_by_origin": selected_by_origin,
        "captures_per_stopped_muon": CAPTURES_PER_STOPPED_MUON,
        "dio_per_stopped_muon": DIO_PER_STOPPED_MUON,
        "cosmic_rate_per_mev": spectra["cosmic_rate_per_mev"],
        "onspill_seconds": spectra["onspill_seconds"],
        "top_windows": top,
    })
    return RunOutcome(metrics=metrics, files=files, log_path=log_path, extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"S/sqrt(B) = {metrics['sensitivity']:.4g} in "
        f"[{metrics['signal_mom_low_mevc']:.1f}, {metrics['signal_mom_high_mevc']:.1f}] MeV/c x "
        f"[{metrics['signal_time_low_ns']:g}, {metrics['signal_time_high_ns']:g}] ns "
        f"(S = {metrics['signal_rate']:.3g}, B = {metrics['total_background']:.3g}: "
        f"DIO {metrics['dio_background']:.3g}, cosmic {metrics['cosmic_background']:.3g}) "
        f"for {metrics['npot']:.3g} POT ({metrics['n_stopped_muons']:.3g} stopped "
        f"mu-, R_mue = {metrics['rmue']:.3g}); signal efficiency "
        f"{metrics['signal_efficiency']:.3g} from "
        f"{metrics['n_events_counted']:.0f} reconstructed CE; "
        + (f"cosmics measured: {metrics['n_cosmic_flat_range']:.0f} cosmic "
           f"tracks in {cosmics.COSMIC_FLAT_RANGE[0]:g}-"
           f"{cosmics.COSMIC_FLAT_RANGE[1]:g} MeV/c in "
           f"{metrics['cosmic_livetime_s']:.3g} s of cosmics"
           + (" (none: 90% CL upper limit used)." if metrics["cosmic_upper_limit"] > 0
              else ".")
           if metrics["n_cosmic_files"] > 0 else "cosmics from the flat rate.")
    )


SPEC = AnalysisSpec(
    name="fullsim_sensitivity",
    description=(
        "Full-simulation CE sensitivity S/sqrt(B), the counterpart of "
        "approx_ce_sensitivity: the signal efficiency, momentum shape and "
        "detector response are measured from reconstructed CE mixed with "
        "pileup (CeMLeadingLogMix1BB EventNtuple files) after pyfitter's "
        "CE-like cuts (run1a_tsda). Signal and DIO (theory spectrum "
        "folded with that response) are normalized to the stopped mu- NPOT "
        "gives, as Production's normalizations.py; cosmics are measured on "
        "the matching official cosmic mix (same cuts, livetime from SAM and "
        "the S1 sample) unless cosmic_source is 'rate'; the best-window "
        "scan is approx_ce_sensitivity's."
    ),
    input_kind="root_file",
    combines_files=True,
    metrics=(
        "sensitivity",
        "signal_mom_low_mevc", "signal_mom_high_mevc",
        "signal_time_low_ns", "signal_time_high_ns",
        "signal_rate", "dio_background", "cosmic_background", "total_background",
        "signal_efficiency", "signal_efficiency_window", "n_signal_window",
        "n_events", "n_events_processed", "n_events_selected", "n_events_counted",
        "n_events_generated", "acceptance",
        "n_files", "npot", "upstream_eff", "cosmic_rate_per_s_per_mev",
        "stopped_muons_per_pot", "n_stopped_muons", "rmue", "n_conversions",
        *OPT_METRICS, *COSMIC_METRICS,
    ),
    units={
        "signal_mom_low_mevc": "MeV/c", "signal_mom_high_mevc": "MeV/c",
        "signal_time_low_ns": "ns", "signal_time_high_ns": "ns",
        "n_signal_window": "events", "n_events": "events",
        "n_events_processed": "events", "n_events_selected": "events",
        "n_events_generated": "events",
        "n_events_counted": "events", "n_files": "files", "npot": "POT",
        "cosmic_rate_per_s_per_mev": "per second per MeV/c",
        "stopped_muons_per_pot": "per POT", "n_stopped_muons": "muons",
        "n_conversions": "events",
        "cosmic_livetime_s": "s", "cosmic_weight": "run events per cosmic event",
        "n_cosmic_files": "files", "n_cosmic_events": "events",
        "n_cosmic_counted": "events", "n_cosmic_flat_range": "events",
        "opt_signal_test": "events", "opt_background_test": "events",
        "opt_seed_signal_test": "events", "opt_seed_background_test": "events",
        "opt_window_low_mevc": "MeV/c", "opt_window_high_mevc": "MeV/c",
        "opt_seed_window_low_mevc": "MeV/c", "opt_seed_window_high_mevc": "MeV/c",
        "opt_n_ce_rows": "events", "opt_n_cosmic_rows": "tracks",
        "opt_cosmic_livetime_s": "s",
        "opt_cut_rmax_low": "mm", "opt_cut_rmax_high": "mm",
        "opt_cut_d0_low": "mm", "opt_cut_d0_high": "mm",
        "opt_cut_t0err_low": "ns", "opt_cut_t0err_high": "ns",
    },
    parameters=(
        ParamSpec(
            name="npot",
            description="Protons on target to assume for the rate normalization.",
            default=NPOT, minimum=0.0,
        ),
        ParamSpec(
            name="stopped_muons_per_pot",
            description="mu- stopped in the target per proton on target. "
                        "Signal and DIO scale with npot times this. 0 (the "
                        "default) computes it in SAM, from the stop chain "
                        "stop_datasets gives: "
                        "MuBeamCat x MuminusStopsCat efficiency x 1000, as "
                        "Production's normalizations.py, each efficiency "
                        "being events in the dataset over events generated, "
                        "as CreateSimEfficiency.sh (mu2eGenFilterEff) has "
                        "it. Set a number to skip SAM: "
                        f"{STOPPED_MUONS_PER_POT:.4g} for the MDC2025 chain.",
            default=0.0, minimum=0.0,
        ),
        ParamSpec(
            name="stop_datasets",
            description="The stop chain stopped_muons_per_pot is computed "
                        "from when it is 0. 'auto' (the default) traces it "
                        "from the input files' SAM ancestry: the "
                        "MuminusStopsCat dataset the CE were generated from, "
                        "and the MuBeamCat dataset that came from, so a new "
                        "iteration of the simulation needs no database "
                        "table and no setting here. Or comma-separated SAM "
                        f"datasets, one per stage ({', '.join(STOP_CHAIN)}), "
                        f"e.g. '{', '.join(STOP_CHAIN_DATASETS)}' (the chain "
                        "of the MDC2025au_best_v1_1 CE mix), for inputs "
                        "whose ancestry is not in SAM.",
            default=STOP_DATASETS_AUTO, kind="text",
        ),
        ParamSpec(
            name="rmue",
            description="mu- -> e- conversion rate relative to muon capture "
                        "(R_mue) assumed for the signal. The expected CE "
                        "count is npot x stopped_muons_per_pot x "
                        f"{CAPTURES_PER_STOPPED_MUON:g} (captures per stopped "
                        "mu-) x rmue, times the measured efficiency.",
            default=RMUE, minimum=0.0,
        ),
        ParamSpec(
            name="mean_pot_per_event",
            description="Mean number of protons on target per event; sets the "
                        "on-spill live time the cosmic background is scaled to.",
            default=MEAN_POT_PER_EVENT, minimum=1.0,
        ),
        ParamSpec(
            name="cosmic_source",
            description="Where the cosmic background comes from. 'auto' (the "
                        "default) measures it on the official cosmic mix that "
                        "matches the inputs (same configuration and mix, "
                        "description CosmicSignal or CosmicCRYAll + the mix, "
                        "e.g. nts.mu2e.CosmicSignalMix1BB.MDC2025au_best_v1_1-001"
                        ".root for the CeMLeadingLogMix1BB one): its files "
                        "pass the same cuts, trigger and time window, the "
                        "events whose first track a cosmic made are counted, "
                        "and each stands for the on-spill time over the "
                        "sample's cosmic livetime. A SAM dataset name measures "
                        "on that sample instead. 'rate' uses the flat "
                        "cosmic_rate_per_s_per_mev and reads no cosmic files. "
                        "Measuring reads every file of the sample (500 for "
                        "MDC2025au, ~6 s each) unless cosmic_max_files caps it.",
            default=cosmics.COSMIC_AUTO, kind="text",
        ),
        ParamSpec(
            name="cosmic_max_files",
            description="Read only the first this-many files (by name) of the "
                        "cosmic sample; the livetime counts only those. 0 (the "
                        "default) reads them all. Each MDC2025au "
                        "CosmicSignalMix1BB file is ~11,400 s of cosmics, about "
                        "a tenth of the on-spill time of 1e18 POT.",
            default=0, minimum=0.0,
        ),
        ParamSpec(
            name="cosmic_shape",
            description="The cosmic momentum spectrum the window scan sees. "
                        "'flat' (the default): the cosmic tracks counted in "
                        f"{cosmics.COSMIC_FLAT_RANGE[0]:g}-"
                        f"{cosmics.COSMIC_FLAT_RANGE[1]:g} MeV/c spread evenly "
                        "per MeV/c, or the 90% CL upper limit "
                        f"({cosmics.ZERO_COUNT_UPPER_LIMIT:g} events) when "
                        "none is, so the scan cannot chase gaps between single "
                        "events. 'measured': the counted tracks binned as they "
                        "are.",
            default=cosmics.SHAPE_FLAT, kind="text",
        ),
        ParamSpec(
            name="cosmic_livetime_per_event",
            description="Seconds of cosmic exposure per event of the S1 CRY "
                        "sample the cosmic mix was resampled from (its summed "
                        "CosmicLivetime over its GenEventCount). 0 (the "
                        "default) takes it from the measured table by the S1 "
                        "dataset the cosmic files trace to in SAM ("
                        + ", ".join(f"{d}: {t:.5g}" for d, t in
                                    cosmics.S1_LIVETIME_PER_EVENT.items())
                        + "). The livetime stored in resampled cosmic files "
                        "is truncated by a Mu2eProductMixer bug and is not "
                        "used.",
            default=0.0, minimum=0.0,
        ),
        ParamSpec(
            name="cosmic_rate_per_s_per_mev",
            description="With cosmic_source 'rate': the cosmic-ray background "
                        "rate, flat in momentum, per second per MeV/c, as in "
                        "approx_ce_sensitivity (whose default this is: the "
                        "rough Run-1A mu- -> e- number), scaled by the "
                        "on-spill live time npot and mean_pot_per_event imply. "
                        "Ignored otherwise; the metric of the same name then "
                        "reports the measured rate.",
            default=COSMIC_RATE_PER_SECOND_PER_MEV, minimum=0.0,
        ),
        ParamSpec(
            name="n_generated",
            description="Events generated to make the input files: the "
                        "efficiency's denominator, so it includes the "
                        "digitization filter's acceptance. 0 (the default) "
                        "looks it up in SAM: the dh.gencount of each input "
                        "file's nearest ancestor that has one (the parent "
                        "mcs file), summed. Set it where SAM cannot be "
                        "reached.",
            default=0, minimum=0.0,
        ),
        ParamSpec(
            name="upstream_eff",
            description="An extra efficiency factor, for a loss before "
                        "generation that dh.gencount does not count. The "
                        "efficiency is measured over the events generated "
                        "(n_generated) and multiplied by this. 1, the "
                        "default, adds nothing.",
            default=1.0, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="time_window",
            description="'t_low,t_high' (ns): the window a CE's time at the "
                        "tracker front must fall in (open interval). Empty, "
                        "the default: cut_set's ("
                        + "; ".join("{}: {:g},{:g}".format(n, *c.time_window_ns)
                                    for n, c in CUT_SETS.items())
                        + "). The momentum window is scanned.",
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="trigger_paths",
            description="Comma-separated trigger paths for the or_trigger "
                        "cut: an event passes if any of them fired. Names as "
                        "in the trigger menu, with or without the trig_ "
                        "prefix of their EventNtuple branch. Default: the "
                        "production tracker paths (APR or CPR), as pyfitter.",
            default=", ".join(DEFAULT_TRIGGERS), kind="text",
        ),
        ParamSpec(
            name="cut_mode",
            description="'fixed' (default): the cut set as it stands. "
                        "'optimize': also tune the selection on these inputs "
                        "(the active cuts' vetoes, as enable_cuts and "
                        "disable_cuts leave them, no stopping-target cuts, "
                        "the time window fixed, and rectangular cuts on "
                        "cos(theta), R(max), d0, TrkQual, TrkPID, t0err and "
                        "N(active)) on the Punzi FoM, against a fixed "
                        "reconstructed cosmic reference (cosmic_reference; "
                        "electrons and muons) and the DIO folded per event; "
                        "chosen on even events, reported on odd ones as the "
                        "opt_* metrics (-1 in fixed mode; an opt_cut_* edge "
                        "left open is NaN). The default reference is CRY "
                        "signal-region cosmics (MDC2025ap) reconstructed "
                        "primary-only, without pileup, with "
                        "Code_ana_v020301_fcl_mdc2025bc: its TrkQual, TrkPID, "
                        "N(active) and t0err match inputs reconstructed the "
                        "same way; for mixed (Mix1BB) or other-release inputs "
                        "they are approximate, and a matching reference should "
                        "be made. The fixed result is reported either way.",
            default="fixed", kind="text",
        ),
        ParamSpec(
            name="cosmic_reference",
            description="cut_mode='optimize': the cosmic reference table "
                        "(.npz, from scripts/make_fullsim_cosmic_reference.py). "
                        "Empty: data/fullsim_cosmic_reference.npz.",
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="cut_set",
            description="The pyfitter cut set (config.py VERSION_CUTS) the "
                        "thresholds and default cuts come from. Only "
                        "run1a_tsda is supported (the Run 1A selection with "
                        "the signed-d0 window and the TSDA veto, which needs "
                        "trk.tsdainter): "
                        + ", ".join(default_cuts(CUT_SET)) + ".",
            default=CUT_SET, kind="text",
        ),
        ParamSpec(
            name="enable_cuts",
            description="Comma-separated cuts to apply on top of cut_set's "
                        "defaults. Known cuts: " + ", ".join(CUT_ORDER) + ". "
                        "in_mom_range and within_t0 are pyfitter's fit ranges, "
                        "which the momentum scan and time_window stand in for.",
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="disable_cuts",
            description="Comma-separated cuts to drop from the default set, "
                        "e.g. 'has_st,no_opa'.",
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="trkqual_min",
            description="good_trkqual threshold: keep tracks with TrkQual "
                        "above this. Negative (the default): cut_set's "
                        f"({', '.join(f'{n} {c.trkqual_min:g}' for n, c in CUT_SETS.items())}).",
            default=-1.0, minimum=-1.0, maximum=1.0,
        ),
        ParamSpec(
            name="trkpid_min",
            description="good_trkpid threshold: keep tracks with TrkPID above "
                        "this. Negative (the default): cut_set's "
                        f"({', '.join(f'{n} {c.trkpid_min:g}' for n, c in CUT_SETS.items())}).",
            default=-1.0, minimum=-1.0, maximum=1.0,
        ),
    ),
    input_hint=(
        f"Reconstructed EventNtuple file(s) (nts.*.root with {TREE_PATH}) of "
        "CE- signal mixed with pileup, with MC truth, e.g. "
        "/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/"
        "MDC2025au_best_v1_1-001/root/*/*/nts.*.root. Several files are "
        "combined into one measurement. Each takes a few seconds per 10k events."
    ),
    run=run,
    summarize=summarize,
)
