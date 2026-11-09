"""Full-simulation CE sensitivity: S/sqrt(B) in the best momentum window.

The full-simulation counterpart of approx_ce_sensitivity, built the same way
but with the signal efficiency, shape and detector response measured from
reconstructed CE events mixed with pileup (CeMLeadingLogMix1BB EventNtuple
files) instead of from EdepAna's truth-level tree:

1. Select CE-like tracks with pyfitter's cut-set 80 (cuts.py), file by file.
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
4. Cosmic background: flat in momentum at `cosmic_rate_per_s_per_mev`,
   scaled by the on-spill live time NPOT implies, as approx_ce_sensitivity
   has it.
5. Scan every momentum window with approx_ce_sensitivity's scan_signal_box
   and keep the one maximizing S/sqrt(B).
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from ...eventntuple import ORIGIN_NAMES, SID_TT_FRONT, TREE_PATH
from ...spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from ...spectrum import Kernel, Spectrum
from ..approx_ce_sensitivity import (COSMIC_RATE_PER_SECOND_PER_MEV,
                                     MEAN_POT_PER_EVENT, NPOT,
                                     ONSPILL_SECONDS_PER_EVENT, SIGNAL_REBIN,
                                     SensitivityError, load_dio_spectrum,
                                     scan_signal_box)
from .cuts import (CUT_DESCRIPTIONS, CUT_SET, DEFAULT_CUTS, TRKPID_MIN,
                   TRKQUAL_MIN, CutError, active_cuts, apply_cuts, cut_masks)
from .eventntuple import (DEFAULT_TRIGGERS, EventNtupleError, origin_codes,
                          parse_trigger_paths, read_eventntuple)
from .normalization import (CAPTURES_PER_STOPPED_MUON, DIO_PER_STOPPED_MUON,
                            STOP_CHAIN, STOP_CHAIN_DATASETS,
                            STOPPED_MUONS_PER_POT, NormalizationError,
                            sim_efficiencies, stop_chain_of_inputs,
                            stopped_muons_per_pot)
from .provenance import ProvenanceError, generated_events

SIGN = "minus"
# stop_datasets' value for tracing the stop chain from the inputs.
STOP_DATASETS_AUTO = "auto"
# The origin label of a true CE- (eventntuple.ORIGIN_NAMES).
CE_ORIGIN = 168

# The mu- -> e- conversion rate relative to capture, R_mue, assumed.
RMUE = 1.0e-13

# The time window at the tracker front (ns) a CE must fall in: pyCount's
# fixed window, which is also pyfitter's signal-region time window.
TIME_WINDOW_NS = (640.0, 1650.0)

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
                trkpid_min: float, triggers=DEFAULT_TRIGGERS) -> dict:
    """One file's selected events, reduced to flat per-event arrays, and its
    cut flow. Done per file so the jagged arrays never all sit in memory."""
    return reduce_data(read_eventntuple(path, triggers), active, trkqual_min,
                       trkpid_min, triggers)


def reduce_data(data: dict, active: list[str], trkqual_min: float,
                trkpid_min: float, triggers=DEFAULT_TRIGGERS) -> dict:
    """reduce_file for EventNtuple groups already in memory."""
    import awkward as ak

    masks = cut_masks(data, SIGN, trkqual_min, trkpid_min, triggers)
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
                  rmue: float = RMUE) -> dict:
    """Signal, DIO and cosmic rates on one momentum binning, from the
    signal events' reconstructed and true momenta.

    Every count is per event generated, times upstream_eff, so the signal
    and the response carry the measured efficiency. Signal and DIO scale
    with the stopped mu- NPOT gives; cosmics with the on-spill time.
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
    cosmic = replace(signal, name="cosmic", title="Cosmics",
                     values=np.full(signal.nbins, cosmic_rate * signal.width))
    return {"signal": signal, "dio": dio, "cosmic": cosmic,
            "response": response, "dio_true": dio_true,
            "efficiency": float(p_reco.size * per_event),
            "stopped_muons": stopped_muons,
            "conversions": stopped_muons * CAPTURES_PER_STOPPED_MUON * rmue,
            "cosmic_rate_per_mev": cosmic_rate,
            "onspill_seconds": onspill_seconds}


# --- plots -------------------------------------------------------------------

def _write_plots(outdir: Path, spectra: dict, best: dict, ev: dict,
                 is_signal: np.ndarray, time_window) -> list[str]:
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
        active = active_cuts(params["enable_cuts"], params["disable_cuts"])
        extra["cut_set"] = CUT_SET
        extra["cuts"] = active
        triggers = parse_trigger_paths(params["trigger_paths"])
        extra["trigger_paths"] = triggers
        t_lo, t_hi = parse_time_window(params["time_window"])
        parts = [reduce_file(path, active, params["trkqual_min"],
                             params["trkpid_min"], triggers)
                 for path in context.input_paths]
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
        spectra = build_spectra(ev["p"][counted], ev["p_true"][counted],
                                n_generated, upstream_eff, npot,
                                params["mean_pot_per_event"],
                                cosmic_rate_per_s_per_mev,
                                stops_per_pot, rmue)
        best, top = scan_signal_box(spectra["signal"], spectra["dio"],
                                    spectra["cosmic"])
    except (FullsimError, EventNtupleError, CutError, SensitivityError) as exc:
        return RunOutcome(error=str(exc), extra=extra)
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
            f"{exc}. The efficiency counts against the events generated, which "
            "come from SAM; without SAM, pass n_generated: the sum of dh.gencount "
            "over the input files' parent mcs files (samweb get-metadata)"
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
    }

    flow_rows = [("No cuts", ev["n_events"], "no selection")] + [
        (name, n, CUT_DESCRIPTIONS[name]) for name, n in zip(active, ev["flow"])]
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
        f"  cut set          pyfitter {CUT_SET}",
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
        f"  cosmic rate      {cosmic_rate_per_s_per_mev:.4g} per s per MeV/c "
        f"-> {spectra['cosmic_rate_per_mev']:.4g} per MeV/c "
        f"({spectra['onspill_seconds']:.4g} s on-spill)",
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
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = [str(flow_path)]
    files += _write_plots(outdir, spectra, best, ev, is_signal, (t_lo, t_hi))

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
        f"{metrics['n_events_counted']:.0f} reconstructed CE."
    )


SPEC = AnalysisSpec(
    name="fullsim_sensitivity",
    description=(
        "Full-simulation CE sensitivity S/sqrt(B), the counterpart of "
        "approx_ce_sensitivity: the signal efficiency, momentum shape and "
        "detector response are measured from reconstructed CE mixed with "
        "pileup (CeMLeadingLogMix1BB EventNtuple files) after pyfitter's "
        f"cut-set {CUT_SET} CE-like cuts. Signal and DIO (theory spectrum "
        "folded with that response) are normalized to the stopped mu- NPOT "
        "gives, as Production's normalizations.py; cosmics are flat per "
        "MeV/c over the on-spill time; the best-window scan is "
        "approx_ce_sensitivity's."
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
                        "on-spill live time the cosmic rate is scaled by.",
            default=MEAN_POT_PER_EVENT, minimum=1.0,
        ),
        ParamSpec(
            name="cosmic_rate_per_s_per_mev",
            description="Cosmic-ray background rate, flat in momentum, per "
                        "second per MeV/c, as in approx_ce_sensitivity (whose "
                        "default this is: the rough Run-1A mu- -> e- number). "
                        "Scaled by the on-spill live time npot and "
                        "mean_pot_per_event imply.",
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
                        "tracker front must fall in (open interval). The "
                        "default is pyfitter's signal-region time window. The "
                        "momentum window is scanned.",
            default="{:g},{:g}".format(*TIME_WINDOW_NS), kind="text",
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
            name="enable_cuts",
            description="Comma-separated cuts to apply on top of the default "
                        f"set (pyfitter cut-set {CUT_SET}: "
                        f"{', '.join(DEFAULT_CUTS)}). The others, pyfitter's "
                        "fit ranges, which the momentum scan and time_window "
                        "stand in for here: "
                        + ", ".join(n for n in CUT_DESCRIPTIONS
                                    if n not in DEFAULT_CUTS) + ".",
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
                        f"above this. Default: pyfitter cut-set {CUT_SET}'s.",
            default=TRKQUAL_MIN, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="trkpid_min",
            description="good_trkpid threshold: keep tracks with TrkPID above "
                        f"this. Default: pyfitter cut-set {CUT_SET}'s.",
            default=TRKPID_MIN, minimum=0.0, maximum=1.0,
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
