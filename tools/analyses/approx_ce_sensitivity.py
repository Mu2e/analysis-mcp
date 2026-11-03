"""Approximate Run-1A CE sensitivity: signal = CE, background = DIO + cosmics.

A Python conversion of Mu2eOptAna/scripts/rough_run1a_sensitivity.C. It reads
the per-event tree EdepAna writes (nts.*.root, EDepAna/tree), fills the
macro's histograms from the events passing a selection, and estimates
S/sqrt(B) for the best momentum window.

The chain, following the original macro:

1. Signal shape: the energy of the primary at the front of the tracker
   (`primary_trk_front_energy`) for the selected events. The default
   selection, `event_calo_edep_vis > 10`, is the macro's: EdepAna's fixed
   "edep 10 MeV" histogram set. Binned as EdepAna's `trk_front_energy`,
   rebinned x2 and scaled to a rate for NPOT protons at the assumed branching
   ratio and signal efficiency.
2. Signal smearing: convolved with a Gaussian tracker resolution.
3. DIO background: the Heeck/Szafron theoretical spectrum, scaled to a rate,
   then convolved with the *measured* energy-loss response of the same
   selected events (`primary_trk_front_energy_diff`, energy at the tracker
   minus energy at birth) and the same tracker resolution.
4. Cosmic background: flat in momentum at a rough rate per second per MeV/c
   (the `cosmic_rate_per_s_per_mev` parameter), scaled by the live on-spill
   time implied by NPOT.
5. Scan every window [x1, x2] with x1 >= 50 MeV and keep the one maximizing
   S/sqrt(B).

Given a reference file too (another configuration's CE sample), the same
chain runs on it with the same assumptions, apart from its own stopping rate
or efficiency, and the two are overlaid: signal and backgrounds, the signal
shape, and the energy-loss response.

The numbers are rough by construction — this is a figure of merit for
comparing beamline configurations, not a sensitivity calculation.
"""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..selection import SelectionError
from ..spectrum import Kernel, Spectrum
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .edep import (TREE_PATH, EdepTreeError, generated_events, read_edep_tree,
                   read_run_record, select_events, selection_help)

# --- assumptions carried over from the macro ---------------------------------

MUON_CAPTURE_RATE = 0.609         # N(muon captures) / N(muon stops) on aluminum
NPOT = 1.0e18                     # protons on target assumed
R_MUE = 1.0e-13                   # conversions per muon capture (the signal assumed)
SIGNAL_BR = R_MUE * MUON_CAPTURE_RATE  # CEs per stopped muon: R_mue per capture x captures per stop
MEAN_POT_PER_EVENT = 1.6e7        # 1BB
ONSPILL_SECONDS_PER_EVENT = 1.695e-6
COSMIC_RATE_PER_SECOND_PER_MEV = 10. / 7.8e5  # rough, per second per MeV/c, taken from Run 1A mu- --> e- analysis
TRK_RESOLUTION_SIGMA_MEV = 0.2
SIGNAL_BOX_MIN_MEV = 50.0         # below this DIO swamps everything anyway
SIGNAL_REBIN = 2

# The Heeck (2016) / Szafron DIO spectrum, finely binned.
DIO_TABLE = Path(
    "/exp/mu2e/app/users/mmackenz/run1b/Run1BAna/data/"
    "heeck_finer_binning_2016_szafron.tbl"
)
DIO_NBINS, DIO_EMIN, DIO_EMAX = 11000, 0.0, 110.0

# The cut EdepAna's "edep 10 MeV" histogram set (hist_2) was fixed to, which
# the macro read its shapes from.
DEFAULT_SELECTION = "event_calo_edep_vis > 10"

# Tree variable -> (nbins, xmin, xmax): the binning of the EdepAna histogram
# each one replaces (EdepAna_module.cc bookHistograms).
SIGNAL_VAR = "primary_trk_front_energy"
RESPONSE_VAR = "primary_trk_front_energy_diff"
BINNING = {
    SIGNAL_VAR: (1500, 0.0, 150.0),
    RESPONSE_VAR: (500, -100.0, 0.0),
    "primary_start_z": (500, 3000.0, 8000.0),
    "primary_start_r": (100, 0.0, 200.0),
}
EXTRA_PLOT_VARS = ("primary_start_z", "primary_start_r")


class SensitivityError(RuntimeError):
    """Raised for a problem the caller should see verbatim."""


# --- pieces of the calculation -----------------------------------------------

def load_dio_spectrum(table: Path = DIO_TABLE) -> Spectrum:
    """The theoretical DIO spectrum as a probability density in energy.

    The table holds (energy, weight) rows on a 0.01 MeV grid; each row sets the
    bin containing that energy. Normalized so the result integrates to 1 over
    energy.
    """
    if not table.exists():
        raise SensitivityError(f"DIO spectrum table not found: {table}")
    data = np.loadtxt(table)
    if data.ndim != 2 or data.shape[1] < 2:
        raise SensitivityError(f"DIO table {table} is not two columns of numbers")
    energy, weight = data[:, 0], data[:, 1]

    width = (DIO_EMAX - DIO_EMIN) / DIO_NBINS
    values = np.zeros(DIO_NBINS)
    # Nudge left by half a bin so a tabulated energy lands in the bin it ends.
    index = np.floor((energy - width / 2.0 - DIO_EMIN) / width).astype(np.int64)
    inside = (index >= 0) & (index < DIO_NBINS)
    values[index[inside]] = weight[inside]

    total = values.sum()
    if total <= 0.0:
        raise SensitivityError(f"DIO table {table} summed to zero weight")
    return Spectrum(values / (total * width), DIO_EMIN, width,
                    name="dio", title="DIO spectrum")


def mpv_fwhm(spectrum: Spectrum) -> tuple[float, float]:
    """Rough most-probable value and full width at half maximum."""
    peak = int(np.argmax(spectrum.values))
    mpv = float(spectrum.centers()[peak])
    above = np.flatnonzero(spectrum.values > spectrum.values[peak] / 2.0)
    if above.size == 0:
        return mpv, 0.0
    edges = spectrum.edges()
    return mpv, float(edges[above[-1] + 1] - edges[above[0]])


def scan_signal_box(signal: Spectrum, dio: Spectrum, cosmic: Spectrum,
                    box_min_mev: float = SIGNAL_BOX_MIN_MEV,
                    top_n: int = 10) -> tuple[dict[str, float], list[dict]]:
    """Find the window [x1, x2] maximizing S/sqrt(B).

    The three spectra share one binning, so a window is a slice and the scan is
    a cumulative sum per lower edge. Those sums run outward from each window's
    own lower edge, never as a difference of whole-spectrum prefix sums: the
    DIO spectrum spans ~18 orders of magnitude, so subtracting two such totals
    to get a count of order 1 loses the answer entirely to float cancellation.
    """
    for other in (dio, cosmic):
        if (other.nbins, other.xmin, other.width) != (signal.nbins, signal.xmin,
                                                      signal.width):
            raise SensitivityError(
                f"'{other.name}' is not on the signal's binning; regrid it first"
            )

    centers = signal.centers()
    first_bin = int(np.searchsorted(centers, box_min_mev))
    best: dict[str, float] | None = None
    per_edge: list[dict] = []

    for ibin in range(first_bin, signal.nbins):
        s = np.cumsum(signal.values[ibin:])
        d = np.cumsum(dio.values[ibin:])
        c = np.cumsum(cosmic.values[ibin:])
        background = d + c
        usable = np.flatnonzero((s > 0.0) & (background > 0.0))
        if usable.size == 0:
            continue
        ratio = s[usable] / np.sqrt(background[usable])
        jbest = usable[int(np.argmax(ratio))]
        entry = {
            "low_mev": float(centers[ibin]),
            "high_mev": float(centers[ibin + jbest]),
            "signal": float(s[jbest]), "dio": float(d[jbest]),
            "cosmic": float(c[jbest]), "background": float(background[jbest]),
            "sensitivity": float(s[jbest] / np.sqrt(background[jbest])),
        }
        per_edge.append(entry)
        if best is None or entry["sensitivity"] > best["sensitivity"]:
            best = entry

    if best is None:
        raise SensitivityError(
            "no signal window had both signal and background > 0 — the input "
            "histograms are probably empty above "
            f"{box_min_mev:g} MeV"
        )
    # The best window for each lower edge, ranked -- more informative in the
    # log than the top 10 overall, which only ever differ by a bin.
    per_edge.sort(key=lambda e: e["sensitivity"], reverse=True)
    return best, per_edge[:top_n]


# --- plots -------------------------------------------------------------------

def _write_plots(outdir: Path, signal: Spectrum, dio_reco: Spectrum,
                 cosmic: Spectrum, response: Spectrum, resolution: Spectrum,
                 dio_true: Spectrum, extras: dict[str, Spectrum],
                 best: dict[str, float], mpv: float, fwhm: float) -> list[str]:
    """Reproduce the macro's figures. Returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def step(ax, spectrum: Spectrum, **kwargs):
        ax.stairs(spectrum.values, spectrum.edges(), **kwargs)

    # signal vs background — the one that matters
    fig, ax = plt.subplots(figsize=(8, 6))
    step(ax, signal, color="tab:blue", lw=1.8, label="Signal (CE)")
    step(ax, dio_reco, color="tab:red", lw=1.8, label="DIO")
    step(ax, cosmic, color="tab:green", lw=1.8, label="Cosmic")
    ax.axvspan(best["low_mev"], best["high_mev"], color="0.85", zorder=0,
               label=f"box [{best['low_mev']:.1f}, {best['high_mev']:.1f}]")
    ax.set_yscale("log")
    ax.set_xlim(min(95.0, mpv - 1.5 * fwhm), max(105.0, mpv + 1.5 * fwhm))
    ax.set_ylim(1e-3, 1e4)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel(f"Rate / {signal.width:.1f} MeV")
    ax.set_title(f"Signal vs. background — S/sqrt(B) = {best['sensitivity']:.3g}")
    ax.legend(ncols=2, fontsize="small")
    fig.tight_layout()
    path = figdir / "sig_vs_bkg.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # DIO before/after smearing
    fig, ax = plt.subplots(figsize=(8, 6))
    step(ax, dio_true, color="tab:blue", lw=1.5, label="DIO (true)")
    step(ax, dio_reco, color="tab:red", lw=1.5, label="DIO (smeared)")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Rate")
    ax.set_title("DIO spectrum")
    ax.set_ylim(1.e-10, 1.e20)
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = figdir / "dio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # inputs: measured energy-loss response and the assumed resolution
    for hist, filename, title, xlabel in (
        (response, "response.png", "Energy loss before the tracker",
         "Energy difference (MeV)"),
        (resolution, "res.png",
         f"Tracker resolution (sigma = {TRK_RESOLUTION_SIGMA_MEV:g} MeV)",
         "Energy offset (MeV)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        step(ax, hist, color="tab:blue", lw=1.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        path = figdir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    # optional primary-vertex distributions, as the macro drew them
    for hist_name, hist in extras.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        step(ax, hist, color="tab:blue", lw=1.5)
        ax.set_xlabel(hist_name)
        ax.set_title(hist.title or hist_name)
        fig.tight_layout()
        path = figdir / f"ce_{hist_name.replace('primary_start_', '')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    return written


def _unit_area(spectrum: Spectrum) -> np.ndarray:
    total = spectrum.values.sum() * spectrum.width
    return spectrum.values / total if total > 0.0 else spectrum.values


def _write_overlay_plots(outdir: Path, new: "SampleResult", ref: "SampleResult",
                         label: str, reference_label: str) -> list[str]:
    """The input (solid) over the reference (dashed). Returns the paths
    written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    samples = ((new, label, "-"), (ref, reference_label, "--"))
    xlim = (min(95.0, *(s.mpv - 1.5 * s.fwhm for s in (new, ref))),
            max(105.0, *(s.mpv + 1.5 * s.fwhm for s in (new, ref))))

    def step(ax, spectrum: Spectrum, values=None, **kwargs):
        ax.stairs(spectrum.values if values is None else values,
                  spectrum.edges(), **kwargs)

    def save(fig, filename: str) -> None:
        fig.tight_layout()
        path = figdir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    # signal vs background, both samples, with the signal ratio underneath
    fig, (ax, rax) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True,
                                  gridspec_kw={"height_ratios": (3, 1)})
    for sample, name, ls in samples:
        best = sample.best
        step(ax, sample.signal, color="tab:blue", ls=ls, lw=1.8,
             label=f"Signal, {name} (S/sqrt(B) = {best['sensitivity']:.3g})")
        step(ax, sample.dio_reco, color="tab:red", ls=ls, lw=1.8,
             label=f"DIO, {name}")
        for edge in (best["low_mev"], best["high_mev"]):
            ax.axvline(edge, color="0.4", ls=ls, lw=1.0)
    # the cosmic rate depends on the run's parameters alone: one line
    step(ax, new.cosmic, color="tab:green", lw=1.8, label="Cosmic (both)")
    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_ylim(1e-3, 1e4)
    ax.set_ylabel(f"Rate / {new.signal.width:.1f} MeV")
    ax.set_title(f"{label} (solid) vs. {reference_label} (dashed), "
                 f"S/sqrt(B) ratio {new.best['sensitivity'] / ref.best['sensitivity']:.3g}")
    ax.legend(ncols=2, fontsize="small")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ref.signal.values > 0.0,
                         new.signal.values / ref.signal.values, np.nan)
    step(rax, new.signal, values=ratio, color="tab:blue", lw=1.5)
    rax.axhline(1.0, color="0.5", lw=0.8)
    rax.set_ylim(0.0, 2.0)
    rax.set_xlabel("Energy (MeV)")
    rax.set_ylabel(f"Signal {label} / {reference_label}", fontsize="small")
    save(fig, "sig_vs_bkg_overlay.png")

    # signal shape (unit area): where the peak sits and how wide it is
    fig, ax = plt.subplots(figsize=(8, 6))
    for sample, name, ls in samples:
        step(ax, sample.signal, values=_unit_area(sample.signal), color="tab:blue",
             ls=ls, lw=1.8, label=f"{name}: MPV {sample.mpv:.2f}, "
                                  f"FWHM {sample.fwhm:.2f} MeV")
    ax.set_xlim(*xlim)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Fraction per MeV")
    ax.set_title("Signal shape after the tracker resolution (unit area)")
    ax.legend(fontsize="small")
    save(fig, "signal_shape_overlay.png")

    # energy loss before the tracker (unit area)
    fig, ax = plt.subplots(figsize=(8, 6))
    for sample, name, ls in samples:
        values = _unit_area(sample.response)
        mean = float((sample.response.centers() * values).sum()
                     * sample.response.width)
        step(ax, sample.response, values=values, color="tab:blue", ls=ls, lw=1.5,
             label=f"{name}: mean {mean:.2f} MeV")
    ax.set_yscale("log")
    ax.set_xlabel("Energy difference (MeV)")
    ax.set_ylabel("Fraction per MeV")
    ax.set_title("Energy loss before the tracker (unit area)")
    ax.legend(fontsize="small")
    save(fig, "response_overlay.png")

    return written


# --- selected events ---------------------------------------------------------

def selected_hist(variables: dict[str, np.ndarray], mask: np.ndarray,
                  name: str) -> Spectrum:
    """`name` for the selected events where it is defined, weighted, binned
    as the EdepAna histogram it replaces."""
    values = variables[name]
    keep = mask & np.isfinite(values)
    nbins, xmin, xmax = BINNING[name]
    return Spectrum.from_values(values[keep], nbins, xmin, xmax,
                                weights=variables["weight"][keep], name=name)


def signal_efficiency(variables: dict[str, np.ndarray], mask: np.ndarray,
                      n_gen_events: float,
                      stops_per_pot: float) -> tuple[float, float]:
    """(sig_eff, CE acceptance) for a stopping rate: the acceptance is the
    file's events passing the selection (`mask`, weighted) per generated CE
    event, so the selection sets the signal's size as well as its shape."""
    if variables["weight"].size == 0:
        raise SensitivityError(
            "the file holds no events, so there is no CE acceptance to work "
            "out; run this on a CE (signal) sample"
        )
    if not n_gen_events > 0.0:
        raise SensitivityError(
            "the file records no generated events (ngen is 0), so the CE "
            "acceptance behind stops_per_pot cannot be worked out; pass "
            "sig_eff instead, or rerun 'edep' on files whose SubRuns carry a "
            "GenEventCount"
        )
    selected = float(variables["weight"][mask].sum())
    if not selected > 0.0:
        raise SensitivityError(
            "no event in the file passes the selection, so the CE acceptance "
            "is 0; loosen the selection"
        )
    acceptance = selected / n_gen_events
    if acceptance > 1.0:
        raise SensitivityError(
            f"the CE acceptance comes out at {acceptance:g} (events per "
            "generated event) — above 1, so the file's generated-event count "
            "(ngen) does not cover its events"
        )
    sig_eff = stops_per_pot * acceptance
    if not 0.0 < sig_eff <= 1.0:
        raise SensitivityError(
            f"stops_per_pot {stops_per_pot:g} times the CE acceptance "
            f"{acceptance:g} gives sig_eff {sig_eff:g}, outside (0, 1]"
        )
    return sig_eff, acceptance


# --- one sample --------------------------------------------------------------

@dataclass
class SampleResult:
    """What one EdepAna file gives: the spectra the plots draw, the best
    window, and the normalization they were built on."""

    path: Path
    n_selected: int
    n_events: int
    selected_weight: float
    sig_eff: float
    ce_acceptance: float | None
    n_gen_events: float | None
    run_record: dict | None
    signal_entries: float
    signal: Spectrum          # smeared, as a rate
    response: Spectrum        # energy-loss density, sig_eff folded in
    resolution: Kernel
    dio_true: Spectrum
    dio_reco: Spectrum
    cosmic: Spectrum
    extras: dict[str, Spectrum]
    best: dict[str, float]
    top: list[dict]
    mpv: float
    fwhm: float
    cosmic_rate: float
    onspill_seconds: float


def evaluate_sample(path: Path, params: dict[str, float | str], sig_eff: float,
                    stops_per_pot: float) -> SampleResult:
    """Steps 1-5 of the module docstring for one file. Exactly one of sig_eff
    and stops_per_pot is > 0. Raises SensitivityError, EdepTreeError or
    SelectionError with a message for the caller."""
    npot = params["npot"]
    selection = params["selection"]
    ce_acceptance = n_gen_events = run_record = None

    variables = read_edep_tree(path)
    mask = select_events(variables, selection)
    if stops_per_pot > 0.0:
        run_record = read_run_record(path)
        if run_record is not None and run_record["max_events"] is not None:
            raise SensitivityError(
                f"this file comes from an 'edep' run with max_events="
                f"{run_record['max_events']}, which cuts a subrun short "
                "while ngen keeps its whole generated-event count, so the "
                "CE acceptance behind stops_per_pot would come out low; "
                "rerun 'edep' without max_events, or pass sig_eff"
            )
        n_gen_events = generated_events(path)
        sig_eff, ce_acceptance = signal_efficiency(variables, mask,
                                                   n_gen_events, stops_per_pot)
    n_selected = int(mask.sum())
    signal = selected_hist(variables, mask, SIGNAL_VAR)
    response = selected_hist(variables, mask, RESPONSE_VAR)
    extras = {name: selected_hist(variables, mask, name)
              for name in EXTRA_PLOT_VARS}

    # Both normalizations divide by the entry count, so empty input is a
    # clean error rather than a division by zero.
    for hist in (signal, response):
        if hist.entries <= 0:
            raise SensitivityError(
                f"no selected event has '{hist.name}': {n_selected} of "
                f"{mask.size} events pass '{selection}', and none of them "
                "reached the tracker front, so there is no signal shape to "
                "work with. Run this on a CE (signal) sample, or loosen "
                "the selection."
            )

    # 1-2. signal shape -> rate, then smeared by the tracker resolution
    signal = signal.rebin(SIGNAL_REBIN)
    signal_entries = float(signal.entries)
    signal = signal.scaled(npot * SIGNAL_BR * sig_eff / signal.entries)
    # A density in energy offset. The efficiency is folded in here, so it
    # rides along with the response into the smeared DIO spectrum.
    response = response.scaled(sig_eff / response.entries / response.width)
    signal_reco = signal.smear(
        Kernel.gaussian(signal.width, TRK_RESOLUTION_SIGMA_MEV)
    )
    mpv, fwhm = mpv_fwhm(signal_reco)

    # 3. DIO: theory spectrum -> rate, smeared by the energy loss and then
    # by the resolution on its own fine binning, then put on signal's bins.
    dio_true = load_dio_spectrum().scaled((1. - MUON_CAPTURE_RATE) * sig_eff * npot)
    resolution = Kernel.gaussian(dio_true.width, TRK_RESOLUTION_SIGMA_MEV)
    dio_reco = (dio_true
                .smear(Kernel.from_density(response, dio_true.width))
                .smear(resolution)
                .regrid(signal_reco))

    # 4. cosmics: flat rate per MeV/c over the implied on-spill time
    events = npot / params["mean_pot_per_event"]
    onspill_seconds = events * ONSPILL_SECONDS_PER_EVENT
    cosmic_rate = params["cosmic_rate_per_s_per_mev"] * onspill_seconds
    cosmic = replace(
        signal_reco, name="cosmic", title="Cosmics",
        values=np.full(signal_reco.nbins, cosmic_rate * signal_reco.width),
    )

    # 5. best window
    best, top = scan_signal_box(signal_reco, dio_reco, cosmic)
    return SampleResult(
        path=path, n_selected=n_selected, n_events=int(mask.size),
        selected_weight=float(variables["weight"][mask].sum()),
        sig_eff=sig_eff, ce_acceptance=ce_acceptance,
        n_gen_events=n_gen_events, run_record=run_record,
        signal_entries=signal_entries, signal=signal_reco, response=response,
        resolution=resolution, dio_true=dio_true, dio_reco=dio_reco,
        cosmic=cosmic, extras=extras, best=best, top=top, mpv=mpv, fwhm=fwhm,
        cosmic_rate=cosmic_rate, onspill_seconds=onspill_seconds,
    )


def _sig_eff_line(sample: SampleResult, stops_per_pot: float) -> str:
    return f"  sig_eff          {sample.sig_eff:g}" + (
        "" if sample.ce_acceptance is None else
        f"  (stops/POT {stops_per_pot:g} x CE acceptance {sample.ce_acceptance:g}"
        f" = {sample.selected_weight:g} selected / "
        f"{sample.n_gen_events:g} generated)")


def _best_line(sample: SampleResult) -> str:
    best = sample.best
    return (f"Best window [{best['low_mev']:.1f}, {best['high_mev']:.1f}] MeV: "
            f"S = {best['signal']:.3g}, DIO = {best['dio']:.3g}, "
            f"cosmic = {best['cosmic']:.3g} -> B = {best['background']:.3g}, "
            f"S/sqrt(B) = {best['sensitivity']:.4g}")


# --- the runner --------------------------------------------------------------

def run(context: RunContext) -> RunOutcome:
    """Compute the approximate CE sensitivity for one EdepAna ROOT file, and
    for a reference file too when one is given, overlaying the two."""
    params = context.params
    sig_eff = params["sig_eff"]
    stops_per_pot = params["stops_per_pot"]
    ref_sig_eff = params["reference_sig_eff"]
    ref_stops_per_pot = params["reference_stops_per_pot"]
    for name, value in (("sig_eff", sig_eff), ("stops_per_pot", stops_per_pot),
                        ("reference_sig_eff", ref_sig_eff),
                        ("reference_stops_per_pot", ref_stops_per_pot)):
        if np.isnan(value):
            return RunOutcome(error=f"{name} must be a number, not NaN")
    if (sig_eff > 0.0) == (stops_per_pot > 0.0):
        return RunOutcome(error=(
            "pass exactly one of sig_eff or stops_per_pot (> 0): sig_eff is "
            "the signal efficiency itself, stops_per_pot works it out with "
            "the file's own CE acceptance"))
    has_reference = bool(context.reference_paths)
    if not has_reference and (ref_sig_eff > 0.0 or ref_stops_per_pot > 0.0):
        return RunOutcome(error=(
            "reference_sig_eff and reference_stops_per_pot apply only with a "
            "reference_file, the baseline sample to overlay"))
    if (ref_sig_eff > 0.0 and stops_per_pot > 0.0) or (
            ref_stops_per_pot > 0.0 and sig_eff > 0.0):
        return RunOutcome(error=(
            "the reference is normalized the same way as the input: with "
            "sig_eff give reference_sig_eff, with stops_per_pot give "
            "reference_stops_per_pot"))
    if len(context.reference_paths) > 1:
        return RunOutcome(error=(
            f"takes one reference file, as it takes one data_file; got "
            f"{len(context.reference_paths)}"))
    selection = params["selection"]
    outdir = context.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        sample = evaluate_sample(context.input_path, params, sig_eff,
                                 stops_per_pot)
    except (SensitivityError, EdepTreeError, SelectionError) as exc:
        return RunOutcome(error=str(exc), extra={"selection": selection})
    ref = None
    if has_reference:
        # 0 means "the input's": two configurations usually differ in their
        # stopping rate, so the caller passes the reference's own.
        ref_stops = ref_stops_per_pot or stops_per_pot
        try:
            ref = evaluate_sample(context.reference_paths[0], params,
                                  ref_sig_eff or sig_eff, ref_stops)
        except (SensitivityError, EdepTreeError, SelectionError) as exc:
            return RunOutcome(error=f"reference file: {exc}",
                              extra={"selection": selection})
    best = sample.best
    nan = float("nan")

    metrics = {
        "sensitivity": float(best["sensitivity"]),
        "signal_box_low_mev": float(best["low_mev"]),
        "signal_box_high_mev": float(best["high_mev"]),
        "signal_rate": float(best["signal"]),
        "dio_background": float(best["dio"]),
        "cosmic_background": float(best["cosmic"]),
        "total_background": float(best["background"]),
        "signal_mpv_mev": float(sample.mpv),
        "signal_fwhm_mev": float(sample.fwhm),
        # The assumptions the rates above are built on, reported with them so
        # a number never travels without the normalization behind it.
        "npot": float(params["npot"]),
        "cosmic_rate_per_s_per_mev": float(params["cosmic_rate_per_s_per_mev"]),
        # The reference's, each in its own best window; NaN without one.
        "reference_sensitivity": nan if ref is None else float(ref.best["sensitivity"]),
        "reference_signal_rate": nan if ref is None else float(ref.best["signal"]),
        "reference_total_background": (nan if ref is None
                                       else float(ref.best["background"])),
        "reference_signal_mpv_mev": nan if ref is None else float(ref.mpv),
        "reference_signal_fwhm_mev": nan if ref is None else float(ref.fwhm),
        "sensitivity_ratio": (nan if ref is None else
                              float(best["sensitivity"] / ref.best["sensitivity"])),
    }

    log_path = outdir / "approx_ce_sensitivity.log"
    lines = [
        "approx_ce_sensitivity",
        f"  input            {context.input_path}",
        f"  selection        {selection or '(none)'}: {sample.n_selected} of "
        f"{sample.n_events} events",
        _sig_eff_line(sample, stops_per_pot),
    ]
    if sample.ce_acceptance is not None and sample.run_record is None:
        lines.append(
            "  WARNING          no edep run record next to the file, so this "
            "cannot confirm it comes from one full 'edep' run (without "
            "max_events, not merged); the CE acceptance is only as good as that")
    lines += [
        f"  NPOT             {params['npot']:g}",
        f"  signal BR        {SIGNAL_BR:.4g} per stop  (R_mue = {R_MUE:g} per capture "
        f"x {MUON_CAPTURE_RATE:g} captures per stop)",
        f"  cosmic rate      {params['cosmic_rate_per_s_per_mev']:.4g} per s per MeV/c "
        f"-> {sample.cosmic_rate:.4g} per MeV/c ({sample.onspill_seconds:.4g} s on-spill)",
        f"  signal entries   {sample.signal_entries:g}",
        f"  MPV / FWHM       {sample.mpv:.3f} / {sample.fwhm:.3f} MeV",
        "",
        _best_line(sample),
        "",
        f"Top {len(sample.top)} windows scanned:",
    ]
    lines += [
        f"  [{e['low_mev']:6.1f}, {e['high_mev']:6.1f}] MeV  S = {e['signal']:9.3g}  "
        f"DIO = {e['dio']:9.3g}  cosmic = {e['cosmic']:9.3g}  "
        f"S/sqrt(B) = {e['sensitivity']:.4g}"
        for e in sample.top
    ]
    if ref is not None:
        lines += [
            "",
            f"Reference ({params['reference_label']})",
            f"  input            {ref.path}",
            f"  selection        {selection or '(none)'}: {ref.n_selected} of "
            f"{ref.n_events} events",
            _sig_eff_line(ref, ref_stops),
            f"  signal entries   {ref.signal_entries:g}",
            f"  MPV / FWHM       {ref.mpv:.3f} / {ref.fwhm:.3f} MeV",
            "",
            _best_line(ref),
            "",
            f"S/sqrt(B) {params['label']} / {params['reference_label']} = "
            f"{metrics['sensitivity_ratio']:.4g}",
        ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = _write_plots(outdir, sample.signal, sample.dio_reco, sample.cosmic,
                         sample.response,
                         sample.resolution.as_spectrum(sample.dio_true.width,
                                                       "resolution"),
                         sample.dio_true, sample.extras, best, sample.mpv,
                         sample.fwhm)
    if ref is not None:
        files += _write_overlay_plots(outdir, sample, ref, params["label"],
                                      params["reference_label"])

    return RunOutcome(
        metrics=metrics,
        files=files,
        log_path=log_path,
        extra={
            "sig_eff": sample.sig_eff,
            **({} if sample.ce_acceptance is None else {
                "stops_per_pot": stops_per_pot,
                "ce_acceptance": sample.ce_acceptance,
                "n_gen_events": sample.n_gen_events,
                "edep_run_recorded": sample.run_record is not None}),
            "selection": selection,
            "n_events_selected": sample.n_selected,
            "signal_br": SIGNAL_BR,
            "cosmic_rate_per_mev": sample.cosmic_rate,
            "onspill_seconds": sample.onspill_seconds,
            "signal_hist_entries": sample.signal_entries,
            "dio_table": str(DIO_TABLE),
            **({} if ref is None else {
                "label": params["label"],
                "reference_label": params["reference_label"],
                "reference_sig_eff": ref.sig_eff,
                "reference_n_events_selected": ref.n_selected,
                "reference_signal_box_low_mev": float(ref.best["low_mev"]),
                "reference_signal_box_high_mev": float(ref.best["high_mev"]),
                **({} if ref.ce_acceptance is None else {
                    "reference_stops_per_pot": ref_stops,
                    "reference_ce_acceptance": ref.ce_acceptance,
                    "reference_n_gen_events": ref.n_gen_events})}),
        },
    )


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"S/sqrt(B) = {metrics['sensitivity']:.4g} in "
        f"[{metrics['signal_box_low_mev']:.1f}, "
        f"{metrics['signal_box_high_mev']:.1f}] MeV "
        f"(S = {metrics['signal_rate']:.3g}, B = {metrics['total_background']:.3g}) "
        f"for {metrics['npot']:.3g} POT and a cosmic rate of "
        f"{metrics['cosmic_rate_per_s_per_mev']:.4g} per s per MeV/c."
        + ("" if np.isnan(metrics["reference_sensitivity"]) else
           f" Reference: S/sqrt(B) = {metrics['reference_sensitivity']:.4g} "
           f"(S = {metrics['reference_signal_rate']:.3g}), ratio "
           f"{metrics['sensitivity_ratio']:.4g}.")
    )


SPEC = AnalysisSpec(
    name="approx_ce_sensitivity",
    description=(
        "Approximate Run-1A conversion-electron sensitivity S/sqrt(B) from "
        "the EdepAna tree, with DIO and cosmic backgrounds, for a "
        "configurable event selection; optionally overlaid with a reference "
        "configuration's."
    ),
    input_kind="root_file",
    produced_by=("edep",),
    metrics=(
        "sensitivity", "signal_box_low_mev", "signal_box_high_mev",
        "signal_rate", "dio_background", "cosmic_background",
        "total_background", "signal_mpv_mev", "signal_fwhm_mev",
        "npot", "cosmic_rate_per_s_per_mev",
        "reference_sensitivity", "reference_signal_rate",
        "reference_total_background", "reference_signal_mpv_mev",
        "reference_signal_fwhm_mev", "sensitivity_ratio",
    ),
    units={
        "signal_box_low_mev": "MeV",
        "signal_box_high_mev": "MeV",
        "signal_mpv_mev": "MeV",
        "signal_fwhm_mev": "MeV",
        "reference_signal_mpv_mev": "MeV",
        "reference_signal_fwhm_mev": "MeV",
        "npot": "POT",
        "cosmic_rate_per_s_per_mev": "per second per MeV/c",
    },
    parameters=(
        ParamSpec(
            name="sig_eff",
            description="Signal (CE) reconstruction+selection efficiency, 0-1. "
                        "Scales signal and both backgrounds. Pass this or "
                        "stops_per_pot, not both; 0 means not given.",
            default=0.0, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="stops_per_pot",
            description="Stopped muons per POT, e.g. muon_stop_rate's "
                        "stops_per_pot for the same configuration. Instead of "
                        "sig_eff: sig_eff becomes stops_per_pot times the CE "
                        "acceptance, the file's events passing `selection` "
                        "per generated event (EdepAna's ngen), so the "
                        "selection sets the signal's size too. The file must "
                        "come from one full 'edep' run: one 'edep' recorded "
                        "as run with max_events is refused, and a merged one "
                        "is refused when ngen shows it (not every merge "
                        "does). 0 means not given.",
            default=0.0, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="npot",
            description="Protons on target to assume for the rate normalization.",
            default=NPOT, minimum=0.0,
        ),
        ParamSpec(
            name="mean_pot_per_event",
            description="Mean number of protons on target per event.",
            default=MEAN_POT_PER_EVENT, minimum=1.0,
        ),
        ParamSpec(
            name="cosmic_rate_per_s_per_mev",
            description="Cosmic-ray background rate, flat in momentum, per "
                        "second per MeV/c. The default is the rough Run-1A "
                        "mu- -> e- number; scaled by the on-spill live time "
                        "that npot and mean npot per event implies.",
            default=COSMIC_RATE_PER_SECOND_PER_MEV, minimum=0.0,
        ),
        ParamSpec(
            name="selection",
            description=selection_help() + " Selects the events the signal "
                        "shape and energy-loss response are taken from. The "
                        f"default, '{DEFAULT_SELECTION}', is the fixed cut "
                        "the original macro's histograms had.",
            default=DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="reference_sig_eff",
            description="With a reference_file and sig_eff only: the "
                        "reference sample's own signal efficiency. 0 means "
                        "the same as sig_eff.",
            default=0.0, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="reference_stops_per_pot",
            description="With a reference_file and stops_per_pot only: the "
                        "reference configuration's own stopped muons per POT "
                        "(its muon_stop_rate stops_per_pot), times the "
                        "reference file's CE acceptance as for "
                        "stops_per_pot. 0 means the same as stops_per_pot.",
            default=0.0, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="label",
            description="Legend name of the input sample in the overlay "
                        "plots, e.g. 'No IPA'.",
            default="input", kind="text",
        ),
        ParamSpec(
            name="reference_label",
            description="Legend name of the reference sample in the overlay "
                        "plots, e.g. 'nominal'.",
            default="reference", kind="text",
        ),
    ),
    input_hint=(
        "An EdepAna nts.*.root file from a CE (signal) sample, holding "
        f"{TREE_PATH} (EdepAna in Offline v13_39_00 or later), with events "
        "passing the selection that reach the tracker front. Optionally a "
        "reference_file of the same kind (another configuration's CE sample, "
        "same selection, npot and cosmic rate), which is evaluated the same "
        "way and overlaid: figures/sig_vs_bkg_overlay.png, "
        "signal_shape_overlay.png and response_overlay.png, plus the "
        "reference_* metrics (NaN without one)."
    ),
    takes_reference=True,
    reference_optional=True,
    run=run,
    summarize=summarize,
)
