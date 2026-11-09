"""Momentum resolution and energy loss from simulated EventNtuple files.

For every track passing a cut expression (tools/selection.py syntax), the MC
truth of the particle that made it gives two distributions:

  resolution   reconstructed minus true momentum at the tracker front
               (TT_Front): how well the fit measures the momentum it sees
  energy_loss  true momentum at birth minus true momentum at TT_Front: what
               the particle lost on the way to the tracker (stopping target,
               proton absorber, ...)

Their difference is fullsim_sensitivity's "response" (reconstructed at
TT_Front minus true at birth), which folds both together.

The truth is tools/eventntuple.py's MC_VARIABLES (from `trkmcsim` and
`trksegsmc`): `resolution` is p_front - mc_p_front and `energy_loss` is
mc_p_start - mc_p_front, both at the downstream-going TT_Front crossing (the
earliest true one with pz > 0, as the true steps are not stored in time
order). Both distributions have long tails, so besides the mean and rms the
metrics give the median and sigma68, half the width of the central 68% (the
Gaussian sigma for a Gaussian, without a fit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..eventntuple import (MC_LEAVES, MC_VARIABLES, TRACK_VARIABLES, NtupleError,
                           open_tree, read_track_arrays, track_variables)
from ..selection import SelectionError, apply_selection
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from ..spectrum import Spectrum
from .trigger_efficiency_ntuple import DEFAULT_SELECTION

# Added to the track and MC variables, for selections as well as the plots.
RESPONSE_VARIABLES: dict[str, str] = {
    "resolution": "p_front - mc_p_front (MeV/c)",
    "energy_loss": "mc_p_start - mc_p_front (MeV/c)",
}

# The central fraction sigma68 spans: +-1 sigma of a Gaussian.
CENTRAL_FRACTION = 0.6827


class ResolutionError(ValueError):
    """An input or request that cannot be used, worded for the caller."""


# --- per-track truth ---------------------------------------------------------------

def response_variables(variables: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """RESPONSE_VARIABLES from the track's reconstructed and MC variables.
    A track without truth gets NaN."""
    return {"resolution": variables["p_front"] - variables["mc_p_front"],
            "energy_loss": variables["mc_p_start"] - variables["mc_p_front"]}


def read_file(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """(reconstructed, MC and response variables per track, number of events)."""
    tree = open_tree(path)
    names = {key.rsplit("/", 1)[-1] for key in tree.keys()}
    absent = [leaf for leaf in MC_LEAVES if leaf not in names]
    if absent:
        raise NtupleError(
            f"{path.name}: no {', '.join(absent)} branch — this analysis needs "
            "MC truth, which only simulated EventNtuples carry")
    variables = track_variables(read_track_arrays(tree, path))
    variables.update(response_variables(variables))
    return variables, tree.num_entries


# --- the analysis ------------------------------------------------------------------

def select_tracks(variables: dict[str, np.ndarray], selection: str) -> np.ndarray:
    """The mask of tracks passing `selection`."""
    return apply_selection(selection, {k: v for k, v in variables.items()
                                       if not k.startswith("_")},
                           variables["_event"].size)


def summary(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Mean, rms, median and sigma68 of the finite values."""
    x = values[np.isfinite(values)]
    tail = (1.0 - CENTRAL_FRACTION) / 2.0
    low, median, high = np.quantile(x, [tail, 0.5, 1.0 - tail])
    return {f"{prefix}_mean": float(x.mean()), f"{prefix}_rms": float(x.std()),
            f"{prefix}_median": float(median),
            f"{prefix}_sigma68": float((high - low) / 2.0)}


METRICS = ("n_events", "n_events_selected", "n_tracks_selected",
           "n_tracks_with_truth",
           "resolution_mean", "resolution_rms", "resolution_median",
           "resolution_sigma68",
           "energy_loss_mean", "energy_loss_rms", "energy_loss_median",
           "energy_loss_sigma68", "energy_loss_p90")

UNITS = {name: "MeV/c" for name in METRICS
         if name.startswith(("resolution_", "energy_loss_"))}


def resolution_metrics(counts: dict[str, int], resolution: np.ndarray,
                       energy_loss: np.ndarray) -> dict[str, float]:
    """The counts and both distributions' summaries, in METRICS order."""
    metrics = {key: float(counts[key]) for key in METRICS[:3]}
    metrics["n_tracks_with_truth"] = float(resolution.size)
    metrics.update(summary(resolution, "resolution"))
    metrics.update(summary(energy_loss, "energy_loss"))
    metrics["energy_loss_p90"] = float(np.quantile(energy_loss, 0.9))
    return metrics


def histogram(values: np.ndarray, low: float, high: float, width: float,
              name: str) -> Spectrum:
    """From `low` to `high` exactly, in the whole number of bins nearest to
    `width`: the effective width (Spectrum.width) labels the plots."""
    nbins = int(round((high - low) / width))
    return Spectrum.from_values(values, nbins, low, high, name=name)


def _write_outputs(outdir: Path, hists: dict[str, Spectrum], metrics: dict,
                   selection: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written = []

    root_path = outdir / "momentum_resolution.root"
    with uproot.recreate(root_path) as f:
        for name, h in hists.items():
            f[name] = (h.values, h.edges())
    written.append(str(root_path))

    labels = {
        "resolution": ("Reconstructed minus true momentum at TT_Front (MeV/c)",
                       "Momentum resolution"),
        "energy_loss": ("True momentum at creation minus at TT_Front (MeV/c)",
                        "Energy loss before the tracker"),
    }
    for name, h in hists.items():
        xlabel, title = labels[name]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.stairs(h.values, h.edges(), color="#2a78d6", lw=1.5)
        ax.set_yscale("log")
        ax.set_xlim(h.xmin, h.edges()[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Tracks / {h.width:g} MeV/c")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        stats = (f"tracks  {metrics['n_tracks_with_truth']:g}\n"
                 f"mean    {metrics[name + '_mean']:.3f}\n"
                 f"rms     {metrics[name + '_rms']:.3f}\n"
                 f"median  {metrics[name + '_median']:.3f}\n"
                 f"sigma68 {metrics[name + '_sigma68']:.3f}")
        ax.text(0.98 if name == "energy_loss" else 0.02, 0.97, stats,
                transform=ax.transAxes, va="top",
                ha="right" if name == "energy_loss" else "left",
                multialignment="left", family="monospace", fontsize=9)
        fig.text(0.01, 0.01, f"selection: {selection or '(all tracks)'}",
                 fontsize=7, color="#52514e")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = figdir / f"{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))
    return written


def _check_binning(params: dict) -> None:
    for name, low, high, width in (
            ("resolution", params["resolution_min"], params["resolution_max"],
             params["resolution_bin_width"]),
            ("energy_loss", params["energy_loss_min"], params["energy_loss_max"],
             params["energy_loss_bin_width"])):
        if high <= low:
            raise ResolutionError(f"{name}_max ({high:g}) must be above "
                                  f"{name}_min ({low:g})")
        if (high - low) / width > 100_000:
            raise ResolutionError(f"{name}_bin_width {width:g} gives over 100000 "
                                  f"bins between {low:g} and {high:g}")
        # histogram() rounds the bin count, so a width over twice the range
        # would give no bins at all.
        if round((high - low) / width) < 1:
            raise ResolutionError(f"{name}_bin_width {width:g} gives no bins "
                                  f"between {low:g} and {high:g}: make it at "
                                  f"most {high - low:g}")


def run(context: RunContext) -> RunOutcome:
    """Fill the resolution and energy-loss distributions over the ntuple(s)."""
    params = context.params
    selection = params["selection"]
    extra: dict = {"selection": selection}
    counts = {"n_events": 0, "n_events_selected": 0, "n_tracks_selected": 0}
    resolution, energy_loss, per_file = [], [], []
    try:
        _check_binning(params)
        for path in context.input_paths:
            variables, n_events = read_file(path)
            passing = select_tracks(variables, selection)
            truth = passing & np.isfinite(variables["resolution"]) \
                & np.isfinite(variables["energy_loss"])
            n_selected = int(np.unique(variables["_event"][passing]).size)
            counts["n_events"] += n_events
            counts["n_events_selected"] += n_selected
            counts["n_tracks_selected"] += int(passing.sum())
            resolution.append(variables["resolution"][truth])
            energy_loss.append(variables["energy_loss"][truth])
            per_file.append({"file": str(path), "n_events": n_events,
                             "n_tracks_selected": int(passing.sum()),
                             "n_tracks_with_truth": int(truth.sum())})
    except (NtupleError, SelectionError, ResolutionError) as exc:
        return RunOutcome(error=str(exc), extra=extra)
    resolution = np.concatenate(resolution)
    energy_loss = np.concatenate(energy_loss)
    if counts["n_tracks_selected"] == 0:
        return RunOutcome(
            error=f"no track in {counts['n_events']} events passes the "
                  "selection, so there is no distribution to fill",
            extra=extra)
    if resolution.size == 0:
        return RunOutcome(
            error=f"none of the {counts['n_tracks_selected']} selected tracks "
                  "has MC truth at TT_Front (a downstream-going trksegsmc step "
                  "with sid 0) to compare with",
            extra=extra)

    metrics = resolution_metrics(counts, resolution, energy_loss)
    hists = {
        "resolution": histogram(resolution, params["resolution_min"],
                                params["resolution_max"],
                                params["resolution_bin_width"], "resolution"),
        "energy_loss": histogram(energy_loss, params["energy_loss_min"],
                                 params["energy_loss_max"],
                                 params["energy_loss_bin_width"], "energy_loss"),
    }
    # The summaries use every track; the plots and histograms only their range.
    extra["outside_histogram_range"] = {
        name: int(resolution.size - h.values.sum()) for name, h in hists.items()}
    if len(per_file) > 1:
        extra["per_file"] = per_file
    files = _write_outputs(context.outdir, hists, metrics, selection)
    return RunOutcome(metrics=metrics, files=files, extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (f"momentum resolution sigma68 {metrics['resolution_sigma68']:.3g} MeV/c "
            f"(median {metrics['resolution_median']:.3g}, rms "
            f"{metrics['resolution_rms']:.3g}); energy loss before the tracker "
            f"median {metrics['energy_loss_median']:.3g} MeV/c (mean "
            f"{metrics['energy_loss_mean']:.3g}, 90% below "
            f"{metrics['energy_loss_p90']:.3g}); from "
            f"{metrics['n_tracks_with_truth']:g} tracks in "
            f"{metrics['n_events_selected']:g} selected events.")


_SELECTION_HELP = (
    "Track selection: every track passing it fills the distributions. "
    "Comparisons, arithmetic, and/or/not (or &&/||/!) and "
    "abs/sqrt/hypot/min/max/log/exp over the reconstructed variables "
    + ", ".join(TRACK_VARIABLES) + "; the MC truth ones "
    + ", ".join(MC_VARIABLES) + "; and " + ", ".join(RESPONSE_VARIABLES)
    + ". *_front quantities are at the "
    "downstream-going TT_Front crossing; a missing value is NaN and fails "
    "any cut on it. The default is a converged downstream e- fit with "
    "p > 80 MeV/c at the tracker front, >= 15 active hits and chisq/ndof < 5; "
    "add e.g. 'and mc_start_code == 168' for true CE only. An empty string "
    "takes every track."
)


def _range(name: str, what: str, default: float) -> ParamSpec:
    return ParamSpec(name=name, default=default,
                     description=f"{what} of the {name.rsplit('_', 1)[0]} "
                                 "histogram and plot (MeV/c). The metrics use "
                                 "every track, whatever the range.")


SPEC = AnalysisSpec(
    name="momentum_resolution_ntuple",
    input_kind="root_file",
    combines_files=True,
    run=run,
    description=(
        "Momentum resolution (reconstructed minus true momentum at the "
        "tracker front) and energy loss before the tracker (true momentum at "
        "creation minus at the tracker front) for the tracks passing a "
        "selection in simulated EventNtuple file(s): plots, histograms in a "
        "ROOT file, and their mean/rms/median/sigma68. No job is run."
    ),
    metrics=METRICS,
    units=UNITS,
    parameters=(
        ParamSpec(name="selection", description=_SELECTION_HELP,
                  default=DEFAULT_SELECTION, kind="text", allow_empty=True),
        _range("resolution_min", "Lower edge", -5.0),
        _range("resolution_max", "Upper edge", 3.0),
        ParamSpec(name="resolution_bin_width", default=0.05, minimum=1e-4,
                  description="Bin width of the resolution histogram (MeV/c)."),
        _range("energy_loss_min", "Lower edge", 0.0),
        _range("energy_loss_max", "Upper edge", 10.0),
        ParamSpec(name="energy_loss_bin_width", default=0.1, minimum=1e-4,
                  description="Bin width of the energy-loss histogram (MeV/c)."),
    ),
    summarize=summarize,
    input_hint=(
        "Simulated EventNtuple file(s) with MC truth (nts.*.root, tree "
        "EventNtuple/ntuple, branches trksegsmc and trkmcsim), e.g. "
        "nts.mu2e.CeMLeadingLogMix1BB.<version>.root; pass data_files for "
        "several, combined into one result."
    ),
)
