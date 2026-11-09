"""Compare one variable between two sets of EventNtuple files.

The input (data_file(s), "new") is compared with a baseline (reference_file(s),
"ref"), e.g. the same events reprocessed before and after a code change, or
two configurations of a sample. The variable is a track or calorimeter
cluster quantity, or an expression of them (tools/selection.py syntax, over
tools/eventntuple.py's TRACK_VARIABLES/MC_VARIABLES/CRV_VARIABLES or
calo_cluster_resolution_ntuple's CLUSTER_VARIABLES, so they mean the same as
in those analyses), taken for every object passing a selection.

Two comparisons are made:

* Distributions, always: every selected object's value in each sample,
  overlaid (per event by default, so a change in how many objects pass shows
  too) with their ratio, the mean/median shifts, and a two-sample KS test.
* Matched events, when the samples share events (run, subrun, event): each
  event's first selected object (the ntuple's order) in one sample against
  the other's. This says how many events gained or lost a selected object,
  how many changed value, and by how much. "First" is only the same object
  in both when the objects come out in the same order, which they do for a
  reprocessing of the same events. With `match_events="auto"`, events are
  matched when at least MATCH_FRACTION of the smaller sample has a
  counterpart; independent samples can repeat run/subrun/event numbers by
  chance, but not wholesale. A sample that repeats an id itself is never
  matched, as its events cannot be paired one to one.

Metrics that do not apply (the matched ones, with nothing matched) are NaN.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .. import eventntuple as tracks
from ..eventntuple import NtupleError, open_tree
from ..selection import SelectionError, apply_selection, evaluate_expression
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from . import calo_cluster_resolution_ntuple as calo
from .momentum_resolution_ntuple import CENTRAL_FRACTION
from .trigger_efficiency_ntuple import DEFAULT_SELECTION
from .trigger_efficiency_ntuple_binned import _th1

EVENT_LEAVES = ("run", "subrun", "event")

# Matched when this fraction of the smaller sample's events is in both.
MATCH_FRACTION = 0.5
# Two values closer than this, relatively, are the same value: the ntuple
# stores floats, so a reprocessing that changed nothing gives exact equality.
SAME_VALUE_RTOL = 1e-6
# The auto range spans these quantiles of both samples' values together, so
# a few far outliers do not squash the plot. Every value counts in the metrics.
AUTO_QUANTILES = (0.005, 0.995)
DEFAULT_N_BINS, MAX_BINS = 50, 1000

FIGURE_NAME = "variable_comparison.png"
DIFFERENCE_FIGURE_NAME = "variable_difference.png"
ROOT_NAME = "variable_comparison.root"

NEW_COLOR, REF_COLOR = "#2a78d6", "#52514e"


@dataclass(frozen=True)
class Level:
    """The objects a comparison is made over: how a file is read into one
    entry per object, which names it offers, and the check that refuses an
    MC name on a file without truth."""

    noun: str
    variables: dict[str, str]
    read: Callable[[Path], tuple[dict[str, np.ndarray], int]]
    check: Callable[..., None]


def _read_tracks(path: Path) -> tuple[dict[str, np.ndarray], int]:
    variables, _, n_events = tracks.read_file(path, [])
    return variables, n_events


LEVELS = {
    "track": Level("track", {**tracks.TRACK_VARIABLES, **tracks.MC_VARIABLES,
                             **tracks.CRV_VARIABLES},
                   _read_tracks, tracks.check_names),
    "cluster": Level("cluster", calo.CLUSTER_VARIABLES,
                     lambda path: calo.read_file(path, require_truth=False),
                     calo.check_truth_names),
}
MATCH_MODES = ("auto", "yes", "no")
NORMALIZATIONS = ("events", "area", "none")


class ComparisonError(ValueError):
    """An input or request that cannot be used, worded for the caller."""


# --- one sample --------------------------------------------------------------------

@dataclass
class Sample:
    """One side of the comparison, all its files together."""

    values: np.ndarray         # every selected object's value (NaN: undefined)
    first: np.ndarray          # per event: its first selected object's value
    selected: np.ndarray       # per event: has a selected object
    keys: np.ndarray           # per event: (run, subrun, event)
    n_objects: int             # objects before the selection

    @property
    def n_events(self) -> int:
        return int(self.keys.size)

    @property
    def finite(self) -> np.ndarray:
        return self.values[np.isfinite(self.values)]


def event_keys(run, subrun, event) -> np.ndarray:
    """(run, subrun, event) as one sortable structured array, for matching."""
    keys = np.empty(len(run), dtype=[("run", "i8"), ("subrun", "i8"), ("event", "i8")])
    keys["run"], keys["subrun"], keys["event"] = run, subrun, event
    return keys


def sample_from_objects(variables: dict[str, np.ndarray], keys: np.ndarray,
                        selection: str, variable: str) -> Sample:
    """One file's Sample from its per-object variables (with `_event`)."""
    objects = {k: v for k, v in variables.items() if not k.startswith("_")}
    n_objects = variables["_event"].size
    passing = apply_selection(selection, objects, n_objects)
    # A true/false variable (has_front, downstream, ...) compares as 0/1. Only
    # a bare name is converted: an expression like `p > 100` is a cut, and
    # evaluate_expression still refuses it as a quantity.
    if objects.get(variable.strip(), np.empty(0)).dtype == bool:
        values = objects[variable.strip()].astype(np.float64)
    else:
        values = evaluate_expression(variable, objects, n_objects)
    index = np.flatnonzero(passing)
    # Objects are in event order, so the first of each event's is its first.
    events, first = np.unique(variables["_event"][index], return_index=True)
    selected = np.zeros(keys.size, dtype=bool)
    selected[events] = True
    first_value = np.full(keys.size, np.nan)
    first_value[events] = values[index[first]]
    return Sample(values[passing], first_value, selected, keys, int(n_objects))


def read_sample(paths: list[Path], level: Level, selection: str,
                variable: str) -> Sample:
    """The Sample of `paths`, read and selected file by file."""
    parts = []
    for path in paths:
        variables, n_events = level.read(path)
        level.check(variables, selection, variable)
        ids = open_tree(path).arrays(filter_name=list(EVENT_LEAVES), library="np")
        keys = event_keys(*(ids[name] for name in EVENT_LEAVES))
        if keys.size != n_events:
            raise ComparisonError(f"{path.name}: {keys.size} event ids for "
                                  f"{n_events} events")
        parts.append(sample_from_objects(variables, keys, selection, variable))
    return Sample(values=np.concatenate([p.values for p in parts]),
                  first=np.concatenate([p.first for p in parts]),
                  selected=np.concatenate([p.selected for p in parts]),
                  keys=np.concatenate([p.keys for p in parts]),
                  n_objects=sum(p.n_objects for p in parts))


# --- comparing ---------------------------------------------------------------------

def summary(values: np.ndarray) -> dict[str, float]:
    """Mean, rms, median and sigma68 of finite values (NaN for none)."""
    if values.size == 0:
        return dict.fromkeys(("mean", "rms", "median", "sigma68"), float("nan"))
    tail = (1.0 - CENTRAL_FRACTION) / 2.0
    low, median, high = np.quantile(values, [tail, 0.5, 1.0 - tail])
    return {"mean": float(values.mean()), "rms": float(values.std()),
            "median": float(median), "sigma68": float((high - low) / 2.0)}


def distribution_metrics(new: Sample, ref: Sample) -> dict[str, float]:
    """The two samples' selected-object distributions, side by side. With
    no values on one side (a reprocessing that lost or first found every
    object), its statistics, the shifts and the KS test are NaN."""
    from scipy.stats import ks_2samp

    a, b = new.finite, ref.finite
    sa, sb = summary(a), summary(b)
    both = a.size > 0 and b.size > 0
    ks = ks_2samp(a, b) if both else None
    return {
        "n_events_new": float(new.n_events), "n_events_ref": float(ref.n_events),
        "n_selected_new": float(a.size), "n_selected_ref": float(b.size),
        "selected_per_event_new": a.size / new.n_events if new.n_events else float("nan"),
        "selected_per_event_ref": b.size / ref.n_events if ref.n_events else float("nan"),
        "mean_new": sa["mean"], "mean_ref": sb["mean"],
        "mean_shift": sa["mean"] - sb["mean"],
        # As for independent samples: for matched ones, diff_mean is sharper.
        "mean_shift_err": (float(np.sqrt(a.var() / a.size + b.var() / b.size))
                           if both else float("nan")),
        "rms_new": sa["rms"], "rms_ref": sb["rms"],
        "median_new": sa["median"], "median_ref": sb["median"],
        "median_shift": sa["median"] - sb["median"],
        "sigma68_new": sa["sigma68"], "sigma68_ref": sb["sigma68"],
        "ks_statistic": float(ks.statistic) if ks else float("nan"),
        "ks_pvalue": float(ks.pvalue) if ks else float("nan"),
    }


def match(new: Sample, ref: Sample, mode: str) -> dict:
    """Pair the events both samples have. Returns the indices into each
    (empty when not matched) and what was found, for the metadata."""
    unique_new, unique_ref = np.unique(new.keys), np.unique(ref.keys)
    common, i_new, i_ref = np.intersect1d(new.keys, ref.keys, return_indices=True)
    smaller = min(unique_new.size, unique_ref.size)
    info = {"mode": mode, "n_common_events": int(common.size),
            "n_only_new": int(unique_new.size - common.size),
            "n_only_ref": int(unique_ref.size - common.size),
            "n_duplicate_ids_new": int(new.n_events - unique_new.size),
            "n_duplicate_ids_ref": int(ref.n_events - unique_ref.size)}
    duplicated = info["n_duplicate_ids_new"] or info["n_duplicate_ids_ref"]
    if mode == "no":
        matched = False
    elif mode == "yes":
        if duplicated:
            raise ComparisonError(
                f"match_events is 'yes', but {info['n_duplicate_ids_new']} event "
                f"id(s) in the new sample and {info['n_duplicate_ids_ref']} in "
                "the reference repeat a (run, subrun, event), so events cannot "
                "be paired unambiguously: use match_events='no' to compare the "
                "distributions only")
        if common.size == 0:
            raise ComparisonError(
                "match_events is 'yes', but no (run, subrun, event) is in both "
                "samples: they are not the same events")
        matched = True
    else:
        # intersect1d pairs only the first of repeated ids, so the rest would
        # silently drop out or be paired with the wrong event.
        matched = (not duplicated and smaller > 0
                   and common.size >= MATCH_FRACTION * smaller)
    info["matched"] = matched
    if not matched:
        i_new = i_ref = np.array([], dtype=np.int64)
    return {**info, "i_new": i_new, "i_ref": i_ref}


def matched_metrics(new: Sample, ref: Sample, i_new: np.ndarray,
                    i_ref: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Event by event over the matched events: (metrics, the differences
    new - ref of the events selected in both with a value in both)."""
    s_new, s_ref = new.selected[i_new], ref.selected[i_ref]
    a, b = new.first[i_new], ref.first[i_ref]
    both = s_new & s_ref
    defined = both & np.isfinite(a) & np.isfinite(b)
    same = np.isclose(a, b, rtol=SAME_VALUE_RTOL, atol=0.0) | (np.isnan(a) & np.isnan(b))
    diff = a[defined] - b[defined]
    s = summary(diff)
    return {
        "n_events_matched": float(i_new.size),
        "n_events_both_selected": float(both.sum()),
        "n_events_gained": float((s_new & ~s_ref).sum()),
        "n_events_lost": float((~s_new & s_ref).sum()),
        "n_events_changed": float((both & ~same).sum()),
        "diff_mean": s["mean"], "diff_rms": s["rms"],
        "diff_median": s["median"], "diff_sigma68": s["sigma68"],
    }, diff


METRICS = ("n_events_new", "n_events_ref", "n_selected_new", "n_selected_ref",
           "selected_per_event_new", "selected_per_event_ref",
           "mean_new", "mean_ref", "mean_shift", "mean_shift_err",
           "rms_new", "rms_ref", "median_new", "median_ref", "median_shift",
           "sigma68_new", "sigma68_ref", "ks_statistic", "ks_pvalue",
           "n_events_matched", "n_events_both_selected", "n_events_gained",
           "n_events_lost", "n_events_changed",
           "diff_mean", "diff_rms", "diff_median", "diff_sigma68")
MATCHED_METRICS = METRICS[METRICS.index("n_events_matched"):]


# --- binning -----------------------------------------------------------------------

def parse_range(text: str) -> tuple[float, float] | None:
    """'80, 110' -> (80, 110); '' -> None (automatic)."""
    if not text.strip():
        return None
    parts = [p for p in re.split(r"[,\s]+", text.strip()) if p]
    try:
        low, high = (float(p) for p in parts)
    except ValueError:
        raise ComparisonError(f"range {text!r} must be two numbers, low and "
                              "high, e.g. '80, 110'") from None
    if not (np.isfinite(low) and np.isfinite(high) and high > low):
        raise ComparisonError(f"range {text!r}: high must be above low")
    return low, high


def auto_range(values: np.ndarray) -> tuple[float, float]:
    """AUTO_QUANTILES of the values, widened where they are all one value."""
    low, high = (float(q) for q in np.quantile(values, AUTO_QUANTILES))
    if high <= low:
        half = max(abs(low) * 0.01, 0.5)
        low, high = low - half, high + half
    return low, high


def difference_range(diff: np.ndarray) -> tuple[float, float]:
    """Symmetric about 0, spanning the changed events' differences: most
    differences of a reprocessing may be exactly 0."""
    changed = diff[diff != 0]
    if changed.size == 0:
        return -1.0, 1.0
    limit = float(np.max(np.abs(np.quantile(changed, AUTO_QUANTILES))))
    return -limit, limit


def edges_for(low: float, high: float, n_bins: float) -> np.ndarray:
    if n_bins != int(n_bins):
        raise ComparisonError(f"n_bins must be a whole number, got {n_bins:g}")
    return np.linspace(low, high, int(n_bins) + 1)


# --- outputs -----------------------------------------------------------------------

def _label(paths: list[Path]) -> str:
    return paths[0].name + (f" (+{len(paths) - 1} more)" if len(paths) > 1 else "")


def ratio_points(h_new: np.ndarray, h_ref: np.ndarray, scale_new: float,
                 scale_ref: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The ratio panel: which bins to show, new/ref and its error.

    Every bin with reference entries is shown, so a population the new sample
    lost shows as ratio 0 rather than vanishing. The error is
    ratio * sqrt(1/n_new + 1/n_ref), written so it does not divide by n_new
    (an empty new bin gets the new side's sqrt(0) = 0, as in the top panel).
    """
    shown = h_ref > 0
    with np.errstate(all="ignore"):
        per_ref = scale_new / (h_ref * scale_ref)
        ratio = h_new * per_ref
        error = per_ref * np.sqrt(h_new + h_new ** 2 / h_ref)
    return shown, ratio, error


def plot_comparison(path: Path, edges: np.ndarray, h_new: np.ndarray,
                    h_ref: np.ndarray, scale_new: float, scale_ref: float,
                    metrics: dict, labels: tuple[str, str], variable: str,
                    selection: str, normalize: str) -> None:
    """Both distributions, scaled, over their ratio new/ref."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True,
                                      gridspec_kw={"height_ratios": (3, 1)})
    centers = (edges[:-1] + edges[1:]) / 2
    top.stairs(h_ref * scale_ref, edges, color=REF_COLOR, lw=1.5, label="reference")
    top.errorbar(centers, h_new * scale_new, yerr=np.sqrt(h_new) * scale_new,
                 fmt="o", markersize=3, color=NEW_COLOR, capsize=0, label="new")
    ylabel = {"events": "per event", "area": "fraction", "none": "entries"}[normalize]
    top.set_ylabel(f"selected / bin ({ylabel})")
    top.set_title(variable, loc="left", fontweight="bold")
    top.legend(fontsize=8, loc="upper right")
    top.grid(color="#e4e3df", linewidth=0.8)
    stats = (f"         new       ref\n"
             f"n      {metrics['n_selected_new']:>9.0f} {metrics['n_selected_ref']:>9.0f}\n"
             f"mean   {metrics['mean_new']:>9.4g} {metrics['mean_ref']:>9.4g}\n"
             f"median {metrics['median_new']:>9.4g} {metrics['median_ref']:>9.4g}\n"
             f"sig68  {metrics['sigma68_new']:>9.4g} {metrics['sigma68_ref']:>9.4g}\n"
             f"KS p = {metrics['ks_pvalue']:.3g}")
    top.text(0.02, 0.97, stats, transform=top.transAxes, va="top", ha="left",
             family="monospace", fontsize=8)

    shown, ratio, error = ratio_points(h_new, h_ref, scale_new, scale_ref)
    bottom.errorbar(centers[shown], ratio[shown], yerr=error[shown], fmt="o",
                    markersize=3, color=NEW_COLOR, capsize=0)
    bottom.axhline(1.0, color=REF_COLOR, lw=1)
    bottom.set_ylabel("new / ref")
    bottom.set_xlabel(variable)
    bottom.grid(color="#e4e3df", linewidth=0.8)
    bottom.set_xlim(edges[0], edges[-1])
    fig.text(0.01, 0.01, f"new: {labels[0]}\nreference: {labels[1]}\n"
             f"selection: {selection or '(all)'}", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_difference(path: Path, edges: np.ndarray, h_diff: np.ndarray,
                    metrics: dict, variable: str, selection: str) -> None:
    """Per matched event: new - ref of its first selected object."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.stairs(h_diff, edges, color=NEW_COLOR, lw=1.5)
    if h_diff.any():
        ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_xlabel(f"new - reference: {variable}")
    ax.set_ylabel("matched events / bin")
    ax.set_title("Same events, new minus reference", loc="left", fontweight="bold")
    ax.grid(color="#e4e3df", linewidth=0.8)
    stats = (f"matched   {metrics['n_events_matched']:g}\n"
             f"in both   {metrics['n_events_both_selected']:g}\n"
             f"changed   {metrics['n_events_changed']:g}\n"
             f"gained    {metrics['n_events_gained']:g}\n"
             f"lost      {metrics['n_events_lost']:g}\n"
             f"median    {metrics['diff_median']:.4g}\n"
             f"sigma68   {metrics['diff_sigma68']:.4g}")
    ax.text(0.02, 0.97, stats, transform=ax.transAxes, va="top", ha="left",
            family="monospace", fontsize=9)
    fig.text(0.01, 0.01, f"selection: {selection or '(all)'}", fontsize=7,
             color="#52514e")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_root(path: Path, hists: dict[str, tuple[np.ndarray, np.ndarray]],
               variable: str) -> None:
    """Unnormalized TH1Ds: `new`, `ref`, and `diff` when events matched."""
    import uproot

    titles = {"new": "new sample", "ref": "reference sample",
              "diff": "matched events, new - ref"}
    with uproot.recreate(path) as f:
        for name, (counts, edges) in hists.items():
            f[name] = _th1(counts, edges, titles[name], variable)


# --- the analysis ------------------------------------------------------------------

def _scale(normalize: str, n_events: int, n_values: int) -> float:
    # An empty side's histogram is all zero whatever its scale: 1 keeps it so.
    if normalize == "events" and n_events:
        return 1.0 / n_events
    if normalize == "area" and n_values:
        return 1.0 / n_values
    return 1.0


def run(context: RunContext) -> RunOutcome:
    """Read both samples, compare their distributions, and match their
    events where they share them."""
    params = context.params
    level_name, variable = params["level"], params["variable"]
    selection, normalize = params["selection"], params["normalize"]
    extra: dict = {"level": level_name, "variable": variable, "selection": selection}
    try:
        if level_name not in LEVELS:
            raise ComparisonError(f"level {level_name!r} must be one of "
                                  f"{', '.join(LEVELS)}")
        if params["match_events"] not in MATCH_MODES:
            raise ComparisonError(f"match_events {params['match_events']!r} must "
                                  f"be one of {', '.join(MATCH_MODES)}")
        if normalize not in NORMALIZATIONS:
            raise ComparisonError(f"normalize {normalize!r} must be one of "
                                  f"{', '.join(NORMALIZATIONS)}")
        fixed_range = parse_range(params["range"])
        level = LEVELS[level_name]
        samples = {}
        for side, paths in (("new", context.input_paths),
                            ("reference", context.reference_paths)):
            try:
                samples[side] = read_sample(paths, level, selection, variable)
            except (NtupleError, SelectionError, ComparisonError) as exc:
                raise ComparisonError(f"{side} sample: {exc}") from None
        new, ref = samples["new"], samples["reference"]
        # One empty side is a result (every object lost, or the first ones
        # gained); only with both empty is there nothing to compare.
        if new.finite.size == 0 and ref.finite.size == 0:
            raise ComparisonError(
                f"no {level.noun} in either sample passes the selection with a "
                f"value of {variable!r} (new sample: {new.n_objects} "
                f"{level.noun}s in {new.n_events} events; reference sample: "
                f"{ref.n_objects} in {ref.n_events}), so there is nothing to "
                "compare")
        matching = match(new, ref, params["match_events"])
        both = np.concatenate([new.finite, ref.finite])
        edges = edges_for(*(fixed_range or auto_range(both)), params["n_bins"])
    except ComparisonError as exc:
        return RunOutcome(error=str(exc), extra=extra)

    metrics = distribution_metrics(new, ref)
    if matching["matched"]:
        matched, diff = matched_metrics(new, ref, matching["i_new"], matching["i_ref"])
    else:
        matched, diff = dict.fromkeys(MATCHED_METRICS, float("nan")), np.array([])
        matched["n_events_matched"] = 0.0
    metrics.update(matched)
    metrics = {key: metrics[key] for key in METRICS}

    h_new = np.histogram(new.finite, bins=edges)[0].astype(np.float64)
    h_ref = np.histogram(ref.finite, bins=edges)[0].astype(np.float64)
    hists = {"new": (h_new, edges), "ref": (h_ref, edges)}
    outdir = context.outdir
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figure = figdir / FIGURE_NAME
    plot_comparison(figure, edges, h_new, h_ref,
                    _scale(normalize, new.n_events, new.finite.size),
                    _scale(normalize, ref.n_events, ref.finite.size), metrics,
                    (_label(context.input_paths), _label(context.reference_paths)),
                    variable, selection, normalize)
    files = [str(figure)]
    if diff.size:
        # An odd count centres a bin on 0, where unchanged events sit.
        diff_edges = edges_for(*difference_range(diff), params["n_bins"] // 2 * 2 + 1)
        h_diff = np.histogram(diff, bins=diff_edges)[0].astype(np.float64)
        hists["diff"] = (h_diff, diff_edges)
        diff_figure = figdir / DIFFERENCE_FIGURE_NAME
        plot_difference(diff_figure, diff_edges, h_diff, metrics, variable, selection)
        files.append(str(diff_figure))
        extra["diff_outside_histogram_range"] = int(diff.size - h_diff.sum())
    root_path = outdir / ROOT_NAME
    write_root(root_path, hists, variable)
    files.insert(0, str(root_path))

    extra["bin_edges"] = [float(edges[0]), float(edges[-1]), int(edges.size - 1)]
    extra["matching"] = {k: v for k, v in matching.items() if not k.startswith("i_")}
    extra["n_selected_undefined"] = {
        "new": int(new.values.size - new.finite.size),
        "ref": int(ref.values.size - ref.finite.size)}
    extra["outside_histogram_range"] = {
        "new": int(new.finite.size - h_new.sum()),
        "ref": int(ref.finite.size - h_ref.sum())}
    return RunOutcome(metrics=metrics, files=files, extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    text = (f"mean {metrics['mean_new']:.4g} (new) vs {metrics['mean_ref']:.4g} "
            f"(ref), shift {metrics['mean_shift']:+.3g} +- "
            f"{metrics['mean_shift_err']:.2g}; median shift "
            f"{metrics['median_shift']:+.3g}; sigma68 {metrics['sigma68_new']:.3g} "
            f"vs {metrics['sigma68_ref']:.3g}; {metrics['n_selected_new']:g} vs "
            f"{metrics['n_selected_ref']:g} selected in {metrics['n_events_new']:g} "
            f"vs {metrics['n_events_ref']:g} events; KS p {metrics['ks_pvalue']:.3g}.")
    if metrics["n_events_matched"] > 0:
        text += (f" Matched {metrics['n_events_matched']:g} events: "
                 f"{metrics['n_events_changed']:g} changed value (median "
                 f"{metrics['diff_median']:+.3g}, sigma68 "
                 f"{metrics['diff_sigma68']:.3g}), {metrics['n_events_gained']:g} "
                 f"gained and {metrics['n_events_lost']:g} lost a selected object.")
    else:
        text += " Events not matched (no common run/subrun/event)."
    return text


def _names(level: str) -> str:
    return "; ".join(f"{k} ({v})" for k, v in LEVELS[level].variables.items())


SPEC = AnalysisSpec(
    name="variable_comparison_ntuple",
    input_kind="root_file",
    combines_files=True,
    takes_reference=True,
    run=run,
    description=(
        "Compare one track or calorimeter-cluster variable (or an expression "
        "of them) between two sets of EventNtuple files: the input (new) and "
        "a reference (baseline), e.g. before and after a code change. "
        "Overlays the selected objects' distributions with their ratio, "
        "mean/median shifts and a KS test; when the samples share events "
        "(run/subrun/event), also compares each event's first selected object "
        "in both: how many changed, gained or lost, and by how much. Plots "
        "and ROOT histograms. No job is run."
    ),
    metrics=METRICS,
    parameters=(
        ParamSpec(
            name="level", kind="text", default="track",
            description="What the variable belongs to: 'track' (EventNtuple "
                        "tracks) or 'cluster' (calorimeter clusters). Sets "
                        "which names `variable` and `selection` may use.",
        ),
        ParamSpec(
            name="variable", kind="text",
            description=(
                "The quantity compared: a name, or an expression of them "
                "(+ - * /, abs/sqrt/hypot/min/max/log/exp), e.g. 'p_front', "
                "'p_front - mc_p_front', 'nactive', or for clusters 'energy', "
                "'dt'. mc_* (and for clusters de, de_frac, dt) need simulated "
                "files. Track names: " + _names("track")
                + ". Cluster names: " + _names("cluster") + "."
            ),
        ),
        ParamSpec(
            name="selection", kind="text", default="", allow_empty=True,
            description=(
                "Which objects enter, as a condition over the same names as "
                "`variable` (comparisons, and/or/not or &&/||/!). Empty (the "
                "default) takes every track or cluster. For good "
                "reconstructed electrons use the trigger analyses' cut: "
                "'" + DEFAULT_SELECTION + "'. A missing value is NaN "
                "and fails any cut on it."
            ),
        ),
        ParamSpec(
            name="match_events", kind="text", default="auto",
            description=(
                "Whether to compare event by event: 'auto' (the default) "
                "does when at least half of the smaller sample's events "
                "(run, subrun, event) are in both, as for the same events "
                "reprocessed; 'yes' insists and fails if none are shared; "
                "'no' compares only the distributions. A sample that repeats "
                "an event id cannot be paired: 'auto' does not match it and "
                "'yes' fails."
            ),
        ),
        ParamSpec(
            name="normalize", kind="text", default="events",
            description=(
                "How the plotted distributions are scaled: 'events' (the "
                "default; per event, so a change in how many objects pass "
                "shows), 'area' (each to unit area, shape only), or 'none'. "
                "The ROOT histograms are never scaled."
            ),
        ),
        ParamSpec(
            name="range", kind="text", default="", allow_empty=True,
            description=(
                "Histogram range 'low, high' in the variable's units, e.g. "
                "'80, 110'. Empty (the default) spans the 0.5%-99.5% "
                "quantiles of both samples. The metrics use every value."
            ),
        ),
        ParamSpec(name="n_bins", default=DEFAULT_N_BINS, minimum=1,
                  maximum=MAX_BINS,
                  description="Number of bins, for the distributions and the "
                              "matched-event difference."),
    ),
    summarize=summarize,
    input_hint=(
        "EventNtuple file(s) (nts.*.root, tree EventNtuple/ntuple) as the new "
        "sample, and reference_file(s) or reference_dataset of the same kind "
        "as the baseline. Several files on either side are combined."
    ),
)
