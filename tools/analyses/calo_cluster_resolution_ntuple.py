"""Calorimeter cluster energy and time resolution from simulated EventNtuples.

For every calorimeter cluster passing a cut expression (tools/selection.py
syntax, over CLUSTER_VARIABLES), the cluster's MC truth (`caloclustersmc`,
one entry per `caloclusters` entry) gives

  de  reconstructed energy minus the true energy deposited in the cluster's
      crystals by every particle (`etot`)
  dt  reconstructed time minus the earliest true deposit of the cluster's
      leading particle, the one that deposited the most (`tprimary`)

The truth is the EventNtuple's CaloClusterInfoMC, filled from Offline's
CaloClusterMC (EventNtuple/src/InfoMCStructHelper.cc). Two things about it
are easy to get wrong:

* "primary" is the cluster's leading contributor, not the event's generated
  particle. How that particle is related to the event's primary is
  `mc_rel` (MCRelationship: -1 none, 0 same, 1 daughter, 2 mother,
  3 sibling, 4-6 the same once removed), so `mc_rel == 0` is a cluster
  made by the generated particle itself (the CE, in a CE sample).
* `tavg` is not used. Newer EventNtuple versions fill it with `tprimary`
  (its own comment calls that a FIXME); the files written before carry
  something else, tens of ns away. Its meaning depends on the version.

Both distributions have tails (leakage, pileup in the cluster), so beside the
mean and rms the metrics give the median and sigma68, as in
momentum_resolution_ntuple. de_frac is de over the true energy.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..eventntuple import NtupleError, _flat, _numpy, open_tree
from ..selection import SelectionError, apply_selection
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from ..spectrum import Spectrum
from .momentum_resolution_ntuple import histogram, summary

RECO_LEAVES = {
    "energy": "caloclusters.energyDep_",
    "energy_err": "caloclusters.energyDepErr_",
    "time": "caloclusters.time_",
    "time_err": "caloclusters.timeErr_",
    "disk": "caloclusters.diskID_",
    "size": "caloclusters.size_",
    "split": "caloclusters.isSplit_",
}
COG_LEAVES = ("caloclusters.cog_.fCoordinates.fX", "caloclusters.cog_.fCoordinates.fY")
MC_LEAVES = {
    "mc_nsim": "caloclustersmc.nsim",
    "mc_etot": "caloclustersmc.etot",
    "mc_eprimary": "caloclustersmc.eprimary",
    "mc_tprimary": "caloclustersmc.tprimary",
    "mc_rel": "caloclustersmc.prel._rel",
}
EVENT_LEAVES = ("event", "run", "subrun")

CLUSTER_VARIABLES: dict[str, str] = {
    "energy": "reconstructed cluster energy (MeV)",
    "energy_err": "its uncertainty (MeV)",
    "time": "reconstructed cluster time (ns)",
    "time_err": "its uncertainty (ns)",
    "disk": "calorimeter disk, 0 or 1",
    "size": "crystals in the cluster",
    "split": "1 if the cluster was split off another",
    "r": "centroid radius in the disk's frame (mm; the crystals span ~375-660)",
    "nclusters": "clusters in the event",
    "event": "event number", "run": "run number", "subrun": "subrun number",
    "mc_nsim": "sim particles depositing in the cluster (0: no truth)",
    "mc_etot": "true energy deposited by all of them (MeV)",
    "mc_eprimary": "true energy of the leading one (MeV)",
    "mc_tprimary": "earliest true deposit time of the leading one (ns)",
    "mc_purity": "mc_eprimary / mc_etot",
    "mc_rel": "the leading particle's relation to the event's generated "
              "primary: -1 none, 0 same, 1 daughter, 2 mother, 3 sibling",
    "de": "energy - mc_etot (MeV)",
    "de_frac": "de / mc_etot",
    "dt": "time - mc_tprimary (ns)",
}

# What needs caloclustersmc: absent when a file without it is read.
TRUTH_VARIABLES = tuple(name for name in CLUSTER_VARIABLES
                        if name.startswith("mc_") or name in ("de", "de_frac", "dt"))

# A cluster from the generated particle itself, e.g. the CE in a CE sample.
DEFAULT_SELECTION = "mc_rel == 0"


class CaloResolutionError(ValueError):
    """An input or request that cannot be used, worded for the caller."""


# --- reading -----------------------------------------------------------------------

def cluster_variables(arrays, with_truth: bool = True) -> dict[str, np.ndarray]:
    """CLUSTER_VARIABLES, one entry per cluster, from one file's leaves
    (events * clusters). A cluster without truth (mc_nsim 0) gets NaN for
    every MC quantity. Also returns `_event`, each cluster's event index.
    `with_truth=False` (a file without caloclustersmc) leaves out the
    TRUTH_VARIABLES."""
    import awkward as ak

    ncl = ak.num(arrays[RECO_LEAVES["energy"]], axis=1)
    event_of = np.repeat(np.arange(len(ncl)), ak.to_numpy(ncl))
    out: dict[str, np.ndarray] = {"_event": event_of}
    for name, leaf in RECO_LEAVES.items():
        out[name] = _flat(arrays[leaf])
    out["r"] = np.hypot(_flat(arrays[COG_LEAVES[0]]), _flat(arrays[COG_LEAVES[1]]))
    out["nclusters"] = ak.to_numpy(ncl).astype(np.float64)[event_of]
    for name in EVENT_LEAVES:
        out[name] = _numpy(arrays[name])[event_of]
    if not with_truth:
        return out

    for name, leaf in MC_LEAVES.items():
        out[name] = _flat(arrays[leaf])
    no_truth = out["mc_nsim"] <= 0
    for name in ("mc_etot", "mc_eprimary", "mc_tprimary", "mc_rel"):
        out[name][no_truth] = np.nan
    with np.errstate(all="ignore"):
        out["mc_purity"] = out["mc_eprimary"] / out["mc_etot"]
        out["de"] = out["energy"] - out["mc_etot"]
        out["de_frac"] = out["de"] / out["mc_etot"]
    out["dt"] = out["time"] - out["mc_tprimary"]
    return out


def read_file(path: Path, require_truth: bool = True) -> tuple[dict[str, np.ndarray], int]:
    """(cluster variables, number of events). A file without caloclustersmc
    is refused, or with `require_truth=False` read without the
    TRUTH_VARIABLES."""
    tree = open_tree(path)
    names = {key.rsplit("/", 1)[-1] for key in tree.keys()} | set(tree.keys())
    with_truth = "caloclustersmc" in names or MC_LEAVES["mc_etot"] in names
    if require_truth and not with_truth:
        raise NtupleError(
            f"{path.name}: no caloclustersmc branch — this analysis needs the "
            "clusters' MC truth, which only simulated EventNtuples carry")
    leaves = [*RECO_LEAVES.values(), *COG_LEAVES, *EVENT_LEAVES,
              *(MC_LEAVES.values() if with_truth else ())]
    batch = tree.arrays(filter_name=leaves, library="ak")
    absent = [leaf for leaf in leaves if leaf not in batch.fields]
    if absent:
        raise NtupleError(f"{path.name}: missing branch(es) {', '.join(absent)} "
                          "— an EventNtuple version this analysis does not know?")
    return cluster_variables(batch, with_truth), tree.num_entries


def check_truth_names(variables: dict, *expressions: str) -> None:
    """Refuse a TRUTH_VARIABLES name on a file without caloclustersmc,
    saying why, rather than as an unknown name."""
    named = {name for text in expressions
             for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)}
    asked = sorted((named & set(TRUTH_VARIABLES)) - set(variables))
    if asked:
        raise NtupleError(
            f"{', '.join(asked)} need the clusters' MC truth, and this file has "
            "no caloclustersmc branch (real data, or an ntuple written without MC)")


# --- the analysis ------------------------------------------------------------------

def select_clusters(variables: dict[str, np.ndarray], selection: str) -> np.ndarray:
    """The mask of clusters passing `selection`."""
    return apply_selection(selection, {k: v for k, v in variables.items()
                                       if not k.startswith("_")},
                           variables["_event"].size)


METRICS = ("n_events", "n_clusters", "n_clusters_selected", "n_clusters_with_truth",
           "de_mean", "de_rms", "de_median", "de_sigma68",
           "de_frac_median", "de_frac_sigma68",
           "dt_mean", "dt_rms", "dt_median", "dt_sigma68")

UNITS = {**{m: "MeV" for m in METRICS if m.startswith("de_")
            and not m.startswith("de_frac")},
         **{m: "ns" for m in METRICS if m.startswith("dt_")}}


def resolution_metrics(counts: dict[str, int], de: np.ndarray, de_frac: np.ndarray,
                       dt: np.ndarray) -> dict[str, float]:
    """The counts and the three distributions' summaries, in METRICS order."""
    metrics = {key: float(counts[key]) for key in METRICS[:3]}
    metrics["n_clusters_with_truth"] = float(de.size)
    metrics.update(summary(de, "de"))
    frac = summary(de_frac, "de_frac")
    metrics["de_frac_median"] = frac["de_frac_median"]
    metrics["de_frac_sigma68"] = frac["de_frac_sigma68"]
    metrics.update(summary(dt, "dt"))
    return {key: metrics[key] for key in METRICS}


def _write_outputs(outdir: Path, hists: dict[str, Spectrum], metrics: dict,
                   selection: str) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import uproot

    written = []
    root_path = outdir / "calo_cluster_resolution.root"
    with uproot.recreate(root_path) as f:
        for name, h in hists.items():
            f[name] = (h.values, h.edges())
    written.append(str(root_path))

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    labels = {
        "de": ("Reconstructed minus true cluster energy (MeV)",
               "Calorimeter cluster energy", "MeV"),
        "dt": ("Reconstructed cluster time minus earliest true deposit (ns)",
               "Calorimeter cluster time", "ns"),
    }
    for name, h in hists.items():
        xlabel, title, unit = labels[name]
        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.stairs(h.values, h.edges(), color="#2a78d6", lw=1.5)
        ax.set_yscale("log")
        ax.set_xlim(h.xmin, h.edges()[-1])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Clusters / {h.width:g} {unit}")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        stats = (f"clusters {metrics['n_clusters_with_truth']:g}\n"
                 f"mean     {metrics[name + '_mean']:.3f}\n"
                 f"rms      {metrics[name + '_rms']:.3f}\n"
                 f"median   {metrics[name + '_median']:.3f}\n"
                 f"sigma68  {metrics[name + '_sigma68']:.3f}")
        ax.text(0.02, 0.97, stats, transform=ax.transAxes, va="top", ha="left",
                family="monospace", fontsize=9)
        fig.text(0.01, 0.01, f"selection: {selection or '(all clusters)'}",
                 fontsize=7, color="#52514e")
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        path = figdir / f"cluster_{name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))
    return written


def _check_binning(params: dict) -> None:
    for name in ("de", "dt"):
        low, high = params[f"{name}_min"], params[f"{name}_max"]
        width = params[f"{name}_bin_width"]
        if high <= low:
            raise CaloResolutionError(f"{name}_max ({high:g}) must be above "
                                      f"{name}_min ({low:g})")
        if (high - low) / width > 100_000:
            raise CaloResolutionError(f"{name}_bin_width {width:g} gives over "
                                      f"100000 bins between {low:g} and {high:g}")
        # histogram() rounds the bin count, so a width over twice the range
        # would give no bins at all.
        if round((high - low) / width) < 1:
            raise CaloResolutionError(f"{name}_bin_width {width:g} gives no bins "
                                      f"between {low:g} and {high:g}: make it at "
                                      f"most {high - low:g}")


def run(context: RunContext) -> RunOutcome:
    """Fill the cluster dE and dT distributions over the ntuple(s)."""
    params = context.params
    selection = params["selection"]
    extra: dict = {"selection": selection}
    counts = {"n_events": 0, "n_clusters": 0, "n_clusters_selected": 0}
    de, de_frac, dt, per_file = [], [], [], []
    try:
        _check_binning(params)
        for path in context.input_paths:
            variables, n_events = read_file(path)
            passing = select_clusters(variables, selection)
            truth = passing & np.isfinite(variables["de"]) & np.isfinite(variables["dt"])
            counts["n_events"] += n_events
            counts["n_clusters"] += int(variables["_event"].size)
            counts["n_clusters_selected"] += int(passing.sum())
            de.append(variables["de"][truth])
            de_frac.append(variables["de_frac"][truth])
            dt.append(variables["dt"][truth])
            per_file.append({"file": str(path), "n_events": n_events,
                             "n_clusters_selected": int(passing.sum()),
                             "n_clusters_with_truth": int(truth.sum())})
    except (NtupleError, SelectionError, CaloResolutionError) as exc:
        return RunOutcome(error=str(exc), extra=extra)
    de, de_frac, dt = np.concatenate(de), np.concatenate(de_frac), np.concatenate(dt)
    if counts["n_clusters_selected"] == 0:
        return RunOutcome(
            error=f"no cluster of {counts['n_clusters']} (in {counts['n_events']} "
                  "events) passes the selection, so there is no distribution to fill",
            extra=extra)
    if de.size == 0:
        return RunOutcome(
            error=f"none of the {counts['n_clusters_selected']} selected clusters "
                  "has MC truth (caloclustersmc.nsim > 0) to compare with",
            extra=extra)

    metrics = resolution_metrics(counts, de, de_frac, dt)
    hists = {name: histogram(values, params[f"{name}_min"], params[f"{name}_max"],
                             params[f"{name}_bin_width"], name)
             for name, values in (("de", de), ("dt", dt))}
    # The summaries use every cluster; the plots and histograms only their range.
    extra["outside_histogram_range"] = {
        name: int(de.size - h.values.sum()) for name, h in hists.items()}
    if len(per_file) > 1:
        extra["per_file"] = per_file
    files = _write_outputs(context.outdir, hists, metrics, selection)
    return RunOutcome(metrics=metrics, files=files, extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (f"calo cluster dE median {metrics['de_median']:.3g} MeV, sigma68 "
            f"{metrics['de_sigma68']:.3g} MeV ({100 * metrics['de_frac_sigma68']:.2g}% "
            f"of the true energy); dT median {metrics['dt_median']:.3g} ns, sigma68 "
            f"{metrics['dt_sigma68']:.3g} ns; from {metrics['n_clusters_with_truth']:g} "
            f"clusters in {metrics['n_events']:g} events.")


def _binning(name: str, what: str, unit: str, low: float, high: float,
             width: float) -> tuple[ParamSpec, ...]:
    note = "The metrics use every cluster, whatever the range."
    return (
        ParamSpec(name=f"{name}_min", default=low,
                  description=f"Lower edge of the {what} histogram and plot ({unit}). {note}"),
        ParamSpec(name=f"{name}_max", default=high,
                  description=f"Upper edge of the {what} histogram and plot ({unit}). {note}"),
        ParamSpec(name=f"{name}_bin_width", default=width, minimum=1e-4,
                  description=f"Bin width of the {what} histogram ({unit})."),
    )


SPEC = AnalysisSpec(
    name="calo_cluster_resolution_ntuple",
    input_kind="root_file",
    combines_files=True,
    run=run,
    description=(
        "Calorimeter cluster energy and time resolution from simulated "
        "EventNtuple file(s): reconstructed minus true cluster energy (dE) and "
        "reconstructed time minus the leading particle's earliest true deposit "
        "(dT), for clusters passing a selection. Plots, ROOT histograms and "
        "mean/rms/median/sigma68. No job is run."
    ),
    metrics=METRICS,
    units=UNITS,
    parameters=(
        ParamSpec(
            name="selection",
            description=(
                "Cluster selection: every cluster passing it fills the "
                "distributions. Comparisons, arithmetic, and/or/not (or "
                "&&/||/!) and abs/sqrt/hypot/min/max/log/exp over: "
                + "; ".join(f"{k} ({v})" for k, v in CLUSTER_VARIABLES.items())
                + ". A cluster without truth has NaN MC values and fails any "
                "cut on them. The default, 'mc_rel == 0', takes the clusters "
                "the generated particle made itself (the CE in a CE sample); "
                "add e.g. 'and energy > 50'. An empty string takes every cluster."
            ),
            default=DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
        *_binning("de", "dE", "MeV", -10.0, 5.0, 0.1),
        *_binning("dt", "dT", "ns", -5.0, 10.0, 0.1),
    ),
    summarize=summarize,
    input_hint=(
        "Simulated EventNtuple file(s) with calorimeter MC truth (nts.*.root, "
        "tree EventNtuple/ntuple, branches caloclusters and caloclustersmc), "
        "e.g. nts.mu2e.CeMLeadingLogMix1BB.<version>.root; pass data_files "
        "for several, combined into one result."
    ),
)
