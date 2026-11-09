"""Trigger efficiency from EventNtuple files in bins of a track variable.

`trigger_efficiency_ntuple` with the selected events also binned in a
quantity of their track, e.g. the reconstructed momentum at the tracker front
(`p_front`), the true one (`mc_p_front`), or the event's beam intensity
(`mc_npot`, true protons on target per microbunch), giving the curve of
any of the given paths and of each path. The variables, the selection and
the reading are that module's; this one only bins.

An event can have several tracks passing the selection. It is entered once,
at the value of its first passing track (the ntuple's track order), so the
binned counts add up to the same selected events as the overall efficiency,
less the ones outside the binning.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

from ..eventntuple import (CRV_VARIABLES, MC_VARIABLES, TRACK_VARIABLES, NtupleError,
                           check_names, parse_trigger_names, read_file)
from ..selection import SelectionError, apply_selection, evaluate_expression
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .trigger import binomial
from .trigger_efficiency_ntuple import DEFAULT_SELECTION, efficiency_metrics

DEFAULT_VARIABLE = "p_front"
# 2 MeV/c bins over the default selection's momentum range, up to past the
# CE endpoint (104.97 MeV/c).
DEFAULT_N_BINS, DEFAULT_LOW, DEFAULT_HIGH = 15, 80.0, 110.0
MAX_BINS = 1000

# What each run writes into output_dir, overwritten by a rerun.
FIGURE_NAME = "trigger_efficiency_binned.png"
TABLE_NAME = "trigger_efficiency_binned.csv"
ROOT_NAME = "trigger_efficiency_binned.root"


# --- the binning ---------------------------------------------------------------

def bin_edges(n_bins: float, low: float, high: float, edges_text: str) -> np.ndarray:
    """The edges: `edges_text` ('80, 90, 100, 105') if given, else n_bins
    uniform bins over [low, high)."""
    if edges_text.strip():
        try:
            edges = np.array([float(e) for e in re.split(r"[,\s]+", edges_text.strip())
                              if e], dtype=np.float64)
        except ValueError:
            raise NtupleError(f"bin_edges {edges_text!r} must be numbers separated "
                              "by commas, e.g. '80, 90, 100, 105'") from None
        if edges.size < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise NtupleError(f"bin_edges {edges_text!r} must be at least two "
                              "finite numbers in increasing order")
        return edges
    if n_bins != int(n_bins):
        raise NtupleError(f"n_bins must be a whole number, got {n_bins:g}")
    if not high > low:
        raise NtupleError(f"high ({high:g}) must be above low ({low:g})")
    return np.linspace(low, high, int(n_bins) + 1)


def event_values(variables: dict[str, np.ndarray], n_events: int, selection: str,
                 variable: str) -> tuple[np.ndarray, np.ndarray, int]:
    """(selected events, each one's value from its first passing track,
    passing tracks). Unselected events have NaN."""
    tracks = {k: v for k, v in variables.items() if not k.startswith("_")}
    n_tracks = variables["_event"].size
    passing = apply_selection(selection, tracks, n_tracks)
    values = evaluate_expression(variable, tracks, n_tracks)
    index = np.flatnonzero(passing)
    # Tracks are in event order, so the first of each event's is its first.
    events, first = np.unique(variables["_event"][index], return_index=True)
    selected = np.zeros(n_events, dtype=bool)
    selected[events] = True
    value = np.full(n_events, np.nan)
    value[events] = values[index[first]]
    return selected, value, int(passing.sum())


def binned_counts(selected: np.ndarray, value: np.ndarray,
                  triggers: dict[str, np.ndarray], edges: np.ndarray) -> dict:
    """Selected and triggered events per bin, overall and per path, with the
    selected events that fall outside the binning or have no value."""
    any_path = np.zeros(selected.size, dtype=bool)
    for decision in triggers.values():
        any_path |= decision
    defined = selected & np.isfinite(value)
    inside = defined & (value >= edges[0]) & (value < edges[-1])

    def histogram(mask):
        return np.histogram(value[inside & mask], bins=edges)[0]

    return {
        "n_events": int(selected.size),
        "n_selected": int(selected.sum()),
        "n_triggered": int((selected & any_path).sum()),
        "per_path": {p: int((selected & d).sum()) for p, d in triggers.items()},
        "selected": histogram(selected),
        "triggered": histogram(any_path),
        "path_triggered": {p: histogram(d) for p, d in triggers.items()},
        "n_undefined": int((selected & ~np.isfinite(value)).sum()),
        "n_underflow": int((defined & (value < edges[0])).sum()),
        "n_overflow": int((defined & (value >= edges[-1])).sum()),
    }


def add_counts(totals: dict | None, counts: dict) -> dict:
    """Sum two files' binned_counts."""
    if totals is None:
        return counts
    out = {}
    for key, value in totals.items():
        if isinstance(value, dict):
            out[key] = {p: value[p] + counts[key][p] for p in value}
        else:
            out[key] = value + counts[key]
    return out


def bin_rows(counts: dict, edges: np.ndarray) -> list[dict]:
    """One row per bin: its range, counts and efficiencies."""
    rows = []
    for i in range(edges.size - 1):
        n = int(counts["selected"][i])
        k = int(counts["triggered"][i])
        eff, err = binomial(k, n)
        paths = {}
        for p, hist in counts["path_triggered"].items():
            pe, perr = binomial(int(hist[i]), n)
            paths[p] = {"passed": int(hist[i]), "efficiency": pe, "efficiency_err": perr}
        rows.append({"low": float(edges[i]), "high": float(edges[i + 1]),
                     "n_selected": n, "n_triggered": k,
                     "efficiency": eff, "efficiency_err": err, "paths": paths})
    return rows


METRICS = ("n_events", "n_selected", "n_triggered", "efficiency",
           "efficiency_err", "n_tracks_selected", "n_bins", "n_in_range",
           "n_triggered_in_range", "efficiency_in_range",
           "efficiency_in_range_err", "n_underflow", "n_overflow", "n_undefined")


def binned_metrics(counts: dict, n_tracks_selected: int, n_bins: int) -> dict[str, float]:
    """The overall efficiency (as trigger_efficiency_ntuple's), and the
    efficiency over the selected events inside the binning."""
    metrics = efficiency_metrics({**counts, "n_tracks_selected": n_tracks_selected})
    n_in = int(counts["selected"].sum())
    k_in = int(counts["triggered"].sum())
    eff, err = binomial(k_in, n_in)
    metrics.update(n_bins=float(n_bins), n_in_range=float(n_in),
                   n_triggered_in_range=float(k_in), efficiency_in_range=eff,
                   efficiency_in_range_err=err,
                   n_underflow=float(counts["n_underflow"]),
                   n_overflow=float(counts["n_overflow"]),
                   n_undefined=float(counts["n_undefined"]))
    return metrics


# --- output ----------------------------------------------------------------------

def write_table(path: Path, rows: list[dict], paths: list[str]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["low", "high", "n_selected", "n_triggered", "efficiency",
                         "efficiency_err"]
                        + [f"{p}_{c}" for p in paths
                           for c in ("passed", "efficiency", "efficiency_err")])
        for row in rows:
            writer.writerow(
                [row["low"], row["high"], row["n_selected"], row["n_triggered"],
                 f"{row['efficiency']:.6g}", f"{row['efficiency_err']:.6g}"]
                + [v for p in paths for v in (
                    row["paths"][p]["passed"],
                    f"{row['paths'][p]['efficiency']:.6g}",
                    f"{row['paths'][p]['efficiency_err']:.6g}")])


def write_root(path: Path, counts: dict, edges: np.ndarray, variable: str) -> None:
    """TH1Ds of the selected and triggered events (any path, and each path's
    `triggered_<path>`), for a TEfficiency or a ratio in ROOT."""
    import uproot

    with uproot.recreate(path) as f:
        f["selected"] = _th1(counts["selected"], edges, "selected events", variable)
        f["triggered"] = _th1(counts["triggered"], edges, "triggered by any path",
                              variable)
        for p, hist in counts["path_triggered"].items():
            f[f"triggered_{p}"] = _th1(hist, edges, f"triggered by {p}", variable)


def _th1(hist, edges, title, variable):
    """A TH1D with sumw2 = counts (unweighted), titled for the variable.
    Under/overflow are left empty: those events are in the metrics. The
    moments are from the bin centres, as ROOT's would be for a rebuilt
    histogram, so its GetMean/GetRMS are the binned ones, not zero."""
    import uproot

    counts = np.asarray(hist, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.float64)
    centers = (edges[:-1] + edges[1:]) / 2
    data = np.concatenate([[0.0], counts, [0.0]])
    return uproot.writing.identify.to_TH1x(
        fName=None, fTitle=f"{title};{variable};events",
        data=data, fEntries=float(counts.sum()), fTsumw=float(counts.sum()),
        fTsumw2=float(counts.sum()), fTsumwx=float((counts * centers).sum()),
        fTsumwx2=float((counts * centers ** 2).sum()),
        fSumw2=data.copy(),
        fXaxis=uproot.writing.identify.to_TAxis(
            fName="xaxis", fTitle=variable, fNbins=edges.size - 1,
            fXmin=float(edges[0]), fXmax=float(edges[-1]), fXbins=edges))


def plot(path: Path, rows: list[dict], paths: list[str], variable: str,
         selection: str) -> None:
    """The efficiency curve: any path, and each path, against the variable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    filled = [r for r in rows if r["n_selected"] > 0]
    x = np.array([(r["low"] + r["high"]) / 2 for r in filled])
    xerr = np.array([(r["high"] - r["low"]) / 2 for r in filled])
    fig, ax = plt.subplots(figsize=(7, 5))
    series = [("any path", [r["efficiency"] for r in filled],
               [r["efficiency_err"] for r in filled])]
    if len(paths) > 1:
        series += [(p, [r["paths"][p]["efficiency"] for r in filled],
                    [r["paths"][p]["efficiency_err"] for r in filled]) for p in paths]
    for label, y, yerr in series:
        ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o", markersize=4,
                    capsize=0, label=label)
    ax.set_xlabel(variable)
    ax.set_ylabel("trigger efficiency")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(selection or "all events", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --- the analysis ------------------------------------------------------------------

def run(context: RunContext) -> RunOutcome:
    """Select and bin events in the ntuple(s), and count the trigger
    decisions in each bin."""
    params = context.params
    selection, variable = params["selection"], params["variable"]
    extra: dict = {"selection": selection, "variable": variable}
    try:
        paths = parse_trigger_names(params["trigger_paths"])
        edges = bin_edges(params["n_bins"], params["low"], params["high"],
                          params["bin_edges"])
        if edges.size - 1 > MAX_BINS:
            raise NtupleError(f"{edges.size - 1} bins is more than {MAX_BINS}")
        totals, n_tracks, per_file = None, 0, []
        for path in context.input_paths:
            variables, triggers, n_events = read_file(path, paths)
            check_names(variables, selection, variable)
            selected, value, n_passing = event_values(variables, n_events,
                                                      selection, variable)
            counts = binned_counts(selected, value, triggers, edges)
            per_file.append({"file": str(path), "n_events": n_events,
                             "n_selected": counts["n_selected"],
                             "n_triggered": counts["n_triggered"]})
            totals = add_counts(totals, counts)
            n_tracks += n_passing
    except (NtupleError, SelectionError) as exc:
        return RunOutcome(error=str(exc), extra=extra)
    extra["bin_edges"] = edges.tolist()
    if totals["n_selected"] == 0:
        return RunOutcome(
            error=f"no event of {totals['n_events']} has a track passing the "
                  "selection, so there is no efficiency to measure",
            extra=extra)

    rows = bin_rows(totals, edges)
    paths_out = {}
    for p in paths:
        eff, err = binomial(totals["per_path"][p], totals["n_selected"])
        paths_out[p] = {"passed": totals["per_path"][p],
                        "efficiency": eff, "efficiency_err": err}
    outdir = context.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    table, root_path, figure = (outdir / TABLE_NAME, outdir / ROOT_NAME,
                                outdir / FIGURE_NAME)
    write_table(table, rows, paths)
    write_root(root_path, totals, edges, variable)
    plot(figure, rows, paths, variable, selection)
    extra.update(bins=rows, paths=paths_out,
                 **({"per_file": per_file} if len(per_file) > 1 else {}))
    return RunOutcome(metrics=binned_metrics(totals, n_tracks, edges.size - 1),
                      files=[str(figure), str(table), str(root_path)], extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (f"trigger efficiency {metrics['efficiency']:.4g} "
            f"+- {metrics['efficiency_err']:.2g} over {metrics['n_selected']:g} "
            f"selected events (of {metrics['n_events']:g}); "
            f"{metrics['efficiency_in_range']:.4g} "
            f"+- {metrics['efficiency_in_range_err']:.2g} over the "
            f"{metrics['n_in_range']:g} inside the {metrics['n_bins']:g} bins "
            f"({metrics['n_underflow']:g} below, {metrics['n_overflow']:g} above, "
            f"{metrics['n_undefined']:g} with no value). Per-bin efficiencies in "
            "metadata.bins, with a figure, CSV table and ROOT histograms in files.")


_VARIABLE_LIST = (", ".join(TRACK_VARIABLES) + "; in simulated files, "
                  + ", ".join(MC_VARIABLES) + "; and where the file has CRV "
                  "coincidences, " + ", ".join(CRV_VARIABLES))

SPEC = AnalysisSpec(
    name="trigger_efficiency_ntuple_binned",
    input_kind="root_file",
    combines_files=True,
    run=run,
    description=(
        "Trigger efficiency from EventNtuple file(s) as a function of a track "
        "variable, e.g. the reconstructed (p_front) or true (mc_p_front) "
        "momentum at the tracker front, or the beam intensity (mc_npot): of "
        "the events with a track passing a "
        "selection, binned in that track's variable, the fraction any of the "
        "given trigger paths accepted (trig_<path> branches), per bin "
        "(metadata.bins) and overall, and each path's own. Writes the curve "
        "as a figure, a CSV table and ROOT histograms. No job is run."
    ),
    metrics=METRICS,
    parameters=(
        ParamSpec(
            name="trigger_paths",
            description="Trigger paths whose decisions to read, separated by "
                        "commas, e.g. 'apr_TrkDe_80m70p, cpr_TrkDe_80m70p' "
                        "(the trig_ prefix is optional). An event counts as "
                        "triggered if any of them accepted it. A name the "
                        "ntuple has no branch for is reported with the list "
                        "it does have.",
            kind="text",
        ),
        ParamSpec(
            name="variable",
            description=(
                "What to bin in: a track variable or an expression of them "
                "(same syntax as selection), e.g. 'p_front', 'mc_p_front', "
                "'p_front - mc_p_front', 'cos_front', or the beam intensity "
                "'mc_npot / 1e6' (millions of protons per microbunch; pass "
                "low/high to match, e.g. 0 and 100). Each selected event is "
                "entered once, at the value of its first track passing the "
                "selection. Variables: " + _VARIABLE_LIST + "."
            ),
            default=DEFAULT_VARIABLE, kind="text",
        ),
        ParamSpec(name="n_bins", description="Number of uniform bins over "
                  "[low, high). Ignored when bin_edges is given.",
                  default=DEFAULT_N_BINS, minimum=1, maximum=MAX_BINS),
        ParamSpec(name="low", description="Lower edge of the uniform binning, "
                  "in the variable's units (MeV/c for momenta). Ignored when "
                  "bin_edges is given.", default=DEFAULT_LOW),
        ParamSpec(name="high", description="Upper edge of the uniform binning "
                  "(excluded). Ignored when bin_edges is given.",
                  default=DEFAULT_HIGH),
        ParamSpec(name="bin_edges", description="Explicit, increasing bin "
                  "edges separated by commas, e.g. '80, 95, 100, 103, 105, "
                  "110', for variable-width bins. Empty (the default) uses "
                  "n_bins/low/high.", default="", kind="text", allow_empty=True),
        ParamSpec(
            name="selection",
            description=(
                "Track selection: an event is selected if at least one track "
                "passes. Comparisons, arithmetic, and/or/not (or &&/||/!) and "
                "abs/sqrt/hypot/min/max/log/exp over the same variables as "
                "`variable`. *_front quantities are at the downstream-going "
                "TT_Front crossing; fit parameters (d0, maxr, tandip, t0err, "
                "...) at TT_Mid; a missing value is NaN and fails any cut on "
                "it. The default is a converged downstream e- fit with p > 80 "
                "MeV/c at the tracker front, >= 15 active hits and chisq/ndof "
                "< 5. For a curve in a true quantity, loosen the cut on the "
                "reconstructed one (e.g. drop p_front > 80) so it does not "
                "sculpt the curve. An empty string selects every event with "
                "a track."
            ),
            default=DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "EventNtuple file(s) (nts.*.root, tree EventNtuple/ntuple) with "
        "trig_<path> branches, e.g. nts.mu2e.CeMLeadingLogMix1BB.<version>.root; "
        "mc_* variables need a simulated one. Pass data_files for several, "
        "combined into one result."
    ),
)
