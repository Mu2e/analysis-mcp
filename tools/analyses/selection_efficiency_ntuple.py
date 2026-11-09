"""Selection efficiency from EventNtuple files: the fraction of events with a
track passing a cut expression (tools/selection.py syntax).

The track variables, and how they are read, are trigger_efficiency_ntuple's
(TRACK_VARIABLES), so a cut means the same thing in both analyses. An
optional `denominator` cut defines the sample the efficiency is measured on;
by default it is every event in the ntuple. An event passes when one and the
same track passes both the denominator and the selection, so the passing
events are always a subset of the denominator ones.

The denominator counts ntuple entries, not generated events: a production
ntuple may already have dropped events (e.g. by a trigger or a filter
upstream), and this analysis cannot see those.
"""

from __future__ import annotations

import numpy as np

from ..eventntuple import (CRV_VARIABLES, MC_VARIABLES, TRACK_VARIABLES, NtupleError,
                           check_names, read_file)
from ..selection import SelectionError, apply_selection
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .trigger import binomial


def efficiency_counts(variables: dict[str, np.ndarray], n_events: int,
                      selection: str, denominator: str = "") -> dict[str, int]:
    """Events (and tracks) passing the denominator, and the selection on top."""
    objects = {k: v for k, v in variables.items() if not k.startswith("_")}
    event_of_track = variables["_event"]
    passing = apply_selection(selection, objects, event_of_track.size)
    if denominator.strip():
        base = apply_selection(denominator, objects, event_of_track.size)
        n_denominator = int(np.unique(event_of_track[base]).size)
        passing = passing & base
    else:
        base = np.ones(event_of_track.size, dtype=bool)
        n_denominator = n_events
    return {"n_events": n_events,
            "n_denominator": n_denominator,
            "n_passed": int(np.unique(event_of_track[passing]).size),
            "n_tracks_denominator": int(base.sum()),
            "n_tracks_passed": int(passing.sum())}


METRICS = ("n_events", "n_denominator", "n_passed", "efficiency",
           "efficiency_err", "n_tracks_denominator", "n_tracks_passed")


def efficiency_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Passed over denominator events, with its binomial uncertainty."""
    efficiency, err = binomial(counts["n_passed"], counts["n_denominator"])
    return {"n_events": float(counts["n_events"]),
            "n_denominator": float(counts["n_denominator"]),
            "n_passed": float(counts["n_passed"]),
            "efficiency": efficiency, "efficiency_err": err,
            "n_tracks_denominator": float(counts["n_tracks_denominator"]),
            "n_tracks_passed": float(counts["n_tracks_passed"])}


def run(context: RunContext) -> RunOutcome:
    """Count the events passing the cuts in each ntuple, and combine them."""
    selection = context.params["selection"]
    denominator = context.params["denominator"]
    extra: dict = {"selection": selection, "denominator": denominator}
    totals = dict.fromkeys(("n_events", "n_denominator", "n_passed",
                            "n_tracks_denominator", "n_tracks_passed"), 0)
    per_file = []
    try:
        for path in context.input_paths:
            variables, _, n_events = read_file(path, [])
            check_names(variables, selection, denominator)
            counts = efficiency_counts(variables, n_events, selection, denominator)
            per_file.append({"file": str(path), "n_events": n_events,
                             "n_denominator": counts["n_denominator"],
                             "n_passed": counts["n_passed"]})
            for key in totals:
                totals[key] += counts[key]
    except (NtupleError, SelectionError) as exc:
        return RunOutcome(error=str(exc), extra=extra)
    if totals["n_denominator"] == 0:
        what = ("has a track passing the denominator" if denominator.strip()
                else "is in the file(s)")
        return RunOutcome(
            error=f"no event of {totals['n_events']} {what}, so there is no "
                  "efficiency to measure",
            extra=extra)
    if len(per_file) > 1:
        extra["per_file"] = per_file
    return RunOutcome(metrics=efficiency_metrics(totals), extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (f"selection efficiency {metrics['efficiency']:.4g} "
            f"+- {metrics['efficiency_err']:.2g}: {metrics['n_passed']:g} of "
            f"{metrics['n_denominator']:g} denominator events (of "
            f"{metrics['n_events']:g} in the ntuple) have a track passing.")


_SYNTAX = (
    "Comparisons, arithmetic, and/or/not (or &&/||/!) and "
    "abs/sqrt/hypot/min/max/log/exp over the track variables: "
    + ", ".join(TRACK_VARIABLES) + "; in simulated files also the MC truth "
    "ones " + ", ".join(MC_VARIABLES) + "; where the file has CRV coincidences "
    "also " + ", ".join(CRV_VARIABLES) + ". *_front quantities are at the "
    "downstream-going TT_Front crossing; fit parameters (d0, maxr, tandip, "
    "t0err, ...) at TT_Mid; a missing value is NaN and fails any cut on it."
)

SPEC = AnalysisSpec(
    name="selection_efficiency_ntuple",
    input_kind="root_file",
    combines_files=True,
    run=run,
    description=(
        "Selection efficiency from EventNtuple file(s): the fraction of "
        "events with a track passing a cut expression, out of all ntuple "
        "events or those passing an optional denominator cut. No job is run."
    ),
    metrics=METRICS,
    parameters=(
        ParamSpec(
            name="selection",
            description=(
                "Track cut whose efficiency to measure: an event passes if one "
                "track passes both it and the denominator, e.g. 'status == 1 "
                "and pdg == 11 and downstream and 100 < p_front < 110'. "
                + _SYNTAX
            ),
            kind="text",
        ),
        ParamSpec(
            name="denominator",
            description=(
                "Track cut defining the events the efficiency is measured on "
                "(an event counts if any track passes). The default, an empty "
                "string, is every event in the ntuple, including those with no "
                "track; that is ntuple entries, not generated events. " + _SYNTAX
            ),
            default="", kind="text", allow_empty=True,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "EventNtuple file(s) (nts.*.root, tree EventNtuple/ntuple), e.g. "
        "nts.mu2e.CeMLeadingLogMix1BB.<version>.root; pass data_files for "
        "several, combined into one result."
    ),
)
