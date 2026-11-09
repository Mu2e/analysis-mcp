"""Trigger efficiency from EventNtuple files, for a configurable track selection.

The EventNtuple records each trigger path's decision per event as a boolean
branch, `trig_<path>`. This analysis selects events with at least one track
passing a cut expression (tools/selection.py syntax) and reports the fraction
of them that any of the given paths accepted, and each path's own fraction.
Nothing is rerun: the decisions are the ones made when the ntuple's input was
triggered.

The track variables are tools/eventntuple.py's (TRACK_VARIABLES, and where
the file has them MC_VARIABLES and CRV_VARIABLES), shared by every
EventNtuple analysis.
"""

from __future__ import annotations

import numpy as np

from ..eventntuple import (CRV_VARIABLES, MC_VARIABLES, TRACK_VARIABLES, NtupleError,
                           check_names, parse_trigger_names, read_file)
from ..selection import SelectionError, apply_selection
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .trigger import binomial

# A downstream electron fit that converged (TrkInfo status 1: Kalman fit
# converged; 2 is only "OK", -1 failed), above 80 MeV/c at the tracker front,
# with at least 15 active hits and chi^2/dof < 5.
DEFAULT_SELECTION = (
    "status == 1 and pdg == 11 and downstream and p_front > 80"
    " and nactive >= 15 and chisq / ndof < 5"
)


# --- the analysis ------------------------------------------------------------------

def efficiency_counts(variables: dict[str, np.ndarray], triggers: dict[str, np.ndarray],
                      n_events: int, selection: str) -> dict:
    """Selected events, and how many of them each path (and any path) kept."""
    passing = apply_selection(selection, {k: v for k, v in variables.items()
                                          if not k.startswith("_")},
                              variables["_event"].size)
    selected = np.zeros(n_events, dtype=bool)
    selected[variables["_event"][passing]] = True
    any_path = np.zeros(n_events, dtype=bool)
    for decision in triggers.values():
        any_path |= decision
    return {
        "n_events": n_events,
        "n_tracks_selected": int(passing.sum()),
        "n_selected": int(selected.sum()),
        "n_triggered": int((selected & any_path).sum()),
        "per_path": {p: int((selected & d).sum()) for p, d in triggers.items()},
    }


METRICS = ("n_events", "n_selected", "n_triggered", "efficiency",
           "efficiency_err", "n_tracks_selected")


def efficiency_metrics(counts: dict) -> dict[str, float]:
    """The overall efficiency: triggered over selected events."""
    efficiency, err = binomial(counts["n_triggered"], counts["n_selected"])
    return {"n_events": float(counts["n_events"]),
            "n_selected": float(counts["n_selected"]),
            "n_triggered": float(counts["n_triggered"]),
            "efficiency": efficiency, "efficiency_err": err,
            "n_tracks_selected": float(counts["n_tracks_selected"])}


def run(context: RunContext) -> RunOutcome:
    """Select events in the ntuple(s) and count the trigger decisions on them."""
    selection = context.params["selection"]
    extra: dict = {"selection": selection}
    try:
        paths = parse_trigger_names(context.params["trigger_paths"])
        totals = {"n_events": 0, "n_tracks_selected": 0, "n_selected": 0,
                  "n_triggered": 0, "per_path": {p: 0 for p in paths}}
        per_file = []
        for path in context.input_paths:
            variables, triggers, n_events = read_file(path, paths)
            check_names(variables, selection)
            counts = efficiency_counts(variables, triggers, n_events, selection)
            per_file.append({"file": str(path), "n_events": n_events,
                             "n_selected": counts["n_selected"],
                             "n_triggered": counts["n_triggered"]})
            for key in ("n_events", "n_tracks_selected", "n_selected", "n_triggered"):
                totals[key] += counts[key]
            for p in paths:
                totals["per_path"][p] += counts["per_path"][p]
    except (NtupleError, SelectionError) as exc:
        return RunOutcome(error=str(exc), extra=extra)
    if totals["n_selected"] == 0:
        return RunOutcome(
            error=f"no event of {totals['n_events']} has a track passing the "
                  "selection, so there is no efficiency to measure",
            extra=extra)
    paths_out = {}
    for p in paths:
        eff, err = binomial(totals["per_path"][p], totals["n_selected"])
        paths_out[p] = {"passed": totals["per_path"][p],
                        "efficiency": eff, "efficiency_err": err}
    extra.update(paths=paths_out,
                 **({"per_file": per_file} if len(per_file) > 1 else {}))
    return RunOutcome(metrics=efficiency_metrics(totals), extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (f"trigger efficiency {metrics['efficiency']:.4g} "
            f"+- {metrics['efficiency_err']:.2g}: {metrics['n_triggered']:g} of "
            f"{metrics['n_selected']:g} selected events (of {metrics['n_events']:g}) "
            "passed any of the paths; per-path efficiencies in metadata.paths.")


SPEC = AnalysisSpec(
    name="trigger_efficiency_ntuple",
    input_kind="root_file",
    combines_files=True,
    run=run,
    description=(
        "Trigger efficiency from EventNtuple file(s): of the events with a "
        "track passing a configurable selection, the fraction any of the "
        "given trigger paths accepted (trig_<path> branches), and each path's "
        "own efficiency (metadata.paths). No job is run."
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
            name="selection",
            description=(
                "Track selection: an event is selected if at least one track "
                "passes. Comparisons, arithmetic, and/or/not (or &&/||/!) and "
                "abs/sqrt/hypot/min/max/log/exp over: "
                + ", ".join(TRACK_VARIABLES) + "; in simulated files, "
                + ", ".join(MC_VARIABLES) + "; and where the file has CRV "
                "coincidences, " + ", ".join(CRV_VARIABLES) + ". *_front "
                "quantities are at "
                "the downstream-going TT_Front crossing; fit parameters (d0, "
                "maxr, tandip, t0err, ...) at TT_Mid; a missing value is NaN "
                "and fails any cut on it. The default is a converged "
                "downstream e- fit with p > 80 MeV/c at the tracker front, "
                ">= 15 active hits and chisq/ndof < 5. An empty string "
                "selects every event."
            ),
            default=DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
    ),
    summarize=summarize,
    input_hint=(
        "EventNtuple file(s) (nts.*.root, tree EventNtuple/ntuple) with "
        "trig_<path> branches, e.g. nts.mu2e.CeMLeadingLogMix1BB.<version>.root; "
        "pass data_files for several, combined into one result."
    ),
)
