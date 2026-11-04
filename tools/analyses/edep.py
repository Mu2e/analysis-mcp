"""Energy deposition: the EdepAna analyzer (fcl/edep.fcl).

Average calorimeter and tracker energy deposition per event and per generated
event, plus the number and rate of events passing a configurable selection.
The averages come from the summary block EdepAna prints at endJob; the
selection is applied to the per-event TTree it writes (EDepAna/tree). The
module lives in Offline (Offline/Analyses/src/EdepAna_module.cc, from
v13_39_00), so it comes with the configured Musing or its backing. Offline
PR #2067 (merged 2026-10-27) added approximate hit counts from detector steps
to it; they are reported only where the configured Offline has them (see
`run`).

This module also owns the reading of that tree — `read_edep_tree` and
`EDEP_VARIABLES` — so every analysis of EdepAna output selects events the
same way, with the same variable names.
"""

import json
import re
from pathlib import Path

import numpy as np

from ..mu2e_job import ModuleProbeError, module_accepts, run_mu2e_job
from ..selection import SelectionError, apply_selection
from ..spec import FCL_DIR, AnalysisSpec, ParamSpec, RunContext, RunOutcome

FCL = FCL_DIR / "edep.fcl"

# The analyzer's label in edep.fcl, and the tree it books.
TREE_PATH = "EDepAna/tree"

# What the >50 MeV count in the summary block used to be fixed to.
DEFAULT_SELECTION = "event_calo_edep_vis > 50"

# DetectorSystem's origin in the Mu2e frame: x_det = x + 3904 mm. EdepAna's
# primary_start_r histogram is the start radius in the detector frame.
DETECTOR_ORIGIN_X_MM = -3904.0

# --- the EdepAna tree --------------------------------------------------------

_EVENT_BRANCHES = ("event_calo_edep", "event_calo_edep_vis", "event_trk_edep",
                   "weight", "run", "subrun", "event", "nprimaries", "ngen")
_PRIMARY_BRANCHES = (
    "primary_start_x", "primary_start_y", "primary_start_z",
    "primary_start_px", "primary_start_py", "primary_start_pz",
    "primary_start_e", "primary_start_m", "primary_start_pdg",
    "primary_calo_edep", "primary_calo_edep_vis",
    "primary_trk_front_p", "primary_trk_front_energy",
)

# The variables a selection may use, and what each one is. Per-primary
# branches are the *first* primary's value, as EdepAna's histograms use it;
# NaN for an event with no primary, or with no tracker-front step for the
# primary_trk_front_* ones, so any cut on them fails for such an event.
EDEP_VARIABLES: dict[str, str] = {
    "event_calo_edep": "total calo energy deposited in the event (MeV)",
    "event_calo_edep_vis": "total visible (Birks) calo energy in the event (MeV)",
    "event_trk_edep": "total tracker ionizing energy in the event (MeV)",
    "weight": "event weight",
    "run": "run number", "subrun": "subrun number", "event": "event number",
    "nprimaries": "number of primary particles",
    "primary_start_x": "primary start x, Mu2e frame (mm)",
    "primary_start_y": "primary start y, Mu2e frame (mm)",
    "primary_start_z": "primary start z, Mu2e frame (mm)",
    "primary_start_r": "primary start radius, detector frame (mm)",
    "primary_start_px": "primary start px (MeV/c)",
    "primary_start_py": "primary start py (MeV/c)",
    "primary_start_pz": "primary start pz (MeV/c)",
    "primary_start_p": "primary start momentum (MeV/c)",
    "primary_start_e": "primary start energy (MeV)",
    "primary_start_m": "primary mass (MeV)",
    "primary_start_pdg": "primary PDG id",
    "primary_calo_edep": "calo energy from the primary and descendants (MeV)",
    "primary_calo_edep_vis": "visible calo energy from the primary and descendants (MeV)",
    "has_trk_front": "the primary reached the tracker front (true/false)",
    "primary_trk_front_p": "primary momentum at the tracker front (MeV/c)",
    "primary_trk_front_energy": "primary energy at the tracker front (MeV)",
    "primary_trk_front_energy_diff":
        "tracker-front energy minus start energy (MeV, <= 0)",
    "primary_energy_edep_diff": "primary visible calo edep minus start energy (MeV)",
    "primary_trk_front_energy_edep_diff":
        "primary visible calo edep minus tracker-front energy (MeV)",
}

# Optional: only in a tree from an EdepAna with Offline PR #2067 (the step
# counts). Read only where the file has them; `check_optional_names` turns a
# selection naming one on an older file into a plain error.
_STEP_BRANCHES = tuple(f"event_{det}_nsteps{cut}" for det in ("calo", "trk")
                       for cut in ("", "_ecut", "_ecut_tcut"))
OPTIONAL_EDEP_VARIABLES: dict[str, str] = {
    "event_calo_nsteps": "CaloShowerSteps in the event (needs Offline PR #2067)",
    "event_calo_nsteps_ecut":
        "... with energyDepBirks above the calo step threshold",
    "event_calo_nsteps_ecut_tcut":
        "... and with step time inside the calo window",
    "event_trk_nsteps": "StrawGasSteps in the event (needs Offline PR #2067)",
    "event_trk_nsteps_ecut":
        "... with ionizingEdep above the tracker step threshold",
    "event_trk_nsteps_ecut_tcut":
        "... and with step time inside the tracker window",
}
assert tuple(OPTIONAL_EDEP_VARIABLES) == _STEP_BRANCHES

# Optional: the helix of the primary at the tracker front, only in a tree from
# an Offline with PR #2092 (not yet in a Musing). Read only where the file has all
# four (`_HELIX_BRANCHES`, per primary), and NaN where undefined: EdepAna
# writes -1 for the radii and the time (no front step, neutral primary, zero
# field) and 0 for the cosine when there is no front step. rmax and d0 are
# derived, with Mu2e's convention d0 = rmax - 2R (signed).
_HELIX_BRANCHES = ("primary_trk_front_r_center", "primary_trk_front_radius",
                   "primary_trk_front_cos", "primary_trk_front_time")
HELIX_EDEP_VARIABLES: dict[str, str] = {
    "primary_trk_front_r_center":
        "transverse radius of the primary's helix center from the tracker "
        "axis at the tracker front (mm; needs the helix branches)",
    "primary_trk_front_radius":
        "primary's helix radius at the tracker front (mm)",
    "primary_trk_front_cos": "pz/p of the primary at the tracker front",
    "primary_trk_front_time": "time of the primary at the tracker front (ns)",
    "primary_trk_front_rmax":
        "helix center radius plus helix radius, R(max) (mm)",
    "primary_trk_front_d0":
        "helix center radius minus helix radius, signed: d0 = rmax - 2R (mm)",
}


class EdepTreeError(RuntimeError):
    """The input is not an EdepAna file with the tree, worded for the caller."""


def edep_variables(branches: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every EDEP_VARIABLES column from the tree's raw branches.

    `branches` holds the event branches as flat arrays and the primary
    branches as one array per event (uproot's library="np").
    """
    nprim = np.asarray(branches["nprimaries"])
    has_primary = nprim > 0
    out: dict[str, np.ndarray] = {
        name: np.asarray(branches[name], dtype=np.float64)
        for name in _EVENT_BRANCHES if name != "ngen"
    }
    for name in _PRIMARY_BRANCHES:
        first = np.full(nprim.size, np.nan)
        if has_primary.any():
            first[has_primary] = [entry[0] for entry in branches[name][has_primary]]
        out[name] = first

    # EdepAna stores 0 where the primary never reached the tracker front.
    has_front = out["primary_trk_front_p"] > 0.0
    for name in ("primary_trk_front_p", "primary_trk_front_energy"):
        out[name] = np.where(has_front, out[name], np.nan)
    out["has_trk_front"] = has_front

    out["primary_start_r"] = np.hypot(out["primary_start_x"] - DETECTOR_ORIGIN_X_MM,
                                      out["primary_start_y"])
    out["primary_start_p"] = np.sqrt(out["primary_start_px"] ** 2
                                     + out["primary_start_py"] ** 2
                                     + out["primary_start_pz"] ** 2)
    out["primary_trk_front_energy_diff"] = (out["primary_trk_front_energy"]
                                            - out["primary_start_e"])
    out["primary_energy_edep_diff"] = (out["primary_calo_edep_vis"]
                                       - out["primary_start_e"])
    out["primary_trk_front_energy_edep_diff"] = (out["primary_calo_edep_vis"]
                                                 - out["primary_trk_front_energy"])
    # The helix at the tracker front, where the tree has it (all four).
    if all(name in branches for name in _HELIX_BRANCHES):
        for name in _HELIX_BRANCHES:
            first = np.full(nprim.size, np.nan)
            if has_primary.any():
                first[has_primary] = [entry[0] for entry in branches[name][has_primary]]
            out[name] = first
        for name in ("primary_trk_front_r_center", "primary_trk_front_radius",
                     "primary_trk_front_time"):
            out[name] = np.where(has_front & (out[name] >= 0.0), out[name], np.nan)
        out["primary_trk_front_cos"] = np.where(has_front,
                                                out["primary_trk_front_cos"], np.nan)
        out["primary_trk_front_rmax"] = (out["primary_trk_front_r_center"]
                                         + out["primary_trk_front_radius"])
        out["primary_trk_front_d0"] = (out["primary_trk_front_r_center"]
                                       - out["primary_trk_front_radius"])
    # Step counts, where the tree has them (the caller passes only those).
    for name in _STEP_BRANCHES:
        if name in branches:
            out[name] = np.asarray(branches[name], dtype=np.float64)
    return out


def read_edep_tree(paths: list[Path] | Path) -> dict[str, np.ndarray]:
    """EDEP_VARIABLES for every event in the EdepAna file(s), concatenated.

    The optional step-count and helix variables are included only if every
    file has their branches (a mix would leave events without a value).
    """
    import uproot

    paths = [paths] if isinstance(paths, Path) else list(paths)
    parts = []
    for path in paths:
        try:
            with uproot.open(path) as rootfile:
                tree = rootfile[TREE_PATH]
                present = [name for name in _STEP_BRANCHES if name in tree]
                if all(name in tree for name in _HELIX_BRANCHES):
                    present += _HELIX_BRANCHES
                branches = tree.arrays(_EVENT_BRANCHES + _PRIMARY_BRANCHES
                                       + tuple(present), library="np")
        except KeyError as exc:
            raise EdepTreeError(
                f"{path}: no {TREE_PATH} with the EdepAna branches ({exc}) — is "
                "this an nts.*.root from EdepAna in Offline v13_39_00 or later? "
                "Rerun the 'edep' analysis to make one."
            ) from None
        parts.append(edep_variables(branches))
    return {name: np.concatenate([part[name] for part in parts])
            for name in parts[0] if all(name in part for part in parts)}


def generated_events(paths: list[Path] | Path) -> float:
    """The generated events behind the EdepAna file(s).

    `ngen` is EdepAna's running count, raised by each subrun's GenEventCount
    as the job reaches it, so a file's total is its largest value; files from
    separate jobs add up. A subrun after a file's last event is not in it, nor
    is the rest of a subrun cut off by max_events: read it from a full run
    (read_run_record says whether 'edep' made the file with max_events).
    The count restarts with each job, so a file in which it falls holds
    several jobs' output merged, and is refused. Not every merge shows this
    way: one whose later part starts at or above the earlier part's total
    keeps the count rising and is not caught, so do not merge edep outputs.
    """
    import uproot

    paths = [paths] if isinstance(paths, Path) else list(paths)
    total = 0.0
    for path in paths:
        try:
            with uproot.open(path) as rootfile:
                ngen = rootfile[TREE_PATH]["ngen"].array(library="np")
        except KeyError as exc:
            raise EdepTreeError(
                f"{path}: no {TREE_PATH} with an ngen branch ({exc}) — is "
                "this an nts.*.root from EdepAna in Offline v13_39_00 or later? "
                "Rerun the 'edep' analysis to make one."
            ) from None
        if ngen.size > 1 and (np.diff(ngen) < 0).any():
            raise EdepTreeError(
                f"{path}: EdepAna's running generated-event count (ngen) "
                "falls, so this file holds several edep outputs merged (e.g. "
                "by hadd) and their generated events cannot be added up; run "
                "'edep' once over all the art files instead (and do not merge "
                "its outputs: not every merge can be caught this way)"
            )
        total += float(ngen.max()) if ngen.size else 0.0
    return total


RUN_RECORD_SUFFIX = ".edep.json"


def run_record_path(path: Path) -> Path:
    """Where the 'edep' analysis records how it made the EdepAna file `path`."""
    path = Path(path)
    return path.with_name(path.name + RUN_RECORD_SUFFIX)


def write_run_record(path: Path, max_events: int | None) -> Path:
    """Record, next to an EdepAna file 'edep' wrote, the max_events its job
    ran with (None for a full run). ngen keeps a cut-short subrun's whole
    GenEventCount, so a file from a max_events run undercounts its events
    per generated event, and that cannot be seen from the file itself."""
    record = run_record_path(path)
    record.write_text(json.dumps({"max_events": max_events}) + "\n",
                      encoding="utf-8")
    return record


def read_run_record(path: Path) -> dict | None:
    """The record 'edep' wrote next to `path`, or None for a file it did not
    make (e.g. one from a production job)."""
    record = run_record_path(path)
    if not record.exists():
        return None
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EdepTreeError(f"{record}: unreadable edep run record ({exc})") from None
    if not isinstance(data, dict) or "max_events" not in data:
        raise EdepTreeError(f"{record}: an edep run record holds max_events, "
                            f"got {data!r}")
    return data


def check_optional_names(variables: dict[str, np.ndarray], selection: str) -> None:
    """Refuse a selection naming a step-count variable the tree does not have.

    Raises EdepTreeError, so the caller gets what is missing and why rather
    than the selection parser's unknown-name message.
    """
    names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", selection))
    missing = sorted(n for n in names
                     if n in OPTIONAL_EDEP_VARIABLES and n not in variables)
    if missing:
        raise EdepTreeError(
            f"the selection uses {', '.join(missing)}, but this EdepAna tree has no "
            "step-count branches: they come from an EdepAna with Offline PR #2067 "
            "(merged 2026-10-27). Rerun 'edep' in an Offline that has it, or drop "
            "them from the selection.")
    missing = sorted(n for n in names
                     if n in HELIX_EDEP_VARIABLES and n not in variables)
    if missing:
        raise EdepTreeError(
            f"the selection uses {', '.join(missing)}, but this EdepAna tree has no "
            "helix branches: they come from EdepAna in an Offline with PR #2092 "
            "(not yet in a Musing). Rerun 'edep' with such a build, or drop them from the "
            "selection.")


def select_events(variables: dict[str, np.ndarray], selection: str) -> np.ndarray:
    """The mask of events passing `selection`, a cut over EDEP_VARIABLES."""
    check_optional_names(variables, selection)
    nevents = variables["event_calo_edep_vis"].size
    return apply_selection(selection, variables, nevents)


def selection_help() -> str:
    """The parameter description shared by every analysis taking a selection."""
    return (
        "Event selection applied to the EdepAna tree, e.g. "
        "'event_calo_edep_vis > 10 && primary_start_z > 5400'. Comparisons, "
        "arithmetic, and/or/not (or &&/||/!) and abs/sqrt/hypot/min/max/log/"
        "exp over: " + ", ".join(EDEP_VARIABLES) + ". Per-primary variables "
        "are the first primary's. Also, only for an EdepAna tree made with "
        "Offline PR #2067 (the step counts): "
        + ", ".join(OPTIONAL_EDEP_VARIABLES) + "; on an older tree a "
        "selection naming one is an error. Likewise, only for a tree from an "
        "Offline with PR #2092, which adds the tracker-front helix branches (NaN where "
        "undefined): " + ", ".join(HELIX_EDEP_VARIABLES) + ". "
        "An empty string selects every event."
    )

# Matches the block EdepAna_module.cc prints, e.g.:
#   EdepAna summary:
#     Saw 998.5 events (1000 gen events) --> output rate = 0.9985 events / gen event
#     Average calo energy deposition per event: 12.3 MeV
#     Average calo energy deposition per gen event: 12.29 MeV
#     Events with calo Edep > 50 MeV: 42
#     Average tracker energy deposition per event: 3.21 MeV
#     Average tracker energy deposition per gen event: 3.2 MeV
# n_events and n_events_calo_edep_above_50mev are weighted sums (doubles in
# the module), so every field is parsed as a float.
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SUMMARY_FIELDS = [
    ("n_events", re.compile(rf"Saw\s+({_NUM})\s+events\b")),
    ("n_gen_events", re.compile(rf"\(\s*({_NUM})\s+gen events\)")),
    ("event_rate", re.compile(rf"output rate\s*=\s*({_NUM})\s+events / gen event")),
    ("avg_calo_edep_per_event_mev",
     re.compile(rf"Average calo energy deposition per event:\s*({_NUM})\s*MeV")),
    ("avg_calo_edep_per_gen_event_mev",
     re.compile(rf"Average calo energy deposition per gen event:\s*({_NUM})\s*MeV")),
    ("n_events_calo_edep_above_50mev",
     re.compile(rf"Events with calo Edep > 50 MeV:\s*({_NUM})")),
    ("avg_trk_edep_per_event_mev",
     re.compile(rf"Average tracker energy deposition per event:\s*({_NUM})\s*MeV")),
    ("avg_trk_edep_per_gen_event_mev",
     re.compile(rf"Average tracker energy deposition per gen event:\s*({_NUM})\s*MeV")),
]

_SELECTION_METRICS = ("n_events_selected", "selected_per_gen_event")

# --- step counts (Offline PR #2067) --------------------------------------------
# Approximate hit counts from detector steps, printed after the summary block:
#   Calo step counts (E > 0.5 MeV, 500 <= t <= 1e+09 ns):
#     all       : total = 19           per event = 0.095       per gen event = 0.019
#     E cut     : total = 16 ...
#     E + t cut : total = 12 ...
# then the same for "Tracker". The module's per-event and per-gen-event
# values are -1 when their denominator is 0, and that is passed through.
STEP_DETECTORS = (("Calo", "calo"), ("Tracker", "trk"))
STEP_CUTS = (("all", ""), ("E cut", "_ecut"), ("E + t cut", "_ecut_tcut"))
STEP_QUANTITIES = (("total", "total", "steps"),
                   ("per event", "per_event", "steps / event"),
                   ("per gen event", "per_gen_event", "steps / gen event"))

STEP_METRICS = tuple(f"{det}_nsteps{cut}_{key}" for _, det in STEP_DETECTORS
                     for _, cut in STEP_CUTS for _, key, _ in STEP_QUANTITIES)
STEP_UNITS = {f"{det}_nsteps{cut}_{key}": unit for _, det in STEP_DETECTORS
              for _, cut in STEP_CUTS for _, key, unit in STEP_QUANTITIES}

_STEP_HEADER = re.compile(
    rf"(Calo|Tracker) step counts \(E > ({_NUM}) MeV, ({_NUM}) <= t <= ({_NUM}) ns\):")
_STEP_ROW = re.compile(
    rf"^\s*(all|E cut|E \+ t cut)\s*:\s*total = ({_NUM})\s+per event = ({_NUM})\s+"
    rf"per gen event = ({_NUM})\s*$", re.MULTILINE)

# The step parameters EdepAna takes from PR #2067, all required there (no
# module defaults): (analysis parameter, EdepAna key, default, minimum,
# maximum, description). Defaults are the PR's example fcl.
_STEP_NOTE = ("Needs an Offline with PR #2067 (merged 2026-10-27); on an older "
              "one only the default is accepted, and the step-count metrics are "
              "left out. The energy cut is strict (E must be above it) and the "
              "time window is inclusive. Step times are raw Geant4 times: no "
              "time offsets and no microbunch folding.")
STEP_PARAMS = (
    ("trk_step_energy_threshold_mev", "TrkStepEnergyThreshold", 0.0005, 0.0, 1000.0,
     "Tracker step count: a StrawGasStep counts when its ionizingEdep (MeV) is "
     "above this. A StrawGasStep is roughly one particle crossing one straw, so "
     "this approximates a straw hit. " + _STEP_NOTE),
    ("trk_step_time_min_ns", "TrkStepTimeMin", 500.0, 0.0, 1e12,
     "Tracker step count: earliest step time (ns) counted in the E + t cut. "
     + _STEP_NOTE),
    ("trk_step_time_max_ns", "TrkStepTimeMax", 1e9, 0.0, 1e12,
     "Tracker step count: latest step time (ns) counted in the E + t cut. "
     + _STEP_NOTE),
    ("calo_step_energy_threshold_mev", "CaloStepEnergyThreshold", 0.5, 0.0, 1e4,
     "Calorimeter step count: a CaloShowerStep counts when its energyDepBirks "
     "(MeV) is above this. A CaloShowerStep is per particle per crystal, so it "
     "is not a crystal-hit count. " + _STEP_NOTE),
    ("calo_step_time_min_ns", "CaloStepTimeMin", 500.0, 0.0, 1e12,
     "Calorimeter step count: earliest step time (ns) counted in the E + t cut. "
     + _STEP_NOTE),
    ("calo_step_time_max_ns", "CaloStepTimeMax", 1e9, 0.0, 1e12,
     "Calorimeter step count: latest step time (ns) counted in the E + t cut. "
     + _STEP_NOTE),
)

# The folded-time windows of EdepAna's occupancy "hit" histograms, from Offline
# PR #2097 (merged 2026-11-03), all required there: (analysis parameter,
# EdepAna key, default, minimum, maximum, description). Defaults are the PR's
# Analyses/test/edep.fcl: the digitization windows measured from the marker
# (tracker digitizationStart/End 450/1705 ns, calo 450/1795 ns) minus 200 ns.
_WINDOW_NOTE = ("Used only by the occupancy hit histograms (Offline PR #2097); "
                "the tree, the summary and the step counts keep the raw-time "
                "step windows.")
HIT_WINDOW_PARAMS = (
    ("fold_time_offset_ns", "FoldTimeOffset", -200.0, -1e6, 1e6,
     "Added to the raw step time before it is folded modulo the microbunch "
     "period, like StrawDigisFromStrawGasSteps::microbunchTime. The default, "
     "-200 ns (the nominal proton bunch time minus TimeFromProtonsToDRMarker), "
     "puts folded times in the digitizers' event-window frame; 0 folds the raw "
     "step time. " + _WINDOW_NOTE),
    ("trk_hit_time_min_ns", "TrkHitTimeMin", 250.0, -1e6, 1e6,
     "Tracker hit histograms: earliest folded step time (ns) counted. "
     + _WINDOW_NOTE),
    ("trk_hit_time_max_ns", "TrkHitTimeMax", 1505.0, -1e6, 1e6,
     "Tracker hit histograms: latest folded step time (ns) counted. "
     + _WINDOW_NOTE),
    ("calo_hit_time_min_ns", "CaloHitTimeMin", 250.0, -1e6, 1e6,
     "Calorimeter hit histograms: earliest folded step time (ns) counted. "
     + _WINDOW_NOTE),
    ("calo_hit_time_max_ns", "CaloHitTimeMax", 1595.0, -1e6, 1e6,
     "Calorimeter hit histograms: latest folded step time (ns) counted. "
     + _WINDOW_NOTE),
)

# The module whose parameters decide whether the step counts exist, and the
# one it gained them with; likewise the occupancy hit windows.
MODULE_TYPE = "EdepAna"
STEP_PROBE_PARAMETER = "TrkStepEnergyThreshold"
HIT_WINDOW_PROBE_PARAMETER = "FoldTimeOffset"
ANALYZER_LABEL = "EDepAna"           # in edep.fcl
JOB_FCL_NAME = "edep_job.fcl"


class StepCountError(ValueError):
    """The step-count request cannot run, worded for the caller."""

METRIC_UNITS = {
    "event_rate": "events / gen event",
    "selected_per_gen_event": "events / gen event",
    "avg_calo_edep_per_event_mev": "MeV",
    "avg_calo_edep_per_gen_event_mev": "MeV",
    "avg_trk_edep_per_event_mev": "MeV",
    "avg_trk_edep_per_gen_event_mev": "MeV",
}


def parse_edep_summary(stdout: str) -> dict[str, float] | None:
    """Extract the 8 EdepAna summary fields from mu2e stdout.

    Returns None if the "EdepAna summary:" block isn't there or is incomplete
    (e.g. the job died before EdepAna::endJob ran).
    """
    if "EdepAna summary:" not in stdout:
        return None
    metrics: dict[str, float] = {}
    for name, pattern in _SUMMARY_FIELDS:
        match = pattern.search(stdout)
        if not match:
            return None
        metrics[name] = float(match.group(1))
    return metrics


def parse_step_counts(stdout: str) -> tuple[dict[str, float], dict[str, dict[str, float]]] | None:
    """The step-count blocks of the EdepAna printout (Offline PR #2067).

    Returns (metrics, windows): the 18 STEP_METRICS, and per detector ("calo",
    "trk") the energy threshold and time window the module printed, i.e. what
    it actually used. None if either block is missing or incomplete, which is
    what an EdepAna without the step counts prints.
    """
    metrics: dict[str, float] = {}
    windows: dict[str, dict[str, float]] = {}
    headers = {m.group(1): m for m in _STEP_HEADER.finditer(stdout)}
    for label, det in STEP_DETECTORS:
        header = headers.get(label)
        if header is None:
            return None
        end = min((m.start() for m in headers.values() if m.start() > header.start()),
                  default=len(stdout))
        rows = {m.group(1): m for m in _STEP_ROW.finditer(stdout, header.end(), end)}
        for cut_label, cut in STEP_CUTS:
            row = rows.get(cut_label)
            if row is None:
                return None
            for i, (_, key, _) in enumerate(STEP_QUANTITIES, start=2):
                metrics[f"{det}_nsteps{cut}_{key}"] = float(row.group(i))
        windows[det] = {"energy_threshold_mev": float(header.group(2)),
                        "time_min_ns": float(header.group(3)),
                        "time_max_ns": float(header.group(4))}
    return metrics, windows


def job_fcl_text(step_params: dict[str, float], base: Path = FCL,
                 hit_windows: dict[str, float] | None = None) -> str:
    """The run's fcl: the shipped edep.fcl plus the step-count parameters,
    and the occupancy hit windows when `hit_windows` is given.

    EdepAna with PR #2067 requires the step parameters (and with PR #2097 the
    hit windows) and an older one rejects them, so they cannot live in
    fcl/edep.fcl. FHiCL cannot `#include` an absolute path, so this is a copy
    with the lines appended, as the trigger analyses do.
    """
    lines = [base.read_text(encoding="utf-8").rstrip(), "",
             "# --- set by tools/analyses/edep.py for this run (Offline PR #2067) ---"]
    for name, key, *_ in STEP_PARAMS:
        lines.append(f"physics.analyzers.{ANALYZER_LABEL}.{key} : "
                     f"{float(step_params[name])!r}")
    if hit_windows is not None:
        lines.append("# --- occupancy hit windows (Offline PR #2097) ---")
        for name, key, *_ in HIT_WINDOW_PARAMS:
            lines.append(f"physics.analyzers.{ANALYZER_LABEL}.{key} : "
                         f"{float(hit_windows[name])!r}")
    return "\n".join(lines) + "\n"


def hit_window_values(params: dict) -> dict[str, float]:
    """The hit windows to run with: the caller's where set, else the defaults.

    Only edep_occupancy exposes them; plain `edep` runs the defaults, which
    an EdepAna with PR #2097 needs set even when nothing reads the histograms.
    """
    return {name: float(params.get(name, default))
            for name, _, default, *_ in HIT_WINDOW_PARAMS}


def hit_window_problem(windows: dict[str, float]) -> str | None:
    """A complaint if a requested hit window is empty, else None."""
    for det in ("trk", "calo"):
        lo, hi = windows[f"{det}_hit_time_min_ns"], windows[f"{det}_hit_time_max_ns"]
        if lo > hi:
            return (f"{det}_hit_time_min_ns ({lo:g}) is above "
                    f"{det}_hit_time_max_ns ({hi:g}): the window would be empty")
    return None


def step_param_problem(params: dict[str, float | str]) -> str | None:
    """A complaint if the requested step window is empty, else None."""
    for det, label in (("trk", "tracker"), ("calo", "calo")):
        if params[f"{det}_step_time_min_ns"] > params[f"{det}_step_time_max_ns"]:
            return (f"{det}_step_time_min_ns ({params[f'{det}_step_time_min_ns']:g}) "
                    f"is above {det}_step_time_max_ns "
                    f"({params[f'{det}_step_time_max_ns']:g}): the {label} step "
                    "time window would be empty")
    return None


def non_default_step_params(params: dict[str, float | str]) -> list[str]:
    """The step parameters the caller set away from their defaults."""
    return [name for name, _, default, *_ in STEP_PARAMS if params[name] != default]


def selected_metrics(variables: dict[str, np.ndarray], selection: str,
                     n_gen_events: float) -> dict[str, float]:
    """Weighted count of events passing `selection`, and that per gen event."""
    mask = select_events(variables, selection)
    selected = float(variables["weight"][mask].sum())
    return {
        "n_events_selected": selected,
        "selected_per_gen_event": selected / n_gen_events if n_gen_events > 0 else -1.0,
    }


def summarize_edep(metrics: dict[str, float]) -> str:
    text = (
        f"saw {metrics['n_events']:g} events ({metrics['n_gen_events']:g} gen): "
        f"avg calo edep {metrics['avg_calo_edep_per_event_mev']:.4g} MeV/event, "
        f"avg tracker edep {metrics['avg_trk_edep_per_event_mev']:.4g} MeV/event; "
        f"{metrics['n_events_selected']:g} selected "
        f"({metrics['selected_per_gen_event']:.4g} / gen event)."
    )
    calo = metrics.get("calo_nsteps_ecut_tcut_per_event")
    trk = metrics.get("trk_nsteps_ecut_tcut_per_event")
    if calo is not None and trk is not None:
        text += (f" Steps per event after the E + t cuts: calo {calo:.4g}, "
                 f"tracker {trk:.4g} (-1 means no events).")
    return text


def run(context: RunContext) -> RunOutcome:
    """Run edep.fcl, parse the summary block, and apply the selection.

    The step counts need an EdepAna with Offline PR #2067, which requires
    their six parameters, while an older EdepAna rejects them as unknown (and
    the default environment may be either). So the configured Offline is asked
    once (`mu2e --print-description EdepAna`, cached per environment) whether
    the module declares them, rather than running and parsing a fhicl error:
    it is cheap, exact, and keeps the job's own failures out of the decision.
    On an older one the shipped fcl runs unchanged and the step metrics are
    left out of the result, as an optional part of it.
    """
    try:
        with_steps = module_accepts(MODULE_TYPE, STEP_PROBE_PARAMETER, context.outdir)
        # PR #2097 came after PR #2067, so only an EdepAna with steps can have it
        with_windows = with_steps and module_accepts(
            MODULE_TYPE, HIT_WINDOW_PROBE_PARAMETER, context.outdir)
    except ModuleProbeError as exc:
        return RunOutcome(error=f"could not tell whether the configured Offline's "
                                f"EdepAna has the step counts: {exc}")
    extra: dict = {}
    fcl = FCL
    if with_steps:
        problem = step_param_problem(context.params)
        windows = hit_window_values(context.params) if with_windows else None
        if problem is None and windows is not None:
            problem = hit_window_problem(windows)
        if problem:
            return RunOutcome(error=problem)
        context.outdir.mkdir(parents=True, exist_ok=True)
        fcl = context.outdir / JOB_FCL_NAME
        fcl.write_text(job_fcl_text(context.params, hit_windows=windows),
                       encoding="utf-8")
        extra["job_fcl"] = str(fcl)
    else:
        changed = non_default_step_params(context.params)
        if changed:
            return RunOutcome(error=(
                "the configured Offline's EdepAna predates the step counts "
                f"(Offline PR #2067), so {', '.join(changed)} cannot be set; "
                "leave the step parameters at their defaults, or run in an "
                "Offline that has the PR"))
        try:        # no job needed to know the tree will lack the branches
            check_optional_names({}, context.params["selection"])
        except EdepTreeError as exc:
            return RunOutcome(error=str(exc))
        extra["step_counts"] = ("not available: the configured Offline's EdepAna "
                                "predates Offline PR #2067, so the *_nsteps_* "
                                "metrics are absent")
    outcome = run_mu2e_job(
        fcl=fcl,
        input_paths=context.input_paths,
        outdir=context.outdir,
        single=len(context.input_paths) == 1 and not context.wants_file_list,
        timeout_s=context.timeout_s,
        max_events=context.max_events,
    )
    extra["returncode"] = outcome.returncode
    if outcome.file_list_path is not None:
        extra["file_list_path"] = str(outcome.file_list_path)

    if outcome.timed_out:
        return RunOutcome(
            files=outcome.written_root_files, log_path=outcome.log_path, extra=extra,
            error=f"mu2e timed out after {context.timeout_s}s on "
                  f"{len(context.input_paths)} input file(s)",
        )
    if outcome.failed:
        extra["stdout_tail"] = outcome.stdout_tail()
        return RunOutcome(
            files=outcome.written_root_files, log_path=outcome.log_path, extra=extra,
            error=f"mu2e exited {outcome.returncode}",
        )

    metrics = parse_edep_summary(outcome.stdout)
    if metrics is None:
        extra["stdout_tail"] = outcome.stdout_tail()
        return RunOutcome(
            files=outcome.written_root_files, log_path=outcome.log_path, extra=extra,
            error="EdepAna summary block not found in mu2e output",
        )

    if with_steps:
        steps = parse_step_counts(outcome.stdout)
        if steps is None:
            extra["stdout_tail"] = outcome.stdout_tail()
            return RunOutcome(
                files=outcome.written_root_files, log_path=outcome.log_path,
                extra=extra,
                error="the step-count blocks not found in mu2e output, though "
                      "this EdepAna declares the step parameters")
        # What the module printed, i.e. what it used.
        extra["step_count_windows"] = steps[1]

    selection = context.params["selection"]
    extra["selection"] = selection
    ntuples = [Path(f) for f in outcome.written_root_files]
    if not ntuples:
        return RunOutcome(log_path=outcome.log_path, extra=extra,
                          error="mu2e wrote no ROOT file, so there is no "
                                f"{TREE_PATH} to apply the selection to")
    # Beside each file, not in `files`: a chained analysis takes the ROOT
    # file alone, and finds the record next to it.
    for ntuple in ntuples:
        write_run_record(ntuple, context.max_events)
    try:
        metrics.update(selected_metrics(read_edep_tree(ntuples), selection,
                                        metrics["n_gen_events"]))
    except (EdepTreeError, SelectionError) as exc:
        return RunOutcome(files=outcome.written_root_files,
                          log_path=outcome.log_path, extra=extra, error=str(exc))
    if with_steps:
        metrics.update(steps[0])        # after the selection: the spec's order
    return RunOutcome(metrics=metrics, files=outcome.written_root_files,
                      log_path=outcome.log_path, extra=extra)


SPEC = AnalysisSpec(
    name="edep",
    fcl=FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Average calorimeter and tracker energy deposition per event and per "
        "generated event (EdepAna), and the number and rate per generated "
        "event of events passing a configurable selection. With an Offline "
        "that has PR #2067 it also reports approximate hit counts from "
        "detector steps (calo_nsteps_*, trk_nsteps_*: all steps, above an "
        "energy threshold, and also inside a time window); with an older one "
        "those metrics are absent."
    ),
    # The *_nsteps_* metrics come last and only where the configured Offline's
    # EdepAna has Offline PR #2067: otherwise they are absent from the result.
    metrics=(tuple(name for name, _ in _SUMMARY_FIELDS) + _SELECTION_METRICS
             + STEP_METRICS),
    units={**METRIC_UNITS, **STEP_UNITS},
    parameters=(
        ParamSpec(
            name="selection",
            description=selection_help() + " The default, "
                        f"'{DEFAULT_SELECTION}', is the cut behind "
                        "n_events_calo_edep_above_50mev.",
            default=DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
        *(ParamSpec(name=name, description=desc, default=default,
                    minimum=lo, maximum=hi)
          for name, _, default, lo, hi, desc in STEP_PARAMS),
    ),
    summarize=summarize_edep,
    input_hint=(
        "art file(s) holding compressDetStepMCs, CaloClusterMaker and "
        "FindMCPrimary — e.g. TargetStops/mcs/dts files."
    ),
)
