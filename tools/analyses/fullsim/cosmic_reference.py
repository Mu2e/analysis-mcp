"""The cosmic reference sample of fullsim_sensitivity's cut_mode="optimize".

The search needs the cosmics' distribution in every tuned variable, TrkQual
and TrkPID included, whose MVAs change between releases: so the reference
is the CRY signal-region cosmics (dts.mu2e.CosmicSignal.MDC2025ap.art, the
primary of CosmicSignalMix1BB) reconstructed with the same job, code and
conditions as the points (Mu2eBO's full_nts), reduced once by
scripts/make_fullsim_cosmic_reference.py to a small table:

- every loose (set-81) track, the first per event, 95-115 MeV/c, with the
  tuned variables, its MC pdg and event number, and flags for the extra
  preselection cuts (no_opa, no_tsda, no_crv_veto), so one table serves any
  preselection;
- a JSON sidecar with the inputs, their reconstruction, the count that
  normalizes it and, once measured, the CRY livetime behind it.

Normalization, with the livetime (add_livetime; the sidecar's livetime_s):
each track stands for (on-spill time / livetime) run tracks, as in the
fixed-cut mode's cosmic mix measurement (cosmics.py). The livetime is the S2
events behind the reference's dts.mu2e.CosmicSignal inputs (their SAM
dh.gencount, which sums their CosmicCRYAll parents') times the S1 livetime
per event of the CRY sample they descend from (cosmics.S1_LIVETIME_PER_EVENT).
The ntuples are not in SAM, so the inputs come from their jobs' logs (art's
"Opened input file" lines): a job reads several dts files.

Without it (an old sidecar), the loose rate anakit uses for cosmics
(approx_ce_sensitivity.recommended_cosmic_rate("box"), electrons passing
set 81 in 100-110 MeV/c, per second per MeV/c) times the on-spill time,
spread over the reference's true electrons passing set 81 in 100-110 MeV/c.

Either way the preselection's further cuts (CRV, TSDA, OPA, calorimeter)
act on the reference as cuts.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .tune import CALO_FLAG, COSMIC_FLAT_MEV, EXTRA_PRESELECTION, VARIABLE_NAMES

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_REFERENCE = DATA_DIR / "fullsim_cosmic_reference.npz"
# Momentum kept in the table, a margin around the flat range.
KEEP_P_MEV = (95.0, 115.0)
COLUMNS = VARIABLE_NAMES + ("t0", "p", "event", "mc_pdg", CALO_FLAG) + EXTRA_PRESELECTION
# An input art opened, in an art job's log.
OPENED_INPUT = re.compile(r'Opened input file "(?:[^"]*/)?([^"/]+)"')
# The last line of an art job that finished.
ART_COMPLETED = "Art has completed and will exit with status 0"


class ReferenceError(ValueError):
    """The reference is missing or unusable; worded for the caller."""


def reduce(tracks: dict) -> dict:
    """A file's candidate tracks (tune.candidate_tracks with the loose cuts
    and the extra preselection as flags), cut to KEEP_P_MEV."""
    keep = (tracks["p"] > KEEP_P_MEV[0]) & (tracks["p"] < KEEP_P_MEV[1])
    return {k: tracks[k][keep] for k in COLUMNS}


def save(path: Path, rows: dict, provenance: dict) -> None:
    path = Path(path)
    np.savez_compressed(path, **{k: rows[k] for k in COLUMNS})
    norm = normalizing_count(rows)
    meta = dict(provenance, n_rows=int(rows["p"].size), n_normalizing=int(norm),
                columns=list(COLUMNS), flat_mev=list(COSMIC_FLAT_MEV),
                normalized_to="loose set-81 cosmic-electron rate (approx_ce_sensitivity "
                              "recommended_cosmic_rate('box')) x on-spill time")
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def normalizing_count(rows: dict) -> int:
    """True electrons passing set 81 in the flat range: what the loose rate
    counts."""
    p = rows["p"]
    return int(((p > COSMIC_FLAT_MEV[0]) & (p < COSMIC_FLAT_MEV[1])
                & (np.abs(rows["mc_pdg"]) == 11)).sum())


def load(path: Path | str = "") -> tuple[dict, dict]:
    path = Path(path) if str(path).strip() else DEFAULT_REFERENCE
    if not path.exists():
        raise ReferenceError(
            f"no cosmic reference at {path}. cut_mode='optimize' needs the "
            "cosmics reconstructed like the points: run Mu2eBO's full_nts over "
            "dts.mu2e.CosmicSignal.MDC2025ap.art and reduce the ntuples with "
            "scripts/make_fullsim_cosmic_reference.py, or pass cosmic_reference")
    with np.load(path) as f:
        rows = {k: f[k] for k in f.files}
    missing = [c for c in COLUMNS if c not in rows]
    if missing:
        raise ReferenceError(f"{path} lacks column(s) {', '.join(missing)}: "
                             "remake it with scripts/make_fullsim_cosmic_reference.py")
    side = path.with_suffix(".json")
    meta = json.loads(side.read_text()) if side.exists() else {}
    meta["path"] = str(path)
    return rows, meta


def inputs_from_log(text: str, log: str = "the log") -> list[str]:
    """The input art files (names) a finished art job's log says it
    opened, in order. A job that did not finish is refused: its ntuple may
    hold part of its inputs only."""
    if ART_COMPLETED not in text:
        raise ReferenceError(f"{log} has no '{ART_COMPLETED}' line: the job did "
                             "not finish, so which of its inputs it read is unknown")
    names = OPENED_INPUT.findall(text)
    if not names:
        raise ReferenceError(f"{log} opened no input file")
    return names


def add_livetime(meta: dict, inputs: list[str], gencounts: dict[str, int],
                 s1_file: str, livetime_per_event: float) -> dict:
    """`meta` with the reference's CRY livetime: the S2 events behind its
    input files `inputs` (`gencounts`: name -> dh.gencount) times the S1
    livetime per event of `s1_file`'s dataset. An input read by two jobs
    would count twice, so is refused."""
    if len(set(inputs)) != len(inputs):
        twice = sorted({n for n in inputs if inputs.count(n) > 1})
        raise ReferenceError(f"input(s) read by more than one job: {', '.join(twice[:5])}")
    missing = [n for n in inputs if not gencounts.get(n)]
    if missing:
        raise ReferenceError(f"no dh.gencount for {len(missing)} input(s), e.g. {missing[0]}")
    s2_events = int(sum(gencounts[n] for n in inputs))
    return dict(meta, n_inputs=len(inputs), s2_events=s2_events, s1_file=s1_file,
                livetime_per_event=livetime_per_event,
                livetime_s=s2_events * livetime_per_event,
                normalized_to="CRY livetime (S2 events from SAM x S1 livetime per event)")


def rows_for(rows: dict, preselection_flags: tuple, rate_per_s_per_mev: float,
             onspill_seconds: float, livetime_s: float = 0.0) -> tuple[dict, float]:
    """The reference's tracks in the flat range passing the extra
    preselection, of every species, and the weight each carries per MeV/c
    of window.

    With `livetime_s` (the sidecar's), each track is (on-spill time /
    livetime) run tracks, spread flat over the range. Without it, the loose
    rate `rate_per_s_per_mev` counts true electrons, so only they normalize
    the weight; but the weight is then the reference's live-time scale, the
    same for every row. Muons (about a third of the loose tracks) are kept:
    the tuned cuts must still reject them, TrkPID above all, and a table
    without them lets the search drop that cut for free."""
    p = rows["p"]
    keep = ((p > COSMIC_FLAT_MEV[0]) & (p < COSMIC_FLAT_MEV[1])
            & rows[CALO_FLAG].astype(bool))
    for flag in preselection_flags:
        keep &= rows[flag].astype(bool)
    if livetime_s > 0:
        per_mev = onspill_seconds / livetime_s / (COSMIC_FLAT_MEV[1] - COSMIC_FLAT_MEV[0])
    else:
        norm = normalizing_count(rows)
        if norm <= 0:
            raise ReferenceError("the cosmic reference has no true electron passing "
                                 f"set 81 in {COSMIC_FLAT_MEV[0]:g}-{COSMIC_FLAT_MEV[1]:g} MeV/c")
        per_mev = rate_per_s_per_mev * onspill_seconds / norm
    return {k: v[keep] for k, v in rows.items()}, per_mev
