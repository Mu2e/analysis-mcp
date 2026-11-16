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
- a JSON sidecar with the inputs, their reconstruction and the count that
  normalizes it.

Normalization: the loose rate anakit uses for cosmics
(approx_ce_sensitivity.recommended_cosmic_rate("box"), electrons passing
set 81 in 100-110 MeV/c, per second per MeV/c) times the on-spill time,
spread over the reference's true electrons passing set 81 in 100-110 MeV/c.
No CRY livetime is needed, and the preselection's further cuts (CRV, TSDA,
OPA, calorimeter) act on it as cuts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tune import CALO_FLAG, COSMIC_FLAT_MEV, EXTRA_PRESELECTION, VARIABLE_NAMES

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_REFERENCE = DATA_DIR / "fullsim_cosmic_reference.npz"
# Momentum kept in the table, a margin around the flat range.
KEEP_P_MEV = (95.0, 115.0)
COLUMNS = VARIABLE_NAMES + ("t0", "p", "event", "mc_pdg", CALO_FLAG) + EXTRA_PRESELECTION


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


def rows_for(rows: dict, preselection_flags: tuple, rate_per_s_per_mev: float,
             onspill_seconds: float) -> tuple[dict, float]:
    """The reference's true electrons in the flat range passing the extra
    preselection, and the weight each carries per MeV/c of window."""
    norm = normalizing_count(rows)
    if norm <= 0:
        raise ReferenceError("the cosmic reference has no true electron passing "
                             f"set 81 in {COSMIC_FLAT_MEV[0]:g}-{COSMIC_FLAT_MEV[1]:g} MeV/c")
    p = rows["p"]
    keep = ((p > COSMIC_FLAT_MEV[0]) & (p < COSMIC_FLAT_MEV[1])
            & (np.abs(rows["mc_pdg"]) == 11) & rows[CALO_FLAG].astype(bool))
    for flag in preselection_flags:
        keep &= rows[flag].astype(bool)
    per_mev = rate_per_s_per_mev * onspill_seconds / norm
    return {k: v[keep] for k, v in rows.items()}, per_mev
