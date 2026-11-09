"""Reading the EventNtuple (EventNtuple/ntuple): one value per track, shared
by every analysis of EventNtuple files, so a variable means the same thing
in all of them.

Track variables (TRACK_VARIABLES) come from the `trk`, `trkqual`, `trkpid`
and `trkcalohit` branches and from the track's segments (`trksegs`,
`trksegpars_lh`) at the tracker front and middle. Two details:

* A track can cross TT_Front twice, and the first crossing is usually the
  upstream-going leg. Every `*_front` quantity is the first crossing with
  pz > 0; `downstream` means pz > 0 at TT_Mid.
* The fit parameters (`trksegpars_lh`) share the segments' indexing, so a
  track's `d0`, `maxr`, `t0err`, ... are the entries at its TT_Mid segment.

Two groups are optional, read only where the file has their branches, and
asking for one on a file without them is a plain error (check_names):

* MC truth (MC_VARIABLES, `mc_*`), in simulated files: the track's own sim
  particle and its ancestors (`trkmcsim`), its true surface crossings
  (`trksegsmc`) and the event's true beam intensity (`evtinfomc.nprotons`).
  The MC TT_Front crossing is the earliest one with pz > 0, since the true
  steps are not stored in time order.
* CRV coincidences (CRV_VARIABLES, `crv_*`): the coincidence nearest in
  time to the track, from `crvcoincs`.

The fullsim analyses read the same tree in pyfitter's grouping
(analyses/fullsim/eventntuple.py) but share the constants and the origin
labelling below.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TREE_PATH = "EventNtuple/ntuple"
# An event's trigger decision for path X is the branch trig_X.
TRIGGER_PREFIX = "trig_"

# Surface ids a track segment can be extrapolated to, from
# Offline/DataProducts/inc/SurfaceId.hh (as pyutils.pyselect maps them).
SID_TT_FRONT = 0
SID_TT_MID = 1
SID_ST_FOILS = 104
# ST_Front, ST_Back, ST_Inner, ST_Outer: the stopping target's envelope.
SID_ST_BOUNDARY = (100, 101, 102, 103)
SID_OPA = 95

# Process codes, Offline/MCDataProducts/inc/ProcessCode.hh.
DIO = 166             # mu2eMuonDecayAtRest
CE_MINUS = (167, 168)  # mu2eCeMinusEndpoint, mu2eCeMinusLeadingLog
DIO_LL = 170          # mu2eDIOLeadingLog
INTERNAL_RMC = 171
EXTERNAL_RMC = 172
FLATE_MINUS = 173
FLATE_PLUS = 174
CE_PLUS = (169, 176)  # mu2eCePlusEndpoint, mu2eCePlusLeadingLog
EXTERNAL_RPC = 178
INTERNAL_RPC = 179
# The stopped-muon generators (mu2eMuonCaptureAtRest .. mu2eCePlusLeadingLog,
# and mu2eMuonBNV): each places its particle where a muon stopped.
MUON_STOP_CODES = (*range(165, 177), 195)
# Generator ids pyCount takes for cosmics (CRY and CORSIKA).
COSMIC_GENS = (38, 44)
# A DIO electron born further out than this (mm, detector frame) is from
# the inner proton absorber rather than the stopping target.
IPA_MIN_RHO_MM = 75.0
# A muon whose end momentum is below this (MeV/c) stopped, rather than left
# the volume or was killed at a simulation stage boundary.
STOPPED_MUON_P_MAX = 1.0

# MCRelationship::relation (Offline/MCDataProducts/inc/MCRelationship.hh).
REL_SAME, REL_MOTHER, REL_UMOTHER = 0, 2, 5

# The origin label each track gets, following pyCount's count_particle_types:
# the process code itself where there is one, plus these three. CE is always
# labelled with 168 and CE+ with 176, whichever generator made it.
ORIGIN_IPA_DIO = 0
ORIGIN_COSMIC = -1
ORIGIN_OTHER = -2
ORIGIN_NAMES = {
    DIO: "DIO", ORIGIN_IPA_DIO: "IPA DIO", 168: "CE-", 176: "CE+",
    EXTERNAL_RPC: "external RPC", INTERNAL_RPC: "internal RPC",
    INTERNAL_RMC: "internal RMC", EXTERNAL_RMC: "external RMC",
    FLATE_MINUS: "flat e-", FLATE_PLUS: "flat e+",
    ORIGIN_COSMIC: "cosmic", ORIGIN_OTHER: "other",
}

# The `trk` (TrkInfo) leaves offered by their own names.
TRK_FIELDS = ("status", "goodfit", "pdg", "nhits", "ndof", "nactive",
              "ndouble", "nplanes", "nnullambig", "nmat", "nmatactive",
              "nipaup", "nipadown", "nstup", "nstdown", "tsdainter",
              "opainter", "chisq", "fitcon", "maxgap", "avggap", "avgedep",
              "seedalg", "fitalg")
_SEGPAR_FIELDS = ("d0", "maxr", "tanDip", "rad", "t0", "t0err")

TRACK_VARIABLES: dict[str, str] = {
    **{name: f"trk.{name}" for name in TRK_FIELDS},
    "status": "trk.status: 1 Kalman fit converged, 2 fit OK, -1 failed",
    "charge": "-1 for e-/mu- fits (pdg > 0), +1 for e+/mu+",
    "trkqual": "track quality MVA output (trkqual.result)",
    "pid": "PID MVA output (trkpid.result)",
    "has_front": "has a downstream-going TT_Front crossing",
    "p_front": "momentum at TT_Front, downstream crossing (MeV/c)",
    "pt_front": "transverse momentum at TT_Front (MeV/c)",
    "pz_front": "pz at TT_Front (MeV/c)",
    "cos_front": "cos(theta) = pz/p at TT_Front",
    "tandip_front": "tan(dip) = pz/pT at TT_Front",
    "t_front": "time at TT_Front (ns)",
    "momerr_front": "momentum uncertainty at TT_Front (MeV/c)",
    "downstream": "pz > 0 at TT_Mid",
    "p_mid": "momentum at TT_Mid (MeV/c)",
    "pt_mid": "transverse momentum at TT_Mid (MeV/c)",
    "cos_mid": "cos(theta) at TT_Mid",
    "t_mid": "time at TT_Mid (ns)",
    "d0": "helix d0 at TT_Mid (mm)",
    "maxr": "helix maximum radius at TT_Mid (mm)",
    "tandip": "helix tan(dip) at TT_Mid",
    "rad": "helix radius at TT_Mid (mm)",
    "t0": "helix t0 at TT_Mid (ns)",
    "t0err": "helix t0 uncertainty at TT_Mid (ns)",
    "has_calo": "has a matched calorimeter cluster",
    "calo_edep": "matched cluster energy (MeV)",
    "calo_dt": "track - cluster time difference (ns)",
    "event": "event number", "run": "run number", "subrun": "subrun number",
    "ntrk": "tracks in the event",
}

# MC truth per track, only in simulated files. Positions are in the
# detector frame (mm; the tracker centre at the origin).
MC_VARIABLES: dict[str, str] = {
    "mc_pdg": "PDG id of the track's sim particle",
    "mc_start_code": "creation process code of the track's sim particle "
                     "(ProcessCode.hh, e.g. 167/168 CE-, 166/170 DIO)",
    "mc_gen": "generator id of the track's sim particle",
    "mc_origin": "what made the track, the first classified particle among "
                 "its sims (its own chain, then any other particle with hits "
                 "on it, as fullsim_sensitivity labels it): 168 CE-, 176 CE+, 166 DIO on the stopping target, "
                 "0 DIO on the IPA, 171/172 internal/external RMC, 178/179 "
                 "external/internal RPC, 173/174 flat e-/e+, -1 cosmic, "
                 "-2 other",
    "mc_mother_pdg": "PDG id of the sim particle's parent (0 if none recorded)",
    "mc_p_start": "true momentum at creation (MeV/c)",
    "mc_t_start": "true creation time (ns)",
    "mc_p_gen": "generated momentum: the momentum at creation of the event's "
                "primary (generated) particle, when the track's particle is "
                "it or descends from it; NaN for a track from pileup (MeV/c)",
    "mc_pdg_gen": "PDG id of that generated particle",
    "mc_start_code_gen": "creation process code of that generated particle",
    "mc_has_stop": "the track's particle comes from a stopped muon whose "
                   "stop position is known",
    "mc_stop_x": "x of the parent muon's stop (mm, detector frame): the end "
                 "point of a muon ancestor that came to rest, else the "
                 "creation point of the particle a stopped-muon generator "
                 "(DIO, CE, RMC, capture, ...) placed at the stop",
    "mc_stop_y": "y of the parent muon's stop (mm, detector frame)",
    "mc_stop_z": "z of the parent muon's stop (mm, detector frame)",
    "mc_stop_rho": "radius of the parent muon's stop (mm; above ~75 mm is "
                   "the IPA rather than the stopping target)",
    "mc_nhits": "straw hits from the sim particle on the track",
    "mc_nactive": "active straw hits from the sim particle on the track",
    "mc_has_front": "the sim particle has a downstream-going TT_Front crossing",
    "mc_p_front": "true momentum at TT_Front, downstream crossing (MeV/c)",
    "mc_pt_front": "true transverse momentum at TT_Front (MeV/c)",
    "mc_pz_front": "true pz at TT_Front (MeV/c)",
    "mc_cos_front": "true cos(theta) at TT_Front",
    "mc_t_front": "true time at TT_Front (ns)",
    "mc_p_mid": "true momentum at TT_Mid (MeV/c)",
    "mc_t_mid": "true time at TT_Mid (ns)",
    "mc_npot": "true protons on target in the event's microbunch "
               "(evtinfomc.nprotons; the beam intensity, ~1e7-1e8 for Run 1)",
}

# CRV per track: the coincidence nearest in time to the track, taking the
# track's time at TT_Front (downstream crossing), or at TT_Mid where it has
# none. Read where the file has crvcoincs, in data as in simulation.
CRV_VARIABLES: dict[str, str] = {
    "crv_n": "CRV coincidences in the event",
    "crv_dt": "track time (TT_Front, else TT_Mid) minus the time of the "
              "nearest CRV coincidence (ns); positive when the coincidence "
              "came first, NaN when the event has none. The fullsim CE "
              "cuts veto 0 < dt < 150",
    "crv_time": "time of the nearest CRV coincidence (ns)",
    "crv_pes": "photoelectrons in the nearest CRV coincidence",
    "crv_nhits": "hits in the nearest CRV coincidence",
    "crv_nlayers": "layers in the nearest CRV coincidence",
    "crv_sector": "sector type of the nearest CRV coincidence (numbering "
                  "set by the CRV geometry version, crs.sectorType*)",
    "crv_x": "x of the nearest CRV coincidence (mm, detector frame)",
    "crv_y": "y of the nearest CRV coincidence (mm, detector frame)",
    "crv_z": "z of the nearest CRV coincidence (mm, detector frame)",
}

# The leaves read, by name (uproot's filter_name): only what TRACK_VARIABLES
# needs. Split struct members are named "<branch>.<member>"; evtinfo's are not.
LEAVES = (
    *(f"trk.{name}" for name in TRK_FIELDS),
    "trkqual.result", "trkpid.result",
    "trkcalohit.did", "trkcalohit.edep", "trkcalohit.dt",
    "trksegs", "trksegpars_lh", "event", "run", "subrun",
)
# Read too where the file has them (simulation), for MC_VARIABLES.
MC_LEAVES = ("trksegsmc", "trkmcsim")
# The true beam intensity. Read by its full path: evtinfo has a leaf of the
# same name (the reconstructed estimate, 0 in simulation).
NPOT_BRANCH = "evtinfomc/nprotons"
# Read too where the file has them, for CRV_VARIABLES.
CRV_LEAVES = ("crvcoincs.time", "crvcoincs.PEs", "crvcoincs.nHits",
              "crvcoincs.nLayers", "crvcoincs.sectorType",
              "crvcoincs.pos.fCoordinates.fX", "crvcoincs.pos.fCoordinates.fY",
              "crvcoincs.pos.fCoordinates.fZ")


class NtupleError(ValueError):
    """An input or request that cannot be used, worded for the caller."""


def parse_trigger_names(text: str) -> list[str]:
    """'apr_TrkDe_80m70p, trig_cpr_TrkDe_80m70p' -> path names, prefix dropped."""
    names = [n for n in re.split(r"[,\s]+", text.strip()) if n]
    if not names:
        raise NtupleError("no trigger paths given")
    out: list[str] = []
    for name in names:
        name = name[len(TRIGGER_PREFIX):] if name.startswith(TRIGGER_PREFIX) else name
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
            raise NtupleError(f"'{name}' does not name a trigger path")
        if name in out:
            raise NtupleError(f"'{name}' is listed twice")
        out.append(name)
    return out


# --- helpers ---------------------------------------------------------------------

def _numpy(array, fill=np.nan) -> np.ndarray:
    import awkward as ak
    return np.asarray(ak.to_numpy(ak.fill_none(array, fill)), dtype=np.float64)


def _flat(array, fill=np.nan) -> np.ndarray:
    """events * tracks -> one value per track, in event order."""
    import awkward as ak
    return _numpy(ak.flatten(array, axis=1), fill)


def _magnitude(vector):
    xyz = vector.fCoordinates
    return np.sqrt(xyz.fX ** 2 + xyz.fY ** 2 + xyz.fZ ** 2)


def _is_any(values, codes):
    result = values == codes[0]
    for c in codes[1:]:
        result = result | (values == c)
    return result


# --- MC truth --------------------------------------------------------------------

def origin_labels(trkmcsim):
    """What made each track: one ORIGIN_NAMES key per track (events *
    tracks), from its sim particles (events * tracks * sims).

    The track's sim particles are labelled in turn and the first that is
    something other than "other" decides, so an electron from a photon
    conversion is labelled by the RMC that made the photon. Follows pyCount's
    count_particle_types, with two differences: DIO from the leading-log
    generator (170) counts as DIO, and a track whose sims are all "other"
    is labelled "other" rather than failing.
    """
    import awkward as ak

    code = trkmcsim["startCode"]
    gen = trkmcsim["gen"]
    rho = np.hypot(trkmcsim["pos"]["fCoordinates"]["fX"],
                   trkmcsim["pos"]["fCoordinates"]["fY"])

    is_dio = _is_any(code, (DIO, DIO_LL))
    # Later assignments win, in pyCount's order: CE beats an RMC ancestor.
    label = ak.zeros_like(code) + ORIGIN_OTHER
    for mask, value in (
        (is_dio & (rho <= IPA_MIN_RHO_MM), DIO),
        (is_dio & (rho > IPA_MIN_RHO_MM), ORIGIN_IPA_DIO),
        (_is_any(gen, COSMIC_GENS), ORIGIN_COSMIC),
        (code == INTERNAL_RPC, INTERNAL_RPC),
        (code == EXTERNAL_RPC, EXTERNAL_RPC),
        (code == INTERNAL_RMC, INTERNAL_RMC),
        (code == EXTERNAL_RMC, EXTERNAL_RMC),
        (_is_any(code, CE_MINUS), 168),
        (_is_any(code, CE_PLUS), 176),
        (code == FLATE_MINUS, FLATE_MINUS),
        (code == FLATE_PLUS, FLATE_PLUS),
    ):
        label = ak.where(mask, value, label)
    return ak.fill_none(ak.firsts(label[label != ORIGIN_OTHER], axis=-1), ORIGIN_OTHER)


def ancestry(trkmcsim):
    """Which of each track's sims are its own particle and that particle's
    ancestors (events * tracks * sims, bool).

    trkmcsim holds, per matched particle in rank order, that particle
    (rank >= 0) and then its ancestors (rank -1), and finally the event's
    primary if no chain contained it. So the track's own chain is the
    leading block before the next match (rank >= 1); within it trkrel is
    relative to the track's particle, which drops the appended primary.
    """
    import awkward as ak

    rank = trkmcsim["rank"]
    index = ak.local_index(rank, axis=-1)
    next_match = ak.fill_none(ak.firsts(index[rank >= 1], axis=-1), np.iinfo(np.int32).max)
    rel = trkmcsim["trkrel"]["_rel"]
    return (index < next_match) & _is_any(rel, (REL_SAME, REL_MOTHER, REL_UMOTHER))


def mc_track_variables(arrays) -> dict[str, np.ndarray]:
    """MC_VARIABLES but mc_npot, one entry per track, from the trkmcsim and
    trksegsmc leaves (events * tracks * sims, events * tracks * true steps)."""
    import awkward as ak

    out: dict[str, np.ndarray] = {}
    sims = arrays["trkmcsim"]
    sim = ak.firsts(sims, axis=-1)                   # the track's own particle
    out["mc_pdg"] = _flat(sim.pdg)
    out["mc_start_code"] = _flat(sim.startCode)
    out["mc_gen"] = _flat(sim.gen)
    out["mc_origin"] = _flat(origin_labels(sims), ORIGIN_OTHER)
    out["mc_p_start"] = _flat(_magnitude(sim.mom))
    out["mc_t_start"] = _flat(sim.time)
    out["mc_nhits"] = _flat(sim.nhits)
    out["mc_nactive"] = _flat(sim.nactive)

    chain = sims[ancestry(sims)]                     # the particle, then its ancestors
    mother = chain[ak.local_index(chain.pdg, axis=-1) == 1]
    out["mc_mother_pdg"] = _flat(ak.firsts(mother.pdg, axis=-1), 0.0)

    generated = ak.firsts(chain[chain.prirel._rel == REL_SAME], axis=-1)
    out["mc_p_gen"] = _flat(_magnitude(generated.mom))
    out["mc_pdg_gen"] = _flat(generated.pdg)
    out["mc_start_code_gen"] = _flat(generated.startCode)

    # The parent muon's stop, from the first sim along the chain that knows
    # it: a muon at rest ends there; a stopped-muon generator's particle
    # starts there. In staged simulation the muon in the chain is often the
    # earlier stage's, killed at the stage boundary (stopCode 54) rather than
    # at rest, so the generated particle's start is the only record.
    stopped_muon = (abs(chain.pdg) == 13) & (_magnitude(chain.endmom) < STOPPED_MUON_P_MAX)
    from_stop = _is_any(chain.startCode, MUON_STOP_CODES)
    knows = stopped_muon | from_stop
    for axis in ("X", "Y", "Z"):
        where = ak.where(stopped_muon, chain.endpos.fCoordinates[f"f{axis}"],
                         chain.pos.fCoordinates[f"f{axis}"])
        out[f"mc_stop_{axis.lower()}"] = _flat(ak.firsts(where[knows], axis=-1))
    out["mc_stop_rho"] = np.hypot(out["mc_stop_x"], out["mc_stop_y"])
    out["mc_has_stop"] = ~np.isnan(out["mc_stop_z"])

    steps = arrays["trksegsmc"]
    steps = steps[ak.argsort(steps.time, axis=-1)]  # stored out of time order
    mom = steps.mom.fCoordinates
    pz = mom.fZ
    pt = np.sqrt(mom.fX ** 2 + mom.fY ** 2)
    p = np.sqrt(mom.fX ** 2 + mom.fY ** 2 + mom.fZ ** 2)

    def first(values, mask):
        return _flat(ak.firsts(values[mask], axis=-1))

    front = (steps.sid == SID_TT_FRONT) & (pz > 0)
    out["mc_has_front"] = _flat(ak.any(front, axis=-1), False).astype(bool)
    out["mc_p_front"] = first(p, front)
    out["mc_pt_front"] = first(pt, front)
    out["mc_pz_front"] = first(pz, front)
    out["mc_t_front"] = first(steps.time, front)
    with np.errstate(all="ignore"):
        out["mc_cos_front"] = out["mc_pz_front"] / out["mc_p_front"]
    mid = steps.sid == SID_TT_MID
    out["mc_p_mid"] = first(p, mid)
    out["mc_t_mid"] = first(steps.time, mid)
    return out


# --- CRV -------------------------------------------------------------------------

def crv_track_variables(arrays, track_time, ntrk) -> dict[str, np.ndarray]:
    """CRV_VARIABLES, one entry per track: the event's coincidence nearest in
    time to `track_time` (one value per track, NaN where it has none)."""
    import awkward as ak

    event_of_track = np.repeat(np.arange(len(ntrk)), ak.to_numpy(ntrk))
    times = arrays["crvcoincs.time"]
    out: dict[str, np.ndarray] = {
        "crv_n": ak.to_numpy(ak.num(times, axis=1)).astype(np.float64)[event_of_track]}

    def per_track(values):
        """events * coincs -> tracks * coincs, each track its event's list."""
        return values[event_of_track]

    dt = np.asarray(track_time, dtype=np.float64)[:, np.newaxis] - per_track(times)
    nearest = ak.argmin(abs(dt), axis=-1, keepdims=True)
    out["crv_dt"] = _numpy(ak.firsts(dt[nearest], axis=-1))
    for name, leaf in (("crv_time", "crvcoincs.time"), ("crv_pes", "crvcoincs.PEs"),
                       ("crv_nhits", "crvcoincs.nHits"), ("crv_nlayers", "crvcoincs.nLayers"),
                       ("crv_sector", "crvcoincs.sectorType"),
                       ("crv_x", "crvcoincs.pos.fCoordinates.fX"),
                       ("crv_y", "crvcoincs.pos.fCoordinates.fY"),
                       ("crv_z", "crvcoincs.pos.fCoordinates.fZ")):
        out[name] = _numpy(ak.firsts(per_track(arrays[leaf])[nearest], axis=-1))
    # A track with no time has no nearest coincidence.
    no_time = np.isnan(np.asarray(track_time, dtype=np.float64))
    for name in CRV_VARIABLES:
        if name != "crv_n":
            out[name][no_time] = np.nan
    return out


# --- per-track variables ---------------------------------------------------------

def track_variables(arrays) -> dict[str, np.ndarray]:
    """TRACK_VARIABLES, one entry per track, from one file's leaves.

    `arrays` maps each name in LEAVES to its awkward array (events * tracks
    [* segments]). Also returns `_event`: each track's event index in the file.
    MC_VARIABLES are added when `arrays` has the MC_LEAVES too, mc_npot
    when it has NPOT_BRANCH (one value per event), and CRV_VARIABLES when it
    has the CRV_LEAVES (events * coincidences).
    """
    import awkward as ak

    ntrk = ak.num(arrays["trk.status"], axis=1)
    event_of_track = np.repeat(np.arange(len(ntrk)), ak.to_numpy(ntrk))
    out: dict[str, np.ndarray] = {"_event": event_of_track}
    for name in TRK_FIELDS:
        out[name] = _flat(arrays[f"trk.{name}"])
    out["charge"] = np.where(out["pdg"] > 0, -1.0, 1.0)
    out["trkqual"] = _flat(arrays["trkqual.result"])
    out["pid"] = _flat(arrays["trkpid.result"])

    segs = arrays["trksegs"]
    sid = segs.sid
    mom = segs.mom.fCoordinates
    pz = mom.fZ
    pt = np.sqrt(mom.fX ** 2 + mom.fY ** 2)
    p = np.sqrt(mom.fX ** 2 + mom.fY ** 2 + mom.fZ ** 2)

    def first(values, mask):
        return _flat(ak.firsts(values[mask], axis=-1))

    front = (sid == SID_TT_FRONT) & (pz > 0)
    out["has_front"] = _flat(ak.any(front, axis=-1), False).astype(bool)
    out["p_front"] = first(p, front)
    out["pt_front"] = first(pt, front)
    out["pz_front"] = first(pz, front)
    out["t_front"] = first(segs.time, front)
    out["momerr_front"] = first(segs.momerr, front)
    with np.errstate(all="ignore"):
        out["cos_front"] = out["pz_front"] / out["p_front"]
        out["tandip_front"] = out["pz_front"] / out["pt_front"]

    mid = sid == SID_TT_MID
    out["p_mid"] = first(p, mid)
    out["pt_mid"] = first(pt, mid)
    out["t_mid"] = first(segs.time, mid)
    pz_mid = first(pz, mid)
    with np.errstate(all="ignore"):
        out["cos_mid"] = pz_mid / out["p_mid"]
    out["downstream"] = pz_mid > 0

    # Fit parameters share the segments' indexing: take the entry at the
    # position of the track's first TT_Mid segment (missing if it has none,
    # or its parameter list is too short to have that entry).
    pars = arrays["trksegpars_lh"]
    mid_index = ak.fill_none(
        ak.firsts(ak.local_index(sid, axis=-1)[mid], axis=-1), -1)
    at_mid = ak.local_index(pars.t0, axis=-1) == mid_index
    for name in _SEGPAR_FIELDS:
        out["tandip" if name == "tanDip" else name] = _flat(
            ak.firsts(pars[name][at_mid], axis=-1))

    has_calo = _flat(arrays["trkcalohit.did"], -1) >= 0
    out["has_calo"] = has_calo
    out["calo_edep"] = np.where(has_calo, _flat(arrays["trkcalohit.edep"]), np.nan)
    out["calo_dt"] = np.where(has_calo, _flat(arrays["trkcalohit.dt"]), np.nan)

    for name in ("event", "run", "subrun"):
        out[name] = _numpy(arrays[name])[event_of_track]
    out["ntrk"] = ak.to_numpy(ntrk).astype(np.float64)[event_of_track]
    if all(leaf in arrays for leaf in MC_LEAVES):
        out.update(mc_track_variables(arrays))
    if NPOT_BRANCH in arrays:
        out["mc_npot"] = _numpy(arrays[NPOT_BRANCH])[event_of_track]
    if all(leaf in arrays for leaf in CRV_LEAVES):
        track_time = np.where(np.isnan(out["t_front"]), out["t_mid"], out["t_front"])
        out.update(crv_track_variables(arrays, track_time, ntrk))
    return out


def check_names(variables: dict, *expressions: str) -> None:
    """Refuse an mc_* or crv_* variable on a file without the branches it
    needs, saying why, rather than as an unknown name."""
    named = {name for text in expressions
             for name in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text)}
    asked = sorted((named & set(MC_VARIABLES)) - set(variables))
    if asked:
        raise NtupleError(
            f"{', '.join(asked)} need MC truth, and this file has no "
            f"{', '.join((*MC_LEAVES, NPOT_BRANCH))} branches (real data, or "
            "an ntuple written without MC)")
    asked = sorted((named & set(CRV_VARIABLES)) - set(variables))
    if asked:
        raise NtupleError(
            f"{', '.join(asked)} need CRV coincidences, and this file has no "
            "crvcoincs branches (an ntuple written without the CRV)")


# --- reading ---------------------------------------------------------------------

def open_tree(path: Path):
    """The file's EventNtuple tree, or an NtupleError saying it has none."""
    import uproot

    try:
        return uproot.open(path)[TREE_PATH]
    except (KeyError, OSError, ValueError) as exc:
        raise NtupleError(f"{path}: no {TREE_PATH} ({exc}) — is this an "
                          "EventNtuple file?") from None


def read_track_arrays(tree, path: Path) -> dict:
    """The LEAVES, plus MC_LEAVES, NPOT_BRANCH and CRV_LEAVES where the file
    has them, as awkward arrays by name (events * tracks ...)."""
    names = {key.rsplit("/", 1)[-1] for key in tree.keys()}
    absent = [leaf for leaf in LEAVES if leaf not in names]
    if absent:
        raise NtupleError(f"{path.name}: missing branch(es) {', '.join(absent)} "
                          "— an EventNtuple version this analysis does not know?")
    optional = []
    for group in (MC_LEAVES, CRV_LEAVES):
        if all(leaf in names for leaf in group):
            optional += group
    batch = tree.arrays(filter_name=list(LEAVES) + optional, library="ak")
    arrays = {name: batch[name] for name in batch.fields}
    if NPOT_BRANCH in tree.keys():
        arrays[NPOT_BRANCH] = tree[NPOT_BRANCH].array(library="ak")
    return arrays


def read_file(path: Path, paths: list[str]) -> tuple[dict, dict[str, np.ndarray], int]:
    """(track variables, {path: per-event decision}, number of events).

    With no `paths`, only the track variables are read (no trig_ branches
    needed), as selection_efficiency_ntuple does.
    """
    tree = open_tree(path)
    available = [k[len(TRIGGER_PREFIX):] for k in tree.keys(recursive=False)
                 if k.startswith(TRIGGER_PREFIX)]
    missing = [p for p in paths if p not in available]
    if missing:
        raise NtupleError(
            f"{path.name}: no trigger branch for {', '.join(missing)}. "
            f"It has: {', '.join(available) or '(none)'}"
        )
    arrays = read_track_arrays(tree, path)
    triggers: dict[str, np.ndarray] = {}
    if paths:
        decisions = tree.arrays(filter_name=[TRIGGER_PREFIX + p for p in paths],
                                library="np")
        triggers = {p: np.asarray(decisions[TRIGGER_PREFIX + p], dtype=bool)
                    for p in paths}
    return track_variables(arrays), triggers, tree.num_entries
