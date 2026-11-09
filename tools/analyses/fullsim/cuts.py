"""The CE-like track selection: pyfitter's cut-set 80 (config.py '80_1d',
cuts as analyze.py Analyze.define_cuts applies them).

Every cut is a track-level mask (events x tracks). A cut on a segment
quantity applies at one surface, as in pyfitter:
`ak.all(~at_surface | condition, axis=-1)`, so a track with no segment there
passes it vacuously and `has_trk_front_seg` is what requires one. An
event-level cut (the trigger, the calorimeter energy) is broadcast to the
event's tracks.

The cuts and their order are pyfitter's, so the cut flow lines up with its
cutflows/*.csv. Where pyfitter's code and its intent differ, this follows
what the code does on today's EventNtuple: the upstream and multi-track
vetoes ask for trk.t0, which pyfitter never reads, so they fall back to the
tracker-front segment times (the first one, and the mean, respectively).
DEFAULT_CUTS is pyfitter's switch set less its fit ranges: in_mom_range and
within_t0 are there to enable, but here the momentum window is scanned and
the time window is the time_window parameter.
"""

import numpy as np

from ...eventntuple import (SID_OPA, SID_ST_BOUNDARY, SID_ST_FOILS, SID_TT_FRONT,
                            SID_TT_MID, TRIGGER_PREFIX)
from .eventntuple import DEFAULT_TRIGGERS

# The pyfitter cut set the thresholds come from (pyfitter config.py
# VERSION_CUTS). '80_1d' and '80_2d' differ only in within_t0's lower edge.
CUT_SET = "80_1d"
TRKPID_MIN = 0.54
TANDIP_WINDOW = (0.575, 0.85)
TRKQUAL_MIN = 0.155
T0ERR_MAX_NS = 0.85
MIN_ACTIVE_HITS = 20
# A CRV coincidence vetoes a track that comes this long after it (track
# time minus coincidence time, open interval): asymmetric, as in the C++.
CRV_DT_NS = (0.0, 150.0)
# A downstream track is vetoed by an upstream one this long before it
# (closed interval): the reflection of the same particle.
UPSTREAM_VETO_DT_NS = (40.0, 110.0)
# Two downstream e+- tracks closer in time than this veto each other.
MULTI_TRK_DT_NS = 150.0
# pyfitter's fit ranges, as cuts: its default momentum fit range, and the
# 1D cut set's time range.
MOM_RANGE_MEVC = (100.0, 110.0)
T0_WINDOW_NS = (540.0, 1650.0)

# name -> description, in pyfitter's order.
CUT_DESCRIPTIONS: dict[str, str] = {
    "has_a_track": "event has at least one reconstructed track",
    "is_good_track": "track fit status >= 0 and goodfit != 0",
    "has_trk_front_seg": "track has a segment at the tracker front (TT_Front)",
    "is_reco_electron_or_positron": "track fit hypothesis is e- or e+ (PDG +-11)",
    "has_downstream": "downstream: p_z > 0 at the earliest tracker-front segment",
    "charge_selection": "track fit hypothesis is the signal lepton (e- for sign "
                        "minus, e+ for plus)",
    "or_trigger": "event passed any of trigger_paths",
    "upstream_veto": ("no good upstream track {:g}-{:g} ns before this downstream "
                      "one (reflection veto)").format(*UPSTREAM_VETO_DT_NS),
    "no_multi_trk_veto": (f"no other good downstream e+- track within "
                          f"{MULTI_TRK_DT_NS:g} ns (sign minus only)"),
    "good_trkpid": (f"TrkPID > {TRKPID_MIN:g} (trkpid_min) and the event has a "
                    "calorimeter cluster with energy > 0"),
    "pz_over_pt": "{:g} < tan(dip) < {:g} at the first tracker-front segment".format(
        *TANDIP_WINDOW),
    "st_boundary": "extrapolates to a stopping-target boundary (ST front, back, "
                   "inner or outer)",
    "has_st": "extrapolates to the stopping-target foils",
    "no_opa": "does not extrapolate to the outer proton absorber",
    "good_trkqual": f"TrkQual > {TRKQUAL_MIN:g} (trkqual_min)",
    "has_hits": f"at least {MIN_ACTIVE_HITS} active tracker hits",
    "within_t0err": f"t0 error < {T0ERR_MAX_NS:g} ns at the tracker middle",
    "no_crv_veto": ("no CRV coincidence {:g}-{:g} ns before the track at the "
                    "tracker front").format(*CRV_DT_NS),
    "in_mom_range": "{:g} < p < {:g} MeV/c at the tracker front".format(*MOM_RANGE_MEVC),
    "within_t0": "{:g} < t0 < {:g} ns at the tracker front".format(*T0_WINDOW_NS),
}

# pyfitter's switch set, less the fit-range cuts (see the module docstring).
DEFAULT_CUTS = tuple(name for name in CUT_DESCRIPTIONS
                     if name not in ("in_mom_range", "within_t0"))


class CutError(ValueError):
    """A cut list that names unknown cuts, worded for the caller."""


def active_cuts(enable: str, disable: str) -> list[str]:
    """DEFAULT_CUTS plus `enable` minus `disable` (comma-separated names),
    in cut order."""
    known = list(CUT_DESCRIPTIONS)
    active = set(DEFAULT_CUTS)
    for text, add in ((enable, True), (disable, False)):
        names = [n.strip() for n in text.split(",") if n.strip()]
        unknown = [n for n in names if n not in known]
        if unknown:
            raise CutError(f"unknown cut(s) {', '.join(unknown)}. "
                           f"Known: {', '.join(known)}")
        active = active | set(names) if add else active - set(names)
    return [n for n in known if n in active]


def _at_all(at_surface, condition):
    """Track-level: `condition` holds at every segment on the surface."""
    import awkward as ak
    return ak.all(~at_surface | condition, axis=-1)


def cut_masks(data: dict, sign: str, trkqual_min: float = TRKQUAL_MIN,
              trkpid_min: float = TRKPID_MIN, triggers=DEFAULT_TRIGGERS) -> dict:
    """name -> track-level boolean mask for every cut, in cut order.
    `triggers`: the paths or_trigger accepts (any of them), as read by
    read_eventntuple."""
    import awkward as ak

    trk = data["trk"]
    segs = data["trkfit"]["trksegs"]
    lh = data["trkfit"]["trksegpars_lh"]
    crv = data["crv"]
    evt = data["evt"]
    pdg = trk["trk.pdg"]

    sid = segs["sid"]
    at_front = sid == SID_TT_FRONT
    at_mid = sid == SID_TT_MID
    time = segs["time"]
    mom = segs["mom"]["fCoordinates"]
    p = np.sqrt(mom["fX"] ** 2 + mom["fY"] ** 2 + mom["fZ"] ** 2)
    front_times = time[at_front]

    # Downstream: the earliest tracker-front segment goes downstream. A track
    # with no front segment is not, so the vetoes take it as upstream.
    earliest = front_times == ak.min(front_times, axis=-1, keepdims=True)
    pz_earliest = ak.where(earliest, mom["fZ"][at_front], 0.0)
    downstream = ak.fill_none(ak.any(pz_earliest > 0, axis=-1), False)
    is_lepton = (pdg == 11) | (pdg == -11)
    good = (trk["trk.status"] >= 0) & (trk["trk.goodfit"] != 0)

    masks = {}
    masks["has_a_track"] = ak.ones_like(pdg, dtype=bool)
    masks["is_good_track"] = good
    masks["has_trk_front_seg"] = ak.any(at_front, axis=-1)
    masks["is_reco_electron_or_positron"] = is_lepton
    masks["has_downstream"] = downstream
    masks["charge_selection"] = pdg == (11 if sign == "minus" else -11)

    triggered = evt[TRIGGER_PREFIX + triggers[0]] == 1
    for name in triggers[1:]:
        triggered = triggered | (evt[TRIGGER_PREFIX + name] == 1)
    masks["or_trigger"] = ak.broadcast_arrays(triggered, pdg)[0]

    # Pairs of tracks in an event: i is the track judged, j the other.
    index = ak.local_index(pdg, axis=1)
    other = index[:, :, None] != index[:, None, :]

    # Upstream veto, timed by each track's first front segment (0 without one).
    t_first = ak.fill_none(ak.firsts(front_times, axis=-1), 0.0)
    dt = t_first[:, :, None] - t_first[:, None, :]
    reflection = ((dt >= UPSTREAM_VETO_DT_NS[0]) & (dt <= UPSTREAM_VETO_DT_NS[1])
                  & (downstream & good)[:, :, None]
                  & (~downstream & good)[:, None, :] & other)
    masks["upstream_veto"] = ~ak.any(reflection, axis=2)

    # Multi-track veto, timed by the mean of each track's front segments.
    if sign == "minus":
        t_mean = ak.fill_none(ak.mean(front_times, axis=-1), 0.0)
        candidate = downstream & is_lepton & good
        coincident = ((abs(t_mean[:, :, None] - t_mean[:, None, :]) < MULTI_TRK_DT_NS)
                      & candidate[:, :, None] & candidate[:, None, :] & other)
        masks["no_multi_trk_veto"] = ~ak.any(coincident, axis=2)
    else:
        masks["no_multi_trk_veto"] = ak.ones_like(pdg, dtype=bool)

    has_calo = ak.any(data["calo"]["caloclusters.energyDep_"] > 0, axis=-1)
    masks["good_trkpid"] = ((trk["trkpid.result"] > trkpid_min)
                            & ak.broadcast_arrays(has_calo, pdg)[0])
    # A track with no front segment has no tan(dip), and fails.
    tandip = ak.fill_none(ak.firsts(lh["tanDip"][at_front], axis=-1), -100.0)
    masks["pz_over_pt"] = (TANDIP_WINDOW[0] < tandip) & (tandip < TANDIP_WINDOW[1])

    on_boundary = sid == SID_ST_BOUNDARY[0]
    for boundary in SID_ST_BOUNDARY[1:]:
        on_boundary = on_boundary | (sid == boundary)
    masks["st_boundary"] = ak.any(on_boundary, axis=-1)
    masks["has_st"] = ak.sum(sid == SID_ST_FOILS, axis=-1) > 0
    masks["no_opa"] = ak.sum(sid == SID_OPA, axis=-1) == 0
    masks["good_trkqual"] = trk["trkqual.result"] > trkqual_min
    masks["has_hits"] = trk["trk.nactive"] >= MIN_ACTIVE_HITS
    masks["within_t0err"] = _at_all(at_mid, lh["t0err"] < T0ERR_MAX_NS)

    # CRV: t_front - t_coinc for every front segment against every
    # coincidence in the event: events x tracks x front segments x coincs.
    dt_crv = front_times[:, :, :, None] - crv["crvcoincs.time"][:, None, None, :]
    near = (dt_crv > CRV_DT_NS[0]) & (dt_crv < CRV_DT_NS[1])
    masks["no_crv_veto"] = ~ak.any(ak.any(near, axis=3), axis=2)

    masks["in_mom_range"] = _at_all(
        at_front, (MOM_RANGE_MEVC[0] < p) & (p < MOM_RANGE_MEVC[1]))
    masks["within_t0"] = _at_all(
        at_front, (T0_WINDOW_NS[0] < time) & (time < T0_WINDOW_NS[1]))

    return {name: masks[name] for name in CUT_DESCRIPTIONS}


def apply_cuts(masks: dict, active: list[str], ntracks) -> tuple:
    """(track mask, events passing after each active cut in turn).

    `ntracks`: any events x tracks array, giving the shape when no cut is
    active. The cut flow counts events with at least one track passing every
    cut so far, as pyutils' CutManager.create_cut_flow does.
    """
    import awkward as ak

    combined = ak.ones_like(ntracks, dtype=bool)
    flow = []
    for name in active:
        combined = combined & masks[name]
        flow.append(int(ak.sum(ak.any(combined, axis=-1))))
    return combined, flow
