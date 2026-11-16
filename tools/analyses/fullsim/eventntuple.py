"""Reading the EventNtuple (EventNtuple/ntuple) for full-simulation analyses.

The branch set is pyfitter's (process.py AnaProcessor.branches), less the
ones only its plots and fits use, read into the same groups so the cut code
reads like the original: data["trk"]["trk.nactive"],
data["trkfit"]["trksegs"], data["crv"]["crvcoincs.time"], ...

Shapes: "evt" is per event; "crv" is events x coincidences; "calo" is
events x clusters; "trk" is
events x tracks; "trkfit" is events x tracks x segments; "trkmc"'s trkmcsim
is events x tracks x sim particles (the track's particle first, then its
ancestors).
"""

import re
from pathlib import Path

import numpy as np

TREE_PATH = "EventNtuple/ntuple"
# A one-bin histogram holding the events the ntuple job processed.
PROCESSED_PATH = "EventNtuple/n_proc_events"

# An event's trigger decision for path X is the branch trig_X.
TRIGGER_PREFIX = "trig_"
# The paths or_trigger accepts by default, either of them: the production
# tracker paths (APR and CPR), as pyfitter's or_trigger has them.
DEFAULT_TRIGGERS = ("apr_TrkDe_80m70p", "cpr_TrkDe_80m70p")

BRANCHES: dict[str, list[str]] = {
    "evt": ["run", "subrun", "event"],
    "crv": ["crvcoincs.time", "crvcoincs.nHits", "crvcoincs.PEs",
            "crvcoincs.timeStart", "crvcoincs.timeEnd"],
    "calo": ["caloclusters.energyDep_"],
    "trk": ["trk.nactive", "trk.pdg", "trk.status", "trk.goodfit",
            "trkqual.result", "trkpid.result"],
    "trkfit": ["trksegs", "trksegpars_lh"],
    "trkmc": ["trkmcsim"],
}
# Read where the file has them: the TSDA intersection flag no_tsda needs
# (cuts.py refuses no_tsda when it is absent, so a cut set without the veto
# still reads older ntuples).
OPTIONAL_BRANCHES: dict[str, list[str]] = {
    "trk": ["trk.tsdainter"],
}

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
# Generator ids pyCount takes for cosmics (CRY and CORSIKA).
COSMIC_GENS = (38, 44)
# A DIO electron born further out than this (mm, detector frame) is from
# the inner proton absorber rather than the stopping target.
IPA_MIN_RHO_MM = 75.0

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


class EventNtupleError(RuntimeError):
    """The input is not an EventNtuple with the branches needed, worded for
    the caller."""


def parse_trigger_paths(text: str) -> list[str]:
    """'apr_TrkDe_80m70p, trig_cpr_TrkDe_80m70p' -> path names, prefix dropped."""
    names = [n for n in re.split(r"[,\s]+", text.strip()) if n]
    if not names:
        raise EventNtupleError("trigger_paths is empty: name at least one path, "
                               f"e.g. '{', '.join(DEFAULT_TRIGGERS)}'")
    out: list[str] = []
    for name in names:
        name = name.removeprefix(TRIGGER_PREFIX)
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
            raise EventNtupleError(f"'{name}' in trigger_paths does not name a trigger path")
        if name not in out:
            out.append(name)
    return out


def read_eventntuple(path: Path, triggers=DEFAULT_TRIGGERS) -> dict:
    """BRANCHES, plus the trig_ branch of each path in `triggers`, from one
    EventNtuple file, as awkward arrays per group."""
    import uproot

    try:
        rootfile = uproot.open(path)
    except Exception as exc:
        raise EventNtupleError(f"{path}: cannot be opened as a ROOT file ({exc})")
    with rootfile:
        try:
            tree = rootfile[TREE_PATH]
        except KeyError:
            raise EventNtupleError(
                f"{path}: no {TREE_PATH} tree — is this an EventNtuple "
                "(nts.*.root) file from a reconstruction job?"
            ) from None
        trigger_branches = [TRIGGER_PREFIX + name for name in triggers]
        have = set(tree.keys())
        missing = [b for b in trigger_branches if b not in have]
        if missing:
            menu = sorted(k.removeprefix(TRIGGER_PREFIX) for k in have
                          if k.startswith(TRIGGER_PREFIX))
            raise EventNtupleError(
                f"{path}: no trigger decision for {', '.join(missing)}. "
                f"trigger_paths must name paths this file records: {', '.join(menu)}"
            )
        branches = dict(BRANCHES, evt=BRANCHES["evt"] + trigger_branches)
        data = {"n_processed": processed_events(rootfile, tree)}
        for group, names in branches.items():
            # uproot lists a sub-branch by its path ("trk/trk.tsdainter")
            leaf_names = have | {k.rsplit("/", 1)[-1] for k in have}
            names = names + [b for b in OPTIONAL_BRANCHES.get(group, [])
                             if b in leaf_names]
            try:
                data[group] = tree.arrays(names, library="ak")
            except uproot.KeyInFileError as exc:
                raise EventNtupleError(
                    f"{path}: {TREE_PATH} has no branch '{exc.key}', which the "
                    f"'{group}' variables need. This EventNtuple version is "
                    "older or newer than the one the cuts were written for."
                ) from None
    return data


def processed_events(rootfile, tree) -> int:
    """Events the job that wrote the file processed: EventNtuple's
    n_proc_events count, or the tree's entries where a file has none. The
    denominator of an efficiency, so a filtered output still counts what it
    was filtered from."""
    try:
        return int(round(float(rootfile[PROCESSED_PATH].values().sum())))
    except Exception:
        return int(tree.num_entries)


def origin_codes(trkmcsim) -> np.ndarray:
    """What made each event's first track: one ORIGIN_NAMES key per event.

    The track's sim particles are labelled in turn and the first that is
    something other than "other" decides, so an electron from a photon
    conversion is labelled by the RMC that made the photon. Follows pyCount's
    count_particle_types, with two differences: DIO from the leading-log
    generator (170) counts as DIO, and an event whose sims are all "other"
    is labelled "other" rather than failing.
    """
    import awkward as ak

    sims = ak.firsts(trkmcsim, axis=1)          # events x sims of track 0
    sims = ak.fill_none(sims, [], axis=0)
    code = sims["startCode"]
    gen = sims["gen"]
    rho = np.hypot(sims["pos"]["fCoordinates"]["fX"],
                   sims["pos"]["fCoordinates"]["fY"])

    def is_any(values, codes):
        result = values == codes[0]
        for c in codes[1:]:
            result = result | (values == c)
        return result

    is_dio = is_any(code, (DIO, DIO_LL))
    # Later assignments win, in pyCount's order: CE beats an RMC ancestor.
    label = ak.zeros_like(code) + ORIGIN_OTHER
    for mask, value in (
        (is_dio & (rho <= IPA_MIN_RHO_MM), DIO),
        (is_dio & (rho > IPA_MIN_RHO_MM), ORIGIN_IPA_DIO),
        (is_any(gen, COSMIC_GENS), ORIGIN_COSMIC),
        (code == INTERNAL_RPC, INTERNAL_RPC),
        (code == EXTERNAL_RPC, EXTERNAL_RPC),
        (code == INTERNAL_RMC, INTERNAL_RMC),
        (code == EXTERNAL_RMC, EXTERNAL_RMC),
        (is_any(code, CE_MINUS), 168),
        (is_any(code, CE_PLUS), 176),
        (code == FLATE_MINUS, FLATE_MINUS),
        (code == FLATE_PLUS, FLATE_PLUS),
    ):
        label = ak.where(mask, value, label)

    first = ak.firsts(label[label != ORIGIN_OTHER], axis=1)
    return ak.to_numpy(ak.fill_none(first, ORIGIN_OTHER)).astype(np.int64)
