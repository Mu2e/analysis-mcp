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

The surface ids, process codes and origin labels are tools/eventntuple.py's,
shared with the other EventNtuple analyses.
"""

import re
from pathlib import Path

import numpy as np

from ...eventntuple import ORIGIN_OTHER, TREE_PATH, TRIGGER_PREFIX, origin_labels

# A one-bin histogram holding the events the ntuple job processed.
PROCESSED_PATH = "EventNtuple/n_proc_events"

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
    """What made each event's first track: one ORIGIN_NAMES key per event
    (tools/eventntuple.origin_labels of track 0)."""
    import awkward as ak

    first = ak.firsts(origin_labels(trkmcsim), axis=1)
    return ak.to_numpy(ak.fill_none(first, ORIGIN_OTHER)).astype(np.int64)
