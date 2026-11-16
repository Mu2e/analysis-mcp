#!/usr/bin/env python3
"""Reduce reconstructed cosmic EventNtuples to fullsim_sensitivity's cosmic
reference (tools/analyses/fullsim/cosmic_reference.py).

The input must be the CRY signal-region cosmics reconstructed with the same
job, code and conditions as the points the reference will serve: Mu2eBO's
full_nts (EventNtuple/fcl/from_dts-primary.fcl and the full study's code
tarball) over dts.mu2e.CosmicSignal.MDC2025ap.art. Record that in --reco: it
goes into the sidecar JSON, which fullsim_sensitivity reports with every
optimized result.

    python3 scripts/make_fullsim_cosmic_reference.py --files 'nts.*.root' \\
        --reco 'Code_ana_v020301_fcl_mdc2025bc, Sim_best v1_5, run 1430' \\
        --out data/fullsim_cosmic_reference.npz

Run with the `ana` python, OPENBLAS_NUM_THREADS=1. Files are read one at a
time.
"""
import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.analyses.fullsim import cosmic_reference as ref  # noqa: E402
from tools.analyses.fullsim.cuts import CUT_SET  # noqa: E402
from tools.analyses.fullsim.eventntuple import (DEFAULT_TRIGGERS,  # noqa: E402
                                                read_eventntuple)
from tools.analyses.fullsim.tune import (EXTRA_PRESELECTION, LOOSE_CUTS,  # noqa: E402
                                         candidate_tracks, combine_tracks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--files", nargs="+", required=True,
                    help="EventNtuple files or globs, or @list.txt")
    ap.add_argument("--reco", required=True,
                    help="how the inputs were reconstructed (code, conditions, run)")
    ap.add_argument("--dataset", default="dts.mu2e.CosmicSignal.MDC2025ap.art",
                    help="the primary the inputs were made from")
    ap.add_argument("--cut-set", default=CUT_SET)
    ap.add_argument("--out", default=str(ref.DEFAULT_REFERENCE))
    a = ap.parse_args(argv)
    paths = []
    for item in a.files:
        if item.startswith("@"):
            paths += [l.strip() for l in Path(item[1:]).read_text().splitlines() if l.strip()]
        else:
            paths += sorted(glob.glob(item)) or [item]
    if not paths:
        sys.exit("no input files")
    parts, n_events = [], 0
    for i, path in enumerate(paths):
        data = read_eventntuple(Path(path), DEFAULT_TRIGGERS)
        n_events += len(data["evt"])
        # set 81 alone: the calorimeter and the extra preselection are flags,
        # so the normalizing count is the loose selection's
        tracks = candidate_tracks(data, LOOSE_CUTS, DEFAULT_TRIGGERS, a.cut_set,
                                  flags=EXTRA_PRESELECTION, require_calo=False)
        parts.append(ref.reduce(tracks))
        print(f"[{i + 1}/{len(paths)}] {Path(path).name}: {parts[-1]['p'].size} tracks",
              flush=True)
    rows = combine_tracks(parts)
    ref.save(Path(a.out), rows, {
        "dataset": a.dataset, "reco": a.reco, "cut_set": a.cut_set,
        "loose_cuts": list(LOOSE_CUTS), "n_files": len(paths),
        "n_events": n_events, "files_first": Path(paths[0]).name,
        "files_last": Path(paths[-1]).name})
    print(f"wrote {a.out}: {rows['p'].size} tracks, "
          f"{ref.normalizing_count(rows)} normalizing (true e-, 100-110 MeV/c)")


if __name__ == "__main__":
    main()
