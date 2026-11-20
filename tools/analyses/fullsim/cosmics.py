"""The cosmic background measured from the official cosmic mix.

The CE mix (e.g. CeMLeadingLogMix1BB) has a cosmic counterpart made with the
same digitization, pileup and reconstruction (CosmicSignalMix1BB, same
configuration). Its events go through the same cuts and time window as the
CE; the tracks they leave that a cosmic made are the cosmic background, and
each stands for (on-spill time / simulated cosmic livetime) of the run.

The livetime is not read from the files' CosmicLivetime products. The S2
resampler stores them truncated by an integer division in
Mu2eProductMixer::endSubRun (zero, or up to ~25% low, for every resampled
cosmic sample since 2021), and EventNtuple files carry none. It is computed
from what the S2 resampler was meant to store instead:

    livetime = (S2 events resampled) x (S1 livetime / S1 events generated)

The first factor is the cosmic files' dh.gencount in SAM: an S2 job's
gencount is the events it resampled (500k for CosmicCRYAll MDC2025ap), and
every later stage sums its parents'. So the gencount of a cosmic mix file's
mcs parent counts the S2 events behind it, whatever was filtered after. The
second factor, tau, is a property of the S1 CRY sample (its CRY settings
and area), measured once from its subrun products (S1_LIVETIME_PER_EVENT;
read_s1_livetime does it for a new one).
"""

import re
import struct
from pathlib import Path

import numpy as np

from .provenance import dataset_of
from .tune import COSMIC_FLAT_MEV

# cosmic_source values.
COSMIC_AUTO = "auto"
COSMIC_RATE = "rate"

# The cosmic mix datasets tried, in order, for a CE mix with mix suffix
# <mix> (e.g. Mix1BB): CosmicSignal<mix> is MDC2025's, CosmicCRYAll<mix>
# Run-1B's.
COSMIC_DESCRIPTIONS = ("CosmicSignal", "CosmicCRYAll")
MIX_SUFFIX = re.compile(r"(Mix[\w-]+)$")

# S1 livetime per S1 generated event (s), per S1 dataset: summed
# CosmicLivetime over summed GenEventCount of its subruns.
# CosmicDSStopsCRYAll.MDC2025ab: 318 subruns (every 50th file, those on
# disk, 2026-11-19): 2.2882e-4, 0.27% rms per subrun, one CRY setting
# (area 90000 m^2, 0.05-100000 GeV).
S1_LIVETIME_PER_EVENT = {
    "sim.mu2e.CosmicDSStopsCRYAll.MDC2025ab.art": 2.2882e-4,
}

# shape values: flat at the count in COSMIC_FLAT_RANGE, or as measured.
SHAPE_FLAT = "flat"
SHAPE_MEASURED = "measured"
# Where the flat rate is counted (MeV/c): pyfitter's fit range
# (in_mom_range), the region the signal window is scanned in; the same
# range cut_mode="optimize" normalizes its cosmic reference over.
COSMIC_FLAT_RANGE = COSMIC_FLAT_MEV
# The 90% CL Poisson upper limit on a mean when nothing is seen: the count
# used for the flat rate when no cosmic passes, so a window never looks
# background-free for want of cosmic statistics.
ZERO_COUNT_UPPER_LIMIT = 2.303

# The subrun products of an S1 CRY file, and their streamed layout after
# the 10-byte ROOT object header (byte count, version, checksum).
LIVETIME_BRANCH = ("mu2e::CosmicLivetime_generate__Primary./"
                   "mu2e::CosmicLivetime_generate__Primary.obj")
GENCOUNT_BRANCH = ("mu2e::GenEventCount_genCounter__Primary./"
                   "mu2e::GenEventCount_genCounter__Primary.obj")
OBJECT_HEADER = 10


class CosmicError(RuntimeError):
    """The cosmic sample cannot be used, worded for the caller."""


def cosmic_datasets_for(input_names: list[str]) -> list[str]:
    """The cosmic mix datasets that would match CE mix files `input_names`
    (SAM file names), most likely first: same tier, owner, configuration
    and mix, description COSMIC_DESCRIPTIONS + mix."""
    datasets = {dataset_of(name) for name in input_names}
    if None in datasets or len(datasets) != 1:
        raise CosmicError(
            "cosmic_source 'auto' finds the cosmic mix from the inputs' SAM "
            "dataset, but the inputs "
            + ("are not all SAM file names" if None in datasets
               else f"come from {len(datasets)} datasets ({', '.join(sorted(datasets))})")
            + "; name the cosmic dataset in cosmic_source, or 'rate' for "
            "the flat rate")
    tier, owner, description, configuration, fmt = datasets.pop().split(".")
    mix = MIX_SUFFIX.search(description)
    if mix is None:
        raise CosmicError(
            f"cosmic_source 'auto' needs a mixed input (description "
            f"ending Mix<...>), but the inputs are {description}; name the "
            "cosmic dataset in cosmic_source, or 'rate' for the flat rate")
    return [f"{tier}.{owner}.{d}{mix.group(1)}.{configuration}.{fmt}"
            for d in COSMIC_DESCRIPTIONS]


def decode_s1_subrun(livetime_bytes: bytes, gencount_bytes: bytes) -> tuple[float, int]:
    """(livetime in s, events generated) of one S1 subrun from its streamed
    CosmicLivetime (uint64 primaries, then float area, lowE, highE,
    fluxConstant, livetime) and GenEventCount (uint64 count), big-endian.
    uproot's own model of these classes reads them misaligned."""
    *_, livetime = struct.unpack(
        ">Qfffff", livetime_bytes[OBJECT_HEADER:OBJECT_HEADER + 28])
    count, = struct.unpack(">Q", gencount_bytes[OBJECT_HEADER:OBJECT_HEADER + 8])
    return float(livetime), int(count)


def read_s1_livetime(paths: list[Path]) -> tuple[float, int]:
    """Summed livetime (s) and events generated over S1 CRY art files' subruns:
    their ratio is the dataset's S1_LIVETIME_PER_EVENT."""
    import uproot

    livetime, generated = 0.0, 0
    for path in paths:
        tree = uproot.open(path)["SubRuns"]
        for i in range(tree[LIVETIME_BRANCH].num_baskets):
            lt, n = decode_s1_subrun(tree[LIVETIME_BRANCH].basket(i).data.tobytes(),
                                     tree[GENCOUNT_BRANCH].basket(i).data.tobytes())
            livetime += lt
            generated += n
    return livetime, generated


def livetime_per_event(s1_file: str) -> float:
    """tau for the S1 dataset of SAM file `s1_file`."""
    dataset = dataset_of(s1_file)
    if dataset not in S1_LIVETIME_PER_EVENT:
        raise CosmicError(
            f"the cosmic sample descends from S1 file {s1_file}, whose "
            f"dataset {dataset} has no measured livetime per event (known: "
            f"{', '.join(S1_LIVETIME_PER_EVENT)}). Measure it with "
            "cosmics.read_s1_livetime on a few of its files (livetime / "
            "events) and pass it as cosmic_livetime_per_event")
    return S1_LIVETIME_PER_EVENT[dataset]


def cosmic_counts(ev: dict, time_window: tuple[float, float]) -> np.ndarray:
    """Which reduced events are cosmic background: a cosmic made the first
    selected track, inside the time window, with a finite momentum."""
    from .eventntuple import ORIGIN_COSMIC

    t_lo, t_hi = time_window
    return ((ev["origin"] == ORIGIN_COSMIC) & (ev["t"] > t_lo) & (ev["t"] < t_hi)
            & np.isfinite(ev["p"]))


def cosmic_spectrum(p: np.ndarray, weight: float, like, shape: str) -> dict:
    """The cosmic count per bin of spectrum `like`, from the cosmic tracks'
    momenta `p` each standing for `weight` run events.

    flat: the count in COSMIC_FLAT_RANGE, spread evenly per MeV/c
    (ZERO_COUNT_UPPER_LIMIT if none); measured: the tracks binned as they are.
    """
    lo, hi = COSMIC_FLAT_RANGE
    n_flat = int(((p >= lo) & (p < hi)).sum())
    upper_limit = n_flat == 0
    per_mev = (ZERO_COUNT_UPPER_LIMIT if upper_limit else n_flat) * weight / (hi - lo)
    if shape == SHAPE_FLAT:
        values = np.full(like.nbins, per_mev * like.width)
    elif shape == SHAPE_MEASURED:
        values, _ = np.histogram(p, bins=like.edges())
        values = values * weight
    else:
        raise CosmicError(f"cosmic_shape {shape!r} must be '{SHAPE_FLAT}' or "
                          f"'{SHAPE_MEASURED}'")
    return {"values": values.astype(np.float64), "n_flat_range": n_flat,
            "upper_limit": upper_limit and shape == SHAPE_FLAT,
            "rate_per_mev": per_mev}
