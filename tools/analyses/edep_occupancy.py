"""Detector occupancy and peak channel rates from a pileup sample: the
`occupancy/` histograms of EdepAna (fcl/edep.fcl).

EdepAna (Offline/Analyses/src/EdepAna_module.cc) fills, once per event over
all StrawGasSteps and CaloShowerSteps, pairs of histograms: `<name>_occ`
(steps) and `<name>_edep` (deposited energy, MeV), plus "hit" histograms that
count only steps above the step energy threshold whose folded time lies in the
hit window: `trk_straw_hits`, `calo_crystal_hits`, `calo_disk{0,1}_rcrys_hits`.
The time is folded like the digitizers' microbunchTime (raw time plus
`fold_time_offset_ns`, modulo the microbunch period), so the windows
(`trk_hit_time_*_ns`, `calo_hit_time_*_ns`) are in the event-window frame and
default to the digitization windows. The step time window (`*_step_time_*_ns`,
raw time) applies only to edep's tree and step counts, not to these. This analysis runs that job
(through `edep`'s own job and fcl machinery) over the dts files of ONE pileup
component, scales its histograms from per event to per microbunch, and turns
them into totals and into the busiest-channel rates that sizing electronics
and radiation tolerance need.

Per event -> per microbunch -> rate:

    per_ub = per_event * events_per_microbunch
    rate   = per_ub / microbunch_ns

where `events_per_microbunch` is how many dts events of this component land
in one microbunch on average (POT per microbunch x the component's events per
POT). It defaults to 1, which leaves every number per event.

From generated events. A caller that makes its own sample (a geometry scan,
where the dts events per POT are what changes) does not know
events_per_microbunch in advance. It can give `gen_events_per_pot` instead,
the generated events of the job that made these dts per POT (the upstream
stage's efficiency: MuBeamCat or EleBeamCat events per POT for a flash sample
resampled from them, stopped muons per POT for a stop sample), and then

    events_per_microbunch = n_events / (n_gen_events * prescale)
                            * gen_events_per_pot * pot_per_microbunch

with n_gen_events the sum of the input's subrun GenEventCounts (EdepAna sums
them in beginSubRun, so a subrun with no events left in it still counts) and
`prescale` the fraction kept by the producing job's prescale filter, read from
the file when `prescale_filter` names it (as `count` does, through its job).
GenEventCount counts before the prescale, so a prescaled sample (the Early*
flash samples) without its filter named comes out low by the prescale.
max_events is refused in this mode: the job would stop inside a subrun whose
whole GenEventCount it has already added, and undercount.

Peaks. The peak of a SUM over components is not the sum of the components'
peaks (they peak in different channels), so the combining analysis
`edep_occupancy_combine` adds the per-channel histograms, never metrics. For
that this analysis writes its per-microbunch histograms to
`occupancy_per_microbunch.root` (variances scaled too, so errors survive), and
the combine step only has to add files. This module also owns the metric
computation and the plots, which the combine analysis reuses.

Averaging over phi, for statistics. A channel peak is a maximum over ~20000
straws or ~1350 crystals, each with few counts, so it is biased high and
noisy. Averaging channels at the same radius reduces both:

* Tracker: a "straw row" is the same plane and the same straw number in all
  six panels. Panels are rotated copies, so the six straws have the same
  radius (checked on geom_common.txt, 1 um spread); only z differs, by +-13.9
  mm between the two panel faces. The row's rate is the mean per straw over
  the six, computed from the per-straw histograms.
* Tracker regions: rows are still noisy at low statistics (a geometry scan's
  few tens of microbunches put tens of hits in the busiest row, and the
  maximum over ~3500 rows is biased high), so the busiest REGION is reported
  too: bins of `region_planes` consecutive planes x `region_straws`
  consecutive straw numbers over all six panels (default 12 planes x 8 straws,
  25 mm in radius), the mean per straw in the bin. Each must divide the
  tracker's 36 planes / 96 straws, so the bins tile it.
* Calorimeter: the crystals do not vary in z in a way that matters (a crystal
  is a unit), so the average is radial: per disk, in bins of the crystal
  CENTRE radius, divided by the number of crystals in the bin
  (`calo_disk<d>_ncrystals_r`, a geometry histogram EdepAna fills once).

Uncertainties are sqrt of the summed weights squared of the bins involved,
scaled to the microbunch. They treat steps as independent. Steps from one
event are correlated (a shower makes many), so they UNDERESTIMATE the true
statistical error.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..mu2e_job import ModuleProbeError
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from . import count as count_mod
from . import edep as edep_mod

# Where EdepAna books them in its TFileService file.
OCCUPANCY_DIR = "EDepAna/occupancy"
SCALED_FILE = "occupancy_per_microbunch.root"

# Microbunch period (ns): PhysicsParams' nominal DR period, which EdepAna uses
# to fold step times, 1695 ns for Run 1.
MICROBUNCH_NS = 1695.0

# POT per microbunch for the generated-event normalization: SimEfficiencies2
# Sim_best v1_1 (Run 1A), the value behind the events_per_microbunch table in
# EVENTS_PER_UB_HELP, so the two normalizations agree on a production sample.
POT_PER_MICROBUNCH = 1.58e7

NEEDS_OFFLINE = ("this needs an Offline whose EdepAna has the occupancy histograms "
                 "with their hit windows: Offline PR #2097, merged 2026-11-03, in "
                 "the Musings released after it (or a work area built on Offline "
                 "main since then)")

# The tracker: StrawId's layout (DataProducts/inc/StrawId.hh), unique straw =
# (plane * 6 + panel) * 96 + straw.
N_PLANES, N_PANELS, N_STRAWS = 36, 6, 96
# A straw's mid-point radius in the detector frame, from geom_common.txt:
# r = 380.0 + 3.125 * straw mm, the same in every plane and panel (fit
# residual 0.5 um).
STRAW_R0_MM, STRAW_DR_MM = 380.0, 3.125
# The default tracker region for the coarse peak: 12 planes x 8 straws (25 mm).
REGION_PLANES, REGION_STRAWS = 12, 8

# The calorimeter: two disks of 674 crystals, numbered disk 0 first.
N_DISKS = 2

# Histograms kept in the per-microbunch file, and which of them must exist in
# EdepAna's output (an older occupancy/ lacks the hit and radial ones).
_PAIRS = ("trk_straw", "calo_crystal", "trk_time_folded", "calo_time_folded",
          "trk_rz", "calo_rz",
          *(f"calo_disk{d}_rcrys" for d in range(N_DISKS)))
HIST_NAMES = (
    *(f"{p}_{k}" for p in _PAIRS for k in ("occ", "edep")),
    "trk_straw_hits", "calo_crystal_hits",
    "trk_time_folded_ecut", "calo_time_folded_ecut",
    *(f"calo_disk{d}_rcrys_hits" for d in range(N_DISKS)),
)
GEOMETRY_HIST_NAMES = tuple(f"calo_disk{d}_ncrystals_r" for d in range(N_DISKS))

# --- metrics -------------------------------------------------------------------

_TOTALS = (("trk_steps_per_ub", "steps / microbunch"),
           ("trk_edep_mev_per_ub", "MeV / microbunch"),
           ("trk_hits_per_ub", "hits / microbunch"),
           ("calo_steps_per_ub", "steps / microbunch"),
           ("calo_edep_mev_per_ub", "MeV / microbunch"),
           ("calo_hits_per_ub", "hits / microbunch"))
_KHZ, _MEVS = "kHz", "MeV / s"

_RATE_METRICS = (
    ("trk_peak_straw_hit_rate_khz", _KHZ),
    ("trk_peak_straw_plane", ""), ("trk_peak_straw_panel", ""),
    ("trk_peak_straw_straw", ""),
    ("trk_mean_straw_hit_rate_khz", _KHZ),
    ("trk_peak_row_hit_rate_khz", _KHZ),
    ("trk_peak_row_plane", ""), ("trk_peak_row_straw", ""),
    ("trk_peak_row_radius_mm", "mm"),
    ("trk_peak_region_hit_rate_khz", _KHZ),
    ("trk_peak_region_plane", ""), ("trk_peak_region_radius_mm", "mm"),
    ("trk_region_planes", ""), ("trk_region_straws", ""),
    ("trk_peak_straw_edep_rate_mev_per_s", _MEVS),
    ("trk_peak_row_edep_rate_mev_per_s", _MEVS),
    ("trk_peak_region_edep_rate_mev_per_s", _MEVS),
    ("calo_peak_crystal_hit_rate_khz", _KHZ),
    ("calo_peak_crystal_id", ""), ("calo_peak_crystal_disk", ""),
    ("calo_peak_crystal_edep_rate_mev_per_s", _MEVS),
    *(m for d in range(N_DISKS) for m in (
        (f"calo_disk{d}_peak_radial_hit_rate_khz", _KHZ),
        (f"calo_disk{d}_peak_radial_hit_r_mm", "mm"),
        (f"calo_disk{d}_peak_radial_edep_rate_mev_per_s", _MEVS),
        (f"calo_disk{d}_peak_radial_edep_r_mm", "mm"))),
)
# Statistical errors, for the totals and for every peak rate.
_ERR_OF = tuple(m for m, _ in _TOTALS) + tuple(
    m for m, u in _RATE_METRICS if u in (_KHZ, _MEVS)
    and not m.startswith("trk_mean"))

_BASE = (("n_events_analyzed", "events"),
         ("events_per_microbunch", "events / microbunch"))
COMMON_METRICS = tuple(m for m, _ in _BASE) + tuple(m for m, _ in _TOTALS) \
    + tuple(f"{m}_err" for m, _ in _TOTALS) + tuple(m for m, _ in _RATE_METRICS) \
    + tuple(f"{m}_err" for m in _ERR_OF if m not in dict(_TOTALS))
_UNITS = {**dict(_BASE), **dict(_TOTALS), **dict(_RATE_METRICS)}
COMMON_UNITS = {m: u for m, u in _UNITS.items() if u}
COMMON_UNITS.update({f"{m}_err": _UNITS[m] for m in _ERR_OF if _UNITS[m]})
assert set(COMMON_UNITS) <= set(COMMON_METRICS)

# edep_occupancy's own: what the generated-event normalization used. Not in
# COMMON_METRICS, which edep_occupancy_combine reports for a sum of samples.
GEN_METRICS = ("n_gen_events", "prescale")
OCC_METRICS = COMMON_METRICS + GEN_METRICS
OCC_UNITS = {**COMMON_UNITS, "n_gen_events": "generated events"}


class OccupancyError(RuntimeError):
    """The occupancy histograms are missing or inconsistent, worded for the caller."""


# --- histograms ----------------------------------------------------------------

@dataclass
class Hist:
    """A ROOT TH1/TH2 as arrays: contents and variances without flow bins."""

    values: np.ndarray
    variances: np.ndarray
    edges: list[np.ndarray]

    def scaled(self, factor: float) -> "Hist":
        return Hist(self.values * factor, self.variances * factor ** 2, self.edges)

    def plus(self, other: "Hist") -> "Hist":
        if self.values.shape != other.values.shape or not all(
                np.allclose(a, b) for a, b in zip(self.edges, other.edges)):
            raise OccupancyError("histograms with different binning cannot be added")
        return Hist(self.values + other.values, self.variances + other.variances,
                    self.edges)


def read_hist(obj) -> Hist:
    """A uproot TH1/TH2 as a Hist. Variances are the stored sum of squared
    weights, or the contents where the histogram has none (unit weights)."""
    values = np.asarray(obj.values(), dtype=np.float64)
    try:
        variances = np.asarray(obj.variances(), dtype=np.float64)
    except Exception:
        variances = values.copy()
    edges = [np.asarray(obj.axis(i).edges(), dtype=np.float64)
             for i in range(values.ndim)]
    return Hist(values, variances, edges)


def _write_hist(handle, name: str, hist: Hist) -> None:
    import boost_histogram as bh
    axes = [bh.axis.Variable(e) for e in hist.edges]
    boosted = bh.Histogram(*axes, storage=bh.storage.Weight())
    boosted.view(flow=False)["value"] = hist.values
    boosted.view(flow=False)["variance"] = hist.variances
    handle[name] = boosted


def _read_meta_hist(root, name: str) -> float:
    return float(root[name].values()[0])


def write_scaled_file(path: Path, hists: dict[str, Hist], meta: dict) -> None:
    """The per-microbunch histograms and their metadata, as a ROOT file.

    Metadata (label, events per microbunch, microbunch, the step cuts, ...) is
    a JSON string, so `edep_occupancy_combine` can check that the files it
    adds were made with the same cuts.
    """
    import uproot
    with uproot.recreate(path) as handle:
        for name, hist in hists.items():
            _write_hist(handle, name, hist)
        handle["meta_json"] = json.dumps(meta, sort_keys=True)


def read_scaled_file(path: Path) -> tuple[dict[str, Hist], dict]:
    """Read a file `write_scaled_file` wrote: (histograms, metadata)."""
    import uproot
    try:
        root = uproot.open(path)
    except Exception as exc:
        raise OccupancyError(f"{path}: cannot open as a ROOT file ({exc})") from None
    with root:
        keys = {k.split(";")[0] for k in root.keys()}
        if "meta_json" not in keys:
            raise OccupancyError(
                f"{path}: not a per-microbunch occupancy file. Pass "
                f"{SCALED_FILE} from an 'edep_occupancy' run (or an "
                "'edep_occupancy_combine' one), not EdepAna's own output")
        missing = [n for n in (*HIST_NAMES, *GEOMETRY_HIST_NAMES) if n not in keys]
        if missing:
            raise OccupancyError(f"{path}: missing histogram(s) {', '.join(missing)}")
        meta = json.loads(str(root["meta_json"]))
        hists = {n: read_hist(root[n]) for n in (*HIST_NAMES, *GEOMETRY_HIST_NAMES)}
    return hists, meta


def read_edep_occupancy(path: Path) -> tuple[dict[str, Hist], float, float]:
    """EdepAna's occupancy histograms, per event: (hists, n_events, folding period).

    Raises OccupancyError, saying what produces the missing histograms, for
    an EdepAna file from an Offline without them.
    """
    import uproot
    needs = NEEDS_OFFLINE
    with uproot.open(path) as root:
        keys = {k.split(";")[0] for k in root.keys()}
        if OCCUPANCY_DIR not in keys:
            raise OccupancyError(f"{path} has no {OCCUPANCY_DIR} directory: {needs}")
        directory = root[OCCUPANCY_DIR]
        have = {k.split(";")[0] for k in directory.keys()}
        required = ("nevents", *HIST_NAMES, *GEOMETRY_HIST_NAMES)
        missing = [n for n in required if n not in have]
        if missing:
            raise OccupancyError(
                f"{path}: {OCCUPANCY_DIR} lacks {', '.join(missing)}: {needs}")
        n_events = float(directory["nevents"].values()[0])
        hists = {n: read_hist(directory[n]) for n in (*HIST_NAMES, *GEOMETRY_HIST_NAMES)}
    # EdepAna fills the crystal counts once per job, so a file hadd-ed from k
    # jobs holds k copies: scale them back to one geometry (every crystal once).
    n_geom = sum(hists[n].values.sum() for n in GEOMETRY_HIST_NAMES)
    if n_geom <= 0:
        raise OccupancyError(f"{path}: the crystal-count histograms are empty")
    n_cry = hists["calo_crystal_hits"].values.size
    for n in GEOMETRY_HIST_NAMES:
        hists[n] = hists[n].scaled(n_cry / n_geom)
    period = float(hists["trk_time_folded_occ"].edges[0][-1])
    return hists, n_events, period


# --- the computation ---------------------------------------------------------

def per_microbunch(hists: dict[str, Hist], n_events: float,
                   events_per_microbunch: float) -> dict[str, Hist]:
    """Scale per-event histograms to per microbunch (geometry is left alone)."""
    factor = events_per_microbunch / n_events
    return {n: (h if n in GEOMETRY_HIST_NAMES else h.scaled(factor))
            for n, h in hists.items()}


def events_per_microbunch_from_gen(n_events: float, n_gen_events: float,
                                   prescale: float, gen_events_per_pot: float,
                                   pot_per_microbunch: float) -> float:
    """Mean dts events per microbunch from the sample's own generated events.

    n_events / (n_gen_events * prescale) is the events per generated event
    before the prescale threw any away (prescale = fraction kept, 1 for none),
    as `count` computes it; times generated events per POT and POT per
    microbunch.
    """
    return (n_events / (n_gen_events * prescale)
            * gen_events_per_pot * pot_per_microbunch)


def _peak(values: np.ndarray, variances: np.ndarray) -> tuple[int, float, float]:
    """(index, value, error) of the largest bin; index -1 if all are <= 0."""
    index = int(np.argmax(values))
    if values[index] <= 0:
        return -1, 0.0, 0.0
    return index, float(values[index]), float(math.sqrt(variances[index]))


def _rate(per_ub: float, microbunch_ns: float, per_second_scale: float) -> float:
    """per-microbunch -> rate: per_ub / microbunch_ns, in kHz (1e6) or per s (1e9)."""
    return per_ub / microbunch_ns * per_second_scale


def _khz(x: float, ns: float) -> float:
    return _rate(x, ns, 1e6)


def _per_s(x: float, ns: float) -> float:
    return _rate(x, ns, 1e9)


def tracker_rows(hist: Hist) -> Hist:
    """Per-straw histogram -> the mean per straw of each row, shape (plane, straw).

    A row is one straw number in one plane, over its six panels (same r).
    """
    shape = (N_PLANES, N_PANELS, N_STRAWS)
    if hist.values.size != N_PLANES * N_PANELS * N_STRAWS:
        raise OccupancyError(f"expected {N_PLANES * N_PANELS * N_STRAWS} straw "
                             f"bins, got {hist.values.size}")
    return Hist(hist.values.reshape(shape).mean(axis=1),
                hist.variances.reshape(shape).sum(axis=1) / N_PANELS ** 2,
                [np.arange(N_PLANES + 1.0), np.arange(N_STRAWS + 1.0)])


def region_problem(planes: float, straws: float) -> str | None:
    """Why (planes, straws) cannot tile the tracker, or None."""
    for value, name, n in ((planes, "region_planes", N_PLANES),
                           (straws, "region_straws", N_STRAWS)):
        if value != int(value) or int(value) < 1 or n % int(value):
            divisors = [d for d in range(1, n + 1) if n % d == 0]
            return (f"{name} must be a whole number dividing {n} (one of "
                    f"{divisors}) so the regions tile the tracker; got {value:g}")
    return None


def tracker_regions(hist: Hist, planes: int, straws: int) -> Hist:
    """Per-straw histogram -> the mean per straw of each region of `planes`
    consecutive planes x `straws` consecutive straw numbers over all six
    panels, shape (N_PLANES // planes, N_STRAWS // straws). (1, 1) is the row.
    """
    problem = region_problem(planes, straws)
    if problem:
        raise OccupancyError(problem)
    if hist.values.size != N_PLANES * N_PANELS * N_STRAWS:
        raise OccupancyError(f"expected {N_PLANES * N_PANELS * N_STRAWS} straw "
                             f"bins, got {hist.values.size}")
    shape = (N_PLANES // planes, planes, N_PANELS, N_STRAWS // straws, straws)
    n = planes * N_PANELS * straws
    return Hist(hist.values.reshape(shape).sum(axis=(1, 2, 4)) / n,
                hist.variances.reshape(shape).sum(axis=(1, 2, 4)) / n ** 2,
                [np.arange(0.0, N_PLANES + 1.0, planes),
                 np.arange(0.0, N_STRAWS + 1.0, straws)])


def region_radius_mm(first_straw: int, straws: int) -> float:
    """Mean straw radius of a region starting at straw number `first_straw`."""
    return STRAW_R0_MM + STRAW_DR_MM * (first_straw + (straws - 1) / 2.0)


def crystal_disk(cid: int, n_crystals: int) -> int:
    """The disk of a crystal: the crystals are numbered disk 0 first."""
    return int(cid // (n_crystals // N_DISKS))


def radial_per_crystal(hist: Hist, n_crystals: Hist) -> Hist:
    """A radial histogram divided by the crystal count per bin: the mean per
    crystal at that radius. Bins with no crystal are 0."""
    n = n_crystals.values
    safe = np.where(n > 0, n, 1.0)
    return Hist(np.where(n > 0, hist.values / safe, 0.0),
                np.where(n > 0, hist.variances / safe ** 2, 0.0), hist.edges)


def compute(hists: dict[str, Hist], microbunch_ns: float,
            region: tuple[int, int] = (REGION_PLANES, REGION_STRAWS)
            ) -> tuple[dict[str, float], dict]:
    """Totals and peak rates from per-microbunch histograms.

    Returns (metrics without the event bookkeeping, peak channels). The peak
    channels dict holds where each peak is, including those the metrics do
    not carry (the edep peaks), for the combine step's shares.
    """
    ns = microbunch_ns
    m: dict[str, float] = {}
    planes, straws = int(region[0]), int(region[1])
    pos: dict = {"trk_region_shape": (planes, straws)}

    def total(name: str) -> tuple[float, float]:
        return float(hists[name].values.sum()), float(math.sqrt(hists[name].variances.sum()))

    for key, hist in (("trk_steps_per_ub", "trk_straw_occ"),
                      ("trk_edep_mev_per_ub", "trk_straw_edep"),
                      ("trk_hits_per_ub", "trk_straw_hits"),
                      ("calo_steps_per_ub", "calo_crystal_occ"),
                      ("calo_edep_mev_per_ub", "calo_crystal_edep"),
                      ("calo_hits_per_ub", "calo_crystal_hits")):
        m[key], m[f"{key}_err"] = total(hist)

    # Tracker: the busiest straw and the busiest row (mean of the six panels).
    for tag, name, rate, unit in (("hit", "trk_straw_hits", _khz, "khz"),
                                  ("edep", "trk_straw_edep", _per_s, "mev_per_s")):
        h = hists[name]
        rows = tracker_rows(h)
        i, v, e = _peak(h.values, h.variances)
        ri, rv, re_ = _peak(rows.values.ravel(), rows.variances.ravel())
        regions = tracker_regions(h, planes, straws)
        gi, gv, ge = _peak(regions.values.ravel(), regions.variances.ravel())
        key = "hit_rate_khz" if tag == "hit" else "edep_rate_mev_per_s"
        m[f"trk_peak_straw_{key}"], m[f"trk_peak_straw_{key}_err"] = rate(v, ns), rate(e, ns)
        m[f"trk_peak_row_{key}"], m[f"trk_peak_row_{key}_err"] = rate(rv, ns), rate(re_, ns)
        m[f"trk_peak_region_{key}"] = rate(gv, ns)
        m[f"trk_peak_region_{key}_err"] = rate(ge, ns)
        pos[f"trk_straw_{tag}"] = i
        pos[f"trk_row_{tag}"] = ri
        pos[f"trk_region_{tag}"] = gi
        if tag == "hit":
            plane, rem = divmod(i, N_PANELS * N_STRAWS) if i >= 0 else (-1, -1)
            panel, straw = divmod(rem, N_STRAWS) if i >= 0 else (-1, -1)
            m["trk_peak_straw_plane"], m["trk_peak_straw_panel"] = float(plane), float(panel)
            m["trk_peak_straw_straw"] = float(straw)
            rplane, rstraw = divmod(ri, N_STRAWS) if ri >= 0 else (-1, -1)
            m["trk_peak_row_plane"], m["trk_peak_row_straw"] = float(rplane), float(rstraw)
            m["trk_peak_row_radius_mm"] = (STRAW_R0_MM + STRAW_DR_MM * rstraw
                                           if ri >= 0 else -1.0)
            gplane, gstraw = divmod(gi, N_STRAWS // straws) if gi >= 0 else (-1, -1)
            m["trk_peak_region_plane"] = float(gplane * planes) if gi >= 0 else -1.0
            m["trk_peak_region_radius_mm"] = (region_radius_mm(gstraw * straws, straws)
                                              if gi >= 0 else -1.0)
            m["trk_region_planes"], m["trk_region_straws"] = float(planes), float(straws)
    m["trk_mean_straw_hit_rate_khz"] = _khz(
        float(hists["trk_straw_hits"].values.mean()), ns)

    # Calorimeter: the busiest crystal, and the busiest radial bin per disk.
    n_cry = hists["calo_crystal_hits"].values.size
    for tag, name, rate in (("hit", "calo_crystal_hits", _khz),
                            ("edep", "calo_crystal_edep", _per_s)):
        h = hists[name]
        i, v, e = _peak(h.values, h.variances)
        key = "hit_rate_khz" if tag == "hit" else "edep_rate_mev_per_s"
        m[f"calo_peak_crystal_{key}"], m[f"calo_peak_crystal_{key}_err"] = rate(v, ns), rate(e, ns)
        pos[f"calo_crystal_{tag}"] = i
        if tag == "hit":
            m["calo_peak_crystal_id"] = float(i)
            m["calo_peak_crystal_disk"] = float(crystal_disk(i, n_cry)) if i >= 0 else -1.0
    for d in range(N_DISKS):
        ncry = hists[f"calo_disk{d}_ncrystals_r"]
        for tag, name, rate in (("hit", f"calo_disk{d}_rcrys_hits", _khz),
                                ("edep", f"calo_disk{d}_rcrys_edep", _per_s)):
            h = radial_per_crystal(hists[name], ncry)
            i, v, e = _peak(h.values, h.variances)
            key = "hit_rate_khz" if tag == "hit" else "edep_rate_mev_per_s"
            m[f"calo_disk{d}_peak_radial_{key}"] = rate(v, ns)
            m[f"calo_disk{d}_peak_radial_{key}_err"] = rate(e, ns)
            m[f"calo_disk{d}_peak_radial_{tag}_r_mm"] = (
                float(0.5 * (h.edges[0][i] + h.edges[0][i + 1])) if i >= 0 else -1.0)
            pos[f"calo_disk{d}_radial_{tag}"] = i
    return m, pos


def metrics_from(hists: dict[str, Hist], meta: dict,
                 region: tuple[int, int] = (REGION_PLANES, REGION_STRAWS)
                 ) -> tuple[dict[str, float], dict]:
    """Every metric of the occupancy analyses, in COMMON_METRICS order."""
    computed, pos = compute(hists, meta["microbunch_ns"], region)
    computed["n_events_analyzed"] = meta["n_events"]
    computed["events_per_microbunch"] = meta["events_per_microbunch"]
    return {name: computed[name] for name in COMMON_METRICS}, pos


def peak_shares(components: list[tuple[str, dict[str, Hist]]],
                total_pos: dict, microbunch_ns: float) -> dict[str, dict[str, float]]:
    """Each component's share of the TOTAL's peak channels.

    Share = the component's value in the channel (or row, or radial bin) where
    the sum peaks, over the sum's value there. Also the component's rate in
    that channel. A channel with no total gives share 0.
    """
    sums = {name: components[0][1][name].values.copy() for name in HIST_NAMES}
    for _, hists in components[1:]:
        for name in HIST_NAMES:
            sums[name] = sums[name] + hists[name].values

    def at(values: np.ndarray, index: int) -> float:
        return float(values[index]) if index >= 0 else 0.0

    def row_value(values: np.ndarray, index: int) -> float:
        if index < 0:
            return 0.0
        return float(values.reshape(N_PLANES, N_PANELS, N_STRAWS).mean(axis=1).ravel()[index])

    planes, straws = total_pos["trk_region_shape"]

    def region_value(values: np.ndarray, index: int) -> float:
        if index < 0:
            return 0.0
        h = Hist(values, np.zeros_like(values), [])
        return float(tracker_regions(h, planes, straws).values.ravel()[index])

    probes = [("trk_peak_straw_hit", "trk_straw_hits", "trk_straw_hit", at, _khz),
              ("trk_peak_row_hit", "trk_straw_hits", "trk_row_hit", row_value, _khz),
              ("trk_peak_region_hit", "trk_straw_hits", "trk_region_hit", region_value, _khz),
              ("trk_peak_straw_edep", "trk_straw_edep", "trk_straw_edep", at, _per_s),
              ("trk_peak_row_edep", "trk_straw_edep", "trk_row_edep", row_value, _per_s),
              ("trk_peak_region_edep", "trk_straw_edep", "trk_region_edep", region_value,
               _per_s),
              ("calo_peak_crystal_hit", "calo_crystal_hits", "calo_crystal_hit", at, _khz),
              ("calo_peak_crystal_edep", "calo_crystal_edep", "calo_crystal_edep", at, _per_s)]
    for d in range(N_DISKS):
        probes += [(f"calo_disk{d}_peak_radial_hit", f"calo_disk{d}_rcrys_hits",
                    f"calo_disk{d}_radial_hit", None, _khz),
                   (f"calo_disk{d}_peak_radial_edep", f"calo_disk{d}_rcrys_edep",
                    f"calo_disk{d}_radial_edep", None, _per_s)]
    shares: dict[str, dict[str, float]] = {}
    for label, hists in components:
        out: dict[str, float] = {}
        for tag, name, posname, getter, rate in probes:
            index = total_pos[posname]
            if getter is None:   # radial: per crystal, so divide by the crystals in the bin
                d = int(posname[len("calo_disk")])
                n = components[0][1][f"calo_disk{d}_ncrystals_r"].values
                scale = n[index] if index >= 0 and n[index] > 0 else 1.0
                mine, whole = at(hists[name].values, index) / scale, at(sums[name], index) / scale
            else:
                mine, whole = getter(hists[name].values, index), getter(sums[name], index)
            suffix = "_rate_khz" if rate is _khz else "_rate_mev_per_s"
            out[f"{tag}{suffix}"] = rate(mine, microbunch_ns)
            out[f"{tag}_share"] = mine / whole if whole > 0 else 0.0
        shares[label] = out
    return shares


# --- plots ---------------------------------------------------------------------

PLOT_FILES = ("occupancy_trk_row_hit_rate.png", "occupancy_calo_radial_rate.png",
              "occupancy_folded_time.png", "occupancy_rz.png")


def make_plots(hists: dict[str, Hist], meta: dict, outdir: Path,
               title: str) -> list[str]:
    """The modest set of figures, as PNG files in outdir; returns their paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ns = meta["microbunch_ns"]
    paths = []

    def save(fig, name):
        path = outdir / name
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        paths.append(str(path))

    # 1. tracker hit rate per straw row: plane vs straw radius
    rows = tracker_rows(hists["trk_straw_hits"])
    rate = _khz(rows.values, ns)
    fig, ax = plt.subplots(figsize=(7, 5))
    r_edges = STRAW_R0_MM + STRAW_DR_MM * (np.arange(N_STRAWS + 1) - 0.5)
    mesh = ax.pcolormesh(r_edges, np.arange(N_PLANES + 1), rate, cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="hit rate per straw (kHz), mean of the 6 panels")
    ax.set_xlabel("straw radius (mm)")
    ax.set_ylabel("plane")
    save(fig, PLOT_FILES[0])

    # 2. calorimeter rate per crystal vs crystal-centre radius, per disk
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (tag, label, rate_fn) in zip(axes, (("hits", "hit rate per crystal (kHz)", _khz),
                                                ("edep", "edep rate per crystal (MeV/s)", _per_s))):
        for d in range(N_DISKS):
            ncry = hists[f"calo_disk{d}_ncrystals_r"]
            h = radial_per_crystal(hists[f"calo_disk{d}_rcrys_{tag}"], ncry)
            centres = 0.5 * (h.edges[0][:-1] + h.edges[0][1:])
            ax.errorbar(centres, rate_fn(h.values, ns), yerr=rate_fn(np.sqrt(h.variances), ns),
                        marker="o", ls="-", label=f"disk {d}")
        ax.set_xlabel("crystal-centre radius (mm)")
        ax.set_ylabel(label)
        ax.legend()
    save(fig, PLOT_FILES[1])

    # 3. folded time distributions, per microbunch
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, det in zip(axes, ("trk", "calo")):
        for name, label in ((f"{det}_time_folded_occ", "all steps"),
                            (f"{det}_time_folded_ecut", "above the energy threshold")):
            h = hists[name]
            ax.stairs(h.values, h.edges[0], label=label)
        ax.set_xlabel("time mod microbunch period (ns)")
        ax.set_ylabel("steps / microbunch / bin")
        ax.set_title("tracker" if det == "trk" else "calorimeter")
        ax.legend()
    save(fig, PLOT_FILES[2])

    # 4. per-microbunch r vs z, occupancy and dose
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, det in zip(axes, ("trk", "calo")):
        for ax, (kind, label) in zip(row, (("occ", "steps / microbunch"),
                                           ("edep", "MeV / microbunch"))):
            h = hists[f"{det}_rz_{kind}"]
            data = np.ma.masked_less_equal(h.values.T, 0)
            mesh = ax.pcolormesh(h.edges[0], h.edges[1], data,
                                 norm=LogNorm() if data.count() else None)
            fig.colorbar(mesh, ax=ax, label=label)
            ax.set_xlabel("z (mm)")
            ax.set_ylabel("r (mm)")
            ax.set_title(("tracker" if det == "trk" else "calorimeter")
                         + (" occupancy" if kind == "occ" else " dose"))
    save(fig, PLOT_FILES[3])
    return paths


# --- the analysis --------------------------------------------------------------

DOSE_NOTE = (
    "For OCCUPANCY (hits) the regular dts.mu2e.EleBeamFlashCat / "
    "MuBeamFlashCat samples are right. For DOSE (the *_edep_* metrics) they are "
    "not: DetStepFilter keeps only events with a step folded after 250 ns or "
    "before 10 ns, so they miss the prompt flash. Dose needs the unfiltered, "
    "prescaled Early*FlashCat samples (EarlyEleBeamFlashCat, "
    "EarlyMuBeamFlashCat), whose events per POT are the stage efficiencies times "
    "the prescale (1000 / 30 / 20).")

EVENTS_PER_UB_HELP = (
    "Mean number of dts events of THIS component per microbunch: POT per "
    "microbunch times the component's events per POT. The default, 1, leaves "
    "every metric per event, not per microbunch. Run 1A at 1.58e7 POT/microbunch "
    "(SimEfficiencies2 Sim_best v1_1): EleBeamFlashCat 92.78, MuBeamFlashCat "
    "173.25, MuStopPileupCat 226.57, EarlyEleBeamFlashCat 3746.4, "
    "EarlyMuBeamFlashCat 276.6. Leave it at 1 when gen_events_per_pot is "
    "given: it is then computed from the sample. " + DOSE_NOTE)

GEN_PER_POT_HELP = (
    "Use instead of events_per_microbunch when that is not known in advance "
    "(a sample you made yourself, e.g. in a geometry scan): the generated "
    "events per POT of the job that made these dts, i.e. the upstream stage's "
    "efficiency (MuBeamCat or EleBeamCat events per POT for a flash sample "
    "resampled from them; stopped muons per POT for a stop sample). Then "
    "events_per_microbunch = n_events / (n_gen_events * prescale) * "
    "gen_events_per_pot * pot_per_microbunch, with n_gen_events summed from "
    "the input's subrun GenEventCounts. 0 (the default) leaves it unused. "
    "Needs a full run: max_events is refused. A prescaled sample (the Early* "
    "flash samples) also needs prescale_filter, or it comes out low by the "
    "prescale.")

PRESCALE_FILTER_HELP = (
    "Only with gen_events_per_pot: module label of the prescale filter that "
    "thinned this sample's stream in the job that made it (e.g. the Early "
    "flash streams' filter), whose kept fraction is divided out because "
    "GenEventCount counts before it. It is read from the file by a short "
    "print_counts job, as 'count' does, and a file whose name says it is from "
    "another stream is refused. Unset (the default) means no prescale (1); "
    "run 'count' on a file to list the filters it carries.")


def _check_period(period: float, microbunch_ns: float) -> list[str]:
    """Warn when the rate period differs from the one EdepAna folded times into."""
    if abs(period - microbunch_ns) > 0.5:
        return [f"microbunch_ns={microbunch_ns:g} differs from the {period:g} ns "
                "period EdepAna folded step times into, so the time cuts and "
                "the rates refer to different periods"]
    return []


def run(context: RunContext) -> RunOutcome:
    """Run EdepAna over the files (via `edep`), then scale and analyze its occupancy histograms."""
    params = context.params
    if params["events_per_microbunch"] <= 0 or params["microbunch_ns"] <= 0:
        return RunOutcome(error="events_per_microbunch and microbunch_ns must be "
                                "greater than zero")
    gen_per_pot = float(params["gen_events_per_pot"])
    filter_label = str(params["prescale_filter"]) or None
    problem = (_gen_mode_problem(params, context.max_events)
               or region_problem(params["region_planes"], params["region_straws"]))
    if problem:
        return RunOutcome(error=problem)

    # The prescale first: its job is short and refuses a file from another
    # stream before EdepAna spends its time on it.
    prescale, counts_extra, counts_n_gen = 1.0, None, None
    if gen_per_pot > 0 and filter_label:
        counts = count_mod.run_counts_job(
            RunContext(input_paths=context.input_paths,
                       outdir=context.outdir / "counts", params={},
                       timeout_s=context.timeout_s, max_events=None,
                       wants_file_list=context.wants_file_list),
            filter_label)
        if counts.error is not None or counts.metrics is None:
            return RunOutcome(log_path=counts.log_path, extra=counts.extra,
                              error=f"reading the prescale of {filter_label}: "
                                    f"{counts.error or 'no counts'}")
        prescale = counts.metrics["prescale"]
        counts_n_gen = counts.metrics["n_gen_events"]
        counts_extra = {k: counts.extra[k] for k in ("prescale_filters",)
                        if k in counts.extra}
    try:
        with_windows = (
            edep_mod.module_accepts(edep_mod.MODULE_TYPE, edep_mod.STEP_PROBE_PARAMETER,
                                    context.outdir)
            and edep_mod.module_accepts(edep_mod.MODULE_TYPE,
                                        edep_mod.HIT_WINDOW_PROBE_PARAMETER, context.outdir))
    except ModuleProbeError as exc:
        return RunOutcome(error=f"could not describe the configured Offline's "
                                f"EdepAna: {exc}")
    if not with_windows:
        # An EdepAna without the hit windows has no occupancy histograms, or
        # (an unmerged build of the PR) ones with a different time cut.
        return RunOutcome(error=(
            "the configured Offline's EdepAna has no hit-window parameters "
            f"({edep_mod.HIT_WINDOW_PROBE_PARAMETER}): {NEEDS_OFFLINE}"))

    # edep owns the job and its fcl; the selection is irrelevant here.
    cut_names = [name for name, *_ in (*edep_mod.STEP_PARAMS, *edep_mod.HIT_WINDOW_PARAMS)]
    edep_params = {"selection": "", **{name: params[name] for name in cut_names}}
    inner = edep_mod.run(RunContext(
        input_paths=context.input_paths, outdir=context.outdir, params=edep_params,
        timeout_s=context.timeout_s, max_events=context.max_events,
        wants_file_list=context.wants_file_list))
    if inner.error is not None or inner.metrics is None:
        return RunOutcome(files=inner.files, log_path=inner.log_path,
                          extra=inner.extra, error=inner.error or "edep job failed")
    extra = dict(inner.extra)
    extra.pop("selection", None)
    if len(inner.files) != 1:
        return RunOutcome(files=inner.files, log_path=inner.log_path, extra=extra,
                          error=f"expected one EdepAna file from the job, got "
                                f"{len(inner.files)}: {inner.files}")
    edep_file = Path(inner.files[0])
    try:
        raw, n_events, period = read_edep_occupancy(edep_file)
    except OccupancyError as exc:
        return RunOutcome(files=inner.files, log_path=inner.log_path, extra=extra,
                          error=str(exc))
    if n_events <= 0:
        return RunOutcome(files=inner.files, log_path=inner.log_path, extra=extra,
                          error="EdepAna analyzed no events, so there is nothing to scale")
    warnings = _check_period(period, params["microbunch_ns"])

    # EdepAna's summary sums every subrun's GenEventCount it read.
    n_gen = float(inner.metrics.get("n_gen_events", -1.0))
    events_per_ub = params["events_per_microbunch"]
    normalization = {"mode": "events_per_microbunch"}
    if gen_per_pot > 0:
        if counts_n_gen is not None and n_gen > 0 and \
                abs(counts_n_gen - n_gen) > 1e-9 * n_gen:
            warnings.append(
                f"EdepAna counted {n_gen:g} generated events and print_counts "
                f"{counts_n_gen:g}; using print_counts' GenEventCount total")
        if counts_n_gen is not None:
            n_gen = counts_n_gen
        if n_gen <= 0:
            return RunOutcome(files=inner.files, log_path=inner.log_path, extra=extra,
                              error=("the input has no generated-event bookkeeping "
                                     "(GenEventCount), so gen_events_per_pot cannot "
                                     "normalize it: give events_per_microbunch "
                                     "instead"))
        events_per_ub = events_per_microbunch_from_gen(
            n_events, n_gen, prescale, gen_per_pot, params["pot_per_microbunch"])
        normalization = {"mode": "gen_events_per_pot",
                         "gen_events_per_pot": gen_per_pot,
                         "pot_per_microbunch": params["pot_per_microbunch"],
                         "n_gen_events": n_gen, "prescale": prescale,
                         "prescale_filter": filter_label or "",
                         "events_per_gen_event": n_events / (n_gen * prescale)}
        if counts_extra:
            normalization.update(counts_extra)
    extra["normalization"] = normalization
    if warnings:
        extra["warnings"] = warnings

    scaled = per_microbunch(raw, n_events, events_per_ub)
    meta = {"label": params["label"], "microbunch_ns": params["microbunch_ns"],
            "events_per_microbunch": events_per_ub,
            "normalization": normalization,
            "n_events": n_events, "max_events": context.max_events,
            "n_input_files": len(context.input_paths),
            "step_cuts": {name: params[name] for name in cut_names},
            "components": [params["label"]]}
    scaled_path = context.outdir / SCALED_FILE
    write_scaled_file(scaled_path, scaled, meta)
    metrics, pos = metrics_from(scaled, meta, region_of(params))
    metrics["n_gen_events"] = n_gen
    metrics["prescale"] = prescale
    extra["peak_channels"] = _describe_positions(pos, scaled)
    extra["error_note"] = ERROR_NOTE
    plots = make_plots(scaled, meta, context.outdir,
                       title=params["label"] or "occupancy per microbunch")
    return RunOutcome(metrics=metrics, files=[*inner.files, str(scaled_path), *plots],
                      log_path=inner.log_path, extra=extra)


def _gen_mode_problem(params: dict, max_events: int | None) -> str | None:
    """Why the normalization parameters cannot be used together, or None."""
    gen_per_pot = float(params["gen_events_per_pot"])
    if gen_per_pot <= 0:
        if params["prescale_filter"]:
            return ("prescale_filter is used only with gen_events_per_pot: "
                    "events_per_microbunch is the final per-microbunch number, "
                    "prescale included")
        return None
    if params["events_per_microbunch"] != 1.0:
        return ("give either events_per_microbunch or gen_events_per_pot, not "
                "both: with gen_events_per_pot, events_per_microbunch is computed "
                "from the sample's generated events")
    if max_events is not None:
        return ("gen_events_per_pot needs a full run, without max_events: a job "
                "stopped early has already counted the whole GenEventCount of the "
                "subrun it stopped in, so the events per generated event would "
                "come out low. Use fewer files for a quicker run")
    return None


ERROR_NOTE = ("*_err are sqrt of the summed squared weights of the bins involved, "
              "scaled to the microbunch. They treat steps as independent; steps "
              "of one event are correlated, so they underestimate the error.")


def _describe_positions(pos: dict, hists: dict[str, Hist]) -> dict:
    """The peak channels in words, including the edep peaks the metrics omit."""
    out: dict = {}
    for tag in ("hit", "edep"):
        i = pos[f"trk_straw_{tag}"]
        if i >= 0:
            plane, rem = divmod(i, N_PANELS * N_STRAWS)
            panel, straw = divmod(rem, N_STRAWS)
            out[f"trk_peak_straw_{tag}"] = {"unique_straw": i, "plane": plane,
                                            "panel": panel, "straw": straw}
        ri = pos[f"trk_row_{tag}"]
        if ri >= 0:
            plane, straw = divmod(ri, N_STRAWS)
            out[f"trk_peak_row_{tag}"] = {"plane": plane, "straw": straw,
                                          "radius_mm": STRAW_R0_MM + STRAW_DR_MM * straw}
        gi = pos.get(f"trk_region_{tag}", -1)
        if gi >= 0:
            planes, straws = pos["trk_region_shape"]
            gplane, gstraw = divmod(gi, N_STRAWS // straws)
            first_plane, first_straw = gplane * planes, gstraw * straws
            out[f"trk_peak_region_{tag}"] = {
                "planes": [first_plane, first_plane + planes - 1],
                "straws": [first_straw, first_straw + straws - 1],
                "radius_mm": [STRAW_R0_MM + STRAW_DR_MM * first_straw,
                              STRAW_R0_MM + STRAW_DR_MM * (first_straw + straws - 1)]}
        ci = pos[f"calo_crystal_{tag}"]
        if ci >= 0:
            n_cry = hists["calo_crystal_hits"].values.size
            out[f"calo_peak_crystal_{tag}"] = {"crystal": ci,
                                               "disk": crystal_disk(ci, n_cry)}
    return out


def summarize(metrics: dict[str, float]) -> str:
    return (f"{metrics['n_events_analyzed']:g} events at "
            f"{metrics['events_per_microbunch']:g} per microbunch: tracker "
            f"{metrics['trk_hits_per_ub']:.4g} hits/ub, peak straw "
            f"{metrics['trk_peak_straw_hit_rate_khz']:.4g} kHz (plane "
            f"{metrics['trk_peak_straw_plane']:g} panel "
            f"{metrics['trk_peak_straw_panel']:g} straw "
            f"{metrics['trk_peak_straw_straw']:g}), peak row "
            f"{metrics['trk_peak_row_hit_rate_khz']:.4g} kHz, peak "
            f"{metrics['trk_region_planes']:g}-plane x {metrics['trk_region_straws']:g}-straw "
            f"region {metrics['trk_peak_region_hit_rate_khz']:.4g} kHz (planes from "
            f"{metrics['trk_peak_region_plane']:g}, r {metrics['trk_peak_region_radius_mm']:.0f} "
            f"mm), mean {metrics['trk_mean_straw_hit_rate_khz']:.4g} kHz; calorimeter "
            f"{metrics['calo_hits_per_ub']:.4g} hits/ub, peak crystal "
            f"{metrics['calo_peak_crystal_id']:g} at "
            f"{metrics['calo_peak_crystal_hit_rate_khz']:.4g} kHz.")


def region_params() -> tuple[ParamSpec, ...]:
    """The coarse tracker region's size, shared with edep_occupancy_combine."""
    return (
        ParamSpec(name="region_planes",
                  description=("Planes per tracker region for the trk_peak_region_* "
                               "metrics: consecutive planes, a divisor of 36 (1, 2, 3, "
                               "4, 6, 9, 12, 18, 36). Larger regions average more "
                               "straws, so the peak is less noisy and less biased "
                               "high at low statistics, but it is the mean over the "
                               "region, not the hottest row."),
                  default=float(REGION_PLANES), minimum=1.0, maximum=float(N_PLANES)),
        ParamSpec(name="region_straws",
                  description=("Straw numbers per tracker region (all six panels), "
                               "a divisor of 96; 8 straws span 25 mm in radius "
                               "(3.125 mm per straw number)."),
                  default=float(REGION_STRAWS), minimum=1.0, maximum=float(N_STRAWS)),
    )


def region_of(params: dict) -> tuple[int, int]:
    return int(params["region_planes"]), int(params["region_straws"])


def step_params() -> tuple[ParamSpec, ...]:
    """edep's step parameters, as this analysis' own (same names and defaults)."""
    note = (" The occupancy hit histograms use only the energy threshold of "
            "these; their time cut is the folded hit window "
            "(trk_hit_time_*_ns, calo_hit_time_*_ns). The step time window is "
            "accepted for the job's step counts and not used by them.")
    return tuple(ParamSpec(name=name, description=desc + note, default=default,
                           minimum=lo, maximum=hi)
                 for name, _, default, lo, hi, desc in edep_mod.STEP_PARAMS)


SPEC = AnalysisSpec(
    name="edep_occupancy",
    fcl=edep_mod.FCL,
    input_kind="art_files",
    run=run,
    description=(
        "Detector occupancy and peak channel rates of one pileup component "
        "(EdepAna's occupancy histograms): tracker and calorimeter steps, "
        "hits and deposited energy per microbunch, the peak straw and crystal "
        "rates, and peak rates averaged over phi for better statistics (the "
        "tracker's straw rows of six panels; the calorimeter's crystal-centre "
        "radius per disk). Writes occupancy_per_microbunch.root for "
        "edep_occupancy_combine. Needs an Offline with the EdepAna occupancy "
        "histograms (Offline PR #2097)."
    ),
    metrics=OCC_METRICS,
    units=OCC_UNITS,
    parameters=(
        ParamSpec(name="events_per_microbunch", description=EVENTS_PER_UB_HELP,
                  default=1.0, minimum=1e-12, maximum=1e9),
        ParamSpec(name="gen_events_per_pot", description=GEN_PER_POT_HELP,
                  default=0.0, minimum=0.0, maximum=1e6),
        ParamSpec(name="pot_per_microbunch",
                  description=("POT per microbunch, used only with "
                               "gen_events_per_pot. The default, 1.58e7, is "
                               "SimEfficiencies2 Sim_best v1_1 (Run 1A), the value "
                               "behind the events_per_microbunch examples."),
                  default=POT_PER_MICROBUNCH, minimum=1.0, maximum=1e12),
        ParamSpec(name="prescale_filter", description=PRESCALE_FILTER_HELP,
                  default="", kind="text", allow_empty=True),
        ParamSpec(name="microbunch_ns",
                  description=("Microbunch period (ns) that per-microbunch numbers "
                               "are divided by to give rates. EdepAna folds step "
                               "times into its own period, 1695 ns for Run 1; "
                               "change this only with a matching Offline."),
                  default=MICROBUNCH_NS, minimum=1.0, maximum=1e6),
        ParamSpec(name="label",
                  description=("A name for this component (e.g. 'MuBeamFlash'), "
                               "stored in the output file so edep_occupancy_combine "
                               "can report each component's share of the peak."),
                  default="", kind="text", allow_empty=True),
        *region_params(),
        *step_params(),
        *(ParamSpec(name=name, description=desc, default=default,
                    minimum=lo, maximum=hi)
          for name, _, default, lo, hi, desc in edep_mod.HIT_WINDOW_PARAMS),
    ),
    summarize=summarize,
    input_hint=(
        "dts art file(s) of ONE pileup component (e.g. dts.mu2e.MuStopPileupCat, "
        "MuBeamFlashCat, EleBeamFlashCat): compressDetStepMCs. Run one component "
        "per call and combine them with edep_occupancy_combine. "
        "max_events scales nothing: the per-event averages use the events "
        "actually analyzed. " + DOSE_NOTE
    ),
)
