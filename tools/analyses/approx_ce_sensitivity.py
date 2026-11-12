"""Approximate Run-1A CE sensitivity: signal = CE, background = DIO + cosmics.

A Python conversion of Mu2eOptAna/scripts/rough_run1a_sensitivity.C. It reads
the per-event tree EdepAna writes (nts.*.root, EDepAna/tree), fills the
macro's histograms from the events passing a selection, and estimates
S/sqrt(B) for the best momentum window.

The chain, following the original macro:

1. Signal shape: the energy of the primary at the front of the tracker
   (`primary_trk_front_energy`) for the selected events. The default
   selection, `event_calo_edep_vis > 10`, is the macro's: EdepAna's fixed
   "edep 10 MeV" histogram set. Binned as EdepAna's `trk_front_energy`,
   rebinned x2 and scaled to a rate for NPOT protons at the assumed branching
   ratio and signal efficiency.
2. Signal smearing: convolved with a Gaussian tracker resolution.
3. DIO background: the Heeck/Szafron theoretical spectrum, scaled to a rate,
   then convolved with the *measured* energy-loss response of the same
   selected events (`primary_trk_front_energy_diff`, energy at the tracker
   minus energy at birth) and the same tracker resolution.
4. Cosmic background: flat in momentum at a rough rate per second per MeV/c
   (the `cosmic_rate_per_s_per_mev` parameter), scaled by the live on-spill
   time implied by NPOT.
5. Scan every window [x1, x2] with x1 >= 50 MeV and keep the one maximizing
   S/sqrt(B).

Cut modes: with `cut_mode="momentum"` (default) only the momentum window is
scanned and no time or geometry cut is applied. With `cut_mode="box"` the
momentum, time, R(max), d0 and cos(theta) cuts are optimized by coordinate
ascent on S/sqrt(B). In box mode `cosmic_rate_per_s_per_mev` is the
cosmic-electron rate for the LOOSE selection the cosmic shapes were extracted
with (ConvAna set 81: e-, APR/CPR trigger, upstream-reflection and multi-track
vetoes, 100 < p < 110 MeV/c; no t0, cos, R(max) or d0 cut): the box's cosmic
count is that rate summed over the momentum window times the box's cosmic time
efficiency (flat over the shapes' t0 range) and geometry efficiency in that
window (a fixed sample drawn from the shapes, each track with its own momentum,
since R(max) and d0 depend on it), with no further scaling. The box's momentum
window is kept inside the shapes' momentum range, where the rate is defined. Its default is the
Run-1A rate 10/7.8e5 times the loose/tight yield ratio from the shapes file
(`reference_cuts.combined`, the fraction of the loose set passing the tight
Run-1A cuts 540 < t0 < 1650 ns, 0.525 < cos < 0.649, 482 < R(max) < 642.6 mm).
In momentum mode it is the rate after whatever fixed selection the result is meant to represent, applied as-is. Box mode assumes
time is separable from momentum and geometry for CE and cosmics, and
momentum and (rc, cos) separable for cosmics (R(max) and d0 are not: they are
worked out per track from its momentum). The CE is NOT taken as separable
in momentum and geometry: each CE event keeps its own signal bin and the box's
signal is the smeared spectrum of the events passing the geometry cuts (see
`BoxModel`), since a d0 or R(max) cut removes low-energy-loss events more than
others. As a known first-order approximation, DIO's time and geometry
efficiencies equal the CE's momentum-integrated ones. `cosmic_field_scale`
reshapes the cosmic helices for a scaled field (R -> R/s, pinned at R(max));
the CE side comes from a sample simulated at that field.

The numbers are rough by construction — this is a figure of merit for
comparing beamline configurations, not a sensitivity calculation.
"""

from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..selection import SelectionError
from ..spectrum import Kernel, Spectrum
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from .edep import (TREE_PATH, EdepTreeError, generated_events, read_edep_tree,
                   read_run_record, select_events, selection_help)

# --- assumptions carried over from the macro ---------------------------------

MUON_CAPTURE_RATE = 0.609         # N(muon captures) / N(muon stops) on aluminum
NPOT = 1.0e18                     # protons on target assumed
R_MUE = 1.0e-13                   # conversions per muon capture (the signal assumed)
SIGNAL_BR = R_MUE * MUON_CAPTURE_RATE  # CEs per stopped muon: R_mue per capture x captures per stop
MEAN_POT_PER_EVENT = 1.6e7        # 1BB
ONSPILL_SECONDS_PER_EVENT = 1.695e-6
COSMIC_RATE_PER_SECOND_PER_MEV = 10. / 7.8e5  # rough, per second per MeV/c, taken from Run 1A mu- --> e- analysis
TRK_RESOLUTION_SIGMA_MEV = 0.2
SIGNAL_BOX_MIN_MEV = 50.0         # below this DIO swamps everything anyway
SIGNAL_REBIN = 2

# The Heeck (2016) / Szafron DIO spectrum, finely binned.
DIO_TABLE = Path(
    "/exp/mu2e/app/users/mmackenz/run1b/Run1BAna/data/"
    "heeck_finer_binning_2016_szafron.tbl"
)
DIO_NBINS, DIO_EMIN, DIO_EMAX = 11000, 0.0, 110.0

# The cut EdepAna's "edep 10 MeV" histogram set (hist_2) was fixed to, which
# the macro read its shapes from.
DEFAULT_SELECTION = "event_calo_edep_vis > 10"

# Tree variable -> (nbins, xmin, xmax): the binning of the EdepAna histogram
# each one replaces (EdepAna_module.cc bookHistograms).
SIGNAL_VAR = "primary_trk_front_energy"
RESPONSE_VAR = "primary_trk_front_energy_diff"
BINNING = {
    SIGNAL_VAR: (1500, 0.0, 150.0),
    RESPONSE_VAR: (500, -100.0, 0.0),
    "primary_start_z": (500, 3000.0, 8000.0),
    "primary_start_r": (100, 0.0, 200.0),
}
EXTRA_PLOT_VARS = ("primary_start_z", "primary_start_r")

# --- the box-cut model (cut_mode = "box") ------------------------------------

CUT_MODES = ("momentum", "box")

# The cosmic rate a cut_mode assumes when the caller gives none. Momentum mode
# applies it as-is: the rough post-selection (tight) Run-1A rate. Box mode's
# rate is for the LOOSE selection the cosmic shapes were extracted with, so its
# default is that tight rate times the loose/tight yield ratio measured in the
# shapes (`CosmicShapes.loose_to_tight_ratio`).
COSMIC_RATE_MOMENTUM_MODE = COSMIC_RATE_PER_SECOND_PER_MEV


def recommended_cosmic_rate(cut_mode: str) -> float:
    """The cosmic rate (per second per MeV/c) used for `cut_mode` by default."""
    if cut_mode == "box":
        return COSMIC_RATE_MOMENTUM_MODE * load_cosmic_shapes().loose_to_tight_ratio
    return COSMIC_RATE_MOMENTUM_MODE


# The microbunch is the 1695 ns between proton pulses (the on-spill time per
# event above, in ns). CE time follows the stopped muon's lifetime inside it.
# Cosmics are flat over the time range of the loose shapes instead.
MICROBUNCH_NS = 1695.0
# Bound muon lifetime in aluminum (864 ns, the Mu2e stopping-target value).
MUON_LIFETIME_NS = 864.0
# The base time cut, which the time optimization may only tighten.
TIME_MIN_NS = 525.0
TIME_MAX_NS = 1650.0
# The production's time window (box mode): a CE whose time at the tracker front,
# folded into the microbunch, is below this is dropped before reconstruction
# (digitization keeps hits after ~500 ns). Cut per event on EdepAna's
# primary_trk_front_time, it follows the stopped muons' real arrival times,
# which depend on the stopping target (the exponential model above does not);
# the exponential then only sets the shape above it (`BoxModel.ce_time`).
FRONT_TIME_MIN_NS = 500.0

# The tight Run-1A cuts (no d0 cut): the optimization's starting point, and
# the selection whose yield relative to the shapes' loose selection is
# `reference_cuts.combined` in cosmic_shapes.json.
REFERENCE_TIME_NS = (540.0, 1650.0)
REFERENCE_COS = (0.525, 0.649)
REFERENCE_RMAX_MM = (482.0, 642.6)

# Cosmic shapes: made by scripts/extract_cosmic_shapes.py. Found relative to
# this file, so any clone location works.
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
COSMIC_SHAPES_ROOT = DATA_DIR / "cosmic_shapes.root"
COSMIC_SHAPES_JSON = DATA_DIR / "cosmic_shapes.json"
COSMIC_SAMPLE_SIZE = 200_000
COSMIC_SAMPLE_SEED = 20261101
# The range cosmic_field_scale may take: the R(max)-pinned rescaling is a
# first-order reshaping of shapes taken at the nominal field, not a resimulation.
COSMIC_FIELD_SCALE_RANGE = (0.5, 2.0)
# R[mm] = p[MeV/c] sin(theta) / (C_LIGHT B[T])
C_LIGHT_MEV_PER_MM_T = 0.299792458

# The optional fast-sim TSDA veto (tsda_veto = 1). Reco extrapolates each
# track upstream and vetoes it if it reaches the TS downstream absorber (an
# annulus, KinKalGeomMaker's TSDA) instead of reflecting in the DS gradient.
# Rough nominal fields, no map: B_z at the tracker front and at the TSDA.
TRACKER_BZ_T = 1.0
TSDA_BZ_T = 2.0
TSDA_RADII_MM = (235.0, 525.0)

# Coordinate-ascent grids and stopping rule. A cut variable's candidate edges:
# (low, high, step); time's range is the base time cut instead.
TIME_STEP_NS = 5.0
# R(max) must exceed this for the track to have hits in the tracker, so its
# lower cut never goes below it (CE and cosmics) and is never open.
RMAX_MIN_MM = 400.0
RMAX_GRID_MM = (RMAX_MIN_MM, 1100.0, 5.0)
D0_GRID_MM = (-700.0, 700.0, 5.0)
COS_GRID = (0.0, 1.0, 0.005)
MAX_SWEEPS = 20
SWEEP_TOLERANCE = 1.0e-6       # relative FoM gain below which it stops
# The box optimization maximizes the Punzi figure of merit S/(a/2 + sqrt(B))
# (Punzi, physics/0308063), not S/sqrt(B): with cuts on five variables the
# background can be driven to ~0 while keeping a sliver of signal, where
# S/sqrt(B) diverges. a = 1.28 sigma matches the 90% CL limits Mu2e quotes;
# the expected limit scales roughly as 1/FoM.
PUNZI_A = 1.28


def punzi_fom(signal, background):
    """S / (PUNZI_A/2 + sqrt(B)); finite for B = 0."""
    return signal / (PUNZI_A / 2.0 + np.sqrt(background))

BOX_VARIABLES = ("p", "t", "rmax", "d0", "cos")
GEOMETRY_VARIABLES = ("rmax", "d0", "cos")
GEOMETRY_GRIDS = {"rmax": RMAX_GRID_MM, "d0": D0_GRID_MM, "cos": COS_GRID}
# The momentum range the box-mode plots show (cosmics are flat in it).
PLOT_P_RANGE_MEV = (95.0, 110.0)

# The EdepAna columns box mode reads, for the selected events at the tracker front.
HELIX_COLUMNS = {"rmax": "primary_trk_front_rmax", "d0": "primary_trk_front_d0",
                 "cos": "primary_trk_front_cos"}


def tsda_hit_fraction(rmax, d0, cos):
    """The chance a track at the tracker front hits the TSDA, from its helix
    center rc = (rmax + d0)/2, radius R = (rmax - d0)/2 and cos(theta).

    Adiabatically (p_T^2/B conserved) it reaches the TSDA only if
    sin^2(theta) x TSDA_BZ_T / TRACKER_BZ_T < 1; otherwise it reflects in the
    gradient. On the way the helix center and radius both shrink by
    k = sqrt(TRACKER_BZ_T / TSDA_BZ_T), so at the TSDA the track is at radius
    k sqrt(rc^2 + R^2 + 2 rc R cos(phi)), and with the phase phi unknown the
    chance is the fraction of the turn inside TSDA_RADII_MM. NaN never hits.
    Works on arrays."""
    rmax, d0, cos = (np.asarray(x, dtype=np.float64) for x in (rmax, d0, cos))
    ratio = TRACKER_BZ_T / TSDA_BZ_T
    reaches = (1.0 - cos ** 2) < ratio
    center, radius = (rmax + d0) / 2.0, (rmax - d0) / 2.0
    inner, outer = TSDA_RADII_MM
    with np.errstate(divide="ignore", invalid="ignore"):
        base, twice = center ** 2 + radius ** 2, 2.0 * center * radius
        low = np.clip((inner ** 2 / ratio - base) / twice, -1.0, 1.0)
        high = np.clip((outer ** 2 / ratio - base) / twice, -1.0, 1.0)
        fraction = np.maximum(np.arccos(low) - np.arccos(high), 0.0) / np.pi
    return np.where(reaches & np.isfinite(fraction), fraction, 0.0)


class SensitivityError(RuntimeError):
    """Raised for a problem the caller should see verbatim."""


# --- pieces of the calculation -----------------------------------------------

def load_dio_spectrum(table: Path = DIO_TABLE) -> Spectrum:
    """The theoretical DIO spectrum as a probability density in energy.

    The table holds (energy, weight) rows on a 0.01 MeV grid; each row sets the
    bin containing that energy. Normalized so the result integrates to 1 over
    energy.
    """
    if not table.exists():
        raise SensitivityError(f"DIO spectrum table not found: {table}")
    data = np.loadtxt(table)
    if data.ndim != 2 or data.shape[1] < 2:
        raise SensitivityError(f"DIO table {table} is not two columns of numbers")
    energy, weight = data[:, 0], data[:, 1]

    width = (DIO_EMAX - DIO_EMIN) / DIO_NBINS
    values = np.zeros(DIO_NBINS)
    # Nudge left by half a bin so a tabulated energy lands in the bin it ends.
    index = np.floor((energy - width / 2.0 - DIO_EMIN) / width).astype(np.int64)
    inside = (index >= 0) & (index < DIO_NBINS)
    values[index[inside]] = weight[inside]

    total = values.sum()
    if total <= 0.0:
        raise SensitivityError(f"DIO table {table} summed to zero weight")
    return Spectrum(values / (total * width), DIO_EMIN, width,
                    name="dio", title="DIO spectrum")


def mpv_fwhm(spectrum: Spectrum) -> tuple[float, float]:
    """Rough most-probable value and full width at half maximum."""
    peak = int(np.argmax(spectrum.values))
    mpv = float(spectrum.centers()[peak])
    above = np.flatnonzero(spectrum.values > spectrum.values[peak] / 2.0)
    if above.size == 0:
        return mpv, 0.0
    edges = spectrum.edges()
    return mpv, float(edges[above[-1] + 1] - edges[above[0]])


def scan_signal_box(signal: Spectrum, dio: Spectrum, cosmic: Spectrum,
                    box_min_mev: float = SIGNAL_BOX_MIN_MEV,
                    top_n: int = 10) -> tuple[dict[str, float], list[dict]]:
    """Find the window [x1, x2] maximizing S/sqrt(B).

    The three spectra share one binning, so a window is a slice and the scan is
    a cumulative sum per lower edge. Those sums run outward from each window's
    own lower edge, never as a difference of whole-spectrum prefix sums: the
    DIO spectrum spans ~18 orders of magnitude, so subtracting two such totals
    to get a count of order 1 loses the answer entirely to float cancellation.
    """
    for other in (dio, cosmic):
        if (other.nbins, other.xmin, other.width) != (signal.nbins, signal.xmin,
                                                      signal.width):
            raise SensitivityError(
                f"'{other.name}' is not on the signal's binning; regrid it first"
            )

    centers = signal.centers()
    first_bin = int(np.searchsorted(centers, box_min_mev))
    best: dict[str, float] | None = None
    per_edge: list[dict] = []

    for ibin in range(first_bin, signal.nbins):
        s = np.cumsum(signal.values[ibin:])
        d = np.cumsum(dio.values[ibin:])
        c = np.cumsum(cosmic.values[ibin:])
        background = d + c
        usable = np.flatnonzero((s > 0.0) & (background > 0.0))
        if usable.size == 0:
            continue
        ratio = s[usable] / np.sqrt(background[usable])
        jbest = usable[int(np.argmax(ratio))]
        entry = {
            "low_mev": float(centers[ibin]),
            "high_mev": float(centers[ibin + jbest]),
            "signal": float(s[jbest]), "dio": float(d[jbest]),
            "cosmic": float(c[jbest]), "background": float(background[jbest]),
            "sensitivity": float(s[jbest] / np.sqrt(background[jbest])),
        }
        per_edge.append(entry)
        if best is None or entry["sensitivity"] > best["sensitivity"]:
            best = entry

    if best is None:
        raise SensitivityError(
            "no signal window had both signal and background > 0 — the input "
            "histograms are probably empty above "
            f"{box_min_mev:g} MeV"
        )
    # The best window for each lower edge, ranked -- more informative in the
    # log than the top 10 overall, which only ever differ by a bin.
    per_edge.sort(key=lambda e: e["sensitivity"], reverse=True)
    return best, per_edge[:top_n]


# --- the expected limit ---------------------------------------------------------
# The median expected 90% CL CLs upper limit of a counting experiment, worked
# out on the Asimov data (the background alone, without fluctuations): the
# signal count s at which the exclusion significance reaches EXCLUSION_Z, since
# then CLs = 2 (1 - Phi(Z)) = 0.10. S/sqrt(B) is its large-background limit
# without a background uncertainty; at a few background events, or with one, the
# two rank windows and designs differently.
EXCLUSION_Z = 1.6448536
# Above this many background events the exact formulas lose float precision,
# and their Gaussian limit, Z sqrt(b + sigma^2), is exact to far better than 1e-6.
GAUSSIAN_BACKGROUND = 1.0e10


def _exclusion_q(s, b, sigma):
    """Z^2 of the median expected exclusion of signal s over background b with
    an absolute background uncertainty sigma (arrays): Cowan, Cranmer, Gross
    and Vitells (Eur. Phys. J. C 71, 1554) for sigma = 0, and Bhattiprolu,
    Martin and Wells (arXiv:2009.07249) otherwise."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        stat = 2.0 * (s - b * np.log1p(s / b))
        var = np.where(sigma > 0.0, sigma ** 2, 1.0)
        x = np.sqrt((s + b) ** 2 - 4.0 * s * b * var / (b + var))
        syst = (2.0 * (s - b * np.log((b + s + x) / (2.0 * b))
                       - b * b / var * np.log((b - s + x) / (2.0 * b)))
                - (b + s - x) * (1.0 + b / var))
    return np.where(sigma > 0.0, syst, stat)


def expected_signal_limit(background, sigma=0.0):
    """The median expected 90% CL upper limit on the signal count, for a
    background count `background` (> 0) known to an absolute uncertainty
    `sigma` (0: statistics only). Arrays broadcast; solved by bisection."""
    b, sigma = np.broadcast_arrays(np.asarray(background, dtype=float),
                                   np.asarray(sigma, dtype=float))
    target = EXCLUSION_Z ** 2
    gauss = EXCLUSION_Z * np.sqrt(b + sigma ** 2)
    lo, hi = np.zeros(b.shape), 3.0 * gauss + 10.0
    while np.any(_exclusion_q(hi, b, sigma) < target):   # q rises with s
        hi = np.where(_exclusion_q(hi, b, sigma) < target, 2.0 * hi, hi)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        below = _exclusion_q(mid, b, sigma) < target
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    return np.where(b > GAUSSIAN_BACKGROUND, gauss, 0.5 * (lo + hi))


def scan_limit_window(signal: Spectrum, dio: Spectrum, cosmic: Spectrum,
                      background_uncertainty: float = 0.0,
                      box_min_mev: float = SIGNAL_BOX_MIN_MEV) -> dict[str, float]:
    """The momentum window with the lowest expected limit, scanned over the
    same windows as scan_signal_box (edges are bin centers too).

    The background's uncertainty is `background_uncertainty` times the
    background (a relative uncertainty; 0 = statistics only). `limit_per_signal` is the
    limit on the signal count divided by the window's signal count, so the
    limit on a rate is it times the rate the signal spectrum assumes.
    """
    centers = signal.centers()
    first_bin = int(np.searchsorted(centers, box_min_mev))
    best: dict[str, float] | None = None
    for ibin in range(first_bin, signal.nbins):
        s = np.cumsum(signal.values[ibin:])
        d = np.cumsum(dio.values[ibin:])
        c = np.cumsum(cosmic.values[ibin:])
        background = d + c
        usable = np.flatnonzero((s > 0.0) & (background > 0.0))
        if usable.size == 0:
            continue
        sigma = background_uncertainty * background[usable]
        per_signal = expected_signal_limit(background[usable], sigma) / s[usable]
        k = int(np.argmin(per_signal))
        if best is None or per_signal[k] < best["limit_per_signal"]:
            j = usable[k]
            best = {
                "low_mev": float(centers[ibin]),
                "high_mev": float(centers[ibin + j]),
                "signal": float(s[j]), "dio": float(d[j]), "cosmic": float(c[j]),
                "background": float(background[j]),
                "limit_per_signal": float(per_signal[k]),
            }
    if best is None:
        raise SensitivityError(
            "no signal window had both signal and background > 0 — the input "
            f"histograms are probably empty above {box_min_mev:g} MeV")
    return best


def ce_time_efficiency(t1: float, t2: float,
                       lifetime_ns: float = MUON_LIFETIME_NS):
    """Fraction of CEs in [t1, t2) ns of the microbunch [0, T): the density is
    exp(-t/tau), normalized over the microbunch. Works on arrays."""
    norm = -np.expm1(-MICROBUNCH_NS / lifetime_ns)
    return (np.exp(-np.asarray(t1) / lifetime_ns)
            - np.exp(-np.asarray(t2) / lifetime_ns)) / norm


def front_time_kept(front_time: np.ndarray, floor_ns: float) -> np.ndarray:
    """The events the production time window keeps: those whose tracker-front
    time, folded into the microbunch, is at or above `floor_ns`. An event with
    no front time (NaN) is left to the selection."""
    return ~(np.mod(front_time, MICROBUNCH_NS) < floor_ns)


def cosmic_time_efficiency(t1, t2, t_range: tuple[float, float]):
    """Fraction of cosmics in [t1, t2) ns: flat over `t_range`, the time range
    of the loose cosmic shapes, so the overlap of the window with it over its
    width. Works on arrays."""
    low, high = t_range
    overlap = np.minimum(np.asarray(t2), high) - np.maximum(np.asarray(t1), low)
    return np.maximum(overlap, 0.0) / (high - low)


@dataclass(frozen=True)
class CosmicShapes:
    """The extracted cosmic shapes the box model needs."""

    rc_edges: np.ndarray
    cos_edges: np.ndarray
    rc_vs_cos: np.ndarray       # [irc, icos] weighted counts
    p_edges: np.ndarray
    p_counts: np.ndarray
    b_eff_t: float              # effective field, T
    t_range: tuple[float, float]    # ns, the t0 axis: where the cosmics are flat
    loose_to_tight_ratio: float     # loose yield / tight Run-1A yield


@lru_cache(maxsize=None)
def load_cosmic_shapes(root_path: Path = COSMIC_SHAPES_ROOT,
                       json_path: Path = COSMIC_SHAPES_JSON) -> CosmicShapes:
    """Read `rc_vs_cos`, `p` and the effective field from the shapes files."""
    import json

    import uproot

    for path in (root_path, json_path):
        if not path.exists():
            raise SensitivityError(
                f"cosmic shapes file not found: {path}; make it with "
                "scripts/extract_cosmic_shapes.py (see the README)")
    try:
        with uproot.open(root_path) as rootfile:
            hist2, hist1 = rootfile["rc_vs_cos"], rootfile["p"]
            shapes = dict(
                rc_edges=np.asarray(hist2.axis(0).edges(), dtype=np.float64),
                cos_edges=np.asarray(hist2.axis(1).edges(), dtype=np.float64),
                rc_vs_cos=np.asarray(hist2.values(flow=False), dtype=np.float64),
                p_edges=np.asarray(hist1.axis().edges(), dtype=np.float64),
                p_counts=np.asarray(hist1.values(flow=False), dtype=np.float64))
            t_edges = rootfile["t0"].axis().edges()
            t_range = (float(t_edges[0]), float(t_edges[-1]))
        info = json.loads(json_path.read_text(encoding="utf-8"))
        b_eff = float(info["B_eff"]["median_T"])
        tight = float(info["reference_cuts"]["combined"])
    except (KeyError, ValueError, OSError) as exc:
        raise SensitivityError(
            f"cosmic shapes {root_path} / {json_path} are unreadable ({exc!r}); "
            "remake them with scripts/extract_cosmic_shapes.py") from None
    if not 0.0 < tight <= 1.0:
        raise SensitivityError(
            f"cosmic shapes {json_path}: reference_cuts.combined is {tight:g}, "
            "outside (0, 1]; remake them with scripts/extract_cosmic_shapes.py")
    if not b_eff > 0.0 or not (shapes["rc_vs_cos"].sum() > 0.0
                               and shapes["p_counts"].sum() > 0.0):
        raise SensitivityError(
            f"cosmic shapes {root_path} are empty or have no field; remake them "
            "with scripts/extract_cosmic_shapes.py")
    return CosmicShapes(b_eff_t=b_eff, t_range=t_range,
                        loose_to_tight_ratio=1.0 / tight, **shapes)


@dataclass(frozen=True)
class GeometryPopulation:
    """Tracks as (rmax, d0, cos) columns with weights, for counting a box's
    geometry efficiency: the weight passing over `norm`, the total weight
    that defines efficiency 1. NaN never passes a cut."""

    rmax: np.ndarray
    d0: np.ndarray
    cos: np.ndarray
    weights: np.ndarray
    norm: float
    # The chance each track survives a veto that does not depend on the cuts
    # (the TSDA veto), None for no veto. It scales what passes, not `norm`.
    survival: np.ndarray | None = None
    # Each track's momentum bin: for the CE its bin in the rebinned, unsmeared
    # signal histogram, -1 where it has none (see `signal_bins`); for cosmics
    # the bin its sampled momentum is in.
    bins: np.ndarray | None = None

    def column(self, name: str) -> np.ndarray:
        return getattr(self, name)

    @property
    def kept(self) -> np.ndarray:
        """The weights after the veto."""
        return self.weights if self.survival is None else self.weights * self.survival

    def passing(self, cuts: dict[str, tuple[float, float]],
                skip: str | None = None) -> np.ndarray:
        """Mask of the tracks inside every geometry cut but `skip`'s. R(max)
        is never below RMAX_MIN_MM, whatever its cut says."""
        keep = np.ones(self.weights.size, dtype=bool)
        for name in GEOMETRY_VARIABLES:
            if name != skip:
                low, high = cuts[name]
                if name == "rmax":
                    low = max(low, RMAX_MIN_MM)
                values = self.column(name)
                keep &= (values >= low) & (values < high)
        return keep

    def efficiency(self, cuts: dict[str, tuple[float, float]]) -> float:
        return float(self.kept[self.passing(cuts)].sum() / self.norm)


def cosmic_p_bins(axis: Spectrum, shapes: CosmicShapes) -> tuple[int, int]:
    """The first and last bins of `axis` wholly inside the shapes' momentum
    range: where the loose cosmic rate and shapes are defined, so where a box's
    momentum window may lie."""
    edges = axis.edges()
    tol = 1.0e-6 * axis.width
    low = int(np.searchsorted(edges, shapes.p_edges[0] - tol))
    high = int(np.searchsorted(edges, shapes.p_edges[-1] + tol, side="right")) - 2
    if not low <= high:
        raise SensitivityError(
            f"the cosmic shapes' momentum range {shapes.p_edges[0]:g}-"
            f"{shapes.p_edges[-1]:g} MeV/c holds no whole signal bin")
    return low, high


def cosmic_geometry_sample(shapes: CosmicShapes, axis: Spectrum,
                           p_bins: tuple[int, int], n: int = COSMIC_SAMPLE_SIZE,
                           seed: int = COSMIC_SAMPLE_SEED,
                           field_scale: float = 1.0) -> GeometryPopulation:
    """A fixed-seed sample of cosmic tracks.

    (rc, cos) is drawn from `rc_vs_cos` (a bin by its content, uniform inside
    it) and p uniformly over `axis`'s bins p_bins[0]..p_bins[1] (cosmics are
    flat in momentum), taking the two as independent; then R = p sin(theta)/(c
    B), rmax = rc + R, d0 = rc - R. Each track keeps its momentum bin, so the
    geometry efficiency is counted in the box's own momentum window.

    `field_scale` s rescales the field the shapes were taken at: p and cos are
    unchanged and R becomes R/s, with the helix pinned at its R(max) point
    (the shapes' R(max) ends sharply at the tracker's outer edge, an
    acceptance edge that does not move with the field; pinning the center
    instead would push a quarter of the tracks past it at s = 0.9). The new
    center is |R(max) - R/s| from the axis, so a circle that grows past the
    axis is still handled; s = 1 is exactly the unscaled sample.
    """
    rng = np.random.default_rng(seed)

    def draw(counts: np.ndarray, *edges: np.ndarray) -> list[np.ndarray]:
        flat = rng.choice(counts.size, size=n, p=counts.ravel() / counts.sum())
        out = []
        for axis, index in zip(edges, np.unravel_index(flat, counts.shape)):
            low, width = axis[index], np.diff(axis)[index]
            out.append(low + rng.random(n) * width)
        return out

    rc, cos = draw(shapes.rc_vs_cos, shapes.rc_edges, shapes.cos_edges)
    bins = rng.integers(p_bins[0], p_bins[1] + 1, size=n)
    p = axis.edges()[bins] + rng.random(n) * axis.width
    nominal = p * np.sqrt(1.0 - cos ** 2) / (C_LIGHT_MEV_PER_MM_T * shapes.b_eff_t)
    radius = nominal / field_scale
    # rmax - R/s, written so that s = 1 gives rc bit for bit
    center = np.abs(rc + (nominal - radius))
    return GeometryPopulation(rmax=center + radius, d0=center - radius, cos=cos,
                              weights=np.ones(n), norm=float(n), bins=bins)


def scan_bins(signal: np.ndarray, dio: np.ndarray, cosmic: np.ndarray,
              first: int = 0) -> dict[str, float] | None:
    """The bin window [a, b] (inclusive, a >= first) maximizing the Punzi
    FoM of S against DIO + cosmic, or None if no window has signal.

    Sums run outward from each lower edge, as in scan_signal_box (never
    differences of prefix sums).
    """
    best = None
    for a in range(first, signal.size):
        s = np.cumsum(signal[a:])
        d = np.cumsum(dio[a:])
        c = np.cumsum(cosmic[a:])
        background = d + c
        usable = np.flatnonzero(s > 0.0)
        if usable.size == 0:
            continue
        ratio = punzi_fom(s[usable], background[usable])
        k = int(np.argmax(ratio))
        if best is None or ratio[k] > best["fom"]:
            j = int(usable[k])
            best = {"a": a, "b": a + j, "fom": float(ratio[k]),
                    "signal": float(s[j]), "dio": float(d[j]),
                    "cosmic": float(c[j])}
    return best


def grid_edges(grid: tuple[float, float, float]) -> np.ndarray:
    low, high, step = grid
    return low + step * np.arange(int(round((high - low) / step)) + 1)


def time_edges(time_min_ns: float, time_max_ns: float) -> np.ndarray:
    """5 ns steps inside the base time window, ending exactly on its top."""
    n = int(np.ceil((time_max_ns - time_min_ns) / TIME_STEP_NS))
    return np.append(time_min_ns + TIME_STEP_NS * np.arange(n), time_max_ns)


def signal_bins(energy: np.ndarray) -> np.ndarray:
    """Each value's bin in the rebinned signal histogram (SIGNAL_VAR binned per
    BINNING, then merged by SIGNAL_REBIN), or -1 where it has none: NaN, off
    the axis, or in the bins `Spectrum.rebin` drops. Fine bins follow
    `np.histogram` (edges from linspace, the top edge in the last bin), so a
    value lands where `Spectrum.from_values` put it."""
    nbins, xmin, xmax = BINNING[SIGNAL_VAR]
    energy = np.asarray(energy, dtype=np.float64)
    edges = np.linspace(xmin, xmax, nbins + 1)
    fine = np.searchsorted(edges, energy, side="right") - 1
    fine[energy == xmax] = nbins - 1
    index = fine // SIGNAL_REBIN
    inside = (np.isfinite(energy) & (energy >= xmin) & (energy <= xmax)
              & (index < nbins // SIGNAL_REBIN))
    return np.where(inside, index, -1)


def smearing_matrix(template: Spectrum, kernel: Kernel) -> np.ndarray:
    """M[k, j]: the smeared content of bin k from a unit entry in bin j, with
    the very `Spectrum.smear` the signal goes through, so that M @ h is the
    smeared spectrum of any histogram h on the template's grid."""
    n = template.nbins
    matrix = np.empty((n, n))
    unit = np.zeros(n)
    for j in range(n):
        unit[j] = 1.0
        matrix[:, j] = replace(template, values=unit).smear(kernel).values
        unit[j] = 0.0
    return matrix


@dataclass
class BoxModel:
    """Everything the box optimization counts with.

    The DIO and cosmic momentum spectra are rates over the whole microbunch
    and acceptance, with no time or geometry cut; a box multiplies them by its
    efficiencies (see `rates`). The cosmic geometry efficiency is that of the
    sampled tracks in the momentum window (`cosmic_event`), since a track's
    R(max) and d0 follow from its momentum; the window stays inside
    `cosmic_p_bins`, the bins the sample covers.

    The CE signal is not a spectrum times an efficiency. Geometry cuts are
    correlated with the energy loss, so each CE event in `ce_geometry` carries
    its signal bin, and the box's signal is the smeared spectrum of the events
    that pass: `signal_scale` x the sum over them of weight x W(bin), where
    W(j) = sum of `response[k, j]` over the momentum window. With the geometry
    open this is `signal`'s window sum, as the same events fill both.
    """

    signal: Spectrum
    dio: Spectrum
    cosmic: Spectrum
    ce_geometry: GeometryPopulation
    cosmic_geometry: GeometryPopulation
    response: np.ndarray        # smearing_matrix of the signal's resolution
    signal_scale: float         # npot * SIGNAL_BR * sig_eff / signal entries
    lifetime_ns: float
    time_min_ns: float
    time_max_ns: float
    cosmic_t_range: tuple[float, float]     # where cosmics are flat in time
    first_bin: int              # lowest allowed momentum bin (SIGNAL_BOX_MIN_MEV)
    cosmic_p_bins: tuple[int, int]  # the bins the cosmic sample covers uniformly
    ce_time_floor_ns: float = 0.0   # front-time cut the CE events already passed

    def ce_time(self, t1, t2):
        """CE fraction in [t1, t2) ns among the CEs the front-time cut kept:
        the exponential's share of [t1, t2) above the floor over its share above
        the floor, since the events' own times already set the fraction below
        it. Works on arrays; with no floor it is `ce_time_efficiency`."""
        floor = self.ce_time_floor_ns
        return (ce_time_efficiency(np.maximum(t1, floor), np.maximum(t2, floor),
                                   self.lifetime_ns)
                / ce_time_efficiency(floor, MICROBUNCH_NS, self.lifetime_ns))

    def event_signal(self, bins: tuple[int, int]) -> np.ndarray:
        """Each CE event's signal in the momentum window, weight and
        normalization included, before any cut."""
        a, b = bins
        pop = self.ce_geometry
        window = self.response[a:b + 1].sum(axis=0)
        return np.where(pop.bins >= 0, window[pop.bins], 0.0) \
            * pop.kept * self.signal_scale

    def cosmic_event(self, bins: tuple[int, int]) -> np.ndarray:
        """Each cosmic track's count in the momentum window, before any time or
        geometry cut: its bin's cosmic rate times its share of the sample's
        weight in that bin, 0 outside the window. Summed over all tracks it is
        the window's cosmic count exactly."""
        a, b = bins
        pop = self.cosmic_geometry
        inside = (pop.bins >= a) & (pop.bins <= b)
        index = np.clip(pop.bins, 0, self.cosmic.nbins - 1)
        per_bin = np.bincount(index, weights=pop.weights,
                              minlength=self.cosmic.nbins)
        return np.where(inside, self.cosmic.values[index] / per_bin[index], 0.0) \
            * pop.kept

    def cosmic_spectrum(self, keep: np.ndarray) -> np.ndarray:
        """The cosmic momentum spectrum of the sampled tracks in `keep`."""
        pop = self.cosmic_geometry
        return np.bincount(pop.bins[keep],
                           weights=self.cosmic_event(self.cosmic_p_bins)[keep],
                           minlength=self.cosmic.nbins)

    def signal_spectrum(self, keep: np.ndarray) -> np.ndarray:
        """The smeared signal spectrum of the CE events in `keep`."""
        pop = self.ce_geometry
        used = keep & (pop.bins >= 0)
        hist = np.bincount(pop.bins[used], weights=pop.kept[used],
                           minlength=self.signal.nbins)
        return self.response @ hist * self.signal_scale

    def window_sums(self, bins: tuple[int, int]) -> tuple[float, float, float]:
        """(signal, DIO, cosmic) summed over momentum bins a..b, directly."""
        a, b = bins
        return (float(self.signal.values[a:b + 1].sum()),
                float(self.dio.values[a:b + 1].sum()),
                float(self.cosmic.values[a:b + 1].sum()))

    def rates(self, cuts: dict) -> dict[str, float]:
        """Signal, DIO and cosmic in the box, and the efficiencies behind them."""
        t1, t2 = cuts["t"]
        ce_t = float(self.ce_time(t1, t2))
        cos_t = float(cosmic_time_efficiency(t1, t2, self.cosmic_t_range))
        ce_g = self.ce_geometry.efficiency(cuts)
        s, d, c = self.window_sums(cuts["p"])
        cosmic_window = float(self.cosmic_event(cuts["p"])[
            self.cosmic_geometry.passing(cuts)].sum())
        cos_g = cosmic_window / c if c > 0.0 else 0.0     # in the p window
        ce_eff = ce_t * ce_g
        cos_eff = cos_t * cos_g
        in_box = self.ce_geometry.passing(cuts)
        signal = float(self.event_signal(cuts["p"])[in_box].sum()) * ce_t
        dio = d * ce_eff        # the CE's momentum-integrated efficiency
        cosmic = cos_t * cosmic_window
        background = dio + cosmic
        return {"signal": signal, "dio": dio, "cosmic": cosmic,
                "background": background,
                "sensitivity": (signal / np.sqrt(background)
                                if signal > 0.0 and background > 0.0 else 0.0),
                "fom": float(punzi_fom(signal, background)),
                "ce_time": ce_t, "ce_geometry": ce_g, "ce_efficiency": ce_eff,
                # signal with the geometry cuts / without, in the p window:
                # differs from ce_geometry by the energy-loss correlation
                "ce_window_geometry": signal / (s * ce_t) if s > 0.0 else 0.0,
                "cosmic_time": cos_t, "cosmic_geometry": cos_g,
                "cosmic_efficiency": cos_eff}

    def step(self, cuts: dict, variable: str) -> tuple | None:
        """The best (low, high) for `variable` with the other cuts held, as
        (low, high, fom), or None if no window has signal and background.

        The variable is histogrammed once (for the tracks passing the other
        cuts) and every window then comes from cumulative sums, O(N + G^2).
        """
        t1, t2 = cuts["t"]
        ce_t = float(self.ce_time(t1, t2))
        cos_t = float(cosmic_time_efficiency(t1, t2, self.cosmic_t_range))
        _, d, _ = self.window_sums(cuts["p"])
        ce_g = self.ce_geometry.efficiency(cuts)    # DIO's, as in `rates`
        event = self.event_signal(cuts["p"])
        cosmic_event = self.cosmic_event(cuts["p"])
        if variable == "p":
            # only inside the bins the cosmic sample covers
            low, high = self.cosmic_p_bins
            cosmic = self.cosmic_spectrum(self.cosmic_geometry.passing(cuts))
            found = scan_bins(
                (self.signal_spectrum(self.ce_geometry.passing(cuts))
                 * ce_t)[:high + 1],
                (self.dio.values * ce_t * ce_g)[:high + 1],
                (cosmic * cos_t)[:high + 1], max(self.first_bin, low))
            return (None if found is None else
                    ((found["a"], found["b"]), found["fom"]))
        if variable == "t":
            edges = time_edges(self.time_min_ns, self.time_max_ns)
            ce_bins = self.ce_time(edges[:-1], edges[1:])
            cos_bins = cosmic_time_efficiency(edges[:-1], edges[1:],
                                             self.cosmic_t_range)
            cos_c = float(cosmic_event[self.cosmic_geometry.passing(cuts)].sum())
            ce_s = float(event[self.ce_geometry.passing(cuts)].sum())
            found = scan_bins(ce_s * ce_bins, d * ce_g * ce_bins,
                              cos_c * cos_bins)
            if found is None:
                return None
            return ((float(edges[found["a"]]), float(edges[found["b"] + 1])),
                    found["fom"])
        # A geometry variable. Bin 0 and the last bin are the under/overflow,
        # so a window reaching them is open on that side (R(max) may not start
        # in the underflow: its lower cut is at least RMAX_MIN_MM).
        grid = grid_edges(GEOMETRY_GRIDS[variable])
        edges = np.concatenate(([-np.inf], grid, [np.inf]))
        hists = []
        # The CE is histogrammed twice: by weight (the efficiency DIO takes)
        # and by signal in the window (the signal count itself); the cosmics
        # by their count in the window.
        for pop, per_event in ((self.ce_geometry, None),
                               (self.ce_geometry, event),
                               (self.cosmic_geometry, cosmic_event)):
            keep = pop.passing(cuts, skip=variable)
            values = pop.column(variable)[keep]
            weights = (pop.kept[keep] / pop.norm if per_event is None
                       else per_event[keep])
            finite = np.isfinite(values)
            index = np.searchsorted(grid, values[finite], side="right")
            hists.append(np.bincount(index, weights=weights[finite],
                                     minlength=edges.size - 1))
        ce_eff_hist, ce_signal_hist, cos_hist = hists
        found = scan_bins(ce_t * ce_signal_hist, d * ce_t * ce_eff_hist,
                          cos_t * cos_hist,
                          first=1 if variable == "rmax" else 0)
        if found is None:
            return None
        return ((float(edges[found["a"]]), float(edges[found["b"] + 1])),
                found["fom"])

    def start_cuts(self, p_bins: tuple[int, int]) -> dict:
        """The starting box: the momentum-only window (clipped to the cosmic
        sample's bins), the reference cuts (time clipped to the base window)
        and no d0 cut."""
        low = max(REFERENCE_TIME_NS[0], self.time_min_ns)
        high = min(REFERENCE_TIME_NS[1], self.time_max_ns)
        if not low < high:
            low, high = self.time_min_ns, self.time_max_ns
        p_low = max(p_bins[0], self.cosmic_p_bins[0], self.first_bin)
        p_high = min(p_bins[1], self.cosmic_p_bins[1])
        if not p_low <= p_high:
            p_low = max(self.cosmic_p_bins[0], self.first_bin)
            p_high = self.cosmic_p_bins[1]
        return {"p": (p_low, p_high), "t": (low, high), "rmax": REFERENCE_RMAX_MM,
                "d0": (-np.inf, np.inf), "cos": REFERENCE_COS}

    def optimize(self, p_bins: tuple[int, int]) -> tuple[dict, dict, list[dict], dict]:
        """Coordinate ascent on the Punzi FoM from `start_cuts`, or from open
        geometry when no CE passes the reference geometry cuts (a step moves
        one variable, so it could not leave a start with no signal).

        Each step re-optimizes one variable's (low, high) jointly with the
        others held, and is kept only if it does not lower the FoM (the
        start's reference cuts are not on the grids). Sweeps stop when one
        gains less than SWEEP_TOLERANCE relative, or at MAX_SWEEPS.
        Returns (cuts, rates at the cuts, one record per sweep, the start).
        """
        cuts = self.start_cuts(p_bins)
        fom = self.rates(cuts)["fom"]
        if not fom > 0.0:
            cuts.update({name: (-np.inf, np.inf) for name in GEOMETRY_VARIABLES})
            fom = self.rates(cuts)["fom"]
        start = {"cuts": dict(cuts), "fom": fom}
        sweeps = []
        for number in range(1, MAX_SWEEPS + 1):
            before = fom
            for variable in BOX_VARIABLES:
                found = self.step(cuts, variable)
                if found is None or not found[1] > fom:
                    continue
                cuts[variable], fom = found
            sweeps.append({"sweep": number, "fom": fom,
                           "cuts": dict(cuts)})
            gain = fom - before
            if (gain / before if before > 0.0 else float(gain > 0.0)) < SWEEP_TOLERANCE:
                break
        if not fom > 0.0:
            raise SensitivityError(
                "no box had signal > 0: no selected CE "
                "event reaching the tracker front passes any geometry window, "
                "or the input histograms are empty above "
                f"{SIGNAL_BOX_MIN_MEV:g} MeV")
        return cuts, self.rates(cuts), sweeps, start


# --- plots -------------------------------------------------------------------

def _write_plots(outdir: Path, signal: Spectrum, dio_reco: Spectrum,
                 cosmic: Spectrum, response: Spectrum, resolution: Spectrum,
                 dio_true: Spectrum, extras: dict[str, Spectrum],
                 best: dict[str, float], mpv: float, fwhm: float) -> list[str]:
    """Reproduce the macro's figures. Returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def step(ax, spectrum: Spectrum, **kwargs):
        ax.stairs(spectrum.values, spectrum.edges(), **kwargs)

    # signal vs background — the one that matters
    fig, ax = plt.subplots(figsize=(8, 6))
    step(ax, signal, color="tab:blue", lw=1.8, label="Signal (CE)")
    step(ax, dio_reco, color="tab:red", lw=1.8, label="DIO")
    step(ax, cosmic, color="tab:green", lw=1.8, label="Cosmic")
    ax.axvspan(best["low_mev"], best["high_mev"], color="0.85", zorder=0,
               label=f"box [{best['low_mev']:.1f}, {best['high_mev']:.1f}]")
    ax.set_yscale("log")
    ax.set_xlim(min(95.0, mpv - 1.5 * fwhm), max(105.0, mpv + 1.5 * fwhm))
    ax.set_ylim(1e-3, 1e4)
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel(f"Rate / {signal.width:.1f} MeV")
    ax.set_title(f"Signal vs. background — S/sqrt(B) = {best['sensitivity']:.3g}")
    ax.legend(ncols=2, fontsize="small")
    fig.tight_layout()
    path = figdir / "sig_vs_bkg.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # DIO before/after smearing
    fig, ax = plt.subplots(figsize=(8, 6))
    step(ax, dio_true, color="tab:blue", lw=1.5, label="DIO (true)")
    step(ax, dio_reco, color="tab:red", lw=1.5, label="DIO (smeared)")
    ax.set_yscale("log")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Rate")
    ax.set_title("DIO spectrum")
    ax.set_ylim(1.e-10, 1.e20)
    ax.legend(fontsize="small")
    fig.tight_layout()
    path = figdir / "dio.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    # inputs: measured energy-loss response and the assumed resolution
    for hist, filename, title, xlabel in (
        (response, "response.png", "Energy loss before the tracker",
         "Energy difference (MeV)"),
        (resolution, "res.png",
         f"Tracker resolution (sigma = {TRK_RESOLUTION_SIGMA_MEV:g} MeV)",
         "Energy offset (MeV)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 6))
        step(ax, hist, color="tab:blue", lw=1.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        fig.tight_layout()
        path = figdir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    # optional primary-vertex distributions, as the macro drew them
    for hist_name, hist in extras.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        step(ax, hist, color="tab:blue", lw=1.5)
        ax.set_xlabel(hist_name)
        ax.set_title(hist.title or hist_name)
        fig.tight_layout()
        path = figdir / f"ce_{hist_name.replace('primary_start_', '')}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    return written


def box_panels(model: BoxModel, cuts: dict) -> list[dict]:
    """The cosmic and CE distributions of each box variable, for the box-mode
    plots.

    Counts are expected events in the box's normalization (the box rate
    formula with the cuts left out). "nominal" has only the base selection:
    the time window and PLOT_P_RANGE_MEV (inside the cosmic sample's bins), no
    box cuts. "after" has every
    optimized cut but the variable's own. Both include the TSDA veto, if any.
    The CE counts ("ce_nominal", "ce_after") are the correlated per-event signal in the same windows; in
    momentum it is the smeared spectrum of the passing events, in time the
    lifetime model binned on the time axis. Each panel: name, bin edges, the
    count arrays and the selected (low, high).
    """
    pop = model.cosmic_geometry
    centers = model.cosmic.centers()
    bins = np.arange(centers.size)
    in_range = np.flatnonzero((centers >= PLOT_P_RANGE_MEV[0])
                              & (centers <= PLOT_P_RANGE_MEV[1])
                              & (bins >= model.cosmic_p_bins[0])
                              & (bins <= model.cosmic_p_bins[1]))
    p_range = (int(in_range[0]), int(in_range[-1]))
    cosmic_event_nominal = model.cosmic_event(p_range)
    cosmic_event_cut = model.cosmic_event(cuts["p"])
    t1, t2 = cuts["t"]
    time_nominal = float(cosmic_time_efficiency(
        model.time_min_ns, model.time_max_ns, model.cosmic_t_range))
    time_cut = float(cosmic_time_efficiency(t1, t2, model.cosmic_t_range))
    cosmic_in_cut = pop.passing(cuts)
    open_geometry = {name: (-np.inf, np.inf) for name in GEOMETRY_VARIABLES}
    no_geometry = {**cuts, **open_geometry}
    ce = model.ce_geometry
    ce_time_nominal = float(model.ce_time(model.time_min_ns, model.time_max_ns))
    ce_time_cut = float(model.ce_time(t1, t2))
    ce_event_nominal = model.event_signal(p_range)
    ce_event_cut = model.event_signal(cuts["p"])
    ce_in_nominal = ce.passing(no_geometry)
    ce_in_cut = ce.passing(cuts)

    panels = []
    # momentum: flat per bin before the geometry cuts (times the veto's
    # survival in the bin), the passing sample after
    p_edges = model.cosmic.edges()[p_range[0]:p_range[1] + 2]
    p_values = model.cosmic_spectrum(
        np.ones(pop.weights.size, dtype=bool))[p_range[0]:p_range[1] + 1]
    centers_low = centers[cuts["p"][0]]
    centers_high = centers[cuts["p"][1]]
    panels.append({
        "name": "p", "edges": p_edges,
        "nominal": p_values * time_nominal,
        "after": time_cut * model.cosmic_spectrum(
            cosmic_in_cut)[p_range[0]:p_range[1] + 1],
        "ce_nominal": ce_time_nominal * model.signal_spectrum(
            ce_in_nominal)[p_range[0]:p_range[1] + 1],
        "ce_after": ce_time_cut * model.signal_spectrum(
            ce_in_cut)[p_range[0]:p_range[1] + 1],
        "band": (float(centers_low), float(centers_high))})
    # time: flat per bin over the base window
    t_edges = time_edges(model.time_min_ns, model.time_max_ns)
    widths = cosmic_time_efficiency(t_edges[:-1], t_edges[1:],
                                    model.cosmic_t_range)
    ce_widths = model.ce_time(t_edges[:-1], t_edges[1:])
    panels.append({
        "name": "t", "edges": t_edges,
        "nominal": cosmic_event_nominal.sum() * widths,
        "after": cosmic_event_cut[cosmic_in_cut].sum() * widths,
        "ce_nominal": ce_event_nominal[ce_in_nominal].sum() * ce_widths,
        "ce_after": ce_event_cut[ce_in_cut].sum() * ce_widths,
        "band": (float(t1), float(t2))})
    for name in GEOMETRY_VARIABLES:
        grid = grid_edges(GEOMETRY_GRIDS[name])
        values = pop.column(name)
        counts, ce_counts = [], []
        for time, cosmic_event, geometry, event, ce_time in (
                (time_nominal, cosmic_event_nominal, no_geometry,
                 ce_event_nominal, ce_time_nominal),
                (time_cut, cosmic_event_cut, cuts, ce_event_cut, ce_time_cut)):
            keep = pop.passing(geometry, skip=name)
            hist, _ = np.histogram(values[keep], bins=grid,
                                   weights=cosmic_event[keep])
            counts.append(hist * time)
            keep = ce.passing(geometry, skip=name)
            hist, _ = np.histogram(ce.column(name)[keep], bins=grid,
                                   weights=event[keep])
            ce_counts.append(hist * ce_time)
        low, high = cuts[name]
        panels.append({"name": name, "edges": grid, "nominal": counts[0],
                       "after": counts[1],
                       "ce_nominal": ce_counts[0], "ce_after": ce_counts[1],
                       "band": (float(max(low, grid[0])), float(min(high, grid[-1])))})
    return panels


BOX_PLOT_LABELS = {
    "p": "Momentum (MeV/c)", "t": "Time (ns)", "rmax": "R(max) (mm)",
    "d0": "d0 (mm)", "cos": "cos(theta)"}


def _write_box_plots(outdir: Path, panels: list[dict]) -> list[str]:
    """One PNG per box variable: cosmics (left axis) and the CE signal (right
    axis) before and after the other cuts, with the selected range shaded.
    Returns the paths written."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for panel in panels:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.stairs(panel["nominal"], panel["edges"], color="tab:gray", lw=1.5,
                  label="nominal (base selection)")
        ax.stairs(panel["after"], panel["edges"], color="tab:green", lw=1.8,
                  label="after the other cuts")
        low, high = panel["band"]
        ax.axvspan(low, high, color="0.85", zorder=0,
                   label=f"selected [{low:.4g}, {high:.4g}]")
        right = ax.twinx()
        right.stairs(panel["ce_nominal"], panel["edges"], color="tab:blue",
                     lw=1.5, ls=":", label="CE nominal (right axis)")
        right.stairs(panel["ce_after"], panel["edges"], color="tab:blue",
                     lw=1.8, ls="--", label="CE after the other cuts (right axis)")
        right.set_ylabel("Expected CE counts / bin", color="tab:blue")
        ax.set_xlabel(BOX_PLOT_LABELS[panel["name"]])
        ax.set_ylabel("Expected cosmic counts / bin")
        ax.set_title(f"Cosmics and CE signal vs {panel['name']}")
        handles = [*ax.get_legend_handles_labels()[0],
                   *right.get_legend_handles_labels()[0]]
        labels = [*ax.get_legend_handles_labels()[1],
                  *right.get_legend_handles_labels()[1]]
        ax.legend(handles, labels, fontsize="small")
        fig.tight_layout()
        path = figdir / f"box_{panel['name']}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))
    return written


# --- the runner --------------------------------------------------------------

def selected_hist(variables: dict[str, np.ndarray], mask: np.ndarray,
                  name: str) -> Spectrum:
    """`name` for the selected events where it is defined, weighted, binned
    as the EdepAna histogram it replaces."""
    values = variables[name]
    keep = mask & np.isfinite(values)
    nbins, xmin, xmax = BINNING[name]
    return Spectrum.from_values(values[keep], nbins, xmin, xmax,
                                weights=variables["weight"][keep], name=name)


def signal_efficiency(variables: dict[str, np.ndarray], mask: np.ndarray,
                      n_gen_events: float,
                      stops_per_pot: float) -> tuple[float, float]:
    """(sig_eff, CE acceptance) for a stopping rate: the acceptance is the
    file's events passing the selection (`mask`, weighted) per generated CE
    event, so the selection sets the signal's size as well as its shape."""
    if variables["weight"].size == 0:
        raise SensitivityError(
            "the file holds no events, so there is no CE acceptance to work "
            "out; run this on a CE (signal) sample"
        )
    if not n_gen_events > 0.0:
        raise SensitivityError(
            "the file records no generated events (ngen is 0), so the CE "
            "acceptance behind stops_per_pot cannot be worked out; pass "
            "sig_eff instead, or rerun 'edep' on files whose SubRuns carry a "
            "GenEventCount"
        )
    selected = float(variables["weight"][mask].sum())
    if not selected > 0.0:
        raise SensitivityError(
            "no event in the file passes the selection, so the CE acceptance "
            "is 0; loosen the selection"
        )
    acceptance = selected / n_gen_events
    if acceptance > 1.0:
        raise SensitivityError(
            f"the CE acceptance comes out at {acceptance:g} (events per "
            "generated event) — above 1, so the file's generated-event count "
            "(ngen) does not cover its events"
        )
    sig_eff = stops_per_pot * acceptance
    if not 0.0 < sig_eff <= 1.0:
        raise SensitivityError(
            f"stops_per_pot {stops_per_pot:g} times the CE acceptance "
            f"{acceptance:g} gives sig_eff {sig_eff:g}, outside (0, 1]"
        )
    return sig_eff, acceptance


BOX_CUT_METRICS = ("time_low_ns", "time_high_ns", "rmax_low_mm", "rmax_high_mm",
                   "d0_low_mm", "d0_high_mm", "cos_low", "cos_high",
                   "ce_box_efficiency", "cosmic_box_efficiency",
                   "ce_tsda_survival", "cosmic_tsda_survival",
                   "ce_front_time_survival")
NO_BOX_METRICS = {name: -1.0 for name in BOX_CUT_METRICS}


def optimize_box(variables: dict[str, np.ndarray], mask: np.ndarray,
                 signal: Spectrum, dio: Spectrum, cosmic: Spectrum,
                 momentum_best: dict[str, float],
                 params: dict[str, float | str], kernel: Kernel,
                 signal_scale: float, front_survival: float = 1.0) -> dict:
    """Optimize the box from the momentum-only result and the selected events'
    helix geometry; `kernel` and `signal_scale` are what turned the signal
    histogram into `signal` (smearing, normalization per entry), and
    `front_survival` is the share of the selection the front-time cut
    (params["front_time_min_ns"], already in `mask`) kept. Returns the headline numbers, metrics, log lines, plot
    panels and sweep records `run` reports."""
    front = mask & variables["has_trk_front"]
    weights = variables["weight"][front]
    if not weights.sum() > 0.0:
        raise SensitivityError(
            "no selected event reached the tracker front, so there is no CE "
            "geometry to cut on; run this on a CE (signal) sample, or loosen "
            "the selection")
    ce_geometry = GeometryPopulation(
        weights=weights, norm=float(weights.sum()),
        bins=signal_bins(variables[SIGNAL_VAR][front]),
        **{name: variables[column][front] for name, column in HELIX_COLUMNS.items()})
    defined = np.isfinite(ce_geometry.rmax) & np.isfinite(ce_geometry.d0) \
        & np.isfinite(ce_geometry.cos)
    if not defined.any():
        raise SensitivityError(
            "no selected event at the tracker front has a defined helix "
            "(EdepAna writes -1 for a neutral primary or zero field), so "
            "there is no CE geometry to cut on")

    shapes = load_cosmic_shapes()
    p_bins_cosmic = cosmic_p_bins(signal, shapes)
    field_scale = float(params["cosmic_field_scale"])
    cosmic_geometry = cosmic_geometry_sample(shapes, signal, p_bins_cosmic,
                                             field_scale=field_scale)
    # The field the CE helices imply (p ~ E for an electron), to check against
    # the field the cosmics are scaled to: the CE sample must be simulated at it.
    with np.errstate(divide="ignore", invalid="ignore"):
        ce_field = (variables[SIGNAL_VAR][front]
                    * np.sqrt(1.0 - ce_geometry.cos ** 2)
                    / (C_LIGHT_MEV_PER_MM_T * (ce_geometry.rmax - ce_geometry.d0) / 2.0))
    ce_field = ce_field[np.isfinite(ce_field) & (ce_field > 0.0)]
    ce_field_t = float(np.median(ce_field)) if ce_field.size else float("nan")
    tsda = bool(params["tsda_veto"])
    if tsda:
        ce_geometry = replace(ce_geometry, survival=1.0 - tsda_hit_fraction(
            ce_geometry.rmax, ce_geometry.d0, ce_geometry.cos))
        cosmic_geometry = replace(cosmic_geometry, survival=1.0 - tsda_hit_fraction(
            cosmic_geometry.rmax, cosmic_geometry.d0, cosmic_geometry.cos))
    model = BoxModel(
        signal=signal, dio=dio, cosmic=cosmic, ce_geometry=ce_geometry,
        cosmic_geometry=cosmic_geometry,
        response=smearing_matrix(signal, kernel), signal_scale=signal_scale,
        lifetime_ns=params["muon_lifetime_ns"],
        time_min_ns=params["time_min_ns"], time_max_ns=params["time_max_ns"],
        cosmic_t_range=shapes.t_range,
        first_bin=int(np.searchsorted(signal.centers(), SIGNAL_BOX_MIN_MEV)),
        cosmic_p_bins=p_bins_cosmic,
        ce_time_floor_ns=params["front_time_min_ns"])
    centers = signal.centers()
    p_bins = (int(np.argmin(np.abs(centers - momentum_best["low_mev"]))),
              int(np.argmin(np.abs(centers - momentum_best["high_mev"]))))
    cuts, rates, sweeps, start = model.optimize(p_bins)

    def reported(name: str) -> tuple[float, float]:
        """A cut as reported: an end beyond its grid (open) sits on the grid's end."""
        low, high = cuts[name]
        if name == "t":
            return float(low), float(high)
        grid = grid_edges(GEOMETRY_GRIDS[name])
        return float(max(low, grid[0])), float(min(high, grid[-1]))

    p_low, p_high = float(centers[cuts["p"][0]]), float(centers[cuts["p"][1]])
    headline = {"sensitivity": rates["sensitivity"], "fom": rates["fom"],
                "low_mev": p_low,
                "high_mev": p_high, "signal": rates["signal"],
                "dio": rates["dio"], "cosmic": rates["cosmic"],
                "background": rates["background"]}
    metrics = {}
    for name, prefix in (("t", "time_%s_ns"), ("rmax", "rmax_%s_mm"),
                         ("d0", "d0_%s_mm"), ("cos", "cos_%s")):
        metrics[prefix % "low"], metrics[prefix % "high"] = reported(name)
    # time x momentum-integrated geometry, as before the correlation was kept
    metrics["ce_box_efficiency"] = rates["ce_efficiency"]
    metrics["cosmic_box_efficiency"] = rates["cosmic_efficiency"]
    # before any box cut; -1 without the veto
    ce_tsda = float(ce_geometry.kept.sum() / ce_geometry.norm) if tsda else -1.0
    cosmic_tsda = float(cosmic_geometry.kept.sum() / cosmic_geometry.norm) if tsda else -1.0
    metrics["ce_tsda_survival"] = ce_tsda
    metrics["cosmic_tsda_survival"] = cosmic_tsda
    metrics["ce_front_time_survival"] = front_survival

    def span(name: str, unit: str) -> str:
        low, high = reported(name)
        return f"[{low:.6g}, {high:.6g}] {unit}"

    log = [
        f"Box cuts (cut_mode = box), coordinate ascent on the Punzi FoM "
        f"S/({PUNZI_A:g}/2 + sqrt(B)):",
        f"  momentum   [{p_low:.1f}, {p_high:.1f}] MeV",
        f"  time       {span('t', 'ns')}  (base window "
        f"[{params['time_min_ns']:g}, {params['time_max_ns']:g}], "
        f"muon lifetime {params['muon_lifetime_ns']:g} ns)",
        f"  R(max)     {span('rmax', 'mm')}",
        f"  d0         {span('d0', 'mm')}",
        f"  cos(theta) {span('cos', '')}",
        "  (a geometry cut at the end of its grid is open on that side)",
        f"  CE efficiency     time {rates['ce_time']:.4g} x geometry "
        f"{rates['ce_geometry']:.4g} = {rates['ce_efficiency']:.4g}"
        " (geometry integrated over momentum; DIO takes this)",
        f"  CE geometry efficiency in the momentum window (signal with the "
        f"geometry cuts / without): {rates['ce_window_geometry']:.4g}, "
        f"vs {rates['ce_geometry']:.4g} integrated over momentum "
        "(the difference is the energy-loss correlation, which the signal keeps)",
        f"  cosmic efficiency time {rates['cosmic_time']:.4g} (flat over "
        f"{shapes.t_range[0]:g}-{shapes.t_range[1]:g} ns) x geometry "
        f"{rates['cosmic_geometry']:.4g} = {rates['cosmic_efficiency']:.4g}"
        " (geometry in the momentum window)",
        "  cosmic_rate_per_s_per_mev is the rate for the loose selection of the "
        "cosmic shapes (e-, 100 < p < 110 MeV/c, no t0, cos, R(max) or d0 cut)",
        f"  loose/tight Run-1A yield ratio {shapes.loose_to_tight_ratio:.4g} "
        f"(tight: {REFERENCE_TIME_NS[0]:g} < t < {REFERENCE_TIME_NS[1]:g} ns, "
        f"{REFERENCE_COS[0]:g} < cos < {REFERENCE_COS[1]:g}, "
        f"{REFERENCE_RMAX_MM[0]:g} < R(max) < {REFERENCE_RMAX_MM[1]:g} mm); "
        "applied to the default rate only",
        (f"  TSDA veto (fast sim: reaches it if sin^2(theta) x {TSDA_BZ_T:g} T / "
         f"{TRACKER_BZ_T:g} T < 1, hits {TSDA_RADII_MM[0]:g}-{TSDA_RADII_MM[1]:g} mm "
         f"averaged over the helix phase): CE survival {ce_tsda:.4g}, cosmic "
         f"survival {cosmic_tsda:.4g} before the box cuts; DIO takes the CE's"
         if tsda else "  TSDA veto: not applied (tsda_veto = 0)"),
        (f"  cosmic field scale {field_scale:g}: cosmic helix radii x "
         f"1/{field_scale:g} with R(max) pinned (field {shapes.b_eff_t:.4g} T x "
         f"{field_scale:g} = {shapes.b_eff_t * field_scale:.4g} T, reco "
         f"effective); the CE helices imply {ce_field_t:.4g} T (truth at the "
         "tracker front: compare with a nominal-field CE run, not with the "
         "reco value); the loose cosmic rate is not rescaled"),
        (f"  front-time cut: folded tracker-front time >= "
         f"{params['front_time_min_ns']:g} ns kept {front_survival:.4g} of the "
         "selected CEs (in the acceptance and shapes); the box time "
         "efficiency is the exponential's share above it"
         if params["front_time_min_ns"] > 0.0 else
         "  front-time cut: not applied (front_time_min_ns = 0)"),
        f"  S = {rates['signal']:.3g}, DIO = {rates['dio']:.3g}, "
        f"cosmic = {rates['cosmic']:.3g} -> B = {rates['background']:.3g}, "
        f"S/sqrt(B) = {rates['sensitivity']:.4g}, Punzi FoM = {rates['fom']:.4g} "
        f"(start {start['fom']:.4g})",
        "",
        "Punzi FoM after each sweep:",
        *(f"  sweep {record['sweep']:2d}  {record['fom']:.6g}"
          for record in sweeps),
    ]
    return {"headline": headline, "metrics": metrics, "log": log,
            "panels": box_panels(model, cuts),
            "cosmic_loose_to_tight_ratio": shapes.loose_to_tight_ratio,
            "ce_helix_field_t": ce_field_t,
            "start_fom": start["fom"],
            "sweeps": [record["fom"] for record in sweeps]}


def run(context: RunContext) -> RunOutcome:
    """Compute the approximate CE sensitivity for one EdepAna ROOT file."""
    cut_mode = context.params["cut_mode"]
    if cut_mode not in CUT_MODES:
        return RunOutcome(error=(
            f"cut_mode must be one of {', '.join(repr(m) for m in CUT_MODES)}, "
            f"got {cut_mode!r}: 'momentum' scans the momentum window alone, "
            "'box' also optimizes time, R(max), d0 and cos(theta) cuts"))
    time_min_ns = context.params["time_min_ns"]
    time_max_ns = context.params["time_max_ns"]
    if cut_mode == "box" and not time_min_ns < time_max_ns:
        return RunOutcome(error=(
            f"time_min_ns ({time_min_ns:g}) must be below time_max_ns "
            f"({time_max_ns:g}): they are the base time window the box "
            "optimization scans inside"))
    if context.params["tsda_veto"] not in (0, 1):
        return RunOutcome(error=(
            f"tsda_veto must be 0 or 1, got {context.params['tsda_veto']:g}"))
    front_time_min_ns = context.params["front_time_min_ns"]
    if not 0.0 <= front_time_min_ns < MICROBUNCH_NS:
        return RunOutcome(error=(
            f"front_time_min_ns must be in [0, {MICROBUNCH_NS:g}) ns (the "
            f"microbunch), got {front_time_min_ns:g}"))
    field_scale = context.params["cosmic_field_scale"]
    if field_scale != 1.0 and cut_mode != "box":
        return RunOutcome(error=(
            f"cosmic_field_scale = {field_scale:g} needs cut_mode='box': it "
            "reshapes the cosmic helices the box's R(max) and d0 cuts act on, "
            "and momentum mode applies no geometry cut (its cosmic rate is "
            "flat in momentum, which a field change does not alter)"))
    if context.params["tsda_veto"] and cut_mode != "box":
        return RunOutcome(error=(
            "tsda_veto = 1 needs cut_mode='box': the veto is worked out from "
            "the tracker-front helix, which momentum mode does not read"))
    sig_eff = context.params["sig_eff"]
    stops_per_pot = context.params["stops_per_pot"]
    if np.isnan(sig_eff) or np.isnan(stops_per_pot):
        return RunOutcome(error=(
            f"sig_eff ({sig_eff:g}) and stops_per_pot ({stops_per_pot:g}) "
            "must be numbers, not NaN"))
    if (sig_eff > 0.0) == (stops_per_pot > 0.0):
        return RunOutcome(error=(
            "pass exactly one of sig_eff or stops_per_pot (> 0): sig_eff is "
            "the signal efficiency itself, stops_per_pot works it out with "
            "the file's own CE acceptance"))
    ce_acceptance = n_gen_events = run_record = None
    npot = context.params["npot"]
    cosmic_rate_per_s_per_mev = context.params["cosmic_rate_per_s_per_mev"]
    cosmic_rate_source = "user" if cosmic_rate_per_s_per_mev > 0.0 else "mode default"
    mean_pot_per_event = context.params["mean_pot_per_event"]
    selection = context.params["selection"]
    outdir = context.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    try:
        if cosmic_rate_source == "mode default":
            # box mode reads the cosmic shapes, which may be missing
            cosmic_rate_per_s_per_mev = recommended_cosmic_rate(cut_mode)
        variables = read_edep_tree(context.input_path)
        if cut_mode == "box" and HELIX_COLUMNS["rmax"] not in variables:
            raise SensitivityError(
                "cut_mode='box' needs the tracker-front helix branches "
                "(primary_trk_front_r_center, _radius, _cos, _time), which "
                "this file's EdepAna tree lacks: they come from EdepAna in a "
                "an Offline with PR #2092 (not yet in a Musing). Rerun 'edep' "
                "with such a build, or use cut_mode='momentum', which works on "
                "this file")
        mask = select_events(variables, selection)
        front_survival = -1.0     # box mode only: momentum mode has no time model
        if cut_mode == "box":
            selected = float(variables["weight"][mask].sum())
            if front_time_min_ns > 0.0:
                mask = mask & front_time_kept(variables["primary_trk_front_time"],
                                              front_time_min_ns)
            front_survival = (float(variables["weight"][mask].sum()) / selected
                              if selected > 0.0 else 0.0)
        if stops_per_pot > 0.0:
            run_record = read_run_record(context.input_path)
            if run_record is not None and run_record["max_events"] is not None:
                raise SensitivityError(
                    f"this file comes from an 'edep' run with max_events="
                    f"{run_record['max_events']}, which cuts a subrun short "
                    "while ngen keeps its whole generated-event count, so the "
                    "CE acceptance behind stops_per_pot would come out low; "
                    "rerun 'edep' without max_events, or pass sig_eff"
                )
            n_gen_events = generated_events(context.input_path)
            sig_eff, ce_acceptance = signal_efficiency(variables, mask,
                                                       n_gen_events, stops_per_pot)
        n_selected = int(mask.sum())
        signal = selected_hist(variables, mask, SIGNAL_VAR)
        response = selected_hist(variables, mask, RESPONSE_VAR)
        extras = {name: selected_hist(variables, mask, name)
                  for name in EXTRA_PLOT_VARS}

        # Both normalizations divide by the entry count, so empty input is a
        # clean error rather than a division by zero.
        for hist in (signal, response):
            if hist.entries <= 0:
                raise SensitivityError(
                    f"no selected event has '{hist.name}': {n_selected} of "
                    f"{mask.size} events pass '{selection}', and none of them "
                    "reached the tracker front, so there is no signal shape to "
                    "work with. Run this on a CE (signal) sample, or loosen "
                    "the selection."
                )

        # 1-2. signal shape -> rate, then smeared by the tracker resolution
        signal = signal.rebin(SIGNAL_REBIN)
        signal_scale = npot * SIGNAL_BR * sig_eff / signal.entries
        signal = signal.scaled(signal_scale)
        # A density in energy offset. The efficiency is folded in here, so it
        # rides along with the response into the smeared DIO spectrum.
        response = response.scaled(sig_eff / response.entries / response.width)
        signal_kernel = Kernel.gaussian(signal.width, TRK_RESOLUTION_SIGMA_MEV)
        signal_reco = signal.smear(signal_kernel)
        mpv, fwhm = mpv_fwhm(signal_reco)

        # 3. DIO: theory spectrum -> rate, smeared by the energy loss and then
        # by the resolution on its own fine binning, then put on signal's bins.
        dio_true = load_dio_spectrum().scaled((1. - MUON_CAPTURE_RATE) * sig_eff * npot)
        resolution = Kernel.gaussian(dio_true.width, TRK_RESOLUTION_SIGMA_MEV)
        dio_reco = (dio_true
                    .smear(Kernel.from_density(response, dio_true.width))
                    .smear(resolution)
                    .regrid(signal_reco))

        # 4. cosmics: flat rate per MeV/c over the implied on-spill time
        events = npot / mean_pot_per_event
        onspill_seconds = events * ONSPILL_SECONDS_PER_EVENT
        cosmic_rate = cosmic_rate_per_s_per_mev * onspill_seconds
        cosmic = replace(
            signal_reco, name="cosmic", title="Cosmics",
            values=np.full(signal_reco.nbins, cosmic_rate * signal_reco.width),
        )

        # 5. best window
        best, top = scan_signal_box(signal_reco, dio_reco, cosmic)
        # 5b. the window with the lowest expected limit, on the same spectra
        # (momentum only, in either cut mode)
        limit = scan_limit_window(
            signal_reco, dio_reco, cosmic,
            background_uncertainty=context.params["background_uncertainty"])

        # 6. (box mode) the best box in momentum, time, R(max), d0 and cos
        box = None
        if cut_mode == "box":
            box = optimize_box(variables, mask, signal_reco, dio_reco, cosmic,
                               best, context.params, signal_kernel, signal_scale,
                               front_survival)
    except (SensitivityError, EdepTreeError, SelectionError) as exc:
        return RunOutcome(error=str(exc), extra={"selection": selection})

    # What the momentum-only scan found stays in `best`; in box mode the
    # headline metrics describe the optimized box instead.
    headline = best if box is None else box["headline"]
    metrics = {
        "sensitivity": float(headline["sensitivity"]),
        "signal_box_low_mev": float(headline["low_mev"]),
        "signal_box_high_mev": float(headline["high_mev"]),
        "signal_rate": float(headline["signal"]),
        "dio_background": float(headline["dio"]),
        "cosmic_background": float(headline["cosmic"]),
        "total_background": float(headline["background"]),
        "signal_mpv_mev": float(mpv),
        "signal_fwhm_mev": float(fwhm),
        # The assumptions the rates above are built on, reported with them so
        # a number never travels without the normalization behind it.
        "npot": float(npot),
        "cosmic_rate_per_s_per_mev": float(cosmic_rate_per_s_per_mev),
        "momentum_only_sensitivity": float(best["sensitivity"]),
        # The median expected 90% CL limit on R_mue, in its own momentum
        # window (the signal spectrum assumes R_MUE, so the limit is the
        # signal-count limit per signal event times R_MUE).
        "expected_limit": float(limit["limit_per_signal"] * R_MUE),
        "expected_limit_box_low_mev": limit["low_mev"],
        "expected_limit_box_high_mev": limit["high_mev"],
        "expected_limit_signal": limit["signal"],
        "expected_limit_background": limit["background"],
        # The figure of merit the box is optimized on, for the headline
        # window or box in either mode; the expected limit scales ~ 1/punzi_fom.
        "punzi_fom": float(punzi_fom(headline["signal"], headline["background"])),
        # -1 in momentum mode, where no such cut or efficiency exists.
        **(box["metrics"] if box is not None else NO_BOX_METRICS),
    }

    log_path = outdir / "approx_ce_sensitivity.log"
    lines = [
        "approx_ce_sensitivity",
        f"  input            {context.input_path}",
        f"  selection        {selection or '(none)'}: {n_selected} of {mask.size} events",
        f"  sig_eff          {sig_eff:g}" + (
            "" if ce_acceptance is None else
            f"  (stops/POT {stops_per_pot:g} x CE acceptance {ce_acceptance:g}"
            f" = {variables['weight'][mask].sum():g} selected / "
            f"{n_gen_events:g} generated)"),
    ]
    if ce_acceptance is not None and run_record is None:
        lines.append(
            "  WARNING          no edep run record next to the file, so this "
            "cannot confirm it comes from one full 'edep' run (without "
            "max_events, not merged); the CE acceptance is only as good as that")
    lines += [
        f"  NPOT             {npot:g}",
        f"  signal BR        {SIGNAL_BR:.4g} per stop  (R_mue = {R_MUE:g} per capture "
        f"x {MUON_CAPTURE_RATE:g} captures per stop)",
        f"  cosmic rate      {cosmic_rate_per_s_per_mev:.4g} per s per MeV/c "
        f"({cosmic_rate_source}) "
        f"-> {cosmic_rate:.4g} per MeV/c ({onspill_seconds:.4g} s on-spill)",
        f"  signal entries   {signal.entries:g}",
        f"  MPV / FWHM       {mpv:.3f} / {fwhm:.3f} MeV",
        "",
        f"Best window [{best['low_mev']:.1f}, {best['high_mev']:.1f}] MeV: "
        f"S = {best['signal']:.3g}, DIO = {best['dio']:.3g}, "
        f"cosmic = {best['cosmic']:.3g} -> B = {best['background']:.3g}, "
        f"S/sqrt(B) = {best['sensitivity']:.4g}",
        f"Expected 90% CL limit R_mue < {limit['limit_per_signal'] * R_MUE:.4g} in "
        f"[{limit['low_mev']:.1f}, {limit['high_mev']:.1f}] MeV: "
        f"S = {limit['signal']:.3g}, DIO = {limit['dio']:.3g}, "
        f"cosmic = {limit['cosmic']:.3g} -> B = {limit['background']:.3g} "
        f"(relative background uncertainty {context.params['background_uncertainty']:g}; "
        "0 = statistics only)",
        "",
        f"Top {len(top)} windows scanned:",
    ]
    lines += [
        f"  [{e['low_mev']:6.1f}, {e['high_mev']:6.1f}] MeV  S = {e['signal']:9.3g}  "
        f"DIO = {e['dio']:9.3g}  cosmic = {e['cosmic']:9.3g}  "
        f"S/sqrt(B) = {e['sensitivity']:.4g}"
        for e in top
    ]
    if box is not None:
        lines += ["", *box["log"]]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    files = _write_plots(outdir, signal_reco, dio_reco, cosmic, response,
                         resolution.as_spectrum(dio_true.width, "resolution"),
                         dio_true, extras, best, mpv, fwhm)
    if box is not None:
        files += _write_box_plots(outdir, box["panels"])

    return RunOutcome(
        metrics=metrics,
        files=files,
        log_path=log_path,
        extra={
            "sig_eff": sig_eff,
            **({} if ce_acceptance is None else {
                "stops_per_pot": stops_per_pot,
                "ce_acceptance": ce_acceptance,
                "n_gen_events": n_gen_events,
                "edep_run_recorded": run_record is not None}),
            "cut_mode": cut_mode,
            "cosmic_rate_source": cosmic_rate_source,
            **({} if box is None else {
                "cosmic_field_scale": float(field_scale),
                "ce_helix_field_t": box["ce_helix_field_t"],
                "cosmic_loose_to_tight_ratio": box["cosmic_loose_to_tight_ratio"],
                "box_start_fom": box["start_fom"],
                "box_sweeps": box["sweeps"]}),
            "selection": selection,
            "n_events_selected": n_selected,
            "signal_br": SIGNAL_BR,
            "cosmic_rate_per_mev": cosmic_rate,
            "onspill_seconds": onspill_seconds,
            "signal_hist_entries": float(signal.entries),
            "dio_table": str(DIO_TABLE),
        },
    )


def summarize(metrics: dict[str, float]) -> str:
    limit = (
        f" The expected 90% CL limit is R_mue < {metrics['expected_limit']:.3g} "
        f"in [{metrics['expected_limit_box_low_mev']:.1f}, "
        f"{metrics['expected_limit_box_high_mev']:.1f}] MeV/c "
        f"(S = {metrics['expected_limit_signal']:.3g}, "
        f"B = {metrics['expected_limit_background']:.3g}).")
    return _summarize_sensitivity(metrics) + limit


def _summarize_sensitivity(metrics: dict[str, float]) -> str:
    if metrics["ce_box_efficiency"] >= 0.0:     # -1 in momentum mode
        return (
            f"S/sqrt(B) = {metrics['sensitivity']:.4g} in the optimized box "
            f"p [{metrics['signal_box_low_mev']:.1f}, "
            f"{metrics['signal_box_high_mev']:.1f}] MeV/c, "
            f"t [{metrics['time_low_ns']:g}, {metrics['time_high_ns']:g}] ns, "
            f"R(max) [{metrics['rmax_low_mm']:g}, {metrics['rmax_high_mm']:g}] mm, "
            f"d0 [{metrics['d0_low_mm']:g}, {metrics['d0_high_mm']:g}] mm, "
            f"cos [{metrics['cos_low']:g}, {metrics['cos_high']:g}] "
            f"(S = {metrics['signal_rate']:.3g}, B = {metrics['total_background']:.3g}; "
            f"momentum-only S/sqrt(B) = {metrics['momentum_only_sensitivity']:.4g}) "
            f"for {metrics['npot']:.3g} POT and a cosmic rate of "
            f"{metrics['cosmic_rate_per_s_per_mev']:.4g} per s per MeV/c "
            "for the loose cosmic selection.")
    return (
        f"S/sqrt(B) = {metrics['sensitivity']:.4g} in "
        f"[{metrics['signal_box_low_mev']:.1f}, "
        f"{metrics['signal_box_high_mev']:.1f}] MeV "
        f"(S = {metrics['signal_rate']:.3g}, B = {metrics['total_background']:.3g}) "
        f"for {metrics['npot']:.3g} POT and a cosmic rate of "
        f"{metrics['cosmic_rate_per_s_per_mev']:.4g} per s per MeV/c."
    )


SPEC = AnalysisSpec(
    name="approx_ce_sensitivity",
    description=(
        "Approximate Run-1A conversion-electron sensitivity S/sqrt(B) from "
        "the EdepAna tree, with DIO and cosmic backgrounds, for a "
        "configurable event selection."
    ),
    input_kind="root_file",
    produced_by=("edep",),
    metrics=(
        "sensitivity", "signal_box_low_mev", "signal_box_high_mev",
        "signal_rate", "dio_background", "cosmic_background",
        "total_background", "signal_mpv_mev", "signal_fwhm_mev",
        "npot", "cosmic_rate_per_s_per_mev",
        "momentum_only_sensitivity", "punzi_fom", *BOX_CUT_METRICS,
        "expected_limit", "expected_limit_box_low_mev",
        "expected_limit_box_high_mev", "expected_limit_signal",
        "expected_limit_background",
    ),
    units={
        "time_low_ns": "ns", "time_high_ns": "ns",
        "rmax_low_mm": "mm", "rmax_high_mm": "mm",
        "d0_low_mm": "mm", "d0_high_mm": "mm",
        "signal_box_low_mev": "MeV",
        "signal_box_high_mev": "MeV",
        "signal_mpv_mev": "MeV",
        "signal_fwhm_mev": "MeV",
        "npot": "POT",
        "cosmic_rate_per_s_per_mev": "per second per MeV/c",
        "expected_limit": "R_mue (conversions per muon capture), 90% CL",
        "expected_limit_box_low_mev": "MeV",
        "expected_limit_box_high_mev": "MeV",
    },
    parameters=(
        ParamSpec(
            name="sig_eff",
            description="Signal (CE) reconstruction+selection efficiency, 0-1. "
                        "Scales signal and both backgrounds. In cut_mode 'box' "
                        "it must NOT include the efficiency of the time, "
                        "R(max), d0 and cos cuts: the box applies those "
                        "explicitly. Pass this or "
                        "stops_per_pot, not both; 0 means not given.",
            default=0.0, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="stops_per_pot",
            description="Stopped muons per POT, e.g. muon_stop_rate's "
                        "stops_per_pot for the same configuration. Instead of "
                        "sig_eff: sig_eff becomes stops_per_pot times the CE "
                        "acceptance, the file's events passing `selection` "
                        "per generated event (EdepAna's ngen), so the "
                        "selection sets the signal's size too (in cut_mode 'box' "
                        "the selection must not contain the time, R(max), d0 "
                        "or cos cuts, which the box applies). The file must "
                        "come from one full 'edep' run: one 'edep' recorded "
                        "as run with max_events is refused, and a merged one "
                        "is refused when ngen shows it (not every merge "
                        "does). 0 means not given.",
            default=0.0, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="npot",
            description="Protons on target to assume for the rate normalization.",
            default=NPOT, minimum=0.0,
        ),
        ParamSpec(
            name="mean_pot_per_event",
            description="Mean number of protons on target per event.",
            default=MEAN_POT_PER_EVENT, minimum=1.0,
        ),
        ParamSpec(
            name="cosmic_rate_per_s_per_mev",
            description="Cosmic-ray background rate, flat in momentum, per "
                        "second per MeV/c. 0 (the default) means the "
                        "recommended rate for cut_mode: the rough Run-1A "
                        "mu- -> e- number (10/7.8e5) in either mode, which "
                        "the box mode assumes was measured after its "
                        "reference cuts. The metric reports the value used. "
                        "Scaled by the on-spill live time "
                        "that npot and mean npot per event implies. "
                        + "In 'box' mode it is the cosmic-electron rate for the LOOSE selection the cosmic shapes were extracted with (ConvAna set 81: e-, APR/CPR trigger, upstream-reflection and multi-track vetoes, 100 < p < 110 MeV/c; no t0, cos, R(max) or d0 cut); the box's cosmic count is this rate times the box's cosmic time and geometry efficiencies, with no further scaling. In 'momentum' mode it is the rate after whatever fixed selection the result is meant to represent, applied as-is. The default (0) is, in momentum mode, the Run-1A rate 10/7.8e5 and, in box mode, that rate times the loose/tight yield ratio from the shapes file (the tight Run-1A cuts 540 < t0 < 1650 ns, 0.525 < cos < 0.649, 482 < R(max) < 642.6 mm pass 3.2% of the loose set, so ~31x, ~4.0e-4).",
            default=0.0, minimum=0.0,
        ),
        ParamSpec(
            name="background_uncertainty",
            description="Relative uncertainty on the background in a window, "
                        "for expected_limit only (S/sqrt(B) ignores it), e.g. "
                        "0.1 for 10%. The limit's window is "
                        "re-optimized with it, so a larger value favours "
                        "narrower windows with less background. 0 (the "
                        "default) gives the statistics-only limit.",
            default=0.0, minimum=0.0,
        ),
        ParamSpec(
            name="cut_mode",
            description="'momentum' (default): scan the momentum window alone "
                        "(no time or geometry cut is applied). 'box': also "
                        "optimize time, R(max), d0 and cos(theta) cuts by "
                        "coordinate ascent on S/sqrt(B), using the cosmic "
                        "shapes and the CE helix at the tracker front (needs "
                        "an EdepAna tree with the helix branches, from a "
                        "an Offline with PR #2092, not yet in a Musing). In box "
                        "mode cosmic_rate_per_s_per_mev is the cosmic rate "
                        "for the loose selection of the cosmic shapes (set 81: "
                        "e-, 100 < p < 110 MeV/c, no t0, cos, R(max) or d0 "
                        "cut), scaled by the box's time and geometry "
                        "efficiencies; its default then is the Run-1A rate "
                        "times the loose/tight yield ratio (~31x). In "
                        "momentum mode it is applied as-is. DIO is assumed "
                        "to share the CE's time and geometry efficiencies.",
            default="momentum", kind="text",
        ),
        ParamSpec(
            name="muon_lifetime_ns",
            description="Muon lifetime (ns) setting the CE time distribution "
                        "exp(-t/tau) over the microbunch; box mode only. The "
                        "default is the bound-muon lifetime in aluminum.",
            default=MUON_LIFETIME_NS, minimum=1.0,
        ),
        ParamSpec(
            name="time_min_ns",
            description="Lower edge (ns) of the base time cut in box mode; "
                        "the time optimization only tightens it. Ignored in "
                        "momentum mode.",
            default=TIME_MIN_NS, minimum=0.0, maximum=MICROBUNCH_NS,
        ),
        ParamSpec(
            name="time_max_ns",
            description="Upper edge (ns) of the base time cut in box mode "
                        "(the microbunch is 1695 ns long); the time "
                        "optimization only tightens it. Ignored in momentum "
                        "mode.",
            default=TIME_MAX_NS, minimum=0.0, maximum=MICROBUNCH_NS,
        ),
        ParamSpec(
            name="tsda_veto",
            description="1 applies a fast-sim TSDA veto in cut_mode 'box' "
                        "(0, the default, does not): reco vetoes a track whose "
                        "upstream extrapolation reaches the TS downstream "
                        "absorber instead of reflecting in the DS gradient. "
                        "A track reaches it if sin^2(theta) x 2 T / 1 T < 1 "
                        "(rough nominal fields at the TSDA and the tracker "
                        "front); its chance of hitting the 235-525 mm annulus "
                        "is the fraction of its helix turn there, center and "
                        "radius shrunk by sqrt(1 T / 2 T). It weights CE and "
                        "cosmic tracks alike, DIO takes the CE's efficiency, "
                        "and ce_tsda_survival / cosmic_tsda_survival report "
                        "the surviving fractions (-1 when off).",
            default=0, minimum=0, maximum=1,
        ),
        ParamSpec(
            name="cosmic_field_scale",
            description="Box mode: the detector-solenoid field relative to "
                        "the one the cosmic shapes were taken at (1, the "
                        "default, leaves them as they are). Each cosmic "
                        "track keeps its momentum and cos(theta) and its "
                        "helix radius becomes R/scale, pinned at its R(max) "
                        "point, so R(max) is unchanged and d0 = R(max) - "
                        "2R/scale moves (a stronger field raises d0). It "
                        "does NOT touch the CE: pass an EdepAna file "
                        "simulated at the same field (the log prints the "
                        "field the CE helices imply), and it does not rescale "
                        "the loose cosmic rate, nor add tracks that the "
                        "nominal field kept out of the tracker. A first-order "
                        "model, for scans of a few tens of percent.",
            default=1.0, minimum=COSMIC_FIELD_SCALE_RANGE[0],
            maximum=COSMIC_FIELD_SCALE_RANGE[1],
        ),
        ParamSpec(
            name="front_time_min_ns",
            description="Box mode: drop CE events whose time at the tracker "
                        "front (EdepAna primary_trk_front_time, folded into the "
                        "1695 ns microbunch) is below this, as the "
                        "production's digitization window does (default "
                        f"{FRONT_TIME_MIN_NS:g} ns; 0 turns it off). It reduces "
                        "the CE acceptance by the events' own times, which "
                        "shift with the stopping target, and the box's CE "
                        "time efficiency becomes the exponential's share "
                        "above it; ce_front_time_survival reports the kept "
                        "fraction (-1 in momentum mode, where it is ignored).",
            default=FRONT_TIME_MIN_NS, minimum=0.0, maximum=MICROBUNCH_NS,
        ),
        ParamSpec(
            name="selection",
            description=selection_help() + " Selects the events the signal "
                        "shape and energy-loss response are taken from. The "
                        f"default, '{DEFAULT_SELECTION}', is the fixed cut "
                        "the original macro's histograms had.",
            default=DEFAULT_SELECTION, kind="text", allow_empty=True,
        ),
    ),
    input_hint=(
        "An EdepAna nts.*.root file from a CE (signal) sample, holding "
        f"{TREE_PATH} (EdepAna in Offline v13_39_00 or later), with events "
        "passing the selection that reach the tracker front."
    ),
    run=run,
    summarize=summarize,
)
