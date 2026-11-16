"""Generic cut-set optimization: greedy, random grid search, local refinement.

The machinery here knows nothing about physics or file formats — it operates
on in-memory arrays of variable values and per-event signal/background
weights. The analysis module (optimize_selection_ntuple.py) reads EventNtuple
files and builds the ``Events`` and ``CutVar`` objects this code works with.

Algorithms, adapted from optimize_selection_fast.C:

  **Greedy path** — at each step, every variable proposes its best single-edge
  tightening (the one with the highest figure-of-merit, subject to a maximum
  signal-efficiency loss per step). The variable with the steepest gradient
  (FoM gain per signal lost) wins, and that edge is applied. The path records
  every step and stops when signal efficiency drops below ``end_eff`` or
  background reaches zero. An optional ``top_k`` with an RNG makes the
  choice stochastic (pick one of the top-k gradients at random), giving
  different starting points for the local search.

  **Random grid search** — ``n_trials`` random cut-sets are drawn by
  uniformly sampling each variable's edges within its search range, each
  variable applied with 50 % probability. Every point is evaluated and the
  efficiency–rejection pairs are returned.

  **Local refinement** — hill-climbing from a starting selection. Each
  iteration tries three moves: (a) tighten any one variable by one step,
  (b) loosen any one variable by one step, (c) loosen one variable then
  greedily tighten the best other. The move with the largest FoM improvement
  is accepted; the search stops when no move improves (with a 1e-6
  tolerance) or ``max_iter`` is reached.

  **Multi-start optimization** — collects starting points from the best
  greedy-path point, any user-supplied seed selections, and ``n_random``
  randomized greedy runs (each picking from the top-3 candidates per step).
  Every starting point is refined, and the best result is returned.

Figures of merit:

  * S / sqrt(B) — classic significance
  * Asimov Z — sqrt(2 * ((S+B) * log(1 + S/B) - S))
  * Punzi FoM — S / (a/2 + sqrt(B)), with a = 1.28 (90 % CL)
  * Expected upper limit — S / N_UL(B), where N_UL(B) is the median
    expected 90 % CL CLs upper limit (Poisson)
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# ---------------------------------------------------------------------------
# Cut variable
# ---------------------------------------------------------------------------

# Cut types: a window, a one-sided cut, or a veto (exclusion) window.
BOTH = "both"      # two-sided: keep events with lo < x < hi
LEFT = "left"      # lower bound only: keep events with x > lo
RIGHT = "right"    # upper bound only: keep events with x < hi
VETO = "veto"      # exclusion window: keep events with x < lo or x > hi
CUT_TYPES = (BOTH, LEFT, RIGHT, VETO)


@dataclass
class CutVar:
    """One cut variable with its current edges and search bounds.

    Attributes:
        name: Variable name (matches the flat variable names in the event
            arrays, e.g. ``p_front``, ``trkqual``).
        lo, hi: Search-range bounds — the loosest the edges may be.
        cut_lo, cut_hi: Current cut edges (start at ``lo``, ``hi``).
        cut_type: ``"both"``, ``"left"``, ``"right"``, or ``"veto"``.
        discrete: Integer-valued variable — edges sit on half-integers and
            move in whole units.
        applied: Whether this cut is active.
    """

    name: str
    lo: float
    hi: float
    cut_lo: float = float("nan")
    cut_hi: float = float("nan")
    cut_type: str = BOTH
    discrete: bool = False
    applied: bool = False

    def __post_init__(self):
        if math.isnan(self.cut_lo):
            self.cut_lo = self.lo
        if math.isnan(self.cut_hi):
            self.cut_hi = self.hi
        if self.cut_type == VETO:
            # A veto starts with an empty window at the seed (midpoint by
            # default) and widens into [lo, hi].
            if self.cut_lo == self.lo and self.cut_hi == self.hi:
                mid = 0.5 * (self.lo + self.hi)
                self.cut_lo = mid
                self.cut_hi = mid
        if self.cut_type not in CUT_TYPES:
            raise ValueError(f"unknown cut_type {self.cut_type!r}; "
                             f"must be one of {CUT_TYPES}")


def make_discrete(name: str, lo: int, hi: int,
                  cut_type: str = BOTH) -> CutVar:
    """Integer-valued variable: edges shifted by ±0.5 so cuts sit on
    half-integers (the same convention as optimize_selection_fast.C)."""
    return CutVar(name, lo - 0.5, hi + 0.5, cut_type=cut_type, discrete=True)


def passes_cut(var: CutVar, v: np.ndarray) -> np.ndarray:
    """Boolean mask: which events pass this cut variable."""
    if not var.applied:
        return np.ones(len(v), dtype=bool)
    if var.cut_type == BOTH:
        return (v > var.cut_lo) & (v < var.cut_hi)
    if var.cut_type == LEFT:
        return v > var.cut_lo
    if var.cut_type == RIGHT:
        return v < var.cut_hi
    # VETO: keep events outside the exclusion window
    return (v < var.cut_lo) | (v > var.cut_hi)


# ---------------------------------------------------------------------------
# In-memory event store
# ---------------------------------------------------------------------------

@dataclass
class Events:
    """Variable values and signal/background weights for every event.

    ``vals[name]`` is a 1-D float64 array of one value per event.
    ``w_sig`` and ``w_bkg`` are the per-event signal and background weights.
    ``sort_indices[name]`` holds the indices that sort the events by that
    variable's values, computed once by ``prepare()``.

    Use ``split()`` to partition into train/test halves by event-number
    parity, each half weighted by 2 to estimate the full-sample yields.
    """

    vals: dict[str, np.ndarray] = field(default_factory=dict)
    w_sig: np.ndarray = field(default_factory=lambda: np.zeros(0))
    w_bkg: np.ndarray = field(default_factory=lambda: np.zeros(0))
    sort_indices: dict[str, np.ndarray] = field(default_factory=dict)
    event_numbers: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def size(self) -> int:
        return len(self.w_sig)

    def prepare(self) -> None:
        """Pre-sort events by each variable for fast edge scanning."""
        for name, values in self.vals.items():
            self.sort_indices[name] = np.argsort(values, kind="stable")

    def split(self) -> tuple["Events", "Events"]:
        """Split by event-number parity: even → train, odd → test, each ×2."""
        even = (self.event_numbers % 2 == 0)
        odd = ~even
        train = Events(
            vals={n: v[even] for n, v in self.vals.items()},
            w_sig=self.w_sig[even] * 2.0,
            w_bkg=self.w_bkg[even] * 2.0,
            event_numbers=self.event_numbers[even],
        )
        test = Events(
            vals={n: v[odd] for n, v in self.vals.items()},
            w_sig=self.w_sig[odd] * 2.0,
            w_bkg=self.w_bkg[odd] * 2.0,
            event_numbers=self.event_numbers[odd],
        )
        train.prepare()
        test.prepare()
        return train, test


# ---------------------------------------------------------------------------
# Figures of merit
# ---------------------------------------------------------------------------

FigureOfMerit = Callable[[float, float], float]


def s_over_sqrt_b(s: float, b: float) -> float:
    """S / sqrt(B). Returns 0 when S <= 0 or B <= 0."""
    if s <= 0.0 or b <= 0.0:
        return 0.0
    return s / math.sqrt(b)


def asimov_z(s: float, b: float) -> float:
    """Asimov significance: sqrt(2 * ((S+B) * log(1 + S/B) - S)).
    Smooth limit sqrt(2S) at B → 0."""
    if s <= 0.0:
        return 0.0
    if b <= 1e-9:
        return math.sqrt(2.0 * s)
    return math.sqrt(2.0 * ((s + b) * math.log(1.0 + s / b) - s))


def punzi_fom(s: float, b: float, a: float = 1.28) -> float:
    """Punzi figure of merit: S / (a/2 + sqrt(B)), a = 1.28 for 90 % CL."""
    if s <= 0.0:
        return 0.0
    return s / (a / 2.0 + math.sqrt(max(b, 0.0)))


def _poisson_cdf(n: int, lam: float) -> float:
    """Cumulative Poisson probability P(X <= n | lambda), using the
    regularized incomplete gamma function: P(X <= n) = Q(n+1, lam)."""
    from scipy.special import gammaincc
    if lam <= 0.0:
        return 1.0
    return float(gammaincc(n + 1, lam))


def _poisson_pmf(n: int, lam: float) -> float:
    """Poisson probability P(X = n | lambda)."""
    if lam <= 0.0:
        return 1.0 if n == 0 else 0.0
    return math.exp(n * math.log(lam) - lam - math.lgamma(n + 1))


def _poisson_cls_ul(n: int, b: float, cl: float = 0.9) -> float:
    """90 % CL CLs upper limit on signal events for n observed with known
    background b, by bisection."""
    alpha = 1.0 - cl
    cl_b = _poisson_cdf(n, b)
    if cl_b <= 0.0:
        return 0.0

    def cls(s):
        return _poisson_cdf(n, s + b) / cl_b

    lo, hi = 0.0, n + 3.0 + 5.0 * math.sqrt(n + 1)
    while cls(hi) > alpha:
        hi *= 2.0
    for _ in range(100):
        if hi - lo <= 1e-7 * hi:
            break
        mid = 0.5 * (lo + hi)
        if cls(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _expected_ul_events_exact(b: float) -> float:
    """Expected upper limit on signal events: sum_n P(n|b) * N_UL(n, b)."""
    if b <= 0.0:
        return _poisson_cls_ul(0, 0.0)
    n_lo = max(0, int(math.floor(b - 7.0 * math.sqrt(b) - 2.0)))
    n_hi = int(math.ceil(b + 7.0 * math.sqrt(b) + 5.0))
    ul = 0.0
    norm = 0.0
    for n in range(n_lo, n_hi + 1):
        p = _poisson_pmf(n, b)
        ul += p * _poisson_cls_ul(n, b)
        norm += p
    return ul / norm if norm > 0.0 else 0.0


# Lazily-built lookup table in log10(b), as in the C++ code.
_UL_TABLE: list[float] = []
_UL_LMIN = -4.0
_UL_LMAX = 3.0
_UL_NPOINTS = 701


def _ensure_ul_table() -> None:
    if _UL_TABLE:
        return
    for i in range(_UL_NPOINTS):
        b = 10.0 ** (_UL_LMIN + i * (_UL_LMAX - _UL_LMIN) / (_UL_NPOINTS - 1))
        _UL_TABLE.append(_expected_ul_events_exact(b))


def expected_ul_events(b: float) -> float:
    """Expected 90 % CL CLs upper limit on signal events, interpolated from
    a lookup table in log10(b) for speed."""
    _ensure_ul_table()
    b = max(b, 0.0)
    if b <= 10.0 ** _UL_LMIN:
        return _UL_TABLE[0]
    l = math.log10(b)
    if l >= _UL_LMAX:
        return _expected_ul_events_exact(b)
    x = (l - _UL_LMIN) / (_UL_LMAX - _UL_LMIN) * (_UL_NPOINTS - 1)
    i = int(x)
    f = x - i
    return _UL_TABLE[i] * (1.0 - f) + _UL_TABLE[i + 1] * f


def inv_expected_limit(s: float, b: float) -> float:
    """S / N_UL(B): 1 over the expected 90 % CL CLs upper limit on the
    signal strength, independent of the signal normalization."""
    if s <= 0.0:
        return 0.0
    return s / expected_ul_events(max(b, 0.0))


# Registry of named FoM functions.
FOM_FUNCTIONS: dict[str, FigureOfMerit] = {
    "s_over_sqrt_b": s_over_sqrt_b,
    "asimov_z": asimov_z,
    "punzi": punzi_fom,
    "expected_ul": inv_expected_limit,
}


# ---------------------------------------------------------------------------
# Yield evaluation
# ---------------------------------------------------------------------------

def evaluate_yields(cuts: list[CutVar], events: Events
                    ) -> tuple[float, float]:
    """Sum of signal and background weights passing all applied cuts."""
    mask = np.ones(events.size, dtype=bool)
    for var in cuts:
        if var.applied:
            mask &= passes_cut(var, events.vals[var.name])
    return float(events.w_sig[mask].sum()), float(events.w_bkg[mask].sum())


def evaluate_yields_mask(cuts: list[CutVar], events: Events
                         ) -> tuple[float, float, np.ndarray]:
    """Like evaluate_yields, also returning the boolean pass mask."""
    mask = np.ones(events.size, dtype=bool)
    for var in cuts:
        if var.applied:
            mask &= passes_cut(var, events.vals[var.name])
    return float(events.w_sig[mask].sum()), float(events.w_bkg[mask].sum()), mask


# ---------------------------------------------------------------------------
# Edge scanning
# ---------------------------------------------------------------------------

@dataclass
class EdgePoint:
    """One candidate edge position and the signal/background yields there."""
    edge: float
    n_sig: float
    n_bkg: float


def _scan_edge_continuous(entries: np.ndarray, y0: float, y_lim: float,
                          s_fixed: float, b_fixed: float) -> list[EdgePoint]:
    """Candidate edges for a continuous variable. ``entries`` is (N, 3):
    oriented value, signal weight, background weight, sorted by value.
    Just below each signal value, so all background beneath it is removed
    for the same signal loss."""
    n = len(entries)
    if n == 0:
        return []
    # Reverse cumulative sums (yields of entries [i, n))
    s_above = np.zeros(n + 1)
    b_above = np.zeros(n + 1)
    for i in range(n - 1, -1, -1):
        s_above[i] = s_above[i + 1] + entries[i, 1]
        b_above[i] = b_above[i + 1] + entries[i, 2]

    scan: list[EdgePoint] = []
    i = 0
    while i < n:
        j = i
        has_signal = False
        while j < n and entries[j, 0] == entries[i, 0]:
            has_signal = has_signal or entries[j, 1] > 0.0
            j += 1
        u = entries[i, 0]
        if has_signal and u > y0:
            y = max(y0, 0.5 * (entries[i - 1, 0] + u)) if i > 0 else y0
            if y >= y_lim:
                break
            scan.append(EdgePoint(y, s_fixed + s_above[i],
                                  b_fixed + b_above[i]))
        i = j
    return scan


def _scan_edge_discrete(entries: np.ndarray, y0: float, y_lim: float,
                        s_fixed: float, b_fixed: float) -> list[EdgePoint]:
    """Candidate edges for a discrete (integer-valued) variable: whole-unit
    moves from y0 towards y_lim."""
    n = len(entries)
    if n == 0:
        return []
    s_above = np.zeros(n + 1)
    b_above = np.zeros(n + 1)
    for i in range(n - 1, -1, -1):
        s_above[i] = s_above[i + 1] + entries[i, 1]
        b_above[i] = b_above[i + 1] + entries[i, 2]

    n_max = round(y_lim - y0) - 1
    scan: list[EdgePoint] = []
    for move in range(1, n_max + 1):
        y = y0 + move
        # First entry with value > y
        first = int(np.searchsorted(entries[:, 0], y, side="right"))
        scan.append(EdgePoint(y, s_fixed + s_above[first],
                              b_fixed + b_above[first]))
    return scan


def _choose_edge(scan: list[EdgePoint], n_sig_0: float,
                 step: float) -> int:
    """The largest move whose signal loss is within ``step``, or the first
    move if all exceed it. A move losing no signal is taken one further."""
    if not scan:
        return -1
    index = 0
    for i, pt in enumerate(scan):
        if 1.0 - pt.n_sig / n_sig_0 <= step:
            index = i
    if scan[index].n_sig >= n_sig_0 and index + 1 < len(scan):
        index += 1
    return index


# ---------------------------------------------------------------------------
# Single-variable tightening and loosening
# ---------------------------------------------------------------------------

@dataclass
class TestPoint:
    """The result of tightening one variable by one step."""
    ivar: int
    new_lo: float
    new_hi: float
    n_sig_0: float
    n_bkg_0: float
    n_sig: float
    n_bkg: float
    fom_value: float = 0.0
    grad: float = 0.0

    def compute_grad(self, fom: FigureOfMerit) -> None:
        self.fom_value = fom(self.n_sig, self.n_bkg)
        fom_0 = fom(self.n_sig_0, self.n_bkg_0)
        ds = self.n_sig_0 - self.n_sig
        if ds == 0.0 or self.n_bkg_0 <= 0.0:
            self.grad = 0.0
        else:
            self.grad = (self.fom_value - fom_0) / ds


def _find_tighten_point(ivar: int, cuts: list[CutVar], events: Events,
                        pass_mask: np.ndarray, n_sig_0: float,
                        n_bkg_0: float, step: float,
                        fom: FigureOfMerit) -> TestPoint | None:
    """Tighten variable ``ivar`` by about one efficiency step, trying both
    sides. Returns the side with the larger FoM, or None if neither can
    tighten."""
    var = cuts[ivar]
    vals = events.vals[var.name]
    order = events.sort_indices[var.name]
    n = len(order)
    best: TestPoint | None = None

    for side in (0, 1):  # 0 = lower edge, 1 = upper edge
        if side == 0 and var.cut_type == RIGHT:
            continue
        if side == 1 and var.cut_type == LEFT:
            continue
        is_veto = var.cut_type == VETO
        # Orientation: a cut edge tightens upwards for the lower edge,
        # a veto edge for the upper edge.
        sign = (-1.0 if side == 0 else 1.0) if is_veto else (
            1.0 if side == 0 else -1.0)
        if is_veto:
            y0 = -var.cut_lo if side == 0 else var.cut_hi
            y_lim = -var.lo if side == 0 else var.hi
        else:
            y0 = var.cut_lo if side == 0 else -var.cut_hi
            y_lim = var.cut_hi if side == 0 else -var.cut_lo

        entries = []
        s_fixed = 0.0
        b_fixed = 0.0
        indices = order if sign > 0 else order[::-1]
        for k in range(n):
            i = int(indices[k])
            if not pass_mask[i]:
                continue
            v = vals[i]
            if is_veto:
                movable = v < var.cut_lo if side == 0 else v > var.cut_hi
                fixed = v > var.cut_hi if side == 0 else v < var.cut_lo
            else:
                movable = (var.cut_type != BOTH
                           or (v < var.cut_hi if side == 0 else v > var.cut_lo))
                fixed = False
            if movable:
                entries.append([sign * v, events.w_sig[i], events.w_bkg[i]])
            elif fixed:
                s_fixed += events.w_sig[i]
                b_fixed += events.w_bkg[i]

        if not entries:
            continue
        arr = np.array(entries)
        use_discrete = var.discrete and not is_veto
        if use_discrete:
            scan = _scan_edge_discrete(arr, y0, y_lim, s_fixed, b_fixed)
        else:
            scan = _scan_edge_continuous(arr, y0, y_lim, s_fixed, b_fixed)
        idx = _choose_edge(scan, n_sig_0, step)
        if idx < 0:
            continue
        pt = scan[idx]
        edge = sign * pt.edge
        new_lo = edge if side == 0 else var.cut_lo
        new_hi = edge if side == 1 else var.cut_hi
        tp = TestPoint(ivar, new_lo, new_hi, n_sig_0, n_bkg_0,
                       pt.n_sig, pt.n_bkg)
        tp.compute_grad(fom)
        if best is None or tp.fom_value > best.fom_value:
            best = tp
    return best


def _find_loosen_point(ivar: int, cuts: list[CutVar], events: Events,
                       step: float, fom: FigureOfMerit) -> TestPoint | None:
    """Loosen an applied cut until about ``step`` of signal is gained.
    Returns None if the cut cannot be loosened."""
    var = cuts[ivar]
    if not var.applied or var.cut_type == VETO:
        return None
    n_sig_c, n_bkg_c = evaluate_yields(cuts, events)
    # Evaluate with this cut removed to get the pass mask of the other cuts
    others = copy.deepcopy(cuts)
    others[ivar].applied = False
    _, _, pass_mask = evaluate_yields_mask(others, events)

    vals = events.vals[var.name]
    order = events.sort_indices[var.name]
    n = len(order)
    target = n_sig_c * (1.0 + step)
    best: TestPoint | None = None

    for side in (0, 1):
        if side == 0 and var.cut_type == RIGHT:
            continue
        if side == 1 and var.cut_type == LEFT:
            continue
        sign = 1.0 if side == 0 else -1.0
        y0 = var.lo if side == 0 else -var.hi          # loosest edge
        y_cur = var.cut_lo if side == 0 else -var.cut_hi  # current edge
        if y_cur <= y0:
            continue

        entries = []
        s_loose = 0.0
        b_loose = 0.0
        indices = order if sign > 0 else order[::-1]
        for k in range(n):
            i = int(indices[k])
            if not pass_mask[i]:
                continue
            v = vals[i]
            if var.cut_type == BOTH:
                if not (v < var.cut_hi if side == 0 else v > var.cut_lo):
                    continue
            entries.append([sign * v, events.w_sig[i], events.w_bkg[i]])
            if sign * v > y0:
                s_loose += events.w_sig[i]
                b_loose += events.w_bkg[i]

        if not entries:
            continue
        arr = np.array(entries)
        scan = _scan_edge_continuous(arr, y0, y_cur, 0.0, 0.0)
        # Insert the loosest point at the front
        scan.insert(0, EdgePoint(y0, s_loose, b_loose))
        # Take the tightest position that gains the step
        index = 0
        for i, pt in enumerate(scan):
            if pt.n_sig >= target:
                index = i
        if scan[index].n_sig <= n_sig_c:
            continue
        edge = sign * scan[index].edge
        new_lo = edge if side == 0 else var.cut_lo
        new_hi = edge if side == 1 else var.cut_hi
        tp = TestPoint(ivar, new_lo, new_hi, n_sig_c, n_bkg_c,
                       scan[index].n_sig, scan[index].n_bkg)
        tp.compute_grad(fom)
        if best is None or tp.fom_value > best.fom_value:
            best = tp
    return best


# ---------------------------------------------------------------------------
# Selection state
# ---------------------------------------------------------------------------

@dataclass
class Selection:
    """A cut-set and its yields."""
    cuts: list[CutVar]
    n_sig: float
    n_bkg: float


def _apply_point(cuts: list[CutVar], tp: TestPoint) -> None:
    """Apply a TestPoint's edges to the cut list (in place)."""
    cuts[tp.ivar].cut_lo = tp.new_lo
    cuts[tp.ivar].cut_hi = tp.new_hi
    cuts[tp.ivar].applied = True


def _tighten_candidates(cuts: list[CutVar], events: Events, step: float,
                        fom: FigureOfMerit) -> list[TestPoint]:
    """Test points of all variables from the current selection."""
    n_sig, n_bkg, pass_mask = evaluate_yields_mask(cuts, events)
    candidates: list[TestPoint] = []
    for ivar in range(len(cuts)):
        tp = _find_tighten_point(ivar, cuts, events, pass_mask,
                                 n_sig, n_bkg, step, fom)
        if tp is not None and tp.n_sig < tp.n_sig_0:
            candidates.append(tp)
    return candidates


def _choose_greedy(candidates: list[TestPoint],
                   rng: np.random.Generator | None = None,
                   top_k: int = 1) -> int:
    """Index into candidates of the greedy choice: the steepest gradient,
    or a random one of the top_k with ``rng``."""
    if not candidates:
        return -1
    candidates.sort(key=lambda tp: tp.grad, reverse=True)
    if rng is None:
        return 0
    return int(rng.integers(min(top_k, len(candidates))))


# ---------------------------------------------------------------------------
# Greedy path
# ---------------------------------------------------------------------------

def run_greedy(cuts: list[CutVar], events: Events, n_sig_ref: float,
               step: float, end_eff: float, fom: FigureOfMerit,
               rng: np.random.Generator | None = None,
               top_k: int = 1) -> list[Selection]:
    """Greedy path from a starting selection, recording every step.

    ``n_sig_ref`` is the reference signal yield (at no cuts) for the
    efficiency calculation. The path stops when the efficiency drops below
    ``end_eff`` or background reaches zero.
    """
    working = copy.deepcopy(cuts)
    path: list[Selection] = []
    eff = 1.0
    while eff > end_eff:
        candidates = _tighten_candidates(working, events, step, fom)
        idx = _choose_greedy(candidates, rng, top_k)
        if idx < 0:
            break
        _apply_point(working, candidates[idx])
        sel = Selection(copy.deepcopy(working),
                        candidates[idx].n_sig, candidates[idx].n_bkg)
        path.append(sel)
        eff = sel.n_sig / n_sig_ref
        if sel.n_bkg <= 0.0:
            break
    return path


def best_on_path(path: list[Selection], fom: FigureOfMerit) -> Selection:
    """The selection on a greedy path with the highest figure of merit."""
    return max(path, key=lambda s: fom(s.n_sig, s.n_bkg))


# ---------------------------------------------------------------------------
# Local refinement
# ---------------------------------------------------------------------------

def refine(cuts: list[CutVar], events: Events, step: float,
           fom: FigureOfMerit, max_iter: int = 100) -> Selection:
    """Hill-climbing local search from a starting selection.

    Each iteration tries tightening any variable, loosening any variable,
    and loosening one then tightening the best other. The largest
    improvement is accepted; the search stops when no move improves.
    """
    working = copy.deepcopy(cuts)
    n_sig, n_bkg = evaluate_yields(working, events)
    for _ in range(max_iter):
        best_fom = fom(n_sig, n_bkg) * (1.0 + 1e-6)
        improved = False
        best_cuts: list[CutVar] | None = None
        best_s = 0.0
        best_b = 0.0

        def consider(test: list[CutVar], s: float, b: float) -> None:
            nonlocal best_fom, improved, best_cuts, best_s, best_b
            f = fom(s, b)
            if f <= best_fom:
                return
            best_fom = f
            best_cuts = copy.deepcopy(test)
            best_s = s
            best_b = b
            improved = True

        # Try tightening each variable
        for tp in _tighten_candidates(working, events, step, fom):
            test = copy.deepcopy(working)
            _apply_point(test, tp)
            consider(test, tp.n_sig, tp.n_bkg)

        # Try loosening each variable
        for ivar in range(len(working)):
            lp = _find_loosen_point(ivar, working, events, step, fom)
            if lp is None:
                continue
            test = copy.deepcopy(working)
            _apply_point(test, lp)
            s, b = evaluate_yields(test, events)
            consider(test, s, b)

            # Loosen one, then tighten the best other
            tighten = _tighten_candidates(test, events, step, fom)
            tidx = _choose_greedy(tighten)
            if tidx >= 0:
                _apply_point(test, tighten[tidx])
                consider(test, tighten[tidx].n_sig, tighten[tidx].n_bkg)

        if not improved:
            break
        working = best_cuts  # type: ignore[assignment]
        n_sig = best_s
        n_bkg = best_b

    return Selection(working, n_sig, n_bkg)


# ---------------------------------------------------------------------------
# Random grid search
# ---------------------------------------------------------------------------

def random_grid_search(cuts: list[CutVar], events: Events,
                       n_trials: int, fom: FigureOfMerit,
                       rng: np.random.Generator | None = None,
                       ) -> list[dict]:
    """Sample ``n_trials`` random cut-sets and evaluate each one.

    Returns a list of dicts with keys ``eff``, ``rejection``, ``fom``,
    ``n_sig``, ``n_bkg``, and ``cuts`` (the CutVar list).
    """
    if rng is None:
        rng = np.random.default_rng(90)
    # Baselines (all cuts unapplied)
    unapplied = copy.deepcopy(cuts)
    for v in unapplied:
        v.applied = False
    n_sig_0, n_bkg_0 = evaluate_yields(unapplied, events)
    fom_0 = fom(n_sig_0, n_bkg_0)

    results: list[dict] = []
    for _ in range(n_trials):
        test_cuts: list[CutVar] = []
        for var in cuts:
            tc = copy.deepcopy(var)
            tc.applied = rng.random() > 0.5
            if var.cut_type == VETO:
                tc.cut_lo = var.lo + rng.random() * (var.hi - var.lo)
                tc.cut_hi = tc.cut_lo + rng.random() * (var.hi - tc.cut_lo)
                test_cuts.append(tc)
                continue
            if var.discrete:
                span = round(var.hi - var.lo)
                if var.cut_type != RIGHT:
                    tc.cut_lo = var.lo + rng.integers(span)
                if var.cut_type != LEFT:
                    remaining = round(var.hi - tc.cut_lo)
                    if remaining > 0:
                        tc.cut_hi = var.hi - rng.integers(remaining)
                test_cuts.append(tc)
                continue
            if var.cut_type != RIGHT:
                tc.cut_lo = var.lo + rng.random() * (var.hi - var.lo)
            if var.cut_type != LEFT:
                tc.cut_hi = var.hi - rng.random() * (var.hi - tc.cut_lo)
            test_cuts.append(tc)

        n_sig, n_bkg = evaluate_yields(test_cuts, events)
        if n_bkg <= 0.0:
            continue
        eff = n_sig / n_sig_0 if n_sig_0 > 0 else 0.0
        rej = 1.0 - n_bkg / n_bkg_0 if n_bkg_0 > 0 else 0.0
        results.append({
            "eff": eff,
            "rejection": rej,
            "fom": fom(n_sig, n_bkg) / fom_0 if fom_0 > 0 else 0.0,
            "n_sig": n_sig,
            "n_bkg": n_bkg,
            "cuts": test_cuts,
        })
    return results


# ---------------------------------------------------------------------------
# Multi-start optimization
# ---------------------------------------------------------------------------

def multi_start_optimize(cuts: list[CutVar], events: Events, step: float,
                         end_eff: float, fom: FigureOfMerit,
                         seeds: list[list[CutVar]] | None = None,
                         n_random: int = 5,
                         top_k: int = 3,
                         ) -> dict:
    """Multi-start optimization: greedy + seeds + randomized greedy starts,
    all refined by local search.

    Returns a dict with keys ``best`` (Selection), ``greedy_path``
    (list[Selection]), ``starts`` (list of dicts with ``name``,
    ``start``, ``refined``), and ``n_sig_ref``, ``n_bkg_ref`` (the
    baseline yields).
    """
    # Baseline yields (all cuts unapplied)
    unapplied = copy.deepcopy(cuts)
    for v in unapplied:
        v.applied = False
    n_sig_ref, n_bkg_ref = evaluate_yields(unapplied, events)
    fom_ref = fom(n_sig_ref, n_bkg_ref)

    # Greedy path
    path = run_greedy(cuts, events, n_sig_ref, step, end_eff, fom)
    if not path:
        return {"best": None, "greedy_path": [], "starts": [],
                "n_sig_ref": n_sig_ref, "n_bkg_ref": n_bkg_ref,
                "fom_ref": fom_ref}

    starts: list[dict] = []
    starts.append({"name": "greedy", "start": best_on_path(path, fom)})

    # User-supplied seed selections
    for i, seed in enumerate(seeds or []):
        s, b = evaluate_yields(seed, events)
        starts.append({"name": f"seed_{i}", "start": Selection(seed, s, b)})

    # Randomized greedy starts
    for irun in range(n_random):
        rng = np.random.default_rng(1000 + irun)
        rpath = run_greedy(cuts, events, n_sig_ref, step, end_eff, fom,
                           rng=rng, top_k=top_k)
        if rpath:
            starts.append({"name": f"random_{irun}",
                           "start": best_on_path(rpath, fom)})

    # Refine each starting point
    best_fom_val = -1.0
    best_sel: Selection | None = None
    for entry in starts:
        sel = entry["start"]
        refined_sel = refine(sel.cuts, events, step, fom)
        entry["refined"] = refined_sel
        f = fom(refined_sel.n_sig, refined_sel.n_bkg)
        if f > best_fom_val:
            best_fom_val = f
            best_sel = refined_sel

    return {"best": best_sel, "greedy_path": path, "starts": starts,
            "n_sig_ref": n_sig_ref, "n_bkg_ref": n_bkg_ref,
            "fom_ref": fom_ref}


# ---------------------------------------------------------------------------
# Cut-string builder (selection.py-compatible)
# ---------------------------------------------------------------------------

def build_cut_string(cuts: list[CutVar]) -> str:
    """Build a selection expression from the applied cuts, compatible with
    tools/selection.py's syntax."""
    parts: list[str] = []
    for var in cuts:
        if not var.applied:
            continue
        if var.cut_type == BOTH:
            parts.append(f"{var.name} > {var.cut_lo:.8g} and "
                         f"{var.name} < {var.cut_hi:.8g}")
        elif var.cut_type == LEFT:
            parts.append(f"{var.name} > {var.cut_lo:.8g}")
        elif var.cut_type == RIGHT:
            parts.append(f"{var.name} < {var.cut_hi:.8g}")
        elif var.cut_type == VETO:
            parts.append(f"({var.name} < {var.cut_lo:.8g} or "
                         f"{var.name} > {var.cut_hi:.8g})")
    return " and ".join(parts) if parts else "(1)"


# ---------------------------------------------------------------------------
# Cut variable spec parser
# ---------------------------------------------------------------------------

class CutVarSpecError(ValueError):
    """A variable specification string that cannot be parsed, worded for
    the caller."""


def parse_cut_var_specs(text: str) -> list[CutVar]:
    r"""Parse a comma-separated variable specification string into CutVars.

    Format per variable::

        name:type:lo:hi[:seed_lo:seed_hi]

    or for discrete variables::

        name:type:lo:hi:discrete[:seed_lo:seed_hi]

    ``type`` is ``both``, ``left``, ``right``, or ``veto``.

    ``seed_lo:seed_hi`` sets the initial cut edges (``cut_lo``, ``cut_hi``).
    Without them, edges start at the search bounds (loosest), or at the
    midpoint for a veto.

    Examples::

        p_front:both:80:110, trkqual:left:0:1, t_front:both:500:1700
        nactive:left:10:60:discrete
        crv_dt:veto:-150:150:-10:-10

    Raises CutVarSpecError for bad syntax or unknown types.
    """
    specs = [s.strip() for s in text.split(",") if s.strip()]
    if not specs:
        raise CutVarSpecError("optimize_variables is empty: list at least one "
                              "variable as 'name:type:lo:hi', e.g. "
                              "'p_front:both:80:110, trkqual:left:0:1'")
    result: list[CutVar] = []
    for spec in specs:
        parts = [p.strip() for p in spec.split(":")]
        if len(parts) < 4:
            raise CutVarSpecError(
                f"variable spec {spec!r} needs at least 'name:type:lo:hi', "
                f"got {len(parts)} fields"
            )
        name = parts[0]
        cut_type = parts[1].lower()
        if cut_type not in CUT_TYPES:
            raise CutVarSpecError(
                f"variable {name!r}: type {parts[1]!r} must be one of "
                f"{', '.join(CUT_TYPES)}"
            )
        try:
            lo = float(parts[2])
            hi = float(parts[3])
        except ValueError:
            raise CutVarSpecError(
                f"variable {name!r}: lo={parts[2]!r} and hi={parts[3]!r} "
                "must be numbers"
            )
        if lo >= hi:
            raise CutVarSpecError(
                f"variable {name!r}: lo={lo:g} must be less than hi={hi:g}"
            )

        # Remaining parts: 'discrete' flag and seed_lo:seed_hi
        remaining = parts[4:]
        discrete = False
        seed_lo = float("nan")
        seed_hi = float("nan")
        if remaining and remaining[0].lower() == "discrete":
            discrete = True
            if discrete:
                lo = lo - 0.5
                hi = hi + 0.5
            remaining = remaining[1:]
        if len(remaining) == 2:
            try:
                seed_lo = float(remaining[0])
                seed_hi = float(remaining[1])
            except ValueError:
                raise CutVarSpecError(
                    f"variable {name!r}: seed_lo={remaining[0]!r} and "
                    f"seed_hi={remaining[1]!r} must be numbers"
                )
        elif remaining:
            raise CutVarSpecError(
                f"variable {name!r}: unexpected fields after type:lo:hi"
                f"{'[:discrete]' if not discrete else ''}: "
                f"{':'.join(remaining)}"
            )
        cv = CutVar(name, lo, hi, cut_type=cut_type, discrete=discrete)
        if not math.isnan(seed_lo):
            cv.cut_lo = seed_lo
        if not math.isnan(seed_hi):
            cv.cut_hi = seed_hi
        cv.applied = False  # starts unapplied
        result.append(cv)
    return result


def parse_seed_selections(text: str, var_names: list[str]
                          ) -> list[list[CutVar]]:
    """Parse seed selections: semicolon-separated cut-sets, each a
    comma-separated list of ``name:lo:hi``.

    Example::

        trkqual:0.155:1,cos_front:0.5:0.75; trkqual:0.145:1,cos_front:0.56:0.73

    Each seed produces a full CutVar list with the named variables applied
    at the given edges and all others unapplied at their search bounds.
    ``var_names`` is the list of variable names from optimize_variables,
    giving the search bounds.

    Raises CutVarSpecError for bad syntax or unknown variable names.
    """
    if not text.strip():
        return []
    seeds: list[list[CutVar]] = []
    for seed_text in text.split(";"):
        seed_text = seed_text.strip()
        if not seed_text:
            continue
        seed_cuts: dict[str, tuple[float, float]] = {}
        for item in seed_text.split(","):
            item = item.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(":")]
            if len(parts) != 3:
                raise CutVarSpecError(
                    f"seed cut {item!r} must be 'name:lo:hi', got "
                    f"{len(parts)} fields"
                )
            name = parts[0]
            if name not in var_names:
                raise CutVarSpecError(
                    f"seed variable {name!r} is not in the optimize_variables "
                    f"list ({', '.join(var_names)})"
                )
            try:
                seed_cuts[name] = (float(parts[1]), float(parts[2]))
            except ValueError:
                raise CutVarSpecError(
                    f"seed cut {item!r}: lo={parts[1]!r} and hi={parts[2]!r} "
                    "must be numbers"
                )
        seeds.append(seed_cuts)
    return seeds  # type: ignore[return-value] — the dicts are applied below


def apply_seed_to_vars(cut_vars: list[CutVar],
                       seed: dict[str, tuple[float, float]]
                       ) -> list[CutVar]:
    """Apply a parsed seed (name → (lo, hi)) to a copy of the CutVar list."""
    result = copy.deepcopy(cut_vars)
    for var in result:
        if var.name in seed:
            lo, hi = seed[var.name]
            if var.cut_type != RIGHT:
                var.cut_lo = lo
            if var.cut_type != LEFT:
                var.cut_hi = hi
            var.applied = True
    return result
