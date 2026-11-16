"""Rectangular cut optimization on a weighted event table.

A numpy port of M. MacKenzie's optimize_selection_fast.C (mumep_ana,
branch mu2ecombine-workflow, analysis/optimize_selection_fast.C, 62e1def):

- every row is an event with values of the cut variables, a signal weight and
  a background weight (a row may carry both: a CE event is also the DIO
  stand-in at its energy loss, see sensitivity.py);
- each cut edge is found from cumulative yields over the rows sorted by the
  variable, not by re-counting at trial values;
- the selection is chosen on the rows with even event numbers and evaluated
  on the odd ones, each half weighted by 2;
- a greedy path tightens, one step of signal efficiency at a time, the edge
  with the largest figure-of-merit gain per signal lost; the best point on
  the path, and the best points of randomized greedy paths, are refined by a
  local search (tighten one edge, loosen one edge, or loosen one and tighten
  the greedy choice) until nothing improves.

The figure of merit is Punzi's S / (a/2 + sqrt(B)) (approx_ce_sensitivity's
punzi_fom), finite where the background vanishes. Nothing here knows about
Mu2e: the caller builds the table.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# The Punzi a, as approx_ce_sensitivity.PUNZI_A (1.28 sigma: 90% CL).
PUNZI_A = 1.28

# Two-sided, or one edge only: "low" cuts low values (keeps x > lo), "high"
# cuts high values (keeps x < hi).
SIDES = ("both", "low", "high")


def punzi_fom(s, b):
    """S / (PUNZI_A/2 + sqrt(B)); 0 without signal."""
    s = np.asarray(s, dtype=float)
    b = np.maximum(np.asarray(b, dtype=float), 0.0)
    return np.where(s > 0.0, s / (PUNZI_A / 2.0 + np.sqrt(b)), 0.0)


def s_over_sqrt_b(s, b):
    """S / sqrt(B), 0 without background: the macro's default figure of
    merit, here only to compare with it (it diverges as B is cut away)."""
    s = np.asarray(s, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((s > 0.0) & (b > 0.0), s / np.sqrt(np.where(b > 0, b, 1.0)), 0.0)


FOMS = {"punzi": punzi_fom, "s_over_sqrt_b": s_over_sqrt_b}
# The figure of merit the search maximizes; set by optimize(fom=...).
_fom = punzi_fom


class OptimizeError(ValueError):
    """A table or variable list the optimization cannot run on."""


@dataclass(frozen=True)
class Variable:
    """A cut variable and the range its edges may take. An integer-valued
    variable's edges sit on half-integers."""
    name: str
    low: float
    high: float
    side: str = "both"
    discrete: bool = False

    def __post_init__(self):
        if self.side not in SIDES:
            raise OptimizeError(f"{self.name}: side must be one of {SIDES}")
        if not self.low < self.high:
            raise OptimizeError(f"{self.name}: low {self.low} >= high {self.high}")


@dataclass
class Table:
    """The events: `values[name]` per row, signal and background weights,
    and which rows are in the optimization (train) half."""
    values: dict
    signal: np.ndarray
    background: np.ndarray
    train: np.ndarray
    # each column's rows in ascending order, sorted once (the macro's
    # Events_t.order): a scan then filters, never re-sorts
    order: dict = field(init=False, repr=False)

    def __post_init__(self):
        n = len(self.signal)
        for name, v in self.values.items():
            if len(v) != n:
                raise OptimizeError(f"column {name} has {len(v)} rows, not {n}")
        if len(self.background) != n or len(self.train) != n:
            raise OptimizeError("weights and train mask must have one entry per row")
        self.order = {k: np.argsort(v, kind="stable") for k, v in self.values.items()}

    @classmethod
    def from_columns(cls, values: dict, signal, background, event_number):
        """Rows with even event numbers are the train half."""
        return cls({k: np.asarray(v, dtype=float) for k, v in values.items()},
                   np.asarray(signal, dtype=float),
                   np.asarray(background, dtype=float),
                   np.asarray(event_number, dtype=np.int64) % 2 == 0)

    def half(self, train: bool) -> "Table":
        """One half, its weights doubled to estimate the full sample."""
        m = self.train if train else ~self.train
        return Table({k: v[m] for k, v in self.values.items()},
                     2.0 * self.signal[m], 2.0 * self.background[m],
                     np.full(int(m.sum()), train))


@dataclass
class Selection:
    """Cuts {name: (lo, hi)}, open interval, and the yields they pass."""
    cuts: dict
    signal: float = 0.0
    background: float = 0.0

    @property
    def fom(self) -> float:
        return float(_fom(self.signal, self.background))


def passes(table: Table, cuts: dict, skip: str | None = None) -> np.ndarray:
    """Rows inside every cut but `skip`'s."""
    keep = np.ones(len(table.signal), dtype=bool)
    for name, (lo, hi) in cuts.items():
        if name == skip:
            continue
        v = table.values[name]
        keep &= (v > lo) & (v < hi)
    return keep


def evaluate(table: Table, cuts: dict) -> Selection:
    m = passes(table, cuts)
    return Selection(dict(cuts), float(table.signal[m].sum()),
                     float(table.background[m].sum()))


def open_cuts(variables) -> dict:
    """No cut: an edge the search has not moved is infinite, so a variable
    cuts nothing until an edge is placed (the macro's Var_t.applied). Once a
    two-sided variable's edge moves, its other edge sits at the range end;
    a one-sided variable's other edge stays infinite."""
    return {v.name: (-np.inf, np.inf) for v in variables}


def _edge_candidates(var: Variable, x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Where an edge of `var` may sit: half-integers for an integer variable,
    else just past each signal value (so the background beyond a signal
    event is cut for the same signal loss), inside the variable's range."""
    if var.discrete:
        cand = np.arange(np.ceil(var.low - 0.5), np.floor(var.high + 0.5) + 1) + 0.5
        cand = cand[(cand >= var.low) & (cand <= var.high)]
        return cand
    sig = np.unique(x[s > 0])
    return sig[(sig > var.low) & (sig < var.high)]


def _inside(table: Table, cuts: dict) -> dict:
    """Each cut's own pass mask."""
    return {name: (table.values[name] > lo) & (table.values[name] < hi)
            for name, (lo, hi) in cuts.items()}


def _fails(masks: dict) -> np.ndarray:
    """How many cuts each row fails: a row passes every cut but v's when its
    count equals v's own failure (0 or 1)."""
    n = np.zeros(len(next(iter(masks.values()))), dtype=np.int16)
    for m in masks.values():
        n += ~m
    return n


def _scan(table: Table, cuts: dict, var: Variable, upper: bool,
          masks: dict | None = None, fails: np.ndarray | None = None):
    """Yields with `var`'s lower (upper=False) or upper edge moved to every
    candidate, the other cuts fixed. Returns (edges, signal, background),
    edges ordered from loosest to tightest."""
    lo, hi = cuts[var.name]
    if masks is None:
        masks = _inside(table, cuts)
        fails = _fails(masks)
    own_fail = ~masks[var.name]
    others = (fails - own_fail) == 0
    order = table.order[var.name]
    x_all = table.values[var.name]
    # rows passing the other cuts and the fixed edge, in ascending x
    keep = others & ((x_all < hi) if not upper else (x_all > lo))
    idx = order[keep[order]]
    xs, ss, bs = x_all[idx], table.signal[idx], table.background[idx]
    if var.discrete:
        cand = _edge_candidates(var, xs, ss)
    else:
        # just below (lower edge) or above (upper edge) each signal value
        vals = _edge_candidates(var, xs, ss)
        eps = 1e-9 * max(1.0, float(np.max(np.abs(vals)))) if vals.size else 0.0
        cand = vals - eps if not upper else vals + eps
    if upper:
        cand = cand[cand < hi]
        cand = np.concatenate([[var.high], np.sort(cand)[::-1]])
    else:
        cand = cand[cand > lo]
        cand = np.concatenate([[var.low], np.sort(cand)])
    # Kept yields are sums running from the kept side's far end, never a
    # total minus a prefix: weights here span many orders of magnitude (the
    # DIO stand-ins against cosmics), and cancellation would lose the small.
    zero = np.zeros(1)
    if upper:   # keep x < y: the rows below, summed from the bottom
        cs = np.concatenate([zero, np.cumsum(ss)])
        cb = np.concatenate([zero, np.cumsum(bs)])
        i = np.searchsorted(xs, cand, side="left")
    else:       # keep x > y: the rows above, summed from the top
        cs = np.concatenate([np.cumsum(ss[::-1])[::-1], zero])
        cb = np.concatenate([np.cumsum(bs[::-1])[::-1], zero])
        i = np.searchsorted(xs, cand, side="right")
    return cand, cs[i], cb[i]


def _moves(table: Table, cuts: dict, variables, tighten: bool, step: float,
           n_sig_ref: float):
    """One move per edge: tighten (or loosen) it by about `step` of
    `n_sig_ref` signal. Returns [(name, (lo, hi), s, b)]."""
    masks = _inside(table, cuts)
    fails = _fails(masks)
    passing = fails == 0
    current_signal = float(table.signal[passing].sum())
    out = []
    for var in variables:
        for upper in (False, True):
            if (var.side == "low" and upper) or (var.side == "high" and not upper):
                continue
            lo, hi = cuts[var.name]
            if tighten:
                edges, sig, bkg = _scan(table, cuts, var, upper, masks, fails)
                cur = lo if not upper else hi
                ok = ((edges > lo) & (edges < hi)) & ~np.isclose(edges, cur)
                if not ok.any():
                    continue
                e, s, b = edges[ok], sig[ok], bkg[ok]
                # the largest move losing at most a step, else the first;
                # a move losing nothing is taken one further (macro choose_edge)
                loss = (current_signal - s) / n_sig_ref
                within = np.nonzero(loss <= step)[0]
                i = within[-1] if within.size else 0
                if s[i] >= current_signal and i + 1 < e.size:
                    i += 1
            else:
                # loosen from the variable's open range back to the edge
                wide = dict(cuts)
                wide[var.name] = (var.low, hi) if not upper else (lo, var.high)
                wmasks = dict(masks)
                wmasks[var.name] = ((table.values[var.name] > wide[var.name][0])
                                    & (table.values[var.name] < wide[var.name][1]))
                edges, sig, bkg = _scan(table, wide, var, upper, wmasks,
                                        fails - (~masks[var.name]) + (~wmasks[var.name]))
                cur = lo if not upper else hi
                ok = ((edges < cur) if not upper else (edges > cur)) & ~np.isclose(edges, cur)
                if not ok.any():
                    continue
                e, s, b = edges[ok], sig[ok], bkg[ok]
                # e runs loosest first. Gain at least a step: the tightest
                # such edge, else the loosest (macro find_loose_point)
                gain = (s - current_signal) / n_sig_ref
                enough = np.nonzero(gain >= step)[0]
                i = enough[-1] if enough.size else 0
                if not gain[i] > 0:
                    continue
            new = (float(e[i]), hi) if not upper else (lo, float(e[i]))
            if var.side == "both" and not np.all(np.isfinite(new)):
                # first placed edge of a two-sided variable: its other edge
                # comes in to the range end (macro apply_point), so recount
                new = (new[0] if np.isfinite(new[0]) else var.low,
                       new[1] if np.isfinite(new[1]) else var.high)
                if not new[0] < new[1]:
                    continue
                placed = dict(cuts)
                placed[var.name] = new
                sel = evaluate(table, placed)
                out.append((var.name, new, sel.signal, sel.background))
                continue
            out.append((var.name, new, float(s[i]), float(b[i])))
    return out


@dataclass
class Result:
    """The chosen selection on the train half, its path, and the test half's
    yields for the seed and the chosen cuts."""
    best: Selection
    seed_train: Selection
    seed_test: Selection
    best_test: Selection
    path: list = field(default_factory=list)
    starts: list = field(default_factory=list)

    @property
    def gain_test(self) -> float:
        return (self.best_test.fom / self.seed_test.fom
                if self.seed_test.fom > 0 else float("nan"))


def greedy_path(table: Table, variables, cuts: dict, step: float,
                min_efficiency: float, rng=None, top_k: int = 1,
                max_steps: int = 10000) -> list:
    """Tighten one edge at a time, the one with the largest FoM gain per
    signal lost (a random one of the top_k with rng), until the signal falls
    below min_efficiency of the start's."""
    sel = evaluate(table, cuts)
    n_ref = sel.signal
    path = [sel]
    if n_ref <= 0:
        return path
    for _ in range(max_steps):
        cands = []
        # a step is a fraction of the current signal, as in the macro
        for name, new, s, b in _moves(table, sel.cuts, variables, True, step, sel.signal):
            lost = sel.signal - s
            if lost < 0:
                continue
            gain = float(_fom(s, b)) - sel.fom
            rate = gain / lost if lost > 0 else (np.inf if gain >= 0 else -np.inf)
            cands.append((rate, name, new, s, b))
        if not cands:
            break
        cands.sort(key=lambda c: c[0], reverse=True)
        pick = cands[0] if rng is None else cands[rng.integers(min(top_k, len(cands)))]
        _, name, new, s, b = pick
        cuts_next = dict(sel.cuts)
        cuts_next[name] = new
        sel = Selection(cuts_next, s, b)
        path.append(sel)
        if s < min_efficiency * n_ref or s <= 0:
            break
    return path


def best_on_path(path: list) -> Selection:
    return max(path, key=lambda s: s.fom)


def refine(table: Table, variables, sel: Selection, step: float,
           max_iter: int = 200) -> Selection:
    """Local search: tighten any edge by a step, loosen any by a step, or
    loosen one and then take the greedy tightening; keep the largest
    improvement, until none."""
    sel = evaluate(table, sel.cuts)
    for _ in range(max_iter):
        n_ref = max(sel.signal, 1e-300)
        best = sel
        trials = []
        for tighten in (True, False):
            for name, new, s, b in _moves(table, sel.cuts, variables, tighten, step, n_ref):
                c = dict(sel.cuts)
                c[name] = new
                trials.append(Selection(c, s, b))
        for name, new, _, _ in _moves(table, sel.cuts, variables, False, step, n_ref):
            c = dict(sel.cuts)
            c[name] = new
            loose = evaluate(table, c)
            path = greedy_path(table, variables, loose.cuts, step, 0.0, max_steps=1)
            if len(path) > 1:
                trials.append(path[1])
        for t in trials:
            if t.fom > best.fom * (1 + 1e-9):
                best = t
        if best is sel:
            break
        sel = best
    return sel


def optimize(table: Table, variables, seed_cuts: dict, step: float = 0.01,
             min_efficiency: float = 0.3, n_random: int = 5, top_k: int = 3,
             seed: int = 20261116, fom: str = "punzi",
             extra_starts: dict | None = None) -> Result:
    """The best selection on the train half from `seed_cuts`: the greedy
    path's best and n_random randomized paths' bests, each refined; the
    one with the largest train FoM wins. Evaluated on the test half.
    `fom`: "punzi" (default) or "s_over_sqrt_b" (to compare with the macro).
    `extra_starts`: {label: cuts}, more starting points for the local search."""
    global _fom
    if fom not in FOMS:
        raise OptimizeError(f"fom must be one of {', '.join(FOMS)}, got {fom!r}")
    previous, _fom = _fom, FOMS[fom]
    try:
        return _optimize(table, variables, seed_cuts, step, min_efficiency,
                         n_random, top_k, seed, extra_starts)
    finally:
        _fom = previous


def _optimize(table, variables, seed_cuts, step, min_efficiency, n_random,
              top_k, seed, extra_starts=None) -> Result:
    names = {v.name for v in variables}
    missing = names - set(table.values)
    if missing:
        raise OptimizeError(f"no column for {', '.join(sorted(missing))}")
    cuts = open_cuts(variables)
    for name, (lo, hi) in seed_cuts.items():
        if name not in names:
            raise OptimizeError(f"seed cut on {name}, which is not a variable")
        cuts[name] = (lo, hi)
    train, test = table.half(True), table.half(False)
    seed_train = evaluate(train, cuts)
    if seed_train.signal <= 0:
        raise OptimizeError("no signal passes the seed cuts in the train half")
    # The paths start from no cut, as the macro's do: a path from the seed
    # only tightens, and the local search loosens a step at a time, so a
    # seed tighter than the optimum on some variable would trap them. The
    # seed (and any extra_starts) join as starting points of the local search.
    rng = np.random.default_rng(seed)
    nothing = open_cuts(variables)
    path = greedy_path(train, variables, nothing, step, min_efficiency)
    starts = [("greedy", best_on_path(path)), ("seed", seed_train)]
    for label, extra in (extra_starts or {}).items():
        c = open_cuts(variables)
        c.update(extra)
        starts.append((label, evaluate(train, c)))
    for i in range(n_random):
        p = greedy_path(train, variables, nothing, step, min_efficiency, rng, top_k)
        starts.append((f"random{i}", best_on_path(p)))
    refined = [(label, refine(train, variables, s, step)) for label, s in starts]
    label, best = max(refined, key=lambda r: r[1].fom)
    return Result(best=best, seed_train=seed_train,
                  seed_test=evaluate(test, cuts), best_test=evaluate(test, best.cuts),
                  path=path, starts=[(lbl, s.fom) for lbl, s in refined])
