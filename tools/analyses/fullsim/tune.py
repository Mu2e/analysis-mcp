"""Per-point selection tuning for fullsim_sensitivity (cut_mode="optimize").

The selection is a cut set's vetoes and requirements (the preselection:
trigger, upstream and multi-track vetoes, a calorimeter cluster, no OPA, no
TSDA, CRV) with the stopping-target cuts dropped, as M. MacKenzie's
optimize_target_selection_fast does (they assume the standard target), and
rectangular cuts on the tuned variables, chosen on the Punzi FoM by the
shared cut-set optimizer (tools/analyses/optimize.py, M. MacKenzie's port of
optimize_selection_fast.C: greedy path, randomized paths, local search),
started from no cut and from the cut set's own thresholds.

The table's rows:

- CE: the point's own conversion electrons, the first preselected track of
  each event. Signal weight: its share of the expected conversions (NPOT x
  stopped mu- per POT x captures x R_mue / events generated), in the
  momentum window.
- DIO: the same CE rows. A CE event with momentum change delta (reconstructed
  minus true) stands in for a DIO electron reconstructed at E + delta, so its
  DIO weight in a window [a, b] is the theory spectrum integrated over
  [a - delta, b - delta] times the DIO per event generated. Summed over the
  rows this is build_spectra's folded DIO; per row it lets the DIO pass or
  fail the tuned cuts with the CE it stands in for (the same first-order
  assumption as approx_ce_sensitivity's box mode). The integral is taken per
  0.1 MeV bin of reconstructed momentum and a window sums its bins, never a
  difference of cumulative sums (the spectrum spans ~18 decades).
- Cosmic: a fixed reference sample's tracks (cosmic_reference.py), flat in
  momentum over 100-110 MeV/c and scaled to the window's width, as the macro
  treats them. Normalized so the loose set-81 selection gives anakit's loose
  cosmic rate (approx_ce_sensitivity.recommended_cosmic_rate("box")).

The momentum window and the cuts alternate: the window is chosen for the
seed cuts, the cuts for that window, the window again for the cuts, and the
cuts once more. Windows and cuts are chosen on the even events and the
result is reported on the odd ones, together with the seed's on the same
half, so the gain is not the search's own optimism.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import optimize as op
from .cuts import CUT_ORDER, cut_masks, default_cuts, get_cut_set
from .eventntuple import SID_TT_FRONT, SID_TT_MID, origin_codes

# Cuts the search replaces with its own (the tuned variables), and the
# stopping-target cuts it drops.
TUNED_CUTS = ("pz_over_pt", "within_d0", "good_trkpid", "good_trkqual",
              "has_hits", "within_t0err")
ST_CUTS = ("st_boundary", "has_st")
# Set 81, the loose selection anakit's loose cosmic rate is defined for.
LOOSE_CUTS = ("has_a_track", "is_good_track", "has_trk_front_seg",
              "is_reco_electron_or_positron", "has_downstream",
              "charge_selection", "or_trigger", "upstream_veto",
              "no_multi_trk_veto")
# Cuts in the preselection beyond the loose set, kept as flags on the cosmic
# reference so it is built once for any preselection.
EXTRA_PRESELECTION = ("no_opa", "no_tsda", "no_crv_veto")
# The calorimeter-cluster requirement (good_trkpid's event part, which the
# macro applies as trk_cluster > 0), also a flag on the reference.
CALO_FLAG = "has_calo"

# The tuned variables and their search ranges (the macro's, widened where
# the cut set's seed sits near an edge). Not t0: the time window stands
# against the radiative pion captures and the beam flash, which this
# background model (DIO and cosmics) does not hold, so a search would just
# open it. It is applied to every row as the cut set's (time_window).
# side: "both", or "low" (cuts low values, keeps x > lo) or "high" (cuts
# high values, keeps x < hi), the optimizer's both / left / right.
_CUT_TYPES = {"both": op.BOTH, "low": op.LEFT, "high": op.RIGHT}


@dataclass(frozen=True)
class Variable:
    """A tuned variable and the range its edges may take. An integer-valued
    variable's range ends sit on half-integers, and so do its edges."""
    name: str
    low: float
    high: float
    side: str = "both"
    discrete: bool = False

    def cut_var(self, lo: float = -np.inf, hi: float = np.inf) -> op.CutVar:
        """The optimizer's CutVar for the cut lo < x < hi; an infinite edge
        is no cut, and an edge outside the range sits at the range end."""
        cut_type = _CUT_TYPES[self.side]
        applied = ((cut_type != op.RIGHT and np.isfinite(lo))
                   or (cut_type != op.LEFT and np.isfinite(hi)))
        clip = lambda x, end: float(min(max(x, self.low), self.high)) if np.isfinite(x) else end
        return op.CutVar(self.name, self.low, self.high,
                         cut_lo=clip(lo, self.low), cut_hi=clip(hi, self.high),
                         cut_type=cut_type, discrete=self.discrete, applied=applied)


VARIABLES = (
    Variable("cos", 0.30, 0.95),
    Variable("rmax", 300.0, 750.0),
    Variable("d0", -200.0, 200.0),
    Variable("trkqual", -1.0, 1.0, "low"),
    Variable("trkpid", -1.0, 1.0, "low"),
    Variable("t0err", -1.0, 3.0, "high"),
    Variable("nactive", 9.5, 200.5, "low", discrete=True),
)
VARIABLE_NAMES = tuple(v.name for v in VARIABLES)

# Reconstructed-momentum bins of the window scan and of the DIO weights.
P_BIN_MEV = 0.1
P_RANGE_MEV = (100.0, 106.0)
MIN_WINDOW_MEV = 0.3
# Cosmics are flat over this range, scaled to the window's width.
COSMIC_FLAT_MEV = (100.0, 110.0)


class TuneError(ValueError):
    """A problem the caller should see verbatim."""


def _tandip_to_cos(t):
    return t / np.sqrt(1.0 + t * t)


def seed_cuts(cut_set: str, time_window) -> dict:
    """The cut set's thresholds as cuts on the tuned variables (the time
    window is not one: in_time applies it)."""
    cs = get_cut_set(cut_set)
    v = {x.name: x for x in VARIABLES}
    return {
        "cos": tuple(float(_tandip_to_cos(t)) for t in cs.tandip_window),
        # rmax is not in the cut sets: left open (no cut) for the search
        "rmax": (-np.inf, np.inf),
        "d0": tuple(cs.d0_window_mm),
        "trkqual": (cs.trkqual_min, np.inf),
        "trkpid": (cs.trkpid_min, np.inf),
        "t0err": (-np.inf, cs.t0err_max_ns),
        "nactive": (cs.min_active_hits - 0.5, np.inf),
    }


def preselection(cut_set: str) -> tuple:
    """The cut set's default cuts less the tuned and stopping-target ones."""
    return optimize_preselection(default_cuts(cut_set))


def optimize_preselection(active) -> tuple:
    """The active cuts (enable_cuts and disable_cuts applied) less the tuned
    and stopping-target ones. The cosmic reference applies LOOSE_CUTS and
    carries EXTRA_PRESELECTION as flags, nothing else, so a preselection
    outside that would cut the CE and the cosmics differently: refused."""
    drop = set(TUNED_CUTS) | set(ST_CUTS)
    pre = tuple(n for n in active if n not in drop)
    missing = [n for n in LOOSE_CUTS if n not in pre]
    if missing:
        raise TuneError(
            f"disable_cuts removes {', '.join(missing)}, which the cosmic "
            "reference always applies (it holds only tracks passing set 81): "
            "keep them in optimize mode")
    unknown = [n for n in pre if n not in LOOSE_CUTS + EXTRA_PRESELECTION]
    if unknown:
        raise TuneError(
            f"enable_cuts adds {', '.join(unknown)}, which the cosmic reference "
            "has no flag for (it can follow only "
            f"{', '.join(EXTRA_PRESELECTION)} beyond set 81; the momentum and "
            "time windows are the scan and time_window): drop them in optimize mode")
    return pre


def candidate_tracks(data: dict, cuts: tuple, triggers, cut_set: str,
                     flags: tuple = (), require_calo: bool = True) -> dict:
    """The first track of each event passing `cuts` (and having a calorimeter
    cluster, with require_calo), as flat per-event arrays of the tuned
    variables, momentum, MC truth and event number; `flags` are other cuts'
    masks on that track, and has_calo is always one.
    Events whose track has more than one tracker-front segment are dropped,
    as reduce_data does."""
    import awkward as ak

    masks = cut_masks(data, "minus", triggers=triggers, cut_set=cut_set)
    missing = [n for n in tuple(cuts) + tuple(flags) if n not in masks]
    if missing:
        raise TuneError(f"cut(s) {', '.join(missing)} need branches this "
                        "EventNtuple does not have")
    pdg = data["trk"]["trk.pdg"]
    keep = ak.ones_like(pdg, dtype=bool)
    for n in cuts:
        keep = keep & masks[n]
    has_calo = ak.any(data["calo"]["caloclusters.energyDep_"] > 0, axis=-1)
    if require_calo:
        keep = keep & ak.broadcast_arrays(has_calo, pdg)[0]
    has = ak.to_numpy(ak.any(keep, axis=1))
    first = ak.argmax(keep, axis=1, keepdims=True)

    def pick(arr):          # the chosen track's entry, events with one
        return arr[first][has]

    segs = pick(data["trkfit"]["trksegs"])
    lh = pick(data["trkfit"]["trksegpars_lh"])
    sid = segs["sid"]
    at_front = sid == SID_TT_FRONT
    nfront = ak.to_numpy(ak.num(ak.firsts(segs["time"][at_front], axis=1), axis=1)
                         if len(segs) else np.zeros(0, dtype=int))

    def front(values):
        v = ak.firsts(ak.firsts(values[at_front], axis=-1), axis=-1)
        return ak.to_numpy(ak.fill_none(v, np.nan)).astype(np.float64)

    mom = segs["mom"]["fCoordinates"]
    p_seg = np.sqrt(mom["fX"] ** 2 + mom["fY"] ** 2 + mom["fZ"] ** 2)
    tandip = front(lh["tanDip"])
    out = {
        "p": front(p_seg),
        "t0": front(segs["time"]),
        "cos": _tandip_to_cos(tandip),
        "rmax": front(lh["maxr"]),
        "d0": front(lh["maxr"] - 2 * abs(lh["rad"])),
        "t0err": ak.to_numpy(ak.fill_none(ak.firsts(ak.firsts(
            lh["t0err"][sid == SID_TT_MID], axis=-1), axis=-1), np.nan)).astype(np.float64),
        "trkqual": ak.to_numpy(ak.firsts(pick(data["trk"]["trkqual.result"]))).astype(np.float64),
        "trkpid": ak.to_numpy(ak.firsts(pick(data["trk"]["trkpid.result"]))).astype(np.float64),
        "nactive": ak.to_numpy(ak.firsts(pick(data["trk"]["trk.nactive"]))).astype(np.float64),
        "event": ak.to_numpy(data["evt"]["event"][has]).astype(np.int64),
        "has_calo": ak.to_numpy(has_calo[has]).astype(bool),
    }
    sims = pick(data["trkmc"]["trkmcsim"]) if "trkmc" in data else None
    if sims is not None:
        out["origin"] = origin_codes(sims)
        birth = ak.firsts(ak.firsts(sims, axis=1), axis=1)
        b = birth["mom"]["fCoordinates"]
        out["p_true"] = ak.to_numpy(ak.fill_none(
            np.sqrt(b["fX"] ** 2 + b["fY"] ** 2 + b["fZ"] ** 2), np.nan)).astype(np.float64)
        out["mc_pdg"] = (ak.to_numpy(ak.fill_none(birth["pdg"], 0)).astype(np.int64)
                         if "pdg" in ak.fields(birth) else np.zeros(int(has.sum()), np.int64))
    for n in flags:
        out[n] = ak.to_numpy(ak.firsts(pick(masks[n]))).astype(bool)
    single = nfront == 1
    return {k: v[single] for k, v in out.items()}


def in_time(tracks: dict, time_window) -> dict:
    """The rows inside the time window at the tracker front (open)."""
    keep = (tracks["t0"] > time_window[0]) & (tracks["t0"] < time_window[1])
    return {k: v[keep] for k, v in tracks.items()}


def combine_tracks(parts: list) -> dict:
    keys = parts[0].keys()
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


# --- weights -----------------------------------------------------------------

def p_edges() -> np.ndarray:
    n = int(round((P_RANGE_MEV[1] - P_RANGE_MEV[0]) / P_BIN_MEV))
    return P_RANGE_MEV[0] + P_BIN_MEV * np.arange(n + 1)


def dio_bin_weights(delta: np.ndarray, dio_density, edges: np.ndarray) -> np.ndarray:
    """rows x bins: the DIO density integrated over [e_k - delta, e_k+1 -
    delta], from the 0.01 MeV table with the edge bins taken in part. Each
    integral is summed directly over its own few table bins."""
    vals, x0, w = dio_density.values, dio_density.xmin, dio_density.width
    n = vals.size
    out = np.zeros((delta.size, edges.size - 1))
    for k in range(edges.size - 1):
        lo = (edges[k] - delta - x0) / w          # in table-bin units
        hi = (edges[k + 1] - delta - x0) / w
        j_lo = np.floor(lo).astype(np.int64)
        j_hi = np.floor(hi).astype(np.int64)
        span = int(np.max(j_hi - j_lo)) if delta.size else 0
        acc = np.zeros(delta.size)
        for o in range(span + 1):
            j = j_lo + o
            a = np.maximum(lo, j)
            b = np.minimum(hi, j + 1)
            ok = (b > a) & (j >= 0) & (j < n)
            acc[ok] += vals[j[ok]] * (b[ok] - a[ok]) * w
        out[:, k] = acc
    return out


@dataclass
class Rows:
    """The search's rows before a window is chosen."""
    values: dict          # name -> rows (CE rows, then cosmic rows)
    event: np.ndarray
    is_ce: np.ndarray
    p_bin: np.ndarray     # CE rows' reconstructed-momentum bin, -1 outside
    signal_per_row: float # a CE row's signal weight inside the window
    dio: np.ndarray       # CE rows x bins: DIO weight per bin
    cosmic_per_mev: np.ndarray  # cosmic rows' weight per MeV of window

    def events(self, window) -> op.Events:
        """The optimizer's events for a momentum window (bin indices
        [ka, kb)): every row, with its signal and background weight there.
        Split them with .split() (even events train, odd test, each x2)."""
        ka, kb = window
        n_ce = int(self.is_ce.sum())
        s = np.zeros(self.event.size)
        b = np.zeros(self.event.size)
        in_win = (self.p_bin >= ka) & (self.p_bin < kb)
        s[:n_ce] = np.where(in_win, self.signal_per_row, 0.0)
        # summed bin by bin over the few bins of the window
        b[:n_ce] = self.dio[:, ka:kb].sum(axis=1)
        b[n_ce:] = self.cosmic_per_mev * (kb - ka) * P_BIN_MEV
        events = op.Events(vals={k: np.asarray(v, dtype=float) for k, v in self.values.items()},
                           w_sig=s, w_bkg=b,
                           event_numbers=np.asarray(self.event, dtype=np.int64))
        events.prepare()
        return events


def build_rows(ce: dict, cosmic: dict, *, signal_per_event: float,
               dio_per_event: float, cosmic_per_mev_per_row: float,
               dio_density) -> Rows:
    """CE rows (true CE with finite momenta) and cosmic rows (the reference
    tracks passing the preselection, 100-110 MeV/c)."""
    edges = p_edges()
    p, p_true = ce["p"], ce["p_true"]
    delta = p - p_true
    nb = edges.size - 1
    k = np.floor((p - edges[0]) / P_BIN_MEV).astype(np.int64)
    k = np.where((k >= 0) & (k < nb), k, -1)
    dio = dio_bin_weights(delta, dio_density, edges) * dio_per_event
    values = {n: np.concatenate([ce[n], cosmic[n]]) for n in VARIABLE_NAMES}
    n_ce, n_cos = p.size, cosmic["p"].size
    return Rows(values=values,
                event=np.concatenate([ce["event"], cosmic["event"]]),
                is_ce=np.r_[np.ones(n_ce, bool), np.zeros(n_cos, bool)],
                p_bin=k, signal_per_row=signal_per_event, dio=dio,
                cosmic_per_mev=np.full(n_cos, cosmic_per_mev_per_row))


def scan_window(rows: Rows, cuts: dict, train: bool):
    """The momentum window (bin indices [ka, kb)) with the largest Punzi FoM
    for `cuts`, on one half. Returns (ka, kb, fom)."""
    half = (rows.event % 2 == 0) == train
    passing = passes({n: v[half] for n, v in rows.values.items()}, cuts)
    n_ce = int(rows.is_ce.sum())
    ce_half, cos_half = half[:n_ce], half[n_ce:]
    ce_pass = passing[:int(ce_half.sum())]
    cos_pass = passing[int(ce_half.sum()):]
    nb = rows.dio.shape[1]
    sig_bins = np.bincount(rows.p_bin[ce_half][ce_pass & (rows.p_bin[ce_half] >= 0)],
                           minlength=nb)[:nb] * rows.signal_per_row * 2.0
    dio_bins = rows.dio[ce_half][ce_pass].sum(axis=0) * 2.0
    cos_per_mev = rows.cosmic_per_mev[cos_half][cos_pass].sum() * 2.0
    best = (0, 1, -1.0)
    min_bins = int(round(MIN_WINDOW_MEV / P_BIN_MEV))
    for ka in range(nb):
        s = d = 0.0
        for kb in range(ka + 1, nb + 1):
            s += sig_bins[kb - 1]
            d += dio_bins[kb - 1]
            if kb - ka < min_bins:
                continue
            f = float(op.punzi_fom(s, d + cos_per_mev * (kb - ka) * P_BIN_MEV))
            if f > best[2]:
                best = (ka, kb, f)
    return best


# --- the search ------------------------------------------------------------
#
# Cuts here are {name: (lo, hi)}, keeping lo < x < hi, an infinite edge no
# cut: what sensitivity.py reports. The optimizer works on CutVar lists;
# these convert, and a CutVar list converts back exactly.

def open_cuts() -> dict:
    return {v.name: (-np.inf, np.inf) for v in VARIABLES}


def to_cut_vars(cuts: dict) -> list:
    unknown = set(cuts) - set(VARIABLE_NAMES)
    if unknown:
        raise TuneError(f"cut on {', '.join(sorted(unknown))}, which is not a tuned variable")
    return [v.cut_var(*cuts.get(v.name, (-np.inf, np.inf))) for v in VARIABLES]


def from_cut_vars(cut_vars) -> dict:
    out = open_cuts()
    for c in cut_vars:
        if not c.applied:
            continue
        out[c.name] = (c.cut_lo if c.cut_type != op.RIGHT else -np.inf,
                       c.cut_hi if c.cut_type != op.LEFT else np.inf)
    return out


def passes(values: dict, cuts: dict) -> np.ndarray:
    """Rows inside every cut (a NaN fails any finite one)."""
    keep = np.ones(len(next(iter(values.values()))), dtype=bool)
    for name, (lo, hi) in cuts.items():
        v = values[name]
        if np.isfinite(lo):
            keep &= v > lo
        if np.isfinite(hi):
            keep &= v < hi
    return keep


@dataclass
class Yields:
    """Cuts and the signal and background they pass, on one half."""
    cuts: dict
    signal: float
    background: float

    @property
    def fom(self) -> float:
        return float(op.punzi_fom(self.signal, self.background))


def evaluate(events: op.Events, cuts: dict) -> Yields:
    keep = passes(events.vals, cuts)
    return Yields(dict(cuts), float(events.w_sig[keep].sum()),
                  float(events.w_bkg[keep].sum()))


@dataclass
class Tuned:
    seed_cuts: dict
    seed_window: tuple
    cuts: dict
    window: tuple
    seed_train: Yields
    seed_test: Yields
    best_train: Yields
    best_test: Yields
    rounds: list

    @property
    def gain_test(self) -> float:
        return (self.best_test.fom / self.seed_test.fom
                if self.seed_test.fom > 0 else float("nan"))

    def window_mev(self, window) -> tuple:
        e = p_edges()
        return float(e[window[0]]), float(e[window[1]])


def search(train: op.Events, seed: dict, starts: list, step: float,
           min_efficiency: float, n_random: int) -> Yields:
    """The optimizer's best selection on the train half: the greedy path from
    no cut (as the macro's: a path from the seed only tightens, and the local
    search loosens a step at a time, so a seed tighter than the optimum on
    some variable would trap it), n_random randomized paths, and the seed and
    `starts` as further starting points, each refined; Punzi FoM."""
    if evaluate(train, seed).signal <= 0:
        raise TuneError("no signal passes the seed cuts in the train half")
    result = op.multi_start_optimize(
        to_cut_vars({}), train, step, min_efficiency, op.punzi_fom,
        seeds=[to_cut_vars(seed)] + [to_cut_vars(c) for c in starts],
        n_random=n_random)
    best = result["best"]
    if best is None:
        raise TuneError("the cut search found no selection with signal in the train half")
    return Yields(from_cut_vars(best.cuts), best.n_sig, best.n_bkg)


def tune(rows: Rows, seed: dict, step: float = 0.01, min_efficiency: float = 0.3,
         n_random: int = 5, rounds: int = 2) -> Tuned:
    seed = {**open_cuts(), **seed}
    seed_window = scan_window(rows, seed, train=True)[:2]
    cuts, window = dict(seed), seed_window
    history = []
    for _ in range(rounds):
        train, _test = rows.events(window).split()
        best = search(train, seed, [cuts] if cuts != seed else [], step,
                      min_efficiency, n_random)
        cuts = best.cuts
        new_window = scan_window(rows, cuts, train=True)[:2]
        history.append({"window": window, "fom_train": best.fom})
        if new_window == window:
            break
        window = new_window
    seed_train, seed_test = rows.events(seed_window).split()
    best_train, best_test = rows.events(window).split()
    return Tuned(seed_cuts=dict(seed), seed_window=seed_window, cuts=cuts,
                 window=window,
                 seed_train=evaluate(seed_train, seed),
                 seed_test=evaluate(seed_test, seed),
                 best_train=evaluate(best_train, cuts),
                 best_test=evaluate(best_test, cuts),
                 rounds=history)
