#!/usr/bin/env python3
"""Extract cosmic-ray electron background shapes for cut optimization.

The cosmic rate in tools/analyses/approx_ce_sensitivity.py was derived with one
fixed set of cuts (cos, R(max), t0).  To optimize those cuts we need the cosmic
shapes in the cut variables, and we need to know which of them may be treated
as independent.  This script reads CRY cosmics reconstructed as downstream e-
tracks (ConvAna set 81), selects a particle type and momentum window, and writes

  * <outdir>/<name>.root   weighted histograms (raw weighted counts, sumw2
                           stored; consumers normalize).  Under/overflow are
                           kept in the uproot flow bins, so the sum over all
                           bins, flow included, is the sum of weights of the
                           selected tracks, and an efficiency of a cut that
                           reaches outside the plotted range is not lost.
                           Also a `tracks` tree of the selected tracks (weight,
                           p, t0, cos, rmax, d0, rc, radius): the cut
                           efficiencies do not factorize even in (t0, rmax,
                           cos), so a cut optimization should count tracks
                           passing all cuts rather than multiply 1D shapes.
  * <outdir>/<name>.json   selection, B_eff, separability table, reference-cut
                           efficiencies and a recommendation.
  * <outdir>/figures/*.png shapes, heat maps, conditional overlays, pull maps.

Axis convention for the 2D histograms: histogram "A_vs_B" has x = A (the
first-named variable) and y = B (cos).  In uproot terms values()[ix, iy].

Geometry (Mu2eEvtAna Track_t.hh): d0 = rmax - 2R (signed), so the helix-center
radius is rc = (rmax + d0)/2 and the helix radius is R = (rmax - d0)/2.

Run with the `ana` python (uproot, awkward, numpy, scipy, matplotlib; no ROOT
and no `hist` package, so histograms are written with uproot's TH1x/TH2x
helpers).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot
from scipy.stats import rankdata, spearmanr
from uproot.writing.identify import to_TAxis, to_TH1x, to_TH2x

# Shared project output of the ConvAna run over the CRY cosmics sample.
DEFAULT_INPUT = ("/exp/mu2e/data/projects/run1a/mumep_ana/histograms/"
                 "ConvAna.cnv_ana.cry4ab1s5r0101.m1.root")
DEFAULT_TREE = "Ana/Hist/trs_81/tree"
PMIN, PMAX = 100.0, 110.0          # MeV/c, the signal-region momentum window
# Convert pT [MeV/c] to a radius [mm] at 1 T: R = pT / (C_LIGHT * B).
C_LIGHT = 0.299792458              # MeV/(mm T)
# Reference Run-1A cuts the default cosmic rate in approx_ce_sensitivity used.
REF_COS = (0.525, 0.649)
REF_RMAX = (482.0, 642.6)          # mm
REF_T0 = (540.0, 1650.0)           # ns
# Binned mutual information: 30x30 quantile bins; bias estimated by shuffling.
MI_BINS = 30
MI_SHUFFLES = 5
FACTORIZATION_TOLERANCE = 0.10     # a product of efficiencies counts as valid within this
MIN_EXPECTED = 5.0                 # chi2 uses only bins with expected > this
SEED = 12345
# Cos slices for the conditional-shape overlays.
COS_SLICES = [0.0, 0.5, 0.7, 0.85, 0.95, 1.0]

# name -> (nbins, lo, hi, axis title).  The 2D binnings are coarser, ~30x25.
AXES = {
    "t0": (130, 400.0, 1700.0, "track t0 at tracker front [ns]"),
    "cos": (50, 0.0, 1.0, "cos#theta = pz/p"),
    "rc": (60, 0.0, 900.0, "helix-center radius rc [mm]"),
    "rmax": (80, 300.0, 1100.0, "R(max) = rc + R [mm]"),
    "d0": (70, -700.0, 700.0, "d0 = R(max) - 2R [mm]"),
    "radius": (70, 0.0, 700.0, "helix radius R [mm]"),
}
AXES_2D = {  # name -> (x var, y var, nx, ny)
    "rc_vs_cos": ("rc", "cos", 30, 25),
    "rmax_vs_cos": ("rmax", "cos", 32, 25),
    "d0_vs_cos": ("d0", "cos", 28, 25),
    "t0_vs_cos": ("t0", "cos", 26, 25),
}
# Per-track variables stored in the `tracks` tree ("w" is written as "weight").
TRACK_BRANCHES = ("w", "p", "t0", "cos", "rmax", "d0", "rc", "radius")
PAIRS = [("rc", "cos"), ("rmax", "cos"), ("d0", "cos"), ("rc", "radius"),
         ("rmax", "d0"), ("t0", "cos"), ("t0", "rc")]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--tree", default=DEFAULT_TREE)
    ap.add_argument("--outdir", default="output/cosmic_shapes")
    ap.add_argument("--name", default="cosmic_shapes", help="output basename")
    ap.add_argument("--particles", choices=("electron", "all"), default="electron",
                    help="electron keeps |trk_mc_pdg| == 11; all keeps every track")
    ap.add_argument("--pmin", type=float, default=PMIN)
    ap.add_argument("--pmax", type=float, default=PMAX)
    return ap.parse_args(argv)


def load_selected(args):
    """Read the tree and return (variables dict, selection string).

    Derived quantities are computed here so the conventions live in one place.
    """
    branches = ["weight", "trk_p", "trk_t0", "trk_d0", "trk_rmax", "trk_cos", "trk_mc_pdg"]
    with uproot.open(args.input) as f:
        a = f[args.tree].arrays(branches, library="np")
    keep = (a["trk_p"] > args.pmin) & (a["trk_p"] < args.pmax)
    sel = f"{args.pmin:g} < trk_p < {args.pmax:g} MeV/c"
    if args.particles == "electron":
        keep &= np.abs(a["trk_mc_pdg"]) == 11
        sel = "|trk_mc_pdg| == 11, " + sel
    else:
        sel = "all particles, " + sel
    d = {k: a[k][keep].astype(float) for k in a}
    rmax, d0 = d["trk_rmax"], d["trk_d0"]
    return {
        "w": d["weight"], "p": d["trk_p"], "t0": d["trk_t0"], "cos": d["trk_cos"],
        "rmax": rmax, "d0": d0, "rc": 0.5 * (rmax + d0), "radius": 0.5 * (rmax - d0),
    }, sel


def edges_of(name):
    n, lo, hi, _ = AXES[name]
    return np.linspace(lo, hi, n + 1)


def hist1d(x, w, edges):
    """Weighted histogram with flow: returns (sumw, sumw2), length nbins+2."""
    idx = np.clip(np.digitize(x, edges), 0, len(edges))  # 0 = under, nbins+1 = over
    n = len(edges) + 1
    return (np.bincount(idx, weights=w, minlength=n),
            np.bincount(idx, weights=w * w, minlength=n))


def hist2d(x, y, w, ex, ey):
    """Weighted 2D histogram with flow: (sumw, sumw2), shape (nx+2, ny+2)."""
    ix = np.clip(np.digitize(x, ex), 0, len(ex))
    iy = np.clip(np.digitize(y, ey), 0, len(ey))
    nx, ny = len(ex) + 1, len(ey) + 1
    flat = ix * ny + iy
    return (np.bincount(flat, weights=w, minlength=nx * ny).reshape(nx, ny),
            np.bincount(flat, weights=w * w, minlength=nx * ny).reshape(nx, ny))


def th1(name, title, xtitle, edges, sumw, sumw2, x, w):
    """Build a TH1D (with flow bins and sumw2) from a binned result."""
    ax = to_TAxis("xaxis", xtitle, len(edges) - 1, edges[0], edges[-1])
    inr = (x >= edges[0]) & (x < edges[-1])
    xi, wi = x[inr], w[inr]
    return to_TH1x(name, title, sumw, float(len(x)), float(w.sum()), float((w * w).sum()),
                   float((wi * xi).sum()), float((wi * xi * xi).sum()), sumw2, ax)


def th2(name, title, xt, yt, ex, ey, sumw, sumw2, x, y, w):
    """Build a TH2D (with flow bins and sumw2); the data array is y-major."""
    axx = to_TAxis("xaxis", xt, len(ex) - 1, ex[0], ex[-1])
    axy = to_TAxis("yaxis", yt, len(ey) - 1, ey[0], ey[-1])
    inr = (x >= ex[0]) & (x < ex[-1]) & (y >= ey[0]) & (y < ey[-1])
    xi, yi, wi = x[inr], y[inr], w[inr]
    return to_TH2x(name, title, sumw.T.ravel(), float(len(x)), float(w.sum()),
                   float((w * w).sum()), float((wi * xi).sum()), float((wi * xi * xi).sum()),
                   float((wi * yi).sum()), float((wi * yi * yi).sum()),
                   float((wi * xi * yi).sum()), sumw2.T.ravel(), axx, axy)


def write_root(path, v, sel):
    """Write all 1D and 2D histograms; return the in-memory contents for plots."""
    w = v["w"]
    out1, out2 = {}, {}
    with uproot.recreate(path) as f:
        for name, (n, lo, hi, xt) in AXES.items():
            e = edges_of(name)
            sw, sw2 = hist1d(v[name], w, e)
            f[name] = th1(name, sel, xt, e, sw, sw2, v[name], w)
            out1[name] = (e, sw, sw2)
        lo, hi = np.floor(PMIN_USED[0] / 0.5) * 0.5, np.ceil(PMIN_USED[1] / 0.5) * 0.5
        e = np.linspace(lo, hi, int(round((hi - lo) / 0.5)) + 1)
        sw, sw2 = hist1d(v["p"], w, e)
        f["p"] = th1("p", sel, "track momentum p [MeV/c]", e, sw, sw2, v["p"], w)
        out1["p"] = (e, sw, sw2)
        for name, (xv, yv, nx, ny) in AXES_2D.items():
            ex = np.linspace(AXES[xv][1], AXES[xv][2], nx + 1)
            ey = np.linspace(AXES[yv][1], AXES[yv][2], ny + 1)
            sw, sw2 = hist2d(v[xv], v[yv], w, ex, ey)
            f[name] = th2(name, sel, AXES[xv][3], AXES[yv][3], ex, ey, sw, sw2,
                          v[xv], v[yv], w)
            out2[name] = (ex, ey, sw, sw2)
        tracks = {("weight" if k == "w" else k): v[k] for k in TRACK_BRANCHES}
        f.mktree("tracks", {k: np.float32 for k in tracks}, title=sel).extend(
            {k: x.astype(np.float32) for k, x in tracks.items()})
    return out1, out2


def binned_mi(x, y, nb=MI_BINS):
    """Plug-in mutual information (nats) on nb x nb equal-count (rank) bins."""
    n = len(x)
    bx = (rankdata(x, method="ordinal") - 1) * nb // n
    by = (rankdata(y, method="ordinal") - 1) * nb // n
    pxy = np.bincount((bx * nb + by).astype(int), minlength=nb * nb).reshape(nb, nb) / n
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    return float((pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])).sum())


def chi2_vs_marginals(sw):
    """chi2/ndf of an in-range 2D histogram against the outer product of its marginals.

    Only bins with expected > MIN_EXPECTED count.  ndf = bins used - (nx + ny - 1)
    fitted marginal parameters, floored at 1.
    """
    h = sw[1:-1, 1:-1]
    tot = h.sum()
    exp = np.outer(h.sum(1), h.sum(0)) / tot
    use = exp > MIN_EXPECTED
    chi2 = float((((h - exp) ** 2 / np.where(use, exp, 1))[use]).sum())
    ndf = max(int(use.sum()) - (h.shape[0] + h.shape[1] - 1), 1)
    return chi2, ndf, exp, h


def separability(v, rng):
    """Spearman rho, binned MI with its shuffle bias, and (when stored) chi2/ndf."""
    table = {}
    for a, b in PAIRS:
        x, y = v[a], v[b]
        mi = binned_mi(x, y)
        bias = [binned_mi(x, rng.permutation(y)) for _ in range(MI_SHUFFLES)]
        row = {"spearman_rho": float(spearmanr(x, y)[0]), "mi_nats": mi,
               "mi_bias_nats": float(np.mean(bias)), "mi_bias_std": float(np.std(bias)),
               "mi_minus_bias": mi - float(np.mean(bias))}
        table[f"{a},{b}"] = row
    return table


def window_eff(x, w, lo, hi):
    return float(w[(x > lo) & (x < hi)].sum() / w.sum())


def reference_cuts(v):
    """Efficiency of each Run-1A cut, all together, and the factorized products.

    Also gives the (cos, rmax) pair kept jointly times a 1D t0 efficiency, to show
    which part of the factorization breaks.
    """
    w = v["w"]
    cuts = {"cos": REF_COS, "rmax": REF_RMAX, "t0": REF_T0}
    ind = {k: window_eff(v[k], w, *c) for k, c in cuts.items()}
    pass_ = {k: (v[k] > lo) & (v[k] < hi) for k, (lo, hi) in cuts.items()}
    comb = float(w[pass_["cos"] & pass_["rmax"] & pass_["t0"]].sum() / w.sum())
    joint = float(w[pass_["cos"] & pass_["rmax"]].sum() / w.sum())
    fac = float(np.prod(list(ind.values())))
    return {"cuts": {k: list(c) for k, c in cuts.items()}, "individual": ind,
            "combined": comb, "factorized": fac,
            "factorization_ratio": fac / comb if comb > 0 else None,
            "cos_rmax_joint": joint, "cos_rmax_joint_times_t0": joint * ind["t0"],
            "rmax_eff_given_cos_cut": joint / ind["cos"] if ind["cos"] > 0 else None}


def effective_field(v):
    """B_eff = median of pT/(C_LIGHT R) with 16/84 percentiles.

    pT = p sin(theta) = p sqrt(1 - cos^2); B_eff differs from the nominal 1 T
    because of the field map and the fit's R at the tracker front.
    """
    pt = v["p"] * np.sqrt(np.clip(1 - v["cos"] ** 2, 0, None))
    b = pt / (C_LIGHT * v["radius"])
    b = b[np.isfinite(b) & (v["radius"] > 0)]
    q = np.percentile(b, [16, 50, 84])
    return {"median_T": float(q[1]), "p16_T": float(q[0]), "p84_T": float(q[2])}


def out_of_range(v):
    """Fraction (unweighted entries) outside each 1D histogram range."""
    res = {}
    for name in list(AXES) + ["p"]:
        e = edges_of(name) if name in AXES else None
        lo, hi = (e[0], e[-1]) if e is not None else PMIN_USED
        x = v[name]
        res[name] = {"below": float((x < lo).mean()), "above": float((x >= hi).mean())}
    return res


def recommend(sep, ref):
    """State which factorization holds, from the numbers rather than a fixed text.

    A small Spearman rho or mutual information does not guarantee that cut
    efficiencies factorize: rmax is non-monotonic in cos, so a window in cos
    changes the rmax shape.  The test is the product against the true combined
    efficiency.
    """
    tol = FACTORIZATION_TOLERANCE
    full = ref["factorized"] / ref["combined"]
    pair = ref["cos_rmax_joint_times_t0"] / ref["combined"]
    s = (f"rc-cos is correlated (Spearman {sep['rc,cos']['spearman_rho']:+.2f}) and d0-cos "
         f"strongly so: never factorize them. ")
    if abs(full - 1) <= tol:
        s += (f"eff(t0)*eff(rmax)*eff(cos) reproduces the combined reference-cut "
              f"efficiency (ratio {full:.2f}) and is valid. ")
    else:
        s += (f"eff(t0)*eff(rmax)*eff(cos) does NOT reproduce the combined reference-cut "
              f"efficiency (product/combined = {full:.2f}), although the rmax-cos Spearman rho "
              f"is only {sep['rmax,cos']['spearman_rho']:+.2f}: the rmax shape depends on cos "
              f"inside the cut window. ")
    if abs(pair - 1) <= tol:
        s += (f"Use the joint (rmax, cos) shape (rmax_vs_cos) times a 1D t0 efficiency "
              f"(ratio {pair:.2f}); t0 is independent of cos and rc.")
    else:
        s += f"Even joint(rmax, cos)*eff(t0) is off (ratio {pair:.2f}): use the full event list."
    return s


def norm(sw, edges):
    """Density-normalized in-range histogram (area 1) for plotting."""
    h = sw[1:-1]
    return h / max(h.sum() * np.diff(edges)[0], 1e-300)


def make_plots(v, h1, h2, sel, fdir, name):
    """Write the PNGs: 1D shapes, 2D maps, conditional overlays, pull maps."""
    fdir.mkdir(parents=True, exist_ok=True)
    xt = {k: AXES[k][3] for k in AXES}
    xt["p"] = "track momentum p [MeV/c]"

    def save(fig, plot):
        fig.savefig(fdir / f"{name}_{plot}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    for key, (e, sw, _) in h1.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.stairs(norm(sw, e), e, color="C0")
        ax.set(xlabel=xt[key], ylabel="normalized entries per unit", title=sel)
        ax.title.set_fontsize(9)
        save(fig, f"shape_{key}")

    for key in ("rc_vs_cos", "rmax_vs_cos"):
        ex, ey, sw, _ = h2[key]
        xv, yv = AXES_2D[key][:2]
        fig, ax = plt.subplots(figsize=(6.5, 5))
        m = ax.pcolormesh(ex, ey, sw[1:-1, 1:-1].T, cmap="viridis")
        fig.colorbar(m, label="weighted entries")
        ax.set(xlabel=xt[xv], ylabel=xt[yv], title=sel)
        ax.title.set_fontsize(9)
        save(fig, f"map_{key}")

        chi2, ndf, exp, obs = chi2_vs_marginals(sw)
        pull = np.where(exp > MIN_EXPECTED, (obs - exp) / np.sqrt(np.maximum(exp, 1e-12)), np.nan)
        fig, ax = plt.subplots(figsize=(6.5, 5))
        m = ax.pcolormesh(ex, ey, pull.T, cmap="RdBu_r", vmin=-5, vmax=5)
        fig.colorbar(m, label="(obs - expected)/sqrt(expected)")
        ax.set(xlabel=xt[xv], ylabel=xt[yv],
               title=f"{sel}\npull vs product of marginals, chi2/ndf = {chi2:.0f}/{ndf}")
        ax.title.set_fontsize(9)
        save(fig, f"pull_{key}")

    cos = v["cos"]
    for var in ("rc", "rmax", "t0"):
        e = edges_of(var)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for lo, hi in zip(COS_SLICES[:-1], COS_SLICES[1:]):
            m = (cos >= lo) & (cos < hi if hi < 1 else cos <= hi)
            if m.sum() < 10:
                continue
            sw, _ = hist1d(v[var][m], v["w"][m], e)
            ax.stairs(norm(sw, e), e, label=f"{lo:g} < cos < {hi:g} (n={m.sum()})")
        ax.set(xlabel=xt[var], ylabel="normalized entries per unit", title=sel)
        ax.title.set_fontsize(9)
        ax.legend(fontsize=7)
        save(fig, f"cond_{var}_in_cos_slices")


def verify(path, h1, h2, v):
    """Re-open the ROOT file; check every histogram's shape and sum of weights."""
    total, ok = float(v["w"].sum()), True
    with uproot.open(path) as f:
        for name, (e, sw, _) in list(h1.items()) + [(k, (a[0], a[2], a[3])) for k, a in h2.items()]:
            h = f[name]
            vals, flow = h.values(flow=True), h.values(flow=False)
            shape_ok = flow.shape == ((len(e) - 1,) if name in h1 else
                                      (len(h2[name][0]) - 1, len(h2[name][1]) - 1))
            sum_all = float(vals.sum())
            good = shape_ok and abs(sum_all - total) < 1e-6 * total and h.variances(flow=True) is not None
            ok &= good
            print(f"  {name:12s} shape={flow.shape} in-range={flow.sum():.0f} "
                  f"with-flow={sum_all:.0f} (selected {total:.0f}) {'ok' if good else 'FAIL'}")
    return ok


PMIN_USED = (PMIN, PMAX)


def main(argv=None):
    global PMIN_USED
    args = parse_args(argv)
    PMIN_USED = (args.pmin, args.pmax)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    v, sel = load_selected(args)
    if len(v["w"]) == 0:
        sys.exit("No tracks pass the selection: check --particles and the momentum window.")
    rng = np.random.default_rng(SEED)

    root_path = outdir / f"{args.name}.root"
    h1, h2 = write_root(root_path, v, sel)
    sep = separability(v, rng)
    for k, (xv, yv, _, _) in AXES_2D.items():
        chi2, ndf, _, _ = chi2_vs_marginals(h2[k][2])
        pair = f"{xv},{yv}"
        if pair in sep:
            sep[pair].update(chi2=chi2, ndf=ndf, chi2_per_ndf=chi2 / ndf)
    # The chi2 is only defined for pairs with a stored 2D histogram; for the rest
    # build it on the fly with the same binning rule (30x25 over the 1D ranges).
    for a, b in PAIRS:
        row = sep[f"{a},{b}"]
        if "chi2" not in row:
            ex = np.linspace(AXES[a][1], AXES[a][2], 31)
            ey = np.linspace(AXES[b][1], AXES[b][2], 26)
            chi2, ndf, _, _ = chi2_vs_marginals(hist2d(v[a], v[b], v["w"], ex, ey)[0])
            row.update(chi2=chi2, ndf=ndf, chi2_per_ndf=chi2 / ndf)
    ref = reference_cuts(v)
    summary = {
        "input": args.input, "tree": args.tree, "selection": sel,
        "n_selected": int(len(v["w"])), "sum_weights": float(v["w"].sum()),
        "B_eff": effective_field(v), "separability": sep,
        "reference_cuts": ref, "out_of_range_fraction": out_of_range(v),
        "recommendation": recommend(sep, ref),
    }
    (outdir / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    make_plots(v, h1, h2, sel, outdir / "figures", args.name)

    print(f"Selection: {sel}\nSelected {summary['n_selected']} tracks, "
          f"sum of weights {summary['sum_weights']:.1f}")
    b = summary["B_eff"]
    print(f"B_eff = {b['median_T']:.4f} T (16/84%: {b['p16_T']:.4f} / {b['p84_T']:.4f})")
    print(f"{'pair':12s} {'rho':>7s} {'MI':>7s} {'bias':>7s} {'chi2/ndf':>10s}")
    for k, r in sep.items():
        print(f"{k:12s} {r['spearman_rho']:+7.3f} {r['mi_nats']:7.3f} "
              f"{r['mi_bias_nats']:7.3f} {r['chi2_per_ndf']:10.2f}")
    print("Reference cuts: " + ", ".join(f"{k} {x:.4f}" for k, x in ref["individual"].items())
          + f"; combined {ref['combined']:.4f}, factorized {ref['factorized']:.4f}")
    print("Out of range (below/above): " + ", ".join(
        f"{k} {r['below']:.3f}/{r['above']:.3f}" for k, r in summary["out_of_range_fraction"].items()))
    print(f"Joint(cos,rmax)*eff(t0) = {ref['cos_rmax_joint_times_t0']:.4f}; "
          f"eff(rmax | cos cut) = {ref['rmax_eff_given_cos_cut']:.3f}")
    print(summary["recommendation"])
    print("Read-back check:")
    if not verify(root_path, h1, h2, v):
        sys.exit("A histogram did not read back as expected.")


if __name__ == "__main__":
    main()
