"""Total pileup occupancy: add the components of `edep_occupancy`.

Pileup is the sum of several components (the beam flashes, the muon-stop
pileup), each simulated apart and run through `edep_occupancy`. The busiest
channel of the total is NOT the sum of the components' busiest channels: they
peak in different channels. So this adds the per-channel histograms, which
`edep_occupancy` writes already scaled to one microbunch
(`occupancy_per_microbunch.root`), and finds the peaks of the sum. Per-file
scale factors are therefore not needed: each file carries its own
`events_per_microbunch`.

It reports the metrics `edep_occupancy` does, for the total. Each component's
share of the peak channels (its rate in the channel where the SUM peaks, and
that over the sum) is in the result's `components` metadata rather than in
the metrics, since the components are named by the caller.

The files must have been made with the same microbunch period and step cuts,
and with the same kind of sample for what is read: see the dose note in
`edep_occupancy` (the regular flash samples are time-filtered, so use
Early*FlashCat samples for the *_edep_* metrics).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..spec import AnalysisSpec, RunContext, RunOutcome
from . import edep_occupancy as occ

TOTAL_FILE = "occupancy_total_per_microbunch.root"
# crystals per radial bin that still count as the same geometry
GEOMETRY_ATOL = 1.0e-6


def _labels(metas: list[dict], paths: list[Path]) -> list[str]:
    """One distinct name per component: its label, else its directory's name."""
    labels: list[str] = []
    for i, (meta, path) in enumerate(zip(metas, paths)):
        base = meta.get("label") or path.parent.name or f"component{i}"
        label, n = base, 2
        while label in labels:
            label, n = f"{base}_{n}", n + 1
        labels.append(label)
    return labels


def run(context: RunContext) -> RunOutcome:
    problem = occ.region_problem(context.params["region_planes"],
                                 context.params["region_straws"])
    if problem:
        return RunOutcome(error=problem)
    region = occ.region_of(context.params)
    loaded = []
    for path in context.input_paths:
        try:
            loaded.append(occ.read_scaled_file(path))
        except occ.OccupancyError as exc:
            return RunOutcome(error=str(exc))
    metas = [m for _, m in loaded]
    for key in ("microbunch_ns", "step_cuts"):
        if any(m[key] != metas[0][key] for m in metas):
            return RunOutcome(error=(
                f"the files were made with different {key}, so their sum would "
                f"mix cuts: {[m[key] for m in metas]}. Rerun 'edep_occupancy' "
                "with the same parameters for every component"))
    labels = _labels(metas, context.input_paths)
    components = [(label, hists) for label, (hists, _) in zip(labels, loaded)]
    # The crystal counts of a merged file are scaled back to one geometry in
    # floating point (64 crystals can read 63.99999999999999), so compare them
    # to a tolerance well below one crystal.
    first = components[0][1]
    for _, hists in components[1:]:
        for name in occ.GEOMETRY_HIST_NAMES:
            if hists[name].values.shape != first[name].values.shape or not np.allclose(
                    hists[name].values, first[name].values, rtol=0.0, atol=GEOMETRY_ATOL):
                return RunOutcome(error="the files have different calorimeter "
                                        "geometry (crystals per radial bin)")
    total = dict(first)
    try:
        for _, hists in components[1:]:
            for name in total:
                if name not in occ.GEOMETRY_HIST_NAMES:
                    total[name] = total[name].plus(hists[name])
    except occ.OccupancyError as exc:
        return RunOutcome(error=(
            f"the files cannot be added channel by channel ({exc}): they were "
            "made with different detector channel counts or histogram binning. "
            "Rerun 'edep_occupancy' on the same Offline geometry for every component"))
    n_events = sum(m["n_events"] for m in metas)
    meta = {"label": "total", "microbunch_ns": metas[0]["microbunch_ns"],
            "events_per_microbunch": sum(m["events_per_microbunch"] for m in metas),
            "n_events": n_events, "step_cuts": metas[0]["step_cuts"],
            "components": [c for m in metas for c in m.get("components", [m["label"]])]}
    total_path = context.outdir / TOTAL_FILE
    context.outdir.mkdir(parents=True, exist_ok=True)
    occ.write_scaled_file(total_path, total, meta)
    metrics, pos = occ.metrics_from(total, meta, region)
    metrics["n_components"] = float(len(components))
    shares = occ.peak_shares(components, pos, meta["microbunch_ns"])
    plots = occ.make_plots(total, meta, context.outdir, title="total pileup per microbunch")
    totals = {label: {"events_per_microbunch": m["events_per_microbunch"],
                      **{k: float(h[k].values.sum()) for k in
                         ("trk_straw_occ", "trk_straw_hits", "trk_straw_edep",
                          "calo_crystal_occ", "calo_crystal_hits", "calo_crystal_edep")}}
              for label, m, (_, h) in zip(labels, metas, components)}
    extra = {"components": {label: {**totals[label], **shares[label]} for label in labels},
             "peak_channels": occ._describe_positions(pos, total),
             "error_note": occ.ERROR_NOTE}
    return RunOutcome(metrics=metrics, files=[str(total_path), *plots], extra=extra)


METRICS = (*occ.COMMON_METRICS, "n_components")


def summarize(metrics: dict[str, float]) -> str:
    return (f"{metrics['n_components']:g} components at "
            f"{metrics['events_per_microbunch']:g} events per microbunch: peak straw "
            f"{metrics['trk_peak_straw_hit_rate_khz']:.4g} kHz, peak row "
            f"{metrics['trk_peak_row_hit_rate_khz']:.4g} kHz, peak "
            f"{metrics['trk_region_planes']:g}-plane x {metrics['trk_region_straws']:g}-straw "
            f"region {metrics['trk_peak_region_hit_rate_khz']:.4g} kHz, mean straw "
            f"{metrics['trk_mean_straw_hit_rate_khz']:.4g} kHz; peak crystal "
            f"{metrics['calo_peak_crystal_id']:g} at "
            f"{metrics['calo_peak_crystal_hit_rate_khz']:.4g} kHz, disk 0 / 1 radial "
            f"peak {metrics['calo_disk0_peak_radial_hit_rate_khz']:.4g} / "
            f"{metrics['calo_disk1_peak_radial_hit_rate_khz']:.4g} kHz.")


SPEC = AnalysisSpec(
    name="edep_occupancy_combine",
    input_kind="root_file",
    combines_files=True,
    run=run,
    description=(
        "Total pileup occupancy and peak channel rates: adds the "
        "per-microbunch histograms of several 'edep_occupancy' runs "
        "(one per pileup component) channel by channel, then reports the same "
        "metrics for the sum. The peak of a sum is not the sum of the peaks. "
        "Each component's share of the peak channels is in the result's "
        "`components` metadata."
    ),
    metrics=METRICS,
    units={**occ.COMMON_UNITS},
    parameters=occ.region_params(),
    produced_by=("edep_occupancy",),
    input_hint=(
        "data_files: the occupancy_per_microbunch.root of each edep_occupancy "
        "run (one per component, already scaled to a microbunch by its "
        "events_per_microbunch), all made with the same step parameters and "
        "microbunch_ns. For dose use Early*FlashCat components, see "
        "edep_occupancy."
    ),
    summarize=summarize,
)
