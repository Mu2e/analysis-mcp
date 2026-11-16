"""Cut-set optimization from EventNtuple files: greedy, random grid, local
refinement, and multi-start search.

Reads EventNtuple track variables (the same ones trigger_efficiency_ntuple
offers) and optimizes a set of cut variables to maximize a figure of merit
(S/sqrt(B), Asimov significance, Punzi FoM, or 1/expected upper limit)
for signal vs. background discrimination.

Each input file is labeled as a component (e.g. ``signal``, ``DIO``,
``cosmic``, ``RPC``). One label is designated as signal; all others are
background. Each component carries its own weight, so the signal and
different backgrounds can be normalized independently.

Alternatively, when all tracks are in one file (e.g. a mixed signal+pileup
sample), ``signal_selection`` identifies signal tracks by an MC truth
expression instead of by file labels.

The optimization runs entirely in memory on flat per-track arrays, with
pre-sorted indices for fast edge scanning, as in optimize_selection_fast.C.
A train/test split by event-number parity guards against overtraining.

Outputs:
  * The optimized cut string (selection.py-compatible)
  * Greedy-path ROC curve and FoM-vs-efficiency (train and test)
  * Random grid scan overlay
  * Multi-start refinement comparison table
  * Cut flow CSV

The analysis takes EventNtuple root_file(s) with the same interface as
trigger_efficiency_ntuple, and the core optimization lives in optimize.py
(no physics or file-format knowledge).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from ..selection import SelectionError, apply_selection
from ..spec import AnalysisSpec, ParamSpec, RunContext, RunOutcome
from . import optimize as opt
from .optimize import (
    CutVar, CutVarSpecError, Events, FOM_FUNCTIONS,
    apply_seed_to_vars, build_cut_string, evaluate_yields,
    multi_start_optimize, parse_cut_var_specs, parse_seed_selections,
    random_grid_search,
)
from .trigger_efficiency_ntuple import (
    MC_VARIABLES, NtupleError, TRACK_VARIABLES,
    check_mc_names, open_tree, read_track_arrays, track_variables,
)


class OptimizeError(RuntimeError):
    """Raised for a problem the caller should see verbatim."""


# ---------------------------------------------------------------------------
# File labels and component weights
# ---------------------------------------------------------------------------

def parse_file_labels(text: str, n_files: int) -> list[str]:
    """Parse a comma-separated list of file labels, one per input file.

    Example: ``signal, DIO, DIO, cosmic, RPC_ext, RPC_int``

    An empty string means "all files are unlabeled" (use signal_selection
    mode instead). Raises OptimizeError on a count mismatch.
    """
    if not text.strip():
        return []
    labels = [l.strip() for l in text.split(",")]
    # Strip empty trailing entries from a trailing comma
    while labels and not labels[-1]:
        labels.pop()
    if len(labels) != n_files:
        raise OptimizeError(
            f"file_labels has {len(labels)} labels but there are {n_files} "
            f"input files: give one label per file, comma-separated"
        )
    for label in labels:
        if not label:
            raise OptimizeError(
                "file_labels has an empty label: every file must have a "
                "non-empty label"
            )
    return labels


def parse_component_weights(text: str) -> dict[str, float]:
    """Parse ``label:weight`` pairs, comma-separated.

    Example: ``signal:1.0, DIO:0.5, cosmic:0.01, RPC_ext:0.001``

    Returns a dict of label -> weight. Labels not listed default to 1.0
    in the caller.
    """
    if not text.strip():
        return {}
    result: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 2:
            raise OptimizeError(
                f"component_weights entry {item!r} must be 'label:weight', "
                f"e.g. 'DIO:0.5'"
            )
        label = parts[0].strip()
        try:
            weight = float(parts[1].strip())
        except ValueError:
            raise OptimizeError(
                f"component_weights: weight {parts[1]!r} for label "
                f"{label!r} is not a number"
            )
        if weight < 0:
            raise OptimizeError(
                f"component_weights: weight for {label!r} must be >= 0, "
                f"got {weight:g}"
            )
        result[label] = weight
    return result


# ---------------------------------------------------------------------------
# Reading and building the Events object
# ---------------------------------------------------------------------------

def _read_file(path: Path) -> tuple[dict[str, np.ndarray], int]:
    """Read one EventNtuple file into flat per-track variable arrays."""
    tree = open_tree(path)
    arrays = read_track_arrays(tree, path)
    variables = track_variables(arrays)
    return variables, tree.num_entries


def _build_events_labeled(variables: dict[str, np.ndarray],
                          preselection: str,
                          is_signal: bool,
                          weight: float,
                          ) -> Events:
    """Build an Events object from one labeled file's tracks.

    All tracks in this file are either signal (``is_signal=True``) or
    background, weighted by ``weight``.
    """
    public = {k: v for k, v in variables.items() if not k.startswith("_")}
    ntracks = next(iter(public.values())).size

    if preselection.strip():
        pre_mask = apply_selection(preselection, public, ntracks)
    else:
        pre_mask = np.ones(ntracks, dtype=bool)

    pre_public = {k: v[pre_mask] for k, v in public.items()}
    n_pre = int(pre_mask.sum())
    if n_pre == 0:
        return Events()

    if is_signal:
        w_sig = np.full(n_pre, weight)
        w_bkg = np.zeros(n_pre)
    else:
        w_sig = np.zeros(n_pre)
        w_bkg = np.full(n_pre, weight)

    event_numbers = pre_public.get("event", np.arange(n_pre, dtype=float))

    return Events(
        vals=pre_public,
        w_sig=w_sig,
        w_bkg=w_bkg,
        event_numbers=event_numbers.astype(float),
    )


def _build_events_selection(variables: dict[str, np.ndarray],
                            signal_selection: str,
                            preselection: str,
                            sig_weight: float, bkg_weight: float,
                            ) -> Events:
    """Build an Events object from one file, splitting signal/background
    by evaluating ``signal_selection`` on the track variables."""
    public = {k: v for k, v in variables.items() if not k.startswith("_")}
    ntracks = next(iter(public.values())).size

    if preselection.strip():
        pre_mask = apply_selection(preselection, public, ntracks)
    else:
        pre_mask = np.ones(ntracks, dtype=bool)

    pre_public = {k: v[pre_mask] for k, v in public.items()}
    n_pre = int(pre_mask.sum())
    if n_pre == 0:
        return Events()

    is_signal = apply_selection(signal_selection, pre_public, n_pre)
    w_sig = np.where(is_signal, sig_weight, 0.0)
    w_bkg = np.where(is_signal, 0.0, bkg_weight)

    event_numbers = pre_public.get("event", np.arange(n_pre, dtype=float))

    return Events(
        vals=pre_public,
        w_sig=w_sig,
        w_bkg=w_bkg,
        event_numbers=event_numbers.astype(float),
    )


def _merge_events(parts: list[Events]) -> Events:
    """Concatenate multiple Events objects into one."""
    if not parts:
        return Events()
    if len(parts) == 1:
        return parts[0]
    all_names = list(parts[0].vals.keys())
    return Events(
        vals={n: np.concatenate([p.vals[n] for p in parts]) for n in all_names},
        w_sig=np.concatenate([p.w_sig for p in parts]),
        w_bkg=np.concatenate([p.w_bkg for p in parts]),
        event_numbers=np.concatenate([p.event_numbers for p in parts]),
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _write_plots(outdir: Path, result: dict, fom_name: str) -> list[str]:
    """Write diagnostic plots. Returns paths of the written files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def save(fig, name):
        fig.tight_layout()
        path = figdir / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))

    path_data = result.get("greedy_path_data")
    grid_data = result.get("grid_data")

    if path_data:
        # ROC curve: rejection vs efficiency
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(path_data["eff_train"], path_data["rej_train"],
                "r-", lw=2, label="Greedy (train)")
        if grid_data:
            ax.scatter(grid_data["eff"], grid_data["rej"],
                       s=4, c="k", alpha=0.3, label="Random grid", zorder=0)
        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel("Background rejection")
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.set_title("ROC curve")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
        save(fig, "roc_curve.png")

        # FoM vs efficiency (train and test)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(path_data["eff_train"], path_data["fom_train"],
                "-", color="tab:green", lw=2, label=f"{fom_name} (train)")
        if "eff_test" in path_data:
            ax.plot(path_data["eff_test"], path_data["fom_test"],
                    "--", color="tab:green", lw=2, label=f"{fom_name} (test)")
        ax.plot(path_data["eff_train"], path_data["lim_train"],
                "-", color="tab:blue", lw=2, label="1/limit (train)")
        if "eff_test" in path_data:
            ax.plot(path_data["eff_test"], path_data["lim_test"],
                    "--", color="tab:blue", lw=2, label="1/limit (test)")
        if grid_data:
            ax.scatter(grid_data["eff"], grid_data["fom"],
                       s=4, c="k", alpha=0.3, label="Random grid", zorder=0)
        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel(f"Relative {fom_name} and 1/limit")
        ax.set_xlim(0, 1.1)
        ax.set_title(f"{fom_name} vs efficiency")
        ax.legend(fontsize="small", ncols=2)
        ax.grid(True, alpha=0.3)
        save(fig, "fom_vs_efficiency.png")

    return written


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def run(context: RunContext) -> RunOutcome:
    params = context.params
    outdir = context.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    extra: dict = {}

    try:
        # Parse parameters
        preselection = params["preselection"]
        fom_name = params["fom"]
        if fom_name not in FOM_FUNCTIONS:
            raise OptimizeError(
                f"fom {fom_name!r} must be one of "
                f"{', '.join(FOM_FUNCTIONS)}"
            )
        fom = FOM_FUNCTIONS[fom_name]
        n_grid = int(params["n_grid_trials"])
        n_random = int(params["n_random_starts"])
        end_eff = params["end_eff"]
        step = params["step"]
        split = params["split"] > 0.5

        extra["preselection"] = preselection
        extra["fom"] = fom_name
        extra["split"] = split

        # Determine mode: labeled files or signal_selection
        file_labels = parse_file_labels(params["file_labels"],
                                        len(context.input_paths))
        signal_label = params["signal_label"].strip()
        signal_selection = params["signal_selection"].strip()
        component_weights = parse_component_weights(
            params["component_weights"])
        use_labels = bool(file_labels)

        if use_labels:
            # Labeled-file mode
            if not signal_label:
                raise OptimizeError(
                    "file_labels is set but signal_label is empty: name the "
                    "label that identifies signal files"
                )
            if signal_label not in file_labels:
                raise OptimizeError(
                    f"signal_label {signal_label!r} does not appear in "
                    f"file_labels ({', '.join(file_labels)})"
                )
            extra["mode"] = "labeled_files"
            extra["signal_label"] = signal_label
            extra["file_labels"] = file_labels
        else:
            # signal_selection mode
            if not signal_selection:
                raise OptimizeError(
                    "either file_labels (with signal_label) or "
                    "signal_selection must be set to identify signal tracks"
                )
            extra["mode"] = "signal_selection"
            extra["signal_selection"] = signal_selection

        # Parse cut variable specifications
        cut_vars = parse_cut_var_specs(params["optimize_variables"])
        var_names = [v.name for v in cut_vars]
        extra["optimize_variables"] = [
            {"name": v.name, "type": v.cut_type, "lo": v.lo, "hi": v.hi,
             "discrete": v.discrete}
            for v in cut_vars
        ]

        # Parse seed selections
        seed_dicts = parse_seed_selections(params["seed_selections"], var_names)

        # Read EventNtuple files
        all_parts: list[Events] = []
        per_file: list[dict] = []
        component_counts: dict[str, dict] = {}  # label -> counts
        for i, path in enumerate(context.input_paths):
            variables, n_events = _read_file(path)
            check_mc_names(variables, signal_selection, preselection)
            # Check that all optimize variables are available
            available = {k for k in variables if not k.startswith("_")}
            missing = [n for n in var_names if n not in available]
            if missing:
                raise OptimizeError(
                    f"variable(s) {', '.join(missing)} not found in "
                    f"{path.name}. "
                    f"Available: {', '.join(sorted(available))}"
                )

            if use_labels:
                label = file_labels[i]
                is_signal = (label == signal_label)
                weight = component_weights.get(label, 1.0)
                ev = _build_events_labeled(variables, preselection,
                                           is_signal, weight)
                file_info: dict = {
                    "file": str(path), "label": label,
                    "weight": weight,
                    "n_events": n_events, "n_tracks": ev.size,
                }
                # Track per-component counts
                if label not in component_counts:
                    component_counts[label] = {
                        "n_tracks": 0, "weight": weight,
                        "is_signal": is_signal,
                    }
                component_counts[label]["n_tracks"] += ev.size
            else:
                sig_w = component_weights.get("signal", 1.0)
                bkg_w = component_weights.get("background", 1.0)
                ev = _build_events_selection(variables, signal_selection,
                                             preselection, sig_w, bkg_w)
                file_info = {
                    "file": str(path), "n_events": n_events,
                    "n_tracks": ev.size,
                    "n_signal": int((ev.w_sig > 0).sum()),
                    "n_background": int((ev.w_bkg > 0).sum()),
                }

            per_file.append(file_info)
            if ev.size > 0:
                all_parts.append(ev)

        extra["per_file"] = per_file
        if component_counts:
            extra["components"] = {
                label: info for label, info in component_counts.items()
            }

        events = _merge_events(all_parts)
        if events.size == 0:
            raise OptimizeError(
                "no tracks pass the preselection in "
                f"{len(context.input_paths)} file(s)"
            )
        n_signal_total = int((events.w_sig > 0).sum())
        n_bkg_total = int((events.w_bkg > 0).sum())
        if n_signal_total == 0:
            if use_labels:
                raise OptimizeError(
                    f"no signal tracks (label {signal_label!r}) pass the "
                    f"preselection among {events.size} tracks"
                )
            else:
                raise OptimizeError(
                    f"no signal tracks (signal_selection: "
                    f"{signal_selection!r}) among {events.size} "
                    "preselected tracks"
                )
        if n_bkg_total == 0:
            raise OptimizeError(
                f"no background tracks among {events.size} preselected "
                "tracks; nothing to optimize against"
            )
        extra["n_tracks"] = events.size
        extra["n_signal"] = n_signal_total
        extra["n_background"] = n_bkg_total

        # Split into train/test
        if split:
            train, test = events.split()
            extra["n_train"] = train.size
            extra["n_test"] = test.size
        else:
            events.prepare()
            train = events
            test = None

        # Baselines
        unapplied = copy.deepcopy(cut_vars)
        for v in unapplied:
            v.applied = False
        n_sig_0, n_bkg_0 = evaluate_yields(unapplied, train)
        fom_0 = fom(n_sig_0, n_bkg_0)
        lim_0 = opt.inv_expected_limit(n_sig_0, n_bkg_0)
        extra["baseline_train"] = {
            "n_sig": n_sig_0, "n_bkg": n_bkg_0, "fom": fom_0,
        }
        if test is not None:
            n_sig_0_test, n_bkg_0_test = evaluate_yields(unapplied, test)
            fom_0_test = fom(n_sig_0_test, n_bkg_0_test)
            lim_0_test = opt.inv_expected_limit(n_sig_0_test, n_bkg_0_test)
            extra["baseline_test"] = {
                "n_sig": n_sig_0_test, "n_bkg": n_bkg_0_test,
                "fom": fom_0_test,
            }

        # Build seed cut-var lists from parsed seed dicts
        seed_cut_lists: list[list[CutVar]] = []
        for sd in seed_dicts:
            seed_cut_lists.append(apply_seed_to_vars(cut_vars, sd))

        # Run multi-start optimization
        result = multi_start_optimize(
            cut_vars, train, step, end_eff, fom,
            seeds=seed_cut_lists if seed_cut_lists else None,
            n_random=n_random,
        )

        if result["best"] is None:
            raise OptimizeError(
                "the greedy optimizer could not tighten any cut from the "
                "starting selection — the variables may already be at their "
                "bounds, or the step is too large"
            )

        # Build the greedy path data for plots and output
        path_data: dict[str, list[float]] = {
            "eff_train": [1.0], "rej_train": [0.0],
            "fom_train": [1.0], "lim_train": [1.0],
        }
        if test is not None:
            path_data.update({"eff_test": [1.0], "fom_test": [1.0],
                              "lim_test": [1.0]})

        greedy_steps: list[dict] = []
        for sel in result["greedy_path"]:
            eff_tr = sel.n_sig / n_sig_0 if n_sig_0 > 0 else 0.0
            rej_tr = 1.0 - sel.n_bkg / n_bkg_0 if n_bkg_0 > 0 else 0.0
            fom_tr = fom(sel.n_sig, sel.n_bkg) / fom_0 if fom_0 > 0 else 0.0
            lim_tr = (opt.inv_expected_limit(sel.n_sig, sel.n_bkg) / lim_0
                      if lim_0 > 0 else 0.0)
            path_data["eff_train"].append(eff_tr)
            path_data["rej_train"].append(rej_tr)
            path_data["fom_train"].append(fom_tr)
            path_data["lim_train"].append(lim_tr)

            step_info: dict = {
                "eff_train": eff_tr,
                "fom_train": fom_tr,
                "cuts": build_cut_string(sel.cuts),
            }
            if test is not None:
                s_te, b_te = evaluate_yields(sel.cuts, test)
                eff_te = s_te / n_sig_0_test if n_sig_0_test > 0 else 0.0
                fom_te = fom(s_te, b_te) / fom_0_test if fom_0_test > 0 else 0.0
                lim_te = (opt.inv_expected_limit(s_te, b_te) / lim_0_test
                          if lim_0_test > 0 else 0.0)
                path_data["eff_test"].append(eff_te)
                path_data["fom_test"].append(fom_te)
                path_data["lim_test"].append(lim_te)
                step_info.update(eff_test=eff_te, fom_test=fom_te)
            greedy_steps.append(step_info)
        extra["greedy_steps"] = greedy_steps

        # Random grid search
        grid_results = random_grid_search(cut_vars, train, n_grid, fom)
        grid_data = {
            "eff": [r["eff"] for r in grid_results],
            "rej": [r["rejection"] for r in grid_results],
            "fom": [r["fom"] for r in grid_results],
        }

        # Multi-start summary
        starts_summary: list[dict] = []
        for entry in result["starts"]:
            start = entry["start"]
            refined = entry["refined"]
            info = {
                "name": entry["name"],
                "start_eff": start.n_sig / n_sig_0 if n_sig_0 > 0 else 0.0,
                "start_fom": fom(start.n_sig, start.n_bkg) / fom_0 if fom_0 > 0 else 0.0,
                "refined_eff": refined.n_sig / n_sig_0 if n_sig_0 > 0 else 0.0,
                "refined_fom": fom(refined.n_sig, refined.n_bkg) / fom_0 if fom_0 > 0 else 0.0,
                "refined_lim": (opt.inv_expected_limit(refined.n_sig, refined.n_bkg) / lim_0
                                if lim_0 > 0 else 0.0),
                "cuts": build_cut_string(refined.cuts),
            }
            if test is not None:
                s_te, b_te = evaluate_yields(refined.cuts, test)
                info["test_fom"] = fom(s_te, b_te) / fom_0_test if fom_0_test > 0 else 0.0
                info["test_lim"] = (opt.inv_expected_limit(s_te, b_te) / lim_0_test
                                    if lim_0_test > 0 else 0.0)
            starts_summary.append(info)
        extra["starts"] = starts_summary

        # Best selection
        best = result["best"]
        best_cut_string = build_cut_string(best.cuts)
        extra["best_cuts"] = best_cut_string
        best_fom = fom(best.n_sig, best.n_bkg)
        best_eff = best.n_sig / n_sig_0 if n_sig_0 > 0 else 0.0
        best_rej = 1.0 - best.n_bkg / n_bkg_0 if n_bkg_0 > 0 else 0.0

        # Write plots
        files = _write_plots(outdir, {
            "greedy_path_data": path_data,
            "grid_data": grid_data,
        }, fom_name)

        # Write greedy path CSV
        csv_path = outdir / "greedy_path.csv"
        header = "step,eff_train,rej_train,fom_train,lim_train"
        if test is not None:
            header += ",eff_test,fom_test,lim_test"
        header += ",cuts\n"
        rows = []
        for i, step_info in enumerate(greedy_steps):
            row = (f"{i+1},{step_info['eff_train']:.6f},"
                   f"{path_data['rej_train'][i+1]:.6f},"
                   f"{step_info['fom_train']:.4f},"
                   f"{path_data['lim_train'][i+1]:.4f}")
            if test is not None:
                row += (f",{step_info['eff_test']:.6f},"
                        f"{step_info['fom_test']:.4f},"
                        f"{path_data['lim_test'][i+1]:.4f}")
            row += f",\"{step_info['cuts']}\"\n"
            rows.append(row)
        csv_path.write_text(header + "".join(rows), encoding="utf-8")
        files.append(str(csv_path))

        # Write multi-start summary CSV
        starts_path = outdir / "multi_start_summary.csv"
        sh = "name,start_eff,start_fom,refined_eff,refined_fom,refined_lim"
        if test is not None:
            sh += ",test_fom,test_lim"
        sh += ",cuts\n"
        srows = []
        for s in starts_summary:
            sr = (f"{s['name']},{s['start_eff']:.6f},{s['start_fom']:.4f},"
                  f"{s['refined_eff']:.6f},{s['refined_fom']:.4f},"
                  f"{s['refined_lim']:.4f}")
            if test is not None:
                sr += f",{s.get('test_fom', 0.0):.4f},{s.get('test_lim', 0.0):.4f}"
            sr += f",\"{s['cuts']}\"\n"
            srows.append(sr)
        starts_path.write_text(sh + "".join(srows), encoding="utf-8")
        files.append(str(starts_path))

        # Write best selection
        best_path = outdir / "best_selection.txt"
        best_path.write_text(best_cut_string + "\n", encoding="utf-8")
        files.append(str(best_path))

        # Metrics
        metrics: dict[str, float] = {
            "best_fom": float(best_fom),
            "best_fom_relative": float(best_fom / fom_0 if fom_0 > 0 else 0.0),
            "best_efficiency": float(best_eff),
            "best_rejection": float(best_rej),
            "best_signal": float(best.n_sig),
            "best_background": float(best.n_bkg),
            "baseline_signal": float(n_sig_0),
            "baseline_background": float(n_bkg_0),
            "baseline_fom": float(fom_0),
            "n_tracks": float(events.size),
            "n_signal": float(n_signal_total),
            "n_background": float(n_bkg_total),
            "n_greedy_steps": float(len(result["greedy_path"])),
            "n_starts_refined": float(len(result["starts"])),
            "n_grid_trials": float(n_grid),
            "n_files": float(len(context.input_paths)),
        }
        if test is not None:
            s_te, b_te = evaluate_yields(best.cuts, test)
            metrics["best_fom_test"] = float(fom(s_te, b_te))
            metrics["best_fom_test_relative"] = float(
                fom(s_te, b_te) / fom_0_test if fom_0_test > 0 else 0.0)
            metrics["best_efficiency_test"] = float(
                s_te / n_sig_0_test if n_sig_0_test > 0 else 0.0)

        # Log
        log_path = outdir / "optimize_selection.log"
        log_lines = [
            "optimize_selection_ntuple",
            *(f"  input            {p}" for p in context.input_paths),
            f"  figure of merit  {fom_name}",
        ]
        if use_labels:
            log_lines += [
                f"  signal label     {signal_label}",
                f"  file labels      {', '.join(file_labels)}",
                f"  component weights {component_weights or '(all 1.0)'}",
            ]
        else:
            log_lines.append(f"  signal selection {signal_selection}")
        log_lines += [
            f"  preselection     {preselection}",
            f"  variables        {', '.join(var_names)}",
            f"  train/test split {split}",
            f"  step             {step:g}",
            f"  end_eff          {end_eff:g}",
            f"  grid trials      {n_grid}",
            f"  random starts    {n_random}",
            "",
            f"Tracks: {events.size} ({n_signal_total} signal, "
            f"{n_bkg_total} background)",
        ]
        if component_counts:
            for label, info in component_counts.items():
                tag = "signal" if info["is_signal"] else "background"
                log_lines.append(
                    f"  {label:<16s} {info['n_tracks']:>8d} tracks "
                    f"(w={info['weight']:g}, {tag})"
                )
        log_lines += [
            f"Train: N(signal) = {n_sig_0:.3g}, N(background) = {n_bkg_0:.3g}, "
            f"{fom_name} = {fom_0:.4g}",
        ]
        if test is not None:
            log_lines.append(
                f"Test:  N(signal) = {n_sig_0_test:.3g}, "
                f"N(background) = {n_bkg_0_test:.3g}, "
                f"{fom_name} = {fom_0_test:.4g}"
            )
        log_lines += [
            "",
            f"Best selection ({fom_name} = {best_fom:.4g}, "
            f"eff = {best_eff:.3f}, rej = {best_rej:.3f}):",
            f"  {best_cut_string}",
            "",
            "Multi-start results:",
        ]
        for s in starts_summary:
            log_lines.append(
                f"  {s['name']:<12s} start eff={s['start_eff']:.3f} "
                f"fom={s['start_fom']:.3f} -> refined eff={s['refined_eff']:.3f} "
                f"fom={s['refined_fom']:.3f} lim={s['refined_lim']:.3f}"
            )
        log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    except (OptimizeError, NtupleError, SelectionError, CutVarSpecError) as exc:
        return RunOutcome(error=str(exc), extra=extra)

    return RunOutcome(metrics=metrics, files=files, log_path=log_path,
                      extra=extra)


def summarize(metrics: dict[str, float]) -> str:
    return (
        f"best {metrics.get('best_fom', 0):.4g} "
        f"(relative {metrics.get('best_fom_relative', 0):.3f}), "
        f"efficiency {metrics.get('best_efficiency', 0):.3f}, "
        f"rejection {metrics.get('best_rejection', 0):.3f} "
        f"(S = {metrics.get('best_signal', 0):.3g}, "
        f"B = {metrics.get('best_background', 0):.3g}) "
        f"from {metrics.get('n_greedy_steps', 0):.0f} greedy steps, "
        f"{metrics.get('n_starts_refined', 0):.0f} refined starts, "
        f"{metrics.get('n_grid_trials', 0):.0f} grid trials "
        f"on {metrics.get('n_tracks', 0):.0f} tracks."
    )


ALL_VARIABLES = ", ".join(sorted(set(TRACK_VARIABLES) | set(MC_VARIABLES)))
FOM_NAMES = ", ".join(FOM_FUNCTIONS)

SPEC = AnalysisSpec(
    name="optimize_selection_ntuple",
    description=(
        "Cut-set optimization from EventNtuple file(s): greedy path, random "
        "grid search, local refinement, and multi-start search to maximize a "
        "figure of merit (S/sqrt(B), Asimov Z, Punzi, or 1/expected upper "
        "limit) for signal vs. background discrimination. Input files are "
        "labeled by component (signal, DIO, cosmic, ...) with per-component "
        "weights, or signal is identified by an MC truth expression within "
        "each file. The optimized cut string is selection.py-compatible."
    ),
    input_kind="root_file",
    combines_files=True,
    metrics=(
        "best_fom", "best_fom_relative",
        "best_efficiency", "best_rejection",
        "best_signal", "best_background",
        "baseline_signal", "baseline_background", "baseline_fom",
        "n_tracks", "n_signal", "n_background",
        "n_greedy_steps", "n_starts_refined", "n_grid_trials", "n_files",
    ),
    parameters=(
        ParamSpec(
            name="optimize_variables",
            description=(
                "Comma-separated variable specifications: "
                "'name:type:lo:hi[:discrete][:seed_lo:seed_hi]' for each "
                "variable to optimize. type is 'both' (two-sided window), "
                "'left' (lower bound), 'right' (upper bound), or 'veto' "
                "(exclusion window). 'discrete' marks an integer variable. "
                "seed_lo:seed_hi sets initial edges (default: search bounds). "
                "Example: 'p_front:both:80:110, trkqual:left:0:1, "
                "t_front:both:500:1700, nactive:left:10:60:discrete'. "
                "Variables: " + ALL_VARIABLES
            ),
            kind="text",
        ),
        ParamSpec(
            name="file_labels",
            description=(
                "Comma-separated label for each input file, identifying its "
                "component: e.g. 'signal, DIO, DIO, cosmic, RPC_ext'. One "
                "label per file, in the same order as data_files. The label "
                "matching signal_label is signal; all others are background. "
                "Empty to use signal_selection mode instead (identify signal "
                "by an expression within each file)."
            ),
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="signal_label",
            description=(
                "The file_labels label that identifies signal files. All "
                "files with this label contribute signal weight; all others "
                "contribute background weight. Required when file_labels "
                "is set."
            ),
            default="signal", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="signal_selection",
            description=(
                "Expression identifying signal tracks within each file "
                "(evaluated over the track variables). Used when file_labels "
                "is empty. Typically an MC truth cut, e.g. "
                "'mc_start_code == 167 or mc_start_code == 168' for CE-. "
                "Variables: " + ALL_VARIABLES
            ),
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="component_weights",
            description=(
                "Per-component weights as 'label:weight' pairs, comma-"
                "separated. Example: 'signal:1.0, DIO:0.5, cosmic:0.01'. "
                "Labels not listed default to 1.0. In signal_selection mode, "
                "use 'signal:w, background:w' to set signal and background "
                "weights."
            ),
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="preselection",
            description=(
                "Baseline cut applied to all tracks before optimization. "
                "Example: 'status == 1 and pdg == 11 and downstream and "
                "p_front > 80'. An empty string selects everything."
            ),
            default="status == 1 and pdg == 11 and downstream",
            kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="fom",
            description=(
                f"Figure of merit to maximize: {FOM_NAMES}. "
                "'s_over_sqrt_b' is S/sqrt(B); 'asimov_z' is the Asimov "
                "significance; 'punzi' is S/(a/2 + sqrt(B)); 'expected_ul' "
                "is S / N_UL(B), 1 over the expected 90%% CL CLs upper limit."
            ),
            default="s_over_sqrt_b", kind="text",
        ),
        ParamSpec(
            name="seed_selections",
            description=(
                "Starting selections for multi-start refinement, in addition "
                "to the greedy best and randomized starts. Semicolon-separated "
                "cut-sets, each a comma-separated list of 'name:lo:hi'. "
                "Example: 'trkqual:0.155:1,cos_front:0.5:0.75; "
                "trkqual:0.145:1,cos_front:0.56:0.73'. Empty for none."
            ),
            default="", kind="text", allow_empty=True,
        ),
        ParamSpec(
            name="step",
            description=(
                "Target signal-efficiency loss per greedy step. Smaller values "
                "give a finer scan but take longer."
            ),
            default=0.01, minimum=1e-6, maximum=0.5,
        ),
        ParamSpec(
            name="end_eff",
            description=(
                "Stop the greedy path when signal efficiency drops below this."
            ),
            default=0.1, minimum=0.0, maximum=1.0,
        ),
        ParamSpec(
            name="n_grid_trials",
            description="Number of random cut-sets to sample in the grid scan.",
            default=10000, minimum=0,
        ),
        ParamSpec(
            name="n_random_starts",
            description=(
                "Number of randomized greedy starts for the multi-start "
                "search. Each picks randomly from the top-3 candidates per "
                "step, then is refined by local search."
            ),
            default=5, minimum=0,
        ),
        ParamSpec(
            name="split",
            description=(
                "1 to split events into train/test halves by event-number "
                "parity (each half weighted by 2); 0 to use all events for "
                "both optimization and evaluation."
            ),
            default=1, minimum=0, maximum=1,
        ),
    ),
    run=run,
    summarize=summarize,
    input_hint=(
        "EventNtuple file(s) (nts.*.root with EventNtuple/ntuple). Use "
        "file_labels to label each file by component (signal, DIO, cosmic, "
        "...) with per-component weights via component_weights; or use "
        "signal_selection to identify signal tracks within a mixed file."
    ),
)
