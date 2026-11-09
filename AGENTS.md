# AGENTS.md

An MCP server that exposes Mu2e analyses as agent tools: `list_analyses` and
`run_analysis`. An analysis is either a `mu2e` job over art file(s), with its
stdout parsed, a Python computation over a ROOT file, or one over its
parameters alone. README.md is the
full reference. This file holds what you need before changing code.

## Setup, test, run

Use the `ana` python. No installs are needed, and there is no pytest:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana                       # Python 3.12 with mcp, pydantic, uproot, awkward
python3 tests/test_tools.py     # bare asserts; prints ok/FAIL per test, "0 failure(s)"
python3 -m analysis_mcp_server --transport stdio [--work-area DIR | --musing 'SimJob MDC2025ay' | --code-tarball FILE]
```

- The tests never start a mu2e job. Keep it that way: test parsers and
  computations on sample stdout or synthetic ROOT files written with
  `uproot`.
- Write a synthetic tree with `f.mktree(...).extend(...)`. Plain
  `f["x"] = dict` writes an RNTuple with this uproot.
- To check a change against real data, run `run_analysis` with `max_events`
  into a scratch `output_dir`. A real job runs `mu2e` in that directory, and
  `output/` is gitignored.
- These are shared interactive nodes. Set `OPENBLAS_NUM_THREADS=1
  OMP_NUM_THREADS=1` for anything you run, and never fan out many jobs.
- `.mcp.json` and `DEFAULT_WORK_AREA` in `tools/mu2e_env.py` point at the
  original author's work area. Pass your own `--work-area` or `--musing`
  rather than editing them for local use.

## Architecture rules

- **`tools/` never imports MCP.** It is ordinary science code.
  `analysis_mcp_server/` is a generic wrapper that registers every name in
  `tools/__init__.py`'s `__all__` as a tool. Do not add names to `__all__`:
  new analyses go through `run_analysis`, so the tool schema stays the same
  however many analyses exist.
- **One analysis = one module** in `tools/analyses/` that defines `SPEC` (an
  `AnalysisSpec`, see `tools/spec.py`), plus an entry in `_ANALYSIS_MODULES`
  in `tools/registry.py`. The registry tests loop over every analysis and
  fail until it is covered: add a `SAMPLE_STDOUT` entry for an `art_files`
  analysis, or direct tests for a `root_file` or `parameters` one. A
  `parameters` analysis reads no file and computes from its parameters alone.
- **Physics knobs are `ParamSpec`s**, not tool arguments. They are validated,
  have defaults and ranges, and `list_analyses` reports them. Their
  descriptions are what agents read, so write them for an agent.
- **Shared machinery knows nothing about physics.** That covers
  `mu2e_job.py` (running jobs), `mu2e_env.py` (where Offline comes from),
  `catalog.py` (SAM/metacat names and datasets to `/pnfs` paths),
  `spectrum.py` (uniform-binned histograms: rebin, regrid, smear) and
  `selection.py` (cut expressions). Put physics in the analysis modules.
- **Errors are worded for the caller.** A runner returns
  `RunOutcome(error=...)` with a message an agent can act on, such as what
  is missing and what produces it. It never raises a traceback.
  `analysis_tools.run_analysis` wraps the result in the uniform
  `ArtifactResult`.

## Things that are easy to get wrong

- **Metric names are an API.** Downstream code relies on them, for example
  the Mu2eBO workflow uses `edep`'s `n_events_calo_edep_above_50mev`. Add
  new metrics rather than renaming or removing existing ones, and keep
  `metrics`, `units` and the summarizer in step.
- **The fcl ships here, in `fcl/`.** Analyses name it as `FCL_DIR / "x.fcl"`
  (absolute), so it is the same file under every environment. Its
  `#include`s resolve through art's `FHICL_FILE_PATH`.
- **EdepAna is in Offline from v13_39_00** (`SimJob MDC2025ay` and later).
  An environment on an older Offline cannot run `edep`. The step counts
  (`*_nsteps_*` metrics, `event_*_nsteps*` branches) need Offline PR #2067
  (merged 2026-10-27, not yet in a Musing), whose EdepAna *requires* six
  step parameters that an older EdepAna rejects. `edep` therefore probes the
  configured Offline once (`mu2e_job.module_accepts`, via `mu2e
  --print-description`) and, with the PR, runs a per-run `edep_job.fcl` with
  the parameters appended (never put them in `fcl/edep.fcl`); without it,
  the shipped fcl runs, the step metrics are absent and a non-default step
  parameter is an error. The default environment (MDC2025ay) is the old
  case, and the Mu2eBO workflow's metrics must keep working there.
- **Selections over the EdepAna tree** (`EDepAna/tree`) go through
  `read_edep_tree` / `select_events` / `EDEP_VARIABLES` in
  `tools/analyses/edep.py`, so every analysis of EdepAna output uses the same
  variables and semantics:
  - Per-primary variables are the first primary's.
  - A missing value is NaN, so any cut on it fails.
  - A new variable belongs in `EDEP_VARIABLES` and `edep_variables`.
- **A changed selection default must still reproduce the old fixed cut.**
  Today `edep` defaults to `event_calo_edep_vis > 50` and
  `approx_ce_sensitivity` to `event_calo_edep_vis > 10` (the module's
  `hist_2` set). If you change how a result is computed, compare it on real
  data with the old code at the default settings.
- **Selection expressions come from agents.** `tools/selection.py` parses
  them with `ast` and walks a whitelist. Never `eval` them, and extend the
  whitelist only with pure numpy functions.
- **The trigger analyses share one job** (`tools/analyses/trigger.py`): it
  copies `fcl/trigger.fcl` into a per-run fcl, because FHiCL cannot
  `#include` an absolute path. Prescales go into each path's
  `PrescaleEvent` filter, whose label is derived from the path name by the
  menu generator's convention (`prescale_module`). Timing comes from the
  TimeTracker sqlite database, not stdout: an event's time is the sum of its
  module times without the data fetch (`FETCH_MODULE_TYPES`, i.e. Prefetch;
  the source is timed apart), and the first event (DB setup) is skipped.
- **OccupancyAna is in Offline only after PR #2070** (merged 2026-10-28, in no
  Musing release yet), so `occupancy_digi` / `occupancy_reco` need a work area
  built on it. They share `tools/analyses/occupancy.py` (per-run fcl, summary
  table, tree). A skipped collection's metrics are -1, never absent, so every
  listed metric is always reported; keep that when adding collections.
- **`edep_occupancy` needs EdepAna's `occupancy/` histograms** (Offline PR
  #2097, merged 2026-11-03; Musings after it). Its last revision made five
  hit-window parameters required (`FoldTimeOffset`, `Trk/CaloHitTimeMin/Max`):
  `edep.run` probes for `TrkHitTimeMin` (an intermediate revision of the PR
  has `FoldTimeOffset` alone) and appends them to the per-run fcl
  only when the module declares them (never in `fcl/edep.fcl`, which must stay
  valid for released Offline). `edep_occupancy` refuses an EdepAna without
  them, which also covers unmerged builds of the PR. It runs through `edep.run`
  (never a second copy of the job/fcl logic) and then reads
  `EDepAna/occupancy`; a file without them is a plain error. Peaks of a sum
  are not sums of peaks, so `edep_occupancy_combine` adds the per-channel
  histograms in `occupancy_per_microbunch.root`, not metrics. A tracker "row"
  (same plane and straw number over six panels) is one radius; the calo
  average is radial in the crystal-centre radius. Its `gen_events_per_pot`
  mode computes `events_per_microbunch` from the sample's own GenEventCount
  (EdepAna's summary) for callers that cannot know it in advance; a named
  `prescale_filter` reuses `count.run_counts_job` (run first, so a wrong
  stream is refused before EdepAna runs), never a second copy of its parsing.
- **A stop file holds mu+ stops too** (about 1% in MDC2025). A rate for a
  mu- capture process (conversion, DIO, RMC) is `muon_stop_rate`'s
  `muminus_*`, not `stops_per_*`, which keep counting every stop.
- **Never use `tpr_` or `mpr_` paths** in defaults, tests or examples: they
  are in the trigger menu but not meant for the real trigger. Use `apr_`,
  `cpr_` and `calo_` paths.
- **Catalogued inputs** (`dataset`, or a file name with no `/`) become
  paths in `run_analysis` before anything else, so runners only ever see
  paths. All network and dCache access goes through `catalog.Catalogs`;
  tests swap `catalog.CLIENT` for a fake and never reach SAM or metacat.
  An unstaged tape file is refused, not read: reading it blocks on tape.
- **Reference inputs.** An `AnalysisSpec` with `takes_reference` requires
  a baseline (`reference_file(s)` / `reference_dataset` in `run_analysis`,
  resolved like `data_file(s)`), which the runner gets as
  `context.reference_paths`; with `reference_optional` too
  (`approx_ce_sensitivity`) the baseline may be left out and the paths are
  empty. Every other analysis refuses one. Add another
  comparison that way rather than with a new tool argument.
- **Default inputs.** An `AnalysisSpec` with `default_inputs` runs without
  `data_file(s)`; the trigger rate and timing default to
  `mu2e-trig-config/ci/data_files.txt`, found with `Mu2eEnv.find_code_file`
  down the configured code's `backing` chain.
- **EventNtuple track variables** (`tools/eventntuple.py`, shared by every
  EventNtuple analysis; physics, despite living beside the shared machinery)
  are vectorized with awkward and read by leaf name (`filter_name`; uproot's
  `arrays()` would parse a dotted name as an expression). `*_front` is the
  downstream-going TT_Front crossing, and fit parameters are taken at the
  TT_Mid segment's index. Add a variable to both `TRACK_VARIABLES` and
  `track_variables`; a test checks that every advertised one is usable.
  MC truth (`MC_VARIABLES`, `mc_*`) is optional: read only where the file
  has the branches, and `check_names` turns a request for it on data into
  a plain error; CRV coincidences (`CRV_VARIABLES`, `crv_*`) are optional the
  same way. The MC chain of a track is the leading block of `trkmcsim`
  (`ancestry`), not every entry: `trkrel` is relative to each matched
  particle, so it cannot identify ancestors on its own. A muon ancestor's
  `endpos` is its stop only if it ended at rest: in staged simulation the
  muon in a CE's chain was killed at the stage boundary, and the stop is the
  generated particle's start. `evtinfomc/nprotons` is read by full path,
  because a leaf name filter would also match `evtinfo`'s unfilled
  `nprotons`. `trigger_efficiency_ntuple_binned` reuses the shared reading
  and must keep reporting the same overall counts.
- **`approx_ce_sensitivity` has a numerics trap.** Window sums run outward
  from each window's own edge, never as differences of prefix sums: the DIO
  spectrum spans ~18 orders of magnitude, and cancellation silently zeroes
  the background. A test locks this in.
- **`approx_ce_sensitivity`'s `cut_mode="box"`** needs the tracker-front
  helix branches (Offline PR #2092, not yet in a Musing); the default
  `"momentum"` mode must keep reproducing the old numbers (a test locks
  them). Box mode optimizes the Punzi FoM, not S/sqrt(B), which diverges
  as the background is cut to zero. Its cosmic shapes ship in `data/`
  (`cosmic_shapes.root`/`.json`): regenerate them with
  `scripts/extract_cosmic_shapes.py`, never edit them by hand.

## Style

Match the surrounding code: docstrings and comments explain *why*, constants
are named at the top of a module with a short note on where they come from,
and messages are plain sentences. When behaviour changes, update the
README.md section for that analysis, including the test count under "Test".
