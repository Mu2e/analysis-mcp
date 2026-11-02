# analysis-mcp-server

An example MCP server that exposes Mu2e analyses as agent tools. An agent asks
which analyses exist, then runs one — either a mu2e job over art file(s):

```bash
mu2e -c <the analysis' fcl> -s <data file>     # one file
mu2e -c <the analysis' fcl> -S <file list>     # several, one path per line
```

or a Python computation over a ROOT file. Fourteen analyses ship: energy
deposition and approximate step-count occupancy (`edep`), event counts
(`count`), muon stopping rate (`muon_stop_rate`), approximate CE sensitivity
(`approx_ce_sensitivity`) and its full-simulation counterpart
(`fullsim_sensitivity`), muon stops per material (`stop_materials`), the
online trigger's efficiency, rate and processing time (`trigger_efficiency`,
`trigger_rate`, `trigger_timing`), the trigger efficiency read from
EventNtuples, overall and as a function of a variable
(`trigger_efficiency_ntuple`, `trigger_efficiency_ntuple_binned`), and
detector occupancy from digi, reco and EventNtuple files
(`occupancy_digi`, `occupancy_reco`, `occupancy_ntuple`). Adding more is one
small module each.

Built to the same pattern as
[`spectra-mcp-server`](https://github.com/HEP-KE/spectra-mcp-server), so the
`multiagent-client-demo` client in `../multiagent-client-demo` (or Claude
Code, Claude desktop, Codex, Cursor) can drive it unchanged.

## The one idea

> **The science code stays in usual Python. The MCP wrapper only publishes it.**

- `tools/` is an ordinary package that never imports MCP.
- `analysis_mcp_server/` is a generic wrapper: it reads one line of config from
  `pyproject.toml`, imports the science package, and registers every function
  in its `__all__` as an MCP tool.

```toml
[tool.mcp-server]
tool_modules = ["tools"]
```

Type hints, Pydantic `Field` constraints, and docstrings become the tool
schema agents see.

## Layout

```
tools/
  spec.py             AnalysisSpec, ParamSpec, RunContext/RunOutcome, ArtifactResult
  mu2e_env.py         where Offline comes from: work area, Musing, or tarball
  mu2e_job.py         running mu2e: -s/-S inputs, logs, timeouts
  catalog.py          SAM definitions, metacat datasets and registered file
                      names -> /pnfs paths; tape staging check
  spectrum.py         bin contents on a uniform grid: rebin, regrid, smear
  selection.py        event selections: a safe cut expression over arrays
  registry.py         the catalogue: name -> AnalysisSpec
  analyses/
    edep.py                    energy deposition: fcl + summary parser
    count.py                   events, gen events, optional prescale + the
                               parsing/job muon_stop_rate builds on
    muon_stop_rate.py          stopping rate: count.py + the mu-/mu+ split
                               + POT scaling
    approx_ce_sensitivity.py   CE sensitivity from the EdepAna tree
    stop_materials.py          stops per material from <module>/stopmat
    trigger.py                 the trigger job (paths + prescales) and its
                               parsing, shared by the three below
    trigger_efficiency.py      fraction of signal events triggered
    trigger_rate.py            trigger rate on pileup, overall and per path
    trigger_timing.py          trigger processing time per event on pileup
    trigger_efficiency_ntuple.py  trigger efficiency from EventNtuple
                               trig_<path> branches, for a track selection
    trigger_efficiency_ntuple_binned.py  the same, in bins of a track or
                               MC variable (p_front, mc_p_front, mc_npot, ...)
    occupancy.py               the OccupancyAna job (per-run fcl, summary
                               table, tree), shared by the two below
    occupancy_digi.py          digi occupancy of dig files
    occupancy_reco.py          hit/cluster occupancy of mcs/rec files
    occupancy_ntuple.py        the same counts read from an EventNtuple
    fullsim/                   full-simulation analyses of EventNtuple
      eventntuple.py           reading EventNtuple/ntuple; MC origin of a track
      cuts.py                  the CE-like track cuts (pyfitter cut-set 80)
      sensitivity.py           fullsim_sensitivity: S/sqrt(B) in the best window
  analysis_tools.py   the MCP tools: list_analyses, run_analysis
  __init__.py         __all__ — ONLY these names become tools
analysis_mcp_server/  generic drop-in wrapper (FastMCP): server.py, cli.py
fcl/                  the fcl the art_files analyses run (spec.FCL_DIR)
tests/test_tools.py   parsers, registry, input handling — no mu2e, no MCP
```

The split that matters: **the shared machinery (`mu2e_job.py`, `spectrum.py`)
knows how to run jobs and handle binned spectra but nothing about physics;
`analyses/*.py` knows the physics.** The registry joins them, and the two tools
are generic over it.

Each analysis declares an `input_kind`, which is the only thing the generic
layer needs to know about the difference between them:

| input_kind | what it consumes | what its runner does |
|---|---|---|
| `art_files` | mu2e art file(s) | runs an fcl with `mu2e`, parses the job's stdout |
| `root_file` | a ROOT file from an earlier analysis | Python computation over its histograms or tree |

## Tools

| tool | what it does |
|---|---|
| `list_analyses()` | catalogue: description, `input_kind`, metric names + units, parameters and their defaults, what the input must contain |
| `run_analysis(analysis, output_dir, data_file=None, data_files=None, dataset=None, max_files=None, reference_file=None, reference_files=None, reference_dataset=None, parameters=None, max_events=None, timeout_s=900)` | run one analysis, return its metrics |

Both return the same `{status, files, message, metadata}` contract, so a
workflow can chain several runs and collect `metadata` uniformly.

### The analyses

| analysis | input | reports |
|---|---|---|
| `edep` | art file(s) | average calo/tracker energy deposition per event and per generated event, the events passing a `selection`, and (with an Offline that has the step counts) approximate calo/tracker hit counts from detector steps |
| `count` | any art file with subrun bookkeeping | events kept per generated event, dividing out a prescale only if you name the filter |
| `muon_stop_rate` | `sim.*.TargetStops.*.art` | stopped muons per generated event and per POT, from the file's event count, generated-event count and output prescale |
| `approx_ce_sensitivity` | `nts.*.root` from `edep` | `S/sqrt(B)` for the best momentum window (or, with `cut_mode="box"`, the best momentum/time/R(max)/d0/cos box), with the cuts and its signal/DIO/cosmic counts |
| `stop_materials` | `nts.*.root` file(s) from the stop-finding job (e.g. MuBeam) | muon stops per material, and per generated event for the `n_gen_events` you supply |
| `fullsim_sensitivity` | reconstructed CE-mixed-with-pileup EventNtuple `nts.*.root` file(s), e.g. CeMLeadingLogMix1BB | `S/sqrt(B)` for the best momentum window, as `approx_ce_sensitivity`, with the signal efficiency and detector response measured after the CE-like cuts |
| `trigger_efficiency` | signal digi art file(s) | fraction of events passing any of the given trigger paths, and per path |
| `trigger_rate` | pileup digi art file(s); default: mu2e-trig-config's CI sample | trigger rate in Hz averaged over the cycle (duty factor from `batch_mode`) and on spill, overall and per path |
| `trigger_timing` | pileup digi art file(s); default: mu2e-trig-config's CI sample | trigger processing time per event: mean, median, tail; per path and per module |
| `trigger_efficiency_ntuple` | EventNtuple file(s) with `trig_<path>` branches | of events with a track passing a selection, the fraction any given path accepted, and per path |
| `trigger_efficiency_ntuple_binned` | EventNtuple file(s) with `trig_<path>` branches | the same, in bins of a track variable or expression (reco or true momentum at the tracker front, beam intensity, ...), with a figure, CSV table and ROOT histograms |
| `selection_efficiency_ntuple` | EventNtuple file(s) | fraction of events with a track passing a selection, of all ntuple events or of those passing a `denominator` cut |
| `momentum_resolution_ntuple` | simulated EventNtuple file(s) (MC truth) | momentum resolution and energy loss before the tracker for tracks passing a selection: plots, histograms, mean/rms/median/sigma68 |
| `calo_cluster_resolution_ntuple` | simulated EventNtuple file(s) (calo MC truth) | calorimeter cluster dE (reco minus true energy) and dT (reco time minus earliest true deposit) for clusters passing a selection: plots, histograms, mean/rms/median/sigma68 |
| `variable_comparison_ntuple` | two sets of EventNtuple file(s): the input (new) and a `reference_file(s)` (baseline) | one track or cluster variable compared: overlaid distributions with ratio, shifts and a KS test, and, for the same events, how many changed, gained or lost and by how much |
| `muon_stops` | `sim.*.TargetStops.*.art` (or PolyStops / IPAStops) | where (z, radius), when and with what momentum at ST_In the muons stopped: metrics, plots, ROOT histograms |
| `total_edep_per_pot` | none: parameters only (earlier `edep` and `muon_stop_rate` metrics) | the early electron-beam flash, early muon-beam flash and muon-stop pileup depositions per POT, their total and each one's share |

`approx_ce_sensitivity` declares `produced_by = ["edep"]`, so chaining is
discoverable: run `edep`, then pass the `nts.*.root` from its `files` to the
sensitivity.

### Parameters

Analyses declare their own physics knobs, passed as `parameters`:

```python
run_analysis(analysis="approx_ce_sensitivity",
             data_file=".../nts.owner.edep.Run1B.001800_00000000.root",
             output_dir=".../sens",
             parameters={"sig_eff": 0.1})     # npot and the cosmic rate
                                              # fall back to their defaults
```

`list_analyses` reports each parameter's description, default, range, and
whether it is required; unknown, missing, non-finite, or out-of-range values
come back as a plain error naming the offender. A parameter is a number unless
its `kind` is `"text"`, which is for the ones that name something in the job's output —
like `muon_stop_rate`'s `prescale_filter` — or, as with `selection`, are an
expression.

### Selections

`edep` and `approx_ce_sensitivity` take a `selection`: a cut over the
per-event tree EdepAna writes (`EDepAna/tree`), applied by `tools/selection.py`
and read by `read_edep_tree` in `tools/analyses/edep.py`, which also lists the
variables (`EDEP_VARIABLES`, repeated in each parameter's description):

```python
parameters={"selection": "event_calo_edep_vis > 10 && primary_start_z > 5400"}
```

Comparisons, arithmetic, `and`/`or`/`not` (or `&&`/`||`/`!`) and
`abs`/`sqrt`/`hypot`/`min`/`max`/`log`/`exp` are allowed. The expression is
parsed with `ast` and walked against a whitelist, never `eval`ed. Per-primary
variables are the first primary's, as EdepAna's own histograms use; derived
ones (`primary_start_r` in the detector frame, `primary_trk_front_energy_diff`,
`has_trk_front`, ...) are computed in Python. A missing value (no primary, or
no tracker-front step) is NaN, so any cut on it fails. `""` selects every event.

Six more variables, the per-event step counts `event_{calo,trk}_nsteps`,
`..._nsteps_ecut` and `..._nsteps_ecut_tcut` (see `edep` below), exist only in
a tree written by an EdepAna with the step counts. They are read only where the
file has them; a selection naming one against a tree without them is refused
with a plain error (for `edep` on an older Offline, before the job runs).

Likewise six helix variables exist only in a tree from an Offline with PR
#2092 (not yet in a Musing): `primary_trk_front_r_center`, `_radius`, `_cos` and
`_time` (read only if all four branches are in every file; EdepAna's -1 and
"no front step" values are NaN), and the derived `primary_trk_front_rmax`
(= r_center + radius) and `primary_trk_front_d0` (= r_center - radius, signed;
Mu2e's d0 = rmax - 2R). `approx_ce_sensitivity`'s `cut_mode="box"` needs them.

The defaults reproduce the fixed cuts the module's histograms and summary had:
`event_calo_edep_vis > 50` for `edep` (so `n_events_selected` equals
`n_events_calo_edep_above_50mev`) and `event_calo_edep_vis > 10` (the
`hist_2` set) for `approx_ce_sensitivity`. On the Run-1B CE sample the
histograms rebuilt from the tree match `hist_0`..`hist_3` bin for bin, and the
sensitivity is unchanged.

### Inputs

An analysis whose `input_kind` is `"parameters"` (`total_edep_per_pot`)
reads no file: pass only `parameters`, and any `data_file`, `data_files`,
`dataset` or `max_events` is refused.

Otherwise pass **exactly one** of the following — or neither, for an analysis that
`list_analyses` shows with `default_inputs` (`trigger_rate` and
`trigger_timing`, which then run over the files in the configured code's
`mu2e-trig-config/ci/data_files.txt`, found by following its `backing`
links):

- `data_file` — a single input file → `mu2e -s <file>`
- `data_files` — a list of input files → written one per line to
  `filelist.txt` in `output_dir` → `mu2e -S <filelist.txt>`
- `dataset` — a catalogued dataset, whose files run as with `data_files`:
  a SAM definition name, or a metacat DID `<namespace>:<name>`. A bare name
  that is not a SAM definition is looked up in metacat under its owner's
  namespace (`nts.mmackenz.X...` → `mmackenz:`). `max_files=N` takes the
  first N files by name; a dataset over 2000 files needs it.

A file is given either as an absolute path or as the name of a file
registered in SAM or metacat, bare or as a DID (`mu2e:<name>`). Names and
datasets are resolved by `tools/catalog.py` to `/pnfs` paths before
anything else happens:

- Where a file is comes from SAM's location records, then from Mu2e's
  standard dCache layout (mdh's: `<prefix>/<family>/<tier>/<owner>/<desc>/
  <config>/<format>/<h0h1>/<h2h3>/<name>`, `h` the sha256 of the name), so a
  file catalogued only in metacat is found too. The first copy that exists
  wins, preferring persistent disk, then scratch, then tape.
- A tape file that is not staged (dCache locality `NEARLINE`) would leave
  the job waiting on tape, so the run is refused with the command that
  stages it: `samweb prestage-dataset --defname=<def>` for a SAM
  definition, `mdh prestage-files <names>` otherwise.
- SAM and metacat are read over plain HTTPS, without a token or a client.
  A 500-file definition resolves in about 20 s, most of it the locality
  checks.
- The servers' certificates are checked against Python's default CA store.
  Where that store is empty, the catalogs trust the first CA bundle that
  exists: the Python environment's own (`<prefix>/ssl/cacert.pem`),
  certifi's, then the system's. This matters for the CVMFS `ana` 2.7.0
  Python started directly rather than through `pyenv` (which exports
  `SSL_CERT_FILE`): it was relocated from the directory it was built in, so
  its OpenSSL looks for CAs where none exist, and without the fallback every
  lookup fails with `CERTIFICATE_VERIFY_FAILED`. An `SSL_CERT_FILE` or
  `SSL_CERT_DIR` you set is used as it is.

```python
run_analysis(analysis="count",
             dataset="mmackenz:dts.mmackenz.CeEndpoint.MDC2025ax.art",
             max_files=2, max_events=10, output_dir=".../count")
```

With `data_files` the whole set runs as **one** job, so the metrics cover all
the inputs together, not one file each — call the tool once per file for
per-file numbers. Raise `timeout_s` (max 7200) when passing many files.
`max_events` applies only to `art_files` analyses. A `root_file` analysis
takes `data_files` only if `list_analyses` reports `takes_data_files: true`
for it (so far just `stop_materials`, which combines the files' histograms);
the others take a single `data_file`.

`max_events` (mu2e `--nevts`) caps events for a quick check before a full run.
Beware: generated-event counts come from the input's subrun bookkeeping and
cover the whole file regardless, so **any "per gen event" metric is
meaningless when `max_events` is set** — use it to confirm a job runs, not for
physics numbers. `edep` records the `max_events` it ran with next to each file
it writes (`<file>.edep.json`, not listed in `files`), and
`approx_ce_sensitivity` refuses such a file when given `stops_per_pot`.

An analysis that `list_analyses` shows with `takes_reference: true` (so far
`variable_comparison_ntuple`) compares its input with a second, baseline
sample, which it then requires: pass exactly one of `reference_file`,
`reference_files` or `reference_dataset`, given and resolved like
`data_file`, `data_files` and `dataset` (`max_files` cuts either dataset).
Any other analysis refuses them. The result's metadata lists them as
`reference_files`.

### Results

Every result carries `analysis`, `input_kind`, `data_files` (the resolved
paths), `n_input_files`, `log_path`, the resolved `parameters`, and — for the
`-S` case — `file_list_path`. Catalogued inputs add `inputs_from`: the
`dataset`, which `catalog` listed it (`sam` or `metacat`) and its
`n_files_in_dataset`, and how many files came from each dCache
`locations`. `output_dir` can be reused: a rerun overwrites the previous
run's ROOT output, log, file list and figures, and `files` reports what this
run wrote whether or not the name was there before. On success the analysis' metrics are merged into `metadata`
under the names `list_analyses` advertises. `files` lists what the run wrote:
the job's ROOT output for `edep`, the figures for `approx_ce_sensitivity`. On
failure `status="error"`, and for mu2e jobs `metadata.stdout_tail` holds the
last 20 log lines, so an agent can diagnose without re-running.

`edep` reports:

| metric | unit | from the module's print |
|---|---|---|
| `n_events` | | `Saw <N> events` (weighted, so a float) |
| `n_gen_events` | | `(<N> gen events)` |
| `event_rate` | events / gen event | `output rate = <N>` |
| `avg_calo_edep_per_event_mev` | MeV | `Average calo energy deposition per event` |
| `avg_calo_edep_per_gen_event_mev` | MeV | `... per gen event` |
| `n_events_calo_edep_above_50mev` | | `Events with calo Edep > 50 MeV` |
| `avg_trk_edep_per_event_mev` | MeV | `Average tracker energy deposition per event` |
| `avg_trk_edep_per_gen_event_mev` | MeV | `... per gen event` |
| `n_events_selected` | | weighted count of tree events passing `selection` |
| `selected_per_gen_event` | events / gen event | `n_events_selected / n_gen_events` |
| `{calo,trk}_nsteps[_ecut[_tcut]]_{total,per_event,per_gen_event}` (18) | steps, steps / event, steps / gen event | the `Calo` / `Tracker` `step counts` blocks, see below |

**Step counts** (Offline PR [#2067](https://github.com/Mu2e/Offline/pull/2067),
merged 2026-10-27; not yet in a Musing release). EdepAna counts detector steps
as an approximate hit count: `calo` is `CaloShowerStep`s (energy cut on
`energyDepBirks`), `trk` is `StrawGasStep`s (energy cut on `ionizingEdep`).
Three counts per detector: `nsteps` (all), `nsteps_ecut` (energy above the
threshold; strict) and `nsteps_ecut_tcut` (also with the step time inside the
inclusive window). Each is reported as the weighted `total`, `per_event` and
`per_gen_event`; the last two are -1 when the module had no events or no
generated events. Caveats: step times are raw Geant4 times, with no time
offsets and no microbunch folding; a `CaloShowerStep` is per particle per
crystal, so it is not a crystal-hit count; a `StrawGasStep` is roughly one
particle crossing one straw. The six parameters (defaults from the PR's example
fcl, all required by the module) are `trk_step_energy_threshold_mev` (0.0005),
`trk_step_time_min_ns` (500), `trk_step_time_max_ns` (1e9),
`calo_step_energy_threshold_mev` (0.5), `calo_step_time_min_ns` (500) and
`calo_step_time_max_ns` (1e9). What the module printed (thresholds and
windows) is in `metadata.step_count_windows`. The tree gets the same counts per
event, as the optional selection variables above.

The step counts depend on the Offline, which the server asks once per
environment (`mu2e --print-description EdepAna`, cached): with the PR the job
runs a per-run `edep_job.fcl` in the output directory (the shipped
`fcl/edep.fcl` plus the six parameter lines, since FHiCL cannot `#include` an
absolute path and an older EdepAna rejects the keys); without it the shipped fcl
runs unchanged, the step metrics are absent from the result (listed in
`list_analyses`, but not reported; `metadata.step_counts` says so), and
setting any step parameter away from its default is an error naming
Offline PR #2067. Existing metric names are the same either way.

`count` runs `print_counts.fcl` and reports what any art file with subrun
bookkeeping can say about itself:

| metric | unit | from the job's print |
|---|---|---|
| `n_events` | | `<N> Event records found` — events kept in the file |
| `n_gen_events` | | `GenEventCount total: <N> events in <M> SubRuns` |
| `prescale` | | `with prescale fraction <P>`, for the `prescale_filter`, or 1 |
| `saved_per_gen_event` | events / generated event | `n_events / (n_gen_events * prescale)` |

Its one parameter, `prescale_filter`, is **optional and has no assumed name**:
most files were never prescaled, and a job that did prescale one chooses its
own module labels. Left unset (the default, `""`) the prescale is 1 and the
answer is the file's own events per generated event; name a filter and that
filter's prescale is divided out, giving the rate before the prescale threw
events away. A filter that is named but not in the job's output is an error
rather than a silent fallback to 1 — that fallback would report a prescaled
file's rate short by exactly the prescale, with nothing in the output to show
for it. Either way `metadata.prescale_filters` lists every block the job
printed, so an unset run is also how you discover what there was to name.

`muon_stop_rate` is `count` with the target-stop filter, the mu-/mu+ split
and one more factor. It shares `count.py`'s parsing, input check and job run —
`count.py` is the only place that knows how to read `print_counts.fcl`'s lines
— and adds what is about muon stops rather than about counting. Its job is
`fcl/stop_counts.fcl`: `print_counts.fcl` plus the two `ParticleCodeFilter`s of
Production's `MuonStopSelector.fcl`, run as trigger paths (`selectMuminus`: a
mu- captured at rest; `selectMuplus`: a mu+), whose passed counts it reads
from art's TrigReport. A stop file holds both charges (about 1% mu+ in
MDC2025), and only the mu- stops feed a conversion, DIO or RMC
normalization — Production splits them into `MuminusStopsCat` and
`MuplusStopsCat` before any primary is made — so use the `muminus_*` rates
for those. The selectors read the stream's SimParticles, named after its
prescale filter (`TargetStopPrescaleFilter` -> `TargetStopFilter`), through a
per-run `stop_counts_job.fcl`. It reports:

| metric | unit | from the job's print |
|---|---|---|
| `n_events` | | `<N> Event records found` — stopped muons kept in the file |
| `n_gen_events` | | `GenEventCount total: <N> events in <M> SubRuns` |
| `prescale` | | `with prescale fraction <P>`, for the `prescale_filter` |
| `stops_per_gen_event` | stops / generated event | `n_events / (n_gen_events * prescale)` |
| `stops_per_pot` | stops / POT | the above times the required `upstream_eff` parameter |
| `n_muminus`, `n_muplus` | | the events path `selectMuminus` / `selectMuplus` passed (TrigReport) |
| `muminus_stops_per_gen_event`, `muplus_stops_per_gen_event` | stops / generated event | `n_muminus / (n_gen_events * prescale)`, and for mu+ |
| `muminus_stops_per_pot`, `muplus_stops_per_pot` | stops / POT | the above times `upstream_eff` |

`stops_per_gen_event` and `stops_per_pot` count every stop, mu+ included, as
before the split was added.

It takes two parameters:

- `upstream_eff` (required) — the efficiency of everything upstream, i.e.
  generated events of this file's stage per POT, POT -> MuBeam for a Run-1B
  TargetStops file. No default: the rate per POT is only as meaningful as the
  number you supply.
- `prescale_filter` (optional, default `TargetStopPrescaleFilter`) — the
  module label of the filter whose stream the file belongs to. Pass
  `PolyStopPrescaleFilter` to read a poly-stop file's own rate, or another
  label for a job that named its filters differently. Unlike `count`'s
  same-named knob it will not take `""`: a target-stop file is always
  prescaled, so "no filter" is not an answer here — use `count` for a file
  that was not.

`metadata` carries every `PrescaleFilterFraction` block the job printed
(`prescale_filters`), not just the one used, so the other streams' fractions
are there to read off, alongside the `prescale_filter` that was applied.

The production job writes *all* of its filters' products into *every* output
stream, so a poly-stop file parses fine against the target filter and would
silently be divided by the wrong prescale. The input's Mu2e file name is
therefore checked before the job is started, against the stream named by
`prescale_filter` (`TargetStopPrescaleFilter` -> the description must contain
"targetstop"), and anything else comes back as an error naming what was
passed. A file whose name is not in Mu2e's
`<tier>.<owner>.<description>.<config>.<sequencer>.<format>` form is left
alone and run.

`approx_ce_sensitivity` reports `sensitivity` (S/sqrt(B)),
`signal_box_low_mev` / `signal_box_high_mev`, `signal_rate`,
`dio_background`, `cosmic_background`, `total_background`, the signal peak's
`signal_mpv_mev` / `signal_fwhm_mev`, and the two assumptions those rates are
built on — `npot` and `cosmic_rate_per_s_per_mev` — so a number never travels
without its normalization. The summary line carries them too. Its `metadata`
also records the rest of the assumptions (`sig_eff`, `signal_br`, the
`cosmic_rate_per_mev` the cosmic rate works out to over the live time, and
`onspill_seconds`; with `stops_per_pot`, also `stops_per_pot`,
`ce_acceptance`, `n_gen_events` and `edep_run_recorded`), and it writes the macro's figures — `sig_vs_bkg.png`,
`dio.png`, `response.png`, `res.png`, `ce_z.png`, `ce_r.png` — into
`<output_dir>/figures`.

`stop_materials` reads the TH1 `<stop_module>/stopmat` that a stop finder
books in its job's TFileService output — one alphanumeric bin per stopping
material, labelled with the material name, filled once per stopped muon. It
takes two parameters:

- `n_gen_events` (required) — the generated events the input is equivalent
  to, in total over all files when several are passed. The ntuple has no generated-event bookkeeping, so every rate is the
  stop count divided by this number.
- `stop_module` (optional, default `TargetMuonFinder`) — which finder's
  histogram to read, e.g. `PolyMuonFinder` or `IPAMuonFinder`. If you name
  one that has no `stopmat`, the error lists the modules that do.

It reports the totals as metrics:

| metric | unit | |
|---|---|---|
| `n_stops` | | stops summed over the labelled bins |
| `n_gen_events` | | the parameter, repeated so the rate carries its normalization |
| `stops_per_gen_event` | stops / generated event | `n_stops / n_gen_events` |
| `stops_per_gen_event_err` | stops / generated event | from the histogram's bin errors |
| `n_materials` | | materials with at least one stop |

Several files (`data_files`) are combined into one table, as `hadd` would.
Bins are matched by material *name*: each file's axis is labelled in the
order its own job met the materials, so the same bin number can be a
different material in two files. A material that one file never saw counts
as zero there. `metadata.per_file` gives each file's own stop count,
histogram entries and anything outside its labelled bins, and an error in any
one file names that file.

The breakdown is in `metadata.materials`, most stops first: one row per
material with `material`, `stops`, `stops_err`, `stops_per_gen_event`,
`stops_per_gen_event_err` and `fraction`. The same table is written to
`<output_dir>/stop_materials.log`. ROOT extends an alphanumeric axis by
doubling it, so unlabelled empty bins are normal and are dropped. Content in
an unlabelled bin or in the under/overflow is kept out of the total and
reported as `unnamed_stops`, with the details per file in `per_file`
(`unlabelled_bins`, `underflow`, `overflow`).

## The trigger analyses

`trigger_efficiency`, `trigger_rate` and `trigger_timing` run the online
trigger menu's paths over art files, as one job each, built in
`tools/analyses/trigger.py`. All three take `trigger_paths`, the paths and
their prescales:

```python
parameters={"trigger_paths": "apr_TrkDe_80m70p:1, cpr_TrkDe_80m70p:10, calo_photon"}
```

A path without `:N` has prescale 1. The job is `fcl/trigger.fcl`, which
builds on `mu2e-trig-config/test/timingTest.fcl` (the physics menu with a
Prefetch module, so reading the data is not charged to the first
reconstruction module). Each run writes `trigger_job.fcl` into `output_dir`:
that text with `physics.trigger_paths` and each path's prescale filter set.
FHiCL can only `#include` through `FHICL_FILE_PATH`, not an absolute path,
so the base is copied in rather than included.

- **Testing a trigger change.** All three also take `fcl_overrides` (FHiCL
  appended after the base, which may `#include` a tuning fcl found on
  `FHICL_FILE_PATH`), `base_fcl` (an fcl to use instead of the timing test)
  and `leading_paths` (paths put first in `physics.trigger_paths`, unprescaled
  and not reported). Overrides go before the paths and prescales, which
  always win. For example the reduced-tracker tuned trigger, re-digitizing
  an existing digi file:

  ```python
  parameters={"trigger_paths": "apr_TrkDe_80m70p, cpr_TrkDe_80m70p",
              "base_fcl": "trigger/reduced_tracker/redigi_trigger_tuned.fcl",
              "leading_paths": "TriggerablePath"}
  ```

  or a single cut on the default menu:
  `"fcl_overrides": "physics.producers.TrigCalHelixFinderDe.HelixFinderAlg.minNHit : 7"`.
  The files named must be in the configured code. A custom base has no
  Prefetch module, so `trigger_timing` then charges reading the data to the
  first module. The `trigger_job.fcl` written to `output_dir` is the exact
  job, and `metadata` echoes `base_fcl` and `fcl_overrides`.
- **Prescales are applied in the job**, by each path's `PrescaleEvent` filter
  (`event % prescale == 0`), as online. The overall count therefore carries
  the real overlap between prescaled paths, and a prescaled-away path costs
  no time. The filter's label follows the menu generator's convention
  (`apr_TrkDe_80m70p` → `AprTrkDe80m70pPS`); a test checks it against every
  path in the published menu except the `tpr_`/`mpr_` ones, which are not
  meant for the real trigger.
- **Counts** come from art's TrigReport: events, events passing any path,
  and each path's passes. Fractions carry the width of a uniform-prior
  posterior as their uncertainty, so a path that never fired on a short
  sample is not reported as exactly zero ± 0.
- **`trigger_efficiency`** is relative to the events in the input: fold in
  any filtering the sample already had (e.g. a "Triggerable" sample) yourself.
- **`trigger_rate`** turns the accept fraction into Hz with the online event
  rate: the on-spill microbunch rate (`microbunch_rate_hz`, one per 1695 ns)
  times the spill duty factor. The duty factor follows `batch_mode` —
  `1BB` (default) 0.322, `2BB` 0.246 — unless `duty_factor` is given. `rate_hz`
  is averaged over the accelerator cycle, what the online system sees;
  `onspill_rate_hz` is the rate during the spill. The event rate, duty factor
  and microbunch rate are returned with every result.
- **`trigger_timing`** reads the per-event times from the TimeTracker
  database the job writes (`triggerTiming.db`, returned in `files`). An
  event's time is the **sum of its modules' times without fetching the
  data**: the Prefetch module (`PrefetchDAQData`) and the input source, which
  TimeTracker times apart, are left out, as `mu2eTimingPlotsMaker` totals an
  event. TimeTracker's own whole-event time includes the fetch, about 1 ms of
  ~6 ms per event on the CI pileup sample, so it is reported only for
  comparison (`metadata.fetch_time_mean_ms`, `metadata.full_event_time_mean_ms`).
  The first `skip_events` (default 1) are left out too: the first event
  carries the database and geometry initialization the running trigger
  does not pay, ~1 s against a few ms per event, and would double the mean
  of a few hundred events.
  `ignore_modules` (labels or module types, comma or space separated) leaves
  out work the real trigger would not do, such as `makeSD` re-making digis
  from MC truth in a `base_fcl` re-digitization job: they are timed apart
  like the fetch, listed in `metadata.modules` with `counted: false`, and
  their mean time is `metadata.ignored_time_mean_ms`. Names that never ran
  are in `metadata.ignored_modules_not_seen`, to catch typos.
  `metadata.modules` lists every module in the order it ran, with **N(seen)**
  (`n_seen`, the timed events it ran on after the filters and prescale
  upstream of it in its path, and `seen_fraction`), its time per run (mean,
  median, rms, max) and per timed event (`ms_per_event`). The data fetch is
  listed with `counted: false`. A module shared by several paths runs once
  per event, charged to whichever path reached it first, so its runs can be
  split over rows (e.g. `CaloHitMakerFast` under `calo_photon` on the events
  that path's prescale let through, under `apr_` on the rest);
  `n_seen_all_paths` and `ms_per_event_all_paths` add those up. `metadata.path_time_ms` is
  what each path adds: TimeTracker charges a module shared by several paths
  to the first that ran it. The times are wall-clock on whatever node runs
  the job, so they compare configurations run on the same node, not
  absolute online budgets.

Per-path results are in `metadata.paths`: prescale, passes, and the
efficiency or rate with its uncertainty. An unknown path name fails the job
with art's `Unknown path ... has been specified in 'trigger_paths'`, carried
in `metadata.stdout_tail`.

## trigger_efficiency_ntuple

The trigger efficiency without running anything: an EventNtuple records each
path's decision per event in a `trig_<path>` branch. Of the events with at
least one track passing `selection`, it reports the fraction any of
`trigger_paths` accepted, and each path's own (`metadata.paths`):

```python
run_analysis(analysis="trigger_efficiency_ntuple",
             data_file=".../nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root",
             output_dir=".../ntuple_eff",
             parameters={"trigger_paths": "apr_TrkDe_80m70p, cpr_TrkDe_80m70p",
                         "selection": "status == 1 and pdg == 11 and downstream "
                                      "and 100 < p_front < 110"})
```

`trigger_paths` are names separated by commas, with or without the `trig_`
prefix; a name with no branch is reported along with the ones the file has.
`selection` is a per-track cut in the `tools/selection.py` syntax over the
variables in `TRACK_VARIABLES` (repeated in the parameter's description):
the `trk` fields by name (`status`, `pdg`, `nactive`, `chisq`, `ndof`, ...),
`trkqual`, `pid`, momentum, angle and time at the tracker front and middle,
the helix fit parameters at TT_Mid (`d0`, `maxr`, `tandip`, `t0err`, ...),
and the matched calorimeter cluster. Two details:

- `*_front` quantities are at the **downstream-going** TT_Front crossing: a
  track can cross TT_Front twice, and the first crossing is usually the
  upstream leg. `downstream` means pz > 0 at TT_Mid.
- The fit parameters (`trksegpars_lh`) share the segments' indexing, so
  they are taken at the track's TT_Mid segment.

The default selection is a converged (`status == 1`) downstream electron fit
with p > 80 MeV/c at the tracker front, at least 15 active hits and
chi^2/dof < 5. On the CeMLeadingLogMix1BB example file it selects 7178 of
8212 events, of which 96.8% pass apr or cpr `TrkDe_80m70p`. The
track variables were checked against a per-track loop over the raw branches
(the same selected events, and identical TT_Mid fit parameters). Only the
leaves the variables need are read, about 2 s per file; `data_files`
combines several files into one result, with per-file counts in
`metadata.per_file`.

In a simulated file the selection can also use the track's **MC truth**
(`MC_VARIABLES`): its sim particle from `trkmcsim` (`mc_pdg`,
`mc_start_code`, `mc_p_start`, ...), its true crossings from `trksegsmc`
(`mc_p_front`, `mc_cos_front`, `mc_t_front`, `mc_p_mid`, ...; the MC
TT_Front crossing is the *earliest* with pz > 0, as the true steps are not
stored in time order), and the event's true beam intensity `mc_npot`
(`evtinfomc.nprotons`, protons on target in the microbunch; read by its full
path, since `evtinfo.nprotons` is a different, unfilled leaf). A file without
those branches is read without them, and naming an `mc_*` variable on it is
an error saying the file has no MC truth.

## trigger_efficiency_ntuple_binned

`trigger_efficiency_ntuple` as a curve: the selected events are also binned
in `variable`, any track variable or an expression of them in the selection
syntax, and the efficiency (any path, and each path) is reported per bin:

```python
run_analysis(analysis="trigger_efficiency_ntuple_binned",
             data_file=".../nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root",
             output_dir=".../eff_vs_npot",
             parameters={"trigger_paths": "apr_TrkDe_80m70p, cpr_TrkDe_80m70p",
                         "variable": "mc_npot / 1e6", "low": 0, "high": 80,
                         "n_bins": 8})
```

- `variable` (default `p_front`): e.g. `mc_p_front`, `p_front - mc_p_front`,
  `cos_front`, or the beam intensity `mc_npot / 1e6`. An event with several
  passing tracks is entered once, at its **first** passing track's value.
- Binning: `n_bins` uniform bins over `[low, high)` (default 15 over 80-110,
  2 MeV/c), or explicit `bin_edges` (`"80, 95, 100, 103, 105, 110"`).
- `trigger_paths` and `selection` are as in `trigger_efficiency_ntuple`, with
  the same default. For a curve in a true quantity, loosen the cut on the
  reconstructed one so it does not sculpt the curve. Events without any
  track are never in the denominator, so a curve in `mc_p_front` is the
  efficiency given a selected reconstructed track, not the full
  reconstruction-plus-trigger efficiency.

The overall metrics (`n_selected`, `n_triggered`, `efficiency`, ...) are the
same as `trigger_efficiency_ntuple`'s at the same selection. Added to them:
`n_bins`, `n_in_range`, `n_triggered_in_range`, `efficiency_in_range(_err)`,
and the selected events outside the binning (`n_underflow`, `n_overflow`)
or with no value (`n_undefined`, e.g. NaN `mc_p_front` for a track with no
true downstream TT_Front crossing). `metadata.bins` has one row per bin:
`low`, `high`, `n_selected`, `n_triggered`, `efficiency`, `efficiency_err`
and `paths` (each path's `passed` and efficiency). Each run writes into
`output_dir`, overwriting a previous run's:

- `trigger_efficiency_binned.png`: the curve, any path and (for several
  paths) each one, with the selection as title;
- `trigger_efficiency_binned.csv`: the `metadata.bins` table;
- `trigger_efficiency_binned.root`: TH1Ds `selected`, `triggered` and
  `triggered_<path>` (sumw2 = counts), for a `TEfficiency` in ROOT.

On the CeMLeadingLogMix1BB example file at the default selection, apr or cpr
`TrkDe_80m70p` is 97-98% efficient up to 40M protons per microbunch and drops
to about 88% at 50-70M (few events); in `p_front` it rises from about 62% at
80-82 MeV/c to 97% above 98 MeV/c.

## selection_efficiency_ntuple

A plain selection efficiency from EventNtuple file(s): the fraction of events
with a track passing `selection`, using the same track variables as
`trigger_efficiency_ntuple` (`TRACK_VARIABLES`), so a cut means the same thing
in both:

```python
run_analysis(analysis="selection_efficiency_ntuple",
             data_file=".../nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root",
             output_dir=".../sel_eff",
             parameters={"selection": "100 < p_front < 110",
                         "denominator": "status == 1 and pdg == 11 and downstream "
                                        "and p_front > 80 and nactive >= 15 "
                                        "and chisq / ndof < 5"})
```

`selection` is required. `denominator` defaults to `""`, which is every event
in the ntuple, including events with no track. An event passes when **one and
the same track** passes both cuts, so the passing events are always a subset
of the denominator events. The denominator counts ntuple entries, not
generated events: a production ntuple may already have dropped events
upstream. On the CeMLeadingLogMix1BB example file, the
`trigger_efficiency_ntuple` default cut passes 7178 of 8212 events (87.4%).
Of those, 6503 (90.6%) have a track at 100 < p_front < 110 MeV/c. Track
counts are in `n_tracks_denominator` and `n_tracks_passed`, and `data_files`
combines several files, with per-file counts in `metadata.per_file`. In a
simulated file both cuts can also use the MC truth variables (`mc_*`), as
in `trigger_efficiency_ntuple`; naming one on a file without truth is an
error saying so.

## momentum_resolution_ntuple

The two parts of the tracker's momentum response, from the MC truth in a
simulated EventNtuple, for every track passing `selection`:

- **resolution**: reconstructed minus true momentum at the tracker front
  (TT_Front), i.e. how well the fit measures the momentum it sees;
- **energy loss**: true momentum at creation minus true momentum at TT_Front,
  i.e. what the particle lost before reaching the tracker.

Their difference, reconstructed at TT_Front minus true at creation, is the
combined response `fullsim_sensitivity` folds the DIO spectrum with.

```python
run_analysis(analysis="momentum_resolution_ntuple",
             data_file=".../nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root",
             output_dir=".../resolution",
             parameters={"selection": "status == 1 and pdg == 11 and downstream "
                                      "and p_front > 80 and mc_start_code == 168"})
```

The truth is `trigger_efficiency_ntuple`'s MC variables: the true momentum
at creation (`mc_p_start`, the track's own particle in `trkmcsim`) and at the
downstream-going TT_Front crossing (`mc_p_front`, from `trksegsmc`, the
earliest such true step). The selection can use every track and MC variable
plus `resolution` and `energy_loss`, e.g. `mc_start_code == 168` for true CE
only. Its default is `trigger_efficiency_ntuple`'s default cut. A file
without `trksegsmc`/`trkmcsim` (e.g. data) is refused with a message saying
the analysis needs simulation.

Both distributions have long tails, so besides the mean and rms the metrics
give the median and `sigma68`, half the width of the central 68% (the sigma
of a Gaussian, without a fit), and `energy_loss_p90`. The metrics use every
track; the `*_min`/`*_max`/`*_bin_width` parameters set only the histograms
(`momentum_resolution.root`, TH1D `resolution` and `energy_loss`) and the
plots (`figures/resolution.png`, `figures/energy_loss.png`); the histograms
span `*_min` to `*_max` exactly, in the whole number of bins nearest
`*_bin_width` (the axis label gives the width used). How many tracks
fall outside them is in `metadata.outside_histogram_range`. On the
CeMLeadingLogMix1BB example file, with the default selection, 7192 tracks
in 7178 events give a resolution sigma68 of 0.22 MeV/c (median -0.005, rms
1.5) and an energy loss with median 0.73 MeV/c (mean 1.39, 90% below
2.5 MeV/c).

## calo_cluster_resolution_ntuple

The calorimeter's counterpart of `momentum_resolution_ntuple`: for every
cluster passing `selection`, its MC truth (`caloclustersmc`, one entry per
`caloclusters` entry) gives

- **dE** = reconstructed cluster energy minus `etot`, the true energy every
  particle deposited in the cluster's crystals;
- **dT** = reconstructed cluster time minus `tprimary`, the earliest true
  deposit of the cluster's leading particle (the one that deposited most).

```python
run_analysis(analysis="calo_cluster_resolution_ntuple",
             data_file=".../nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root",
             output_dir=".../calo_res",
             parameters={"selection": "mc_rel == 0 and energy > 50"})
```

The selection is per cluster, over `CLUSTER_VARIABLES` (repeated in the
parameter's description): the reconstructed `energy`, `time`, `disk`,
`size`, centroid radius `r` (in the disk's frame), ..., the truth (`mc_etot`,
`mc_eprimary`, `mc_tprimary`, `mc_nsim`, `mc_purity` = eprimary/etot), and
`de`, `de_frac`, `dt` themselves. Two things about the truth are easy to get
wrong:

- The truth's "primary" is the cluster's leading contributor, not the
  event's generated particle. `mc_rel` is how it is related to that
  particle (MCRelationship: -1 none, 0 same, 1 daughter, 2 mother,
  3 sibling), so the default `mc_rel == 0` keeps the clusters the generated
  particle made itself: the CE in a CE sample. Pileup clusters are -1.
- `tavg` is not used. Newer EventNtuple versions fill it with `tprimary`;
  older files carry something else, tens of ns away (on the example file it
  differs from `tprimary` for two thirds of the clusters).

A cluster without truth (`mc_nsim == 0`) has NaN MC values and never enters
the distributions. A file without `caloclustersmc` is refused, saying the
analysis needs simulation. The metrics are the mean, rms, median and
`sigma68` of dE and dT, and the median and sigma68 of `de_frac` (dE over the
true energy). The `de_*`/`dt_*` range and bin-width parameters set only
`calo_cluster_resolution.root` (TH1D `de`, `dt`) and the plots
(`figures/cluster_de.png`, `figures/cluster_dt.png`).

On the CeMLeadingLogMix1BB example file, the 6491 CE clusters above 50 MeV
(of 56159 clusters in 8212 events) give dE median -0.47 MeV, sigma68
1.35 MeV (1.6% of the true energy), and dT median +1.91 ns, sigma68
0.56 ns: the reconstructed time sits about 2 ns after the earliest true
deposit.

## variable_comparison_ntuple

Compares one variable between two sets of EventNtuple files: the input
(`data_file(s)`, "new") and a baseline (`reference_file(s)`, "ref"), e.g.
the same events before and after a code change, or two configurations of a
sample.

```python
run_analysis(analysis="variable_comparison_ntuple",
             data_file=".../nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root",
             reference_file=".../nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.001430_00000000.root",
             output_dir=".../compare",
             parameters={"variable": "p_front",
                         "selection": "status == 1 and pdg == 11 and downstream and p_front > 80"})
```

`level` says what the variable belongs to: `track` (the names of
`trigger_efficiency_ntuple`, `mc_*` included) or `cluster` (those of
`calo_cluster_resolution_ntuple`; the truth ones need `caloclustersmc`, the
rest work on data). `variable` is a name or an expression of them (a
true/false name such as `has_front` compares as 0/1), and
`selection` (empty by default: every object) picks the objects. Two
comparisons come out:

- **Distributions**, always: every selected object's value, overlaid with
  their ratio (`figures/variable_comparison.png`; a bin only the reference
  fills shows as ratio 0), scaled per event by
  default (`normalize`: `events`, `area` or `none`) so a change in how
  many objects pass shows. The metrics give each sample's count, mean, rms,
  median and sigma68, the mean shift (with its error as for independent
  samples) and median shift, and the two-sample KS statistic and p-value.
- **Matched events**, when the samples share events: each event's first
  selected object (the ntuple's order) in one sample against the other's.
  `n_events_gained` / `n_events_lost` count the matched events that have a
  selected object in only the new / only the reference sample,
  `n_events_changed` those whose value moved (by more than 1e-6
  relatively), and `diff_*` summarize new - ref over the events with a value
  in both (`figures/variable_difference.png`). "First" is the same object
  in both only when the objects come out in the same order, as they do for
  a reprocessing of the same events. With `match_events="auto"` events are
  matched when at least half of the smaller sample's (run, subrun, event)
  ids are in both, since independent samples can share a few by chance;
  `yes` insists, `no` skips it. A sample that repeats a (run, subrun,
  event), e.g. from files combined twice, cannot be paired: `auto` does
  not match it and `yes` is refused. Unmatched, the matched metrics are NaN
  (`n_events_matched` is 0).

One sample with nothing selected is still compared, since a reprocessing
can lose every object, or find the first ones: its statistics, the shifts
and the KS test are NaN, and the matched events count as gained or lost.
Only with nothing selected in either sample is it an error.

`range` (`'low, high'`; by default the 0.5-99.5% quantiles of both samples)
and `n_bins` set the histograms, not the metrics. `variable_comparison.root`
holds the unscaled TH1Ds `new`, `ref` and, when matched, `diff`; their
stored mean and RMS are the binned ones.

On the CeMLeadingLogMix1BB file 001430_00000000, the `MDC2025au_best_v1_1-001`
ntuple against the `MDC2025au_best_v1_1` one matches all 8212 events and finds
nothing changed, in tracks or clusters: the re-run reproduces the original.
Against the same CE sample without pileup (`CeMLeadingLogOnSpill.
MDC2025au_best_v1_5`, not the same events), cluster `energy` shows the
pileup clusters (below ~50 MeV, nearly all from the mixing) and the CE peak
near 90 MeV in both. Files from older EventNtuple versions (MDC2020) lack
branches the track and cluster readers need, and are refused saying which.

## muon_stops

The stop profile of a MuBeam-stage stop file: where the muons stopped (z
along the DS, radius from its axis), when, and what momentum they had where
they crossed ST_In, the virtual detector just upstream of the stopping
target.

```python
run_analysis(analysis="muon_stops",
             data_file=".../sim.mmackenz.TargetStops.Run1Bak_local0813111400.001800_00000000.art",
             output_dir=".../stops")
```

One mu2e job (`fcl/muon_stops.fcl`) runs two Offline modules on the stop
filter's collections. `StoppedParticlesDumper` writes each stop's position,
time and codes, as Offline's `TargetStopsDumper.fcl` and Production's
`Tests/TGTstops.fcl` do. `StepPointMCDumper` writes the stopped muons'
virtual-detector steps. `stop_filter` (default `TargetStopFilter`) points
both at another stream's collections: `PolyStopFilter` for PolyStops,
`IPAStopFilter` for IPAStops. The radius is from the DS axis, at
x = -3904 mm (`mu2e.solenoidOffset`). The ST_In momentum is that of every
downstream-going muon crossing of ST_In in the file. The dumpers write no
event number, so a crossing is not paired with its stop; in a stop file the
crossings are the stopped muons', and `n_st_in` is there to compare with
`n_stops`.

Outputs: the metrics (mean/rms/median of z and radius, mean/median time,
mean/median/rms of the ST_In momentum, with the stop and generated-event
counts), `muon_stops_hists.root` (TH1Ds `stop_z`, `stop_r`, `stop_time`,
`p_st_in` and the TH2D `stop_r_vs_z`) and the same as plots in `figures/`.
For a stopping *rate*, use `muon_stop_rate` on the same file: it divides out
the stream's prescale.

On the Run1Bak TargetStops example file, 735 stops from 12500 generated
events sit on the foils of its extended target, between z = 5000 and
6750 mm (mean 5415 mm), at a mean radius of 67 mm and a median time of
238 ns. Every stop has one ST_In crossing, with a median momentum of
32.8 MeV/c (5-49 MeV/c).

## approx_ce_sensitivity

A Python conversion of `Mu2eOptAna/scripts/rough_run1a_sensitivity.C`. Signal
is CE, background is DIO plus cosmics, and it estimates S/sqrt(B) for the best
momentum window:

1. **Signal shape** — `primary_trk_front_energy` (energy at the front of the
   tracker) for the events passing `selection` (default: >10 MeV visible in
   the calorimeter, the macro's `hist_2` cut), filled from `EDepAna/tree` with
   the binning of EdepAna's `trk_front_energy` histogram, rebinned x2 and
   scaled to a rate for `npot` protons at `SIGNAL_BR` (R_mue = 1e-13 per
   capture times 0.609 captures per stop on aluminum) and
   `sig_eff`, then smeared by a Gaussian tracker resolution (sigma = 0.2 MeV).
   Pass `stops_per_pot` instead of `sig_eff` to have it worked out:
   `sig_eff = stops_per_pot * acceptance`, the acceptance being the file's
   events passing `selection` (weighted) per generated CE event, from
   EdepAna's running `ngen` — so the selection sets the signal's size as well
   as its shape. Pass exactly one of the two. The file must come from one
   full `edep` run: one cut short by `max_events` undercounts the acceptance
   and is refused when `edep`'s record says so (a file without the record,
   e.g. from a production job, runs with a warning in the log), and a merged
   (hadd'ed) one is refused when `ngen` falls — not every merge shows that
   way, so do not merge `edep` outputs.
2. **DIO** — the Heeck/Szafron theoretical spectrum, scaled to a rate, then
   smeared by the *measured* energy-loss response
   of the same events (`primary_trk_front_energy_diff`) and the same
   resolution.
3. **Cosmics** — flat in momentum at `cosmic_rate_per_s_per_mev` (default:
   the rough Run-1A mu- -> e- rate), scaled by the on-spill live time implied
   by `npot`.
4. **Window scan** — every `[x1, x2]` with `x1 >= 50 MeV`, keeping the best
   S/sqrt(B). The top 10 windows go to the log.

### Box cuts (`cut_mode="box"`)

`cut_mode="momentum"` (the default) is everything above: the momentum window
alone, with no time or geometry cut, and every number as it was before the box
option existed. `cut_mode="box"` also optimizes cuts on **time, R(max), d0 and
cos(theta)** next to the momentum window, by coordinate ascent on the Punzi
figure of merit S/(a/2 + sqrt(B)) with a = 1.28 (90% CL). Not S/sqrt(B): with
five cut variables the background can be cut to ~0 while keeping a sliver of
signal, where S/sqrt(B) diverges; the expected limit scales roughly as
1/FoM. It
needs a tree with the tracker-front helix branches (see the selection section;
an Offline with PR #2092, not yet in a Musing) and otherwise returns an error
saying so and that `cut_mode="momentum"` works on the file.

New parameters: `cut_mode`, `muon_lifetime_ns` (default 864, the bound-muon
lifetime in aluminum), `time_min_ns` / `time_max_ns` (default 525 / 1650: the
base time cut, which the time optimization may only tighten). In momentum mode
the time window is **not** applied.

`tsda_veto` (box mode only, default 0) set to 1 adds a fast-sim version of the
reco TSDA veto, which rejects a track whose upstream extrapolation reaches the
TS downstream absorber (an annulus, 235 < r < 525 mm) instead of reflecting
in the DS gradient. With rough nominal fields, 1 T at the tracker front and
2 T at the TSDA (no field map), a track reaches it if
sin^2(theta) x 2 T / 1 T < 1 (cos(theta) > 0.707, from p_T^2/B conserved). Its
helix center rc = (R(max) + d0)/2 and radius R = (R(max) - d0)/2 shrink by
sqrt(1/2) on the way, and with the phase at the TSDA unknown its chance of
hitting is the fraction of the turn inside the annulus. Each CE event and
cosmic sample point is weighted by 1 - that chance; DIO takes the CE's
efficiency. `ce_tsda_survival` and `cosmic_tsda_survival` report the
surviving fractions before the box cuts (-1 when off). A box whose cos cut
stays below 0.707 is unaffected, since nothing in it reaches the TSDA.

`front_time_min_ns` (box mode only, default 500; 0 turns it off) models the
production's digitization window: a CE event whose tracker-front time
(EdepAna `primary_trk_front_time`), folded into the 1695 ns microbunch, is
below it is dropped from the selection, so from the CE acceptance, the signal
shape and the DIO response alike. The events' own times carry the stopped
muons' arrival times, which shift with the stopping target (the alternate
designs' CEs arrive ~25-35 ns earlier, and the production keeps ~3% fewer of
them), something the exponential time model cannot see. To not count the
early loss twice, the box's CE time efficiency is then the exponential's share
of [t1, t2] above the cut, eps_t / eps(front_time_min_ns, T).
`ce_front_time_survival` reports the kept fraction of the selection (-1 in
momentum mode, which ignores the parameter).

Model. For the cosmics, momentum, time and the (helix-center radius rc, cos)
distribution are taken as mutually separable; for the CE only time is separable
from the rest, momentum and geometry are not (see S below). For a box [p1,p2] x [t1,t2] x
R(max) x d0 x cos:

- CE time follows the muon lifetime over one 1695 ns microbunch, density
  exp(-t/tau): eps_t = (e^(-t1/tau) - e^(-t2/tau)) / (1 - e^(-T/tau)).
  With `front_time_min_ns` = t_f > 0 this becomes the share above t_f,
  eps_t(max(t1, t_f), max(t2, t_f)) / eps_t(t_f, T), on top of the per-event
  front-time cut.
  Cosmic time is flat over the time range of the loose cosmic shapes (the `t0`
  axis of the shapes file, 400-1700 ns): eps_t = overlap / width.
- CE geometry is counted per event: the weighted fraction of the selected
  events that reached the tracker front whose (R(max), d0, cos) pass.
- Cosmic geometry is a fixed-seed sample (200k points) drawn from the shapes
  file's `rc_vs_cos` (bin by content, uniform inside it) with p flat over the
  signal bins inside the shapes' momentum range (100-110 MeV/c) and
  R = p sin(theta) / (0.299792458 B), B the effective field 1.036 T;
  R(max) = rc + R, d0 = rc - R. Each point keeps its momentum bin, so the
  geometry efficiency is counted in the box's own p window (R(max) and d0 grow
  with p: a high R(max) cut passes only high-momentum cosmics). The p window is
  therefore kept inside 100-110 MeV/c, where the loose rate is defined.
- S = eps_CE,t x the sum, over the CE events passing the geometry cuts, of
  weight x W(bin) x (the signal's per-entry normalization), W(j) being the
  fraction of a unit entry in signal bin j that the tracker-resolution smearing
  puts in the p window (a response matrix built with the same `Spectrum.smear`
  as the signal). That is the smeared signal spectrum of only the passing
  events: a d0 or R(max) cut removes low-energy-loss events preferentially, so
  the in-window fraction is not the momentum-integrated one. With open geometry
  it is the plain window sum. The log reports the in-window geometry
  efficiency (signal with the geometry cuts / without) next to the
  momentum-integrated one; `ce_box_efficiency` stays time x the
  momentum-integrated geometry efficiency.
  DIO = (DIO summed over the p window) x eps_CE,t x the CE's
  momentum-integrated geometry efficiency: **a known first-order
  approximation**, the DIO's time and geometry efficiencies are taken equal to
  the CE's; DIO is a different momentum spectrum, so the CE's in-window
  correlation does not transfer. Cosmic = eps_cos,t x the sum over the p
  window of (cosmic spectrum in the bin) x (the sample's geometry efficiency
  in that bin), with no further scaling; `cosmic_box_efficiency` is eps_cos,t
  x the geometry efficiency in the window. Window sums are direct slice sums.
- Because the box applies the time/R(max)/d0/cos cuts explicitly, `sig_eff`
  must **not** include their efficiency (nor may `selection`, when
  `stops_per_pot` takes the acceptance from it).

**The cosmic rate.** In box mode `cosmic_rate_per_s_per_mev` is the
cosmic-electron rate for the *loose* selection the shapes were extracted with
(ConvAna set 81: e-, APR/CPR trigger, upstream-reflection and multi-track
vetoes, 100 < p < 110 MeV/c; no t0, cos, R(max) or d0 cut). Its default (0
means "the recommended rate for `cut_mode`") is the Run-1A rate 10/7.8e5 times
the loose/tight yield ratio of the shapes file (`reference_cuts.combined`:
0.0320 of the loose set passes the tight Run-1A cuts 540 < t0 < 1650 ns,
0.525 < cos < 0.649, 482 < R(max) < 642.6 mm, so about 31x, ~4.0e-4 per s per
MeV/c). In momentum mode the default is 10/7.8e5, applied as-is: the rate after
whatever fixed selection the result is meant to represent. A value you pass
overrides either; the metric and log report the value used, and the run's
`cosmic_rate_source` says `mode default` or `user` (`cosmic_loose_to_tight_ratio`
reports the ratio). In box mode `momentum_only_sensitivity` is computed with
this loose-selection rate, so it is not the number momentum mode gives by
default.

Optimization. It starts from the momentum-only window (clipped to the cosmic
sample's 100-110 MeV/c), the tight Run-1A cuts (time clipped to the base
window) and no d0 cut; if no CE passes those geometry cuts it starts from open
geometry instead, since a step that moves one variable could not leave it. Each step re-optimizes one
variable's (low, high) pair jointly on a grid with the others fixed, by
histogramming the variable for the events/sample points passing the other cuts
and using cumulative sums outward from each lower edge (O(N + G^2)); a step is
kept only if it does not lower the FoM. Order p, t, R(max), d0, cos.
Grids: p = the spectrum bins in 100-110 MeV/c, t 5 ns steps inside the base window,
R(max) 400-1100 mm in 5 mm, d0 -700..700 mm in 5 mm, cos 0..1 in 0.005 (the
d0 and cos grids have under/overflow bins, so a cut at a grid end is open on
that side). R(max) is never open below: a track needs R(max) > 400 mm to have
hits in the tracker, so `RMAX_MIN_MM` is the floor of its lower cut for CE and
cosmics alike (applied wherever the geometry is counted, so `ce_geometry` is
relative to all selected events, including those below 400 mm). Sweeps stop when one gains < 1e-6 relative or after 20.

Metrics. In box mode `sensitivity`, `signal_box_low_mev` / `_high_mev` (bin
centers, as in momentum mode), `signal_rate`, `dio_background`,
`cosmic_background` and `total_background` describe the optimized box. The new
metrics are always reported, -1 in momentum mode: `time_low_ns`,
`time_high_ns`, `rmax_low_mm`, `rmax_high_mm`, `d0_low_mm`, `d0_high_mm`,
`cos_low`, `cos_high`, `ce_box_efficiency` (eps_CE,t x eps_CE,geom) and
`cosmic_box_efficiency`; `momentum_only_sensitivity` is the momentum-only
result in both modes, and `punzi_fom` is the Punzi FoM of the headline window
or box in both modes. The log lists the final cuts, efficiencies, the
loose/tight ratio and each sweep's FoM. Box mode adds one PNG per
variable in `figures/` (`box_p.png`, `box_t.png`, `box_rmax.png`,
`box_d0.png`, `box_cos.png`): the cosmic distribution as expected counts per
bin in the optimized configuration, "nominal" (base time window, p in 100-110
MeV/c, no box cuts) and "after the other cuts" (all optimized cuts but this
variable's), with the selected range shaded. The CE signal is overlaid on a
right-hand axis ("Expected CE counts / bin", dashed blue for "after", dotted
for "nominal") in the same normalization as `signal_rate`: the smeared
spectrum of the passing events for p, the lifetime model binned in time, and
for the geometry variables the events' in-window signal.

The cosmic shapes are `data/cosmic_shapes.json` and `data/cosmic_shapes.root`,
found relative to the module. Regenerate them with
`scripts/extract_cosmic_shapes.py` (read its docstring for the input and
options).

The numbers are rough by construction: this is a figure of merit for comparing
beamline configurations, not a sensitivity calculation. Note it needs a **CE
signal** sample — given a file where no event passes the selection and
reaches the tracker it reports that plainly instead of dividing by zero. It
also needs an ntuple from EdepAna in Offline v13_39_00 or later (the step
counts, from PR #2067, are not needed): older ones
have only the histograms, and are refused with a message saying to rerun
`edep`.

Three deviations from the macro, all deliberate:

- It logs the best window for each of the 10 best lower edges rather than all
  ~100k scanned — the top 10 overall only ever differ by a bin.
- Window sums are accumulated from each window's own edge, never as
  differences of whole-spectrum prefix sums. The DIO spectrum spans ~18 orders
  of magnitude, and differencing totals of ~4e17 to get a count of order 1
  loses it entirely to float cancellation (it silently reported
  `dio_background = 0`). `tests/test_tools.py` locks this in.
- Smearing is `np.convolve` against a kernel resampled onto the smeared
  spectrum's own binning (`spectrum.py`), not a bin-by-bin redistribution: the
  Gaussian resolution is integrated over each offset bin and the measured
  response is interpolated rather than snapped to the nearest bin. On the
  Run-1B CE sample that moves `dio_background` by ~2% and `sensitivity` by
  0.01%.

## fullsim_sensitivity

The full-simulation counterpart of `approx_ce_sensitivity`, built the same
way: `sensitivity` = S/sqrt(B) in the best momentum window, with a DIO and
a cosmic background. Where `approx_ce_sensitivity` takes the signal shape
and energy loss from EdepAna's truth-level tree and assumes an efficiency
and a Gaussian resolution, this measures all of them from reconstructed CE
mixed with pileup: CeMLeadingLogMix1BB EventNtuple files, e.g.
`/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_1-001/`.
`fullsim/README.md` says how to run it.

1. **Cuts**: pyfitter's cut-set 80 (`80_1d`), track by track, as its
   `Analyze.define_cuts` applies them and in its order (`fullsim/cuts.py`).
   On by default: every cut pyfitter switches on except its fit ranges
   (`in_mom_range`, `within_t0`), which the momentum scan and `time_window`
   stand in for. Add or drop cuts by name with `enable_cuts` /
   `disable_cuts`, and move the `trkqual_min` / `trkpid_min` thresholds
   (default 0.155 and 0.54). `or_trigger` passes an event that any of
   `trigger_paths` accepted, by default the production tracker paths
   `apr_TrkDe_80m70p` or `cpr_TrkDe_80m70p`, as in pyfitter. With every cut
   on, the cut flow on a CE mix file matches pyfitter's cut for cut. The cut
   flow goes to `cut_flow.csv` and to `metadata.cut_flow`.
2. **Signal**: each event with a selected track is represented by its
   first selected track: momentum and time at the tracker front, MC origin,
   and true momentum at birth. An event whose first track has more than one
   front segment is not counted, as in pyCount. The signal is the true CE
   among these with a time inside `time_window` (default 640-1650 ns, open),
   so a pileup track passing the cuts is not signal. Their reconstructed
   momentum, binned as `approx_ce_sensitivity`'s signal (0.2 MeV/c), is
   scaled as Production's `normalizations.py` has it: NPOT x
   `stopped_muons_per_pot` x 0.609 captures per stopped mu- x `rmue`
   (default 1e-13) x efficiency. `stopped_muons_per_pot` is
   MuBeamCat x MuminusStopsCat efficiency x 1000, computed by default from
   the stop chain's SAM datasets (7.67e-4 for the MDC2025 chain) as
   Production's `CreateSimEfficiency.sh` does with
   `mu2eGenFilterEff`: events in each dataset over events generated
   (`fullsim/normalization.py`). The chain is traced from the input files'
   SAM ancestry (`stop_datasets` `auto`): the MuminusStopsCat dataset the
   CE were generated from and the MuBeamCat dataset behind it. No SimEfficiencies2 table is needed, so a
   new iteration of the simulation works before it is in the database;
   pass a number to skip SAM. The
   efficiency is measured: CE counted over the events generated to make the
   input files, so it includes the digitization filter's acceptance. The
   generated count is each file's `dh.gencount` in SAM, from its nearest
   ancestor that has one (the parent mcs file; `fullsim/provenance.py`),
   or the `n_generated` parameter where SAM cannot be reached.
   `upstream_eff` (default 1) multiplies in any further loss.
   No resolution is added; the reconstruction already has it.
3. **DIO**: `approx_ce_sensitivity`'s Heeck/Szafron spectrum, as a count per
   0.01 MeV bin for NPOT x `stopped_muons_per_pot` x 0.391 DIO per stopped
   mu-, folded with the measured response of the signal
   events (reconstructed momentum at the tracker front minus true momentum
   at birth: energy loss and resolution together). The response integrates
   to the measured efficiency, so DIO electrons pass the selection as CE do.
4. **Cosmics**: flat in momentum at `cosmic_rate_per_s_per_mev` over the
   on-spill time NPOT implies, exactly as in `approx_ce_sensitivity`, whose
   defaults (`npot`, `mean_pot_per_event`, the rate) it shares.
5. **Window**: `approx_ce_sensitivity`'s `scan_signal_box` over momentum. The
   time window is fixed.

The metrics carry `approx_ce_sensitivity`'s meanings: `signal_rate`,
`dio_background`, `cosmic_background`, `total_background` in the window,
and `npot`, `stopped_muons_per_pot`, `n_stopped_muons`, `rmue`,
`n_conversions` and `cosmic_rate_per_s_per_mev` behind them. Also
`signal_efficiency` (overall) and `signal_efficiency_window`,
`n_signal_window` (reconstructed CE in the window), and the event counts.
Outputs: `fullsim_sensitivity.log` (with the top windows scanned),
`cut_flow.csv`, and `figures/`: signal vs background, the measured
response, and momentum vs time.

On `nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root`
(20000 generated, 8212 reaching the ntuple: acceptance 0.41; about 6 s),
3010 CE pass the cuts and the time window: efficiency 0.15. For 1e18 POT
(7.67e14 stopped mu-, 46.7 conversions at R_mue = 1e-13) the best window is
[103.5, 104.7] MeV/c with S = 4.69, cosmics 1.90 and DIO 0.0009: with
stopped muons per POT in, the DIO is negligible and the cosmics set the
window. `tests/test_tools.py` locks this in.

Unlike `approx_ce_sensitivity`, whose normalization is its macro's (signal
NPOT x BR and DIO NPOT x (1 - capture fraction), with no stopped muons per
POT), the signal and DIO here are counts for the stopped muons NPOT gives.

## The occupancy analyses

How many hits, digis and clusters an event holds: `occupancy_digi` (dig files),
`occupancy_reco` (mcs/rec files) and `occupancy_ntuple` (EventNtuple files).
The first two run `fcl/occupancy.fcl`, i.e. the `OccupancyAna` analyzer of
Offline PR #2070 (merged 2026-10-28, in no Musing release yet, so they need a
work area built on an Offline with it; on any other, the result says so and
names the PR). It needs no geometry or conditions. Both are configured by
a tag per collection, `<collection>_tag`, where an empty string skips the
collection:

| analysis | collections (default tag) |
|---|---|
| `occupancy_digi` | `strawdigis` (`makeSD`), `calodigis` (`CaloDigiMaker`), `crvdigis` (`CrvDigi`) |
| `occupancy_reco` | `strawhits` (`makeSH`, the single-straw ComboHits), `combohits` (`makePH`, the panel ComboHits), `calohits` (`CaloHitMaker`), `caloclusters` (`CaloClusterMaker`), `calorecodigis` (off) |

The defaults are the production labels; check a file's with
`mu2e -c Offline/Print/fcl/dumpDataProducts.fcl -s FILE -n 1`. An mcs file's
digi, CRV pulse and CRV cluster collections are SelectReco subsets, kept for
the reconstructed objects, not the full occupancy, so `occupancy_reco` does not
offer them, and CaloRecoDigis only as an option that is off by default. Use
`occupancy_digi` on a dig file for digi occupancy. `occupancy_reco` also takes
a window, `hit_time_min`, `hit_time_max` (ns) and `calo_hit_energy_min` (MeV),
default everything, applied to the straw hit and panel ComboHit corrected time
and the calo hit time and energy.

```python
run_analysis(analysis="occupancy_reco",
             data_file=".../mcs.mu2e.CeEndpointMix1BB-KL.Run1Baw_best_v1_5.001470_00000001.art",
             output_dir=".../occupancy", max_events=50,
             parameters={"hit_time_min": 700, "calo_hit_energy_min": 10})
```

Each run writes its own `occupancy_job.fcl` in `output_dir`: the shipped
`fcl/occupancy.fcl` (which configures no collection) with the tags and window
appended. For every collection of the analysis there are `<c>_mean`, `<c>_rms`
and `<c>_max` (per event, over the events that have it), and
`<c>_events_found` and `<c>_events_missing`, from the summary table
OccupancyAna prints. From the per-event tree come `n_events`, the mean count in
the busiest tracker plane (`strawdigis_`/`strawhits_busiest_plane_mean`, with
every plane's mean in `metadata.per_plane_mean`) and in each calorimeter disk
(`calodigis_disk0_mean`, `calohits_disk1_mean`, ...), and for `occupancy_reco`
the windowed counts `strawhits_sel_mean`, `combohits_sel_mean`,
`calohits_sel_mean` and `combohit_straws_mean` (the straw hits in the panel
ComboHits). A skipped collection's metrics are -1 so every listed metric is
always there, and so are the statistics of a collection no event had. A
configured collection missing from every event is flagged in
`metadata.warnings` and the message, since it nearly always means a wrong tag.
There is no per-generated-event normalization: OccupancyAna has no gen count.

`occupancy_ntuple` runs no job. From an EventNtuple it reads `hitcount` (the
whole event, not a SelectReco subset: `nsd`, the straw digis, and the straw
hits flagged energy-, radius-, time-selected and background, `nesel`,
`nrsel`, `ntsel`, `nbkg`), the sizes of `calohits`, `caloclusters` and
`crvcoincs`, `crvsummary` (`nHitCounters`, `totalPEs`) and `tcnt.ntrk`, each
as `<q>_mean`, `<q>_rms` and `<q>_max` (`strawdigis`, `strawhits_esel`,
`calohits`, `crv_total_pes`, `ntracks`, ...). The calo window
parameters give `calohits_sel_mean`. Every group is optional: what a file
lacks is listed in `metadata.missing` and its metrics are -1, and it is an
error only if none is there. `data_files` combines several files; a quantity
missing from one of them is dropped for the set. It has no per-generated-event
normalization either: the `n_gen_events` histogram counts the generated events
behind the file, but the ntuple holds only the events that survived
production (622 of 1e6 in the Run1B example), so a per-gen-event rate would
mix occupancy with that filtering.

On the Run1B `CeEndpointMix1BB-KL` dataset, `occupancy_reco` (50 events of
`...001470_00000001.art`) gives about 779 straw hits, 276 panel ComboHits
(410 straws), 56 calo hits (46 on disk 0) and 5.5 clusters per event, and
`occupancy_ntuple` (622 events of `...001470_00000010.root`) gives
749 straw digis, 54 calo hits and 5.7 clusters, CRV and background
branches all zero.

## total_edep_per_pot

The dose from the three beam-related sources, per POT. Each source is
simulated from its own sample and `edep` reports it per generated event: a
resampled beam electron (early electron-beam flash), a resampled MuBeamCat
event (early muon-beam flash, unprescaled), or a muon stop (MuStopPileup on
the target stops). This puts them on one footing and adds them:

```
per POT = ebeam / pot_per_ebeam_event      (default 11.5367: Run1Baa EleBeamCat, 25.0M POT / 2.17M electrons)
        + mubeam * mubeam_events_per_pot   (default 0.01278168: muon_stop_rate's upstream_eff)
        + mustop * stops_per_pot           (muon_stop_rate on the same geometry)
```

Pass the same detector's deposition (`avg_trk_edep_per_gen_event_mev`, or
the calo one) for all three. It reports `total_edep_per_pot_mev`, each
source's `*_edep_per_pot_mev` and `*_fraction`. A workflow chains it by
passing the earlier metrics as its parameters; Mu2eBO's
`foilsflash_col5bf_ax` uses the tracker total as its dose objective.

## Adding an analysis

1. Write `tools/analyses/<name>.py` with a `run(context) -> RunOutcome` and a
   `SPEC`. For a mu2e job, hand `run_mu2e_job` your fcl and parse its stdout:

```python
def run(context: RunContext) -> RunOutcome:
    outcome = run_mu2e_job(fcl=FCL, input_paths=context.input_paths,
                           outdir=context.outdir, timeout_s=context.timeout_s,
                           single=len(context.input_paths) == 1
                                  and not context.wants_file_list,
                           max_events=context.max_events)
    if outcome.failed:
        return RunOutcome(error=f"mu2e exited {outcome.returncode}",
                          log_path=outcome.log_path)
    return RunOutcome(metrics=parse_stops_summary(outcome.stdout),
                      files=outcome.written_root_files, log_path=outcome.log_path)

SPEC = AnalysisSpec(
    name="stops",
    input_kind="art_files",
    fcl=FCL_DIR / "stops.fcl",          # i.e. fcl/stops.fcl in this repo
    description="Muon stops per POT.",
    metrics=("n_stops", "stops_per_pot"),
    units={"stops_per_pot": "stops / POT"},
    parameters=(ParamSpec(name="pot_per_event", description="...",
                          default=1.6e7, minimum=0.0),),
    run=run, summarize=lambda m: f"{m['n_stops']:g} stops",
    input_hint="art file(s) with ...",
)
```

   For a ROOT-file analysis use `input_kind="root_file"`, name the analysis
   that produces the input in `produced_by`, read `context.input_path`, and
   leave `fcl` unset — `approx_ce_sensitivity.py` is the worked example.

2. Add `"stops"` to `_ANALYSIS_MODULES` in `tools/registry.py`.
3. Extend `tests/test_tools.py` (a stdout sample in `SAMPLE_STDOUT` for an
   `art_files` analysis; direct tests of the computation for a `root_file` one).

That's it — the `analysis` enum, `list_analyses`, and the registry tests pick
it up. Step 3 is not optional in practice: the registry tests loop over every
analysis and fail until each one is covered.

Physics knobs go in `parameters` as `ParamSpec`s (validated, with defaults and
ranges, and reported by `list_analyses`) rather than into the tool signature,
which stays the same however many analyses exist.

## Environment

No installs needed on the mu2e machines — the `ana` python already has the
`mcp` SDK:

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana            # Python 3.12 with mcp + pydantic
```

The server sets up mu2e **itself**, once per job, in a fresh bash subprocess,
so it does not care whether the shell that launched it had the Offline
environment. Where that Offline comes from is one setting, fixed when the
server starts — `tools/mu2e_env.py` is the only place that knows how:

| flag | environment variable | what a job runs |
|---|---|---|
| `--work-area <dir>` | `MU2E_WORK_AREA` | `cd <dir> && muse setup` |
| `--musing 'SimJob MDC2025ay'` | `MU2E_MUSING` | `muse setup SimJob MDC2025ay` |
| `--code-tarball <file>` | `MU2E_CODE_TARBALL` | unpack it, then `muse setup` in the tree |

```bash
python3 -m analysis_mcp_server --transport stdio \
    --work-area /exp/mu2e/app/users/mmackenz/mu2eopt
python3 -m analysis_mcp_server --transport stdio --musing 'SimJob MDC2025ay'
python3 -m analysis_mcp_server --transport stdio --code-tarball ~/code.tar
```

The flags are mutually exclusive; with none of them (and no environment
variable) the server falls back to `DEFAULT_WORK_AREA` in `tools/mu2e_env.py`.
A work area that is not a directory, a tarball that is not a file, or a Musing
without a version is a startup error naming itself, not a failure inside the
first job. `list_analyses` reports the environment in use, and every mu2e
result carries it in `metadata.environment`, so a number can be traced to the
code that produced it.

A tarball is unpacked once — into `<output_dir>/code` by default, so runs stay
self-contained, or into `--code-dir <dir>` to share one unpacking between
them. A `<dir>.unpacked` marker beside it means later jobs reuse it rather
than unpacking again. `muse setup` runs in the unpacked root, or in its single
top-level directory if that is all the tarball holds; `--code-subdir` says so
explicitly when it holds something else.

### Which analyses a given environment can run

The analyses here ship their fcl with the server, in `fcl/` (`FCL_DIR` in
`tools/spec.py`), and name it by absolute path — the same file whichever
environment is configured, checked before the job starts and reported by
`list_analyses` as `fcl_exists`. Its `#include`s (`Offline/...`,
`Production/...`) are resolved by art on `FHICL_FILE_PATH`.

An analysis may instead name its fcl **relative** to the configured code. For
a work area or an unpacked tarball that is a path on disk and is checked the
same way; a Musing has no directory of ours to look in, so the relative path
goes to `mu2e`, art resolves it on `FHICL_FILE_PATH`, `fcl_exists` comes back
`null`, and a fcl that is not there fails in the job with art's own
`Can't find file "..."`, which the result carries in `metadata.stdout_tail`.

> **What the analyses here need from the environment**: `EdepAna` is in
> Offline (`Offline/Analyses/src/EdepAna_module.cc`) from v13_39_00, so any
> Musing built on it — `SimJob MDC2025ay` and later — or a work area backed by
> one runs `edep` as is, without the step counts: those need an EdepAna with
> Offline PR #2067 (merged 2026-10-27), and `SimJob MDC2025ay` predates it.
> With an older Offline, art stops with `Library
> specification "EdepAna" does not correspond to any library`, and the result
> carries it in `metadata.stdout_tail`.

> `occupancy_digi` and `occupancy_reco` need `OccupancyAna`, which is in Offline
> only after PR #2070 (merged 2026-10-28, in no Musing release yet). On an
> Offline without it the job fails with the same `Library specification
> "OccupancyAna" does not correspond to any library`, and the result says that
> it needs the PR. `occupancy_ntuple` runs no job and works anywhere.

## Test

```bash
python3 tests/test_tools.py
```

231 tests, none of which start a mu2e job or reach SAM or metacat (the
catalog tests use a fake). (The `ana` env has no pytest, so
these are bare asserts.)

## Run the server

**stdio** — the client spawns the server; nothing to start by hand:

```bash
python3 -m analysis_mcp_server --transport stdio
```

**Streamable HTTP** — the server is a visible process with a URL:

```bash
python3 -m analysis_mcp_server --transport streamable-http --port 8000
```

Clients connect to `http://127.0.0.1:8000/mcp`. Stop it with **Ctrl+C**
(Ctrl+Z only suspends it and keeps the port; just start again — a leftover
`analysis_mcp_server` holding the port is cleared automatically).

## A worked example

`examples/simple_client.py` is a small MCP client that does what an agent
would: start the server, list the tools, `list_analyses`, run `edep` over an
art file you name, optionally run `muon_stop_rate` over a target-stop file,
then feed the `nts.*.root` `edep` wrote to `approx_ce_sensitivity`.

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana 2.7.0

# the client starts the server itself over stdio
python3 examples/simple_client.py ../dts.mmackenz.CeEndpoint.<...>.art
```

A quick check that skips the full job (and so the chained sensitivity, whose
per-gen-event normalization `--max-events` invalidates):

```bash
python3 examples/simple_client.py path/to/some.art --max-events 200 --no-chain
```

Against a server you started by hand, same code over HTTP:

```bash
python3 -m analysis_mcp_server --transport streamable-http --port 8000 &
python3 examples/simple_client.py path/to/some.art --url http://127.0.0.1:8000/mcp
```

The input path may be relative or use `~`; the client resolves it, since the
tool itself takes only absolute paths (the job runs in `output_dir`, not in
your shell's directory).

`muon_stop_rate` needs an input of its own, so it runs only when you name
one:

```bash
python3 examples/simple_client.py path/to/a/CeEndpoint/dts.art \
    --stops-file ../sim.mmackenz.TargetStops.<...>.art
```

That run passes `upstream_eff` (`--upstream-eff`, default 0.012) and leaves
`prescale_filter` out, so the server's default target-stop label applies;
`--prescale-filter PolyStopPrescaleFilter` overrides it, for a poly-stop file.

`stop_materials` works the same way, over the ntuple of a job that ran the
stop finders. It needs `--n-gen-events`, the generated events that file is
equivalent to. The art file is optional, so leave it out and nothing but
`stop_materials` runs (no mu2e job):

```bash
python3 examples/simple_client.py \
    --stopmat-file ../nts.mmackenz.mubeam.Run1Bak_local0818120248.001800_00000000.root \
    --n-gen-events 1e5
```

`--stopmat-file` takes several paths too (a shell glob works), combined into
one table; `--n-gen-events` is then their total. `--stop-module
PolyMuonFinder` (or `IPAMuonFinder`) reads another finder's histogram; left
out, the server's default `TargetMuonFinder` applies.

Other flags: `--output-dir` (defaults to `output/example`), `--sig-eff`
(handed to `approx_ce_sensitivity`), `--timeout-s`. There is no default input
file: `edep` runs over any art file with the right products, while
`approx_ce_sensitivity` only means anything for a CE signal sample.

The trigger analyses run only with `--trigger` (`--batch-mode 2BB` for the
two-batch duty factor in the rate; the timing prints the module table). Rate and timing then run
over the server's default pileup sample (or `--trigger-pileup-file`), and the
efficiency over `--trigger-signal-file` when given. The default
`--trigger-paths` are `cpr_TrkDe_80m70p` and `apr_TrkDe_80m70p`. The menu's
TPR (`tpr_`) and MPR (`mpr_`) paths are left out of the defaults and the
tests: they are not meant for the real trigger.

```bash
python3 examples/simple_client.py --trigger --max-events 1000 \
    --trigger-signal-file /pnfs/.../dig.mu2e.CeEndpointOnSpill.<...>.art \
    [--trigger-paths 'apr_TrkDe_80m70p, cpr_TrkDe_80m70p:10, calo_photon:10']
```

The trigger efficiency from EventNtuples needs no mu2e job and runs on its
own with `--trigger-ntuple` (one file or several). It reads the decisions
already in the ntuple's `trig_<path>` branches for `--trigger-paths`, minus
their prescales, over events with a track passing the server's default
selection or `--ntuple-selection`:

```bash
python3 examples/simple_client.py \
    --trigger-ntuple /pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_1-001/root/a8/50/nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root \
    [--ntuple-selection 'status == 1 and pdg == 11 and downstream and 100 < p_front < 110']
```

On that file the defaults give 0.968: 6951 of 7178 selected events pass apr
or cpr `TrkDe_80m70p`.

## Use it from a client

**Claude Code** — the checked-in `.mcp.json` already wires it up, work area
and all; or:

```bash
claude mcp add mu2e-analysis -- python3 -m analysis_mcp_server \
    --transport stdio --work-area /exp/mu2e/app/users/mmackenz/mu2eopt
```

**`multiagent-client-demo`** (`../multiagent-client-demo`), stdio — the
client spawns the server:

```python
CONFIG = {
    "mu2e-analysis": {
        "transport": "stdio",
        "command": "python3",
        "args": ["-m", "analysis_mcp_server"],
        "cwd": "/exp/mu2e/app/users/mmackenz/mu2eopt/analysis-mcp-server",
    }
}
tools = await load_tools(CONFIG)
```

or over HTTP, against a server started as above:

```python
CONFIG = {"mu2e-analysis": {"transport": "streamable_http",
                            "url": "http://127.0.0.1:8000/mcp"}}
```

(Note the spelling: the adapter config says `streamable_http`, the server CLI
says `--transport streamable-http`.)

A task for the agent, once connected:

```
List the available analyses, then evaluate the average energy deposition in
the detectors for
/exp/mu2e/app/users/mmackenz/mu2eopt/dts.mmackenz.EarlyMuBeamFlash.Run1Bak_local0818120248.001800_00000000.art
writing outputs to /exp/mu2e/data/users/mmackenz/localtest/agent-output.
Report the average calo and tracker Edep per event and per generated event.
```

Chaining `edep` into the sensitivity, which is what `produced_by` is for:

```
Run the edep analysis on <a CE signal art file>, then feed the nts.*.root it
produces to approx_ce_sensitivity with sig_eff = 0.1, and tell me the best
momentum window and its S/sqrt(B).
```
