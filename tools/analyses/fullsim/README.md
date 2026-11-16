# Full-simulation analyses (`fullsim`)

Analyses of reconstructed EventNtuple files (`EventNtuple/ntuple`) with MC
truth. Today there is one, `fullsim_sensitivity`, the full-simulation
counterpart of `approx_ce_sensitivity`. It computes the same S/sqrt(B) in
the best momentum window, with the same DIO and cosmic backgrounds, but
measures the signal efficiency, shape and detector response from
reconstructed CE mixed with pileup, after a pyfitter CE-like cut set:
`run1a_tsda` with a 540-1650 ns time window by default, or `80_1d`.
The repo README.md, under `fullsim_sensitivity`, says how each step is
computed.

| module | what it holds |
|---|---|
| `sensitivity.py` | `fullsim_sensitivity` (the registered `SPEC`): per-file reduction, the signal/DIO/cosmic spectra (`build_spectra`) |
| `eventntuple.py` | the branch set and reader (`read_eventntuple`), trigger paths, surface ids and process codes, and `origin_codes`: what made each event's first track |
| `normalization.py` | stopped mu- per POT from the stop chain's stage efficiencies, computed from their SAM datasets as `CreateSimEfficiency.sh` does (`sim_efficiencies`, `stopped_muons_per_pot`), and the captures/DIO per stopped mu-, as Production's `normalizations.py` |
| `provenance.py` | events generated to make a file: `dh.gencount` of its nearest SAM ancestor that has one (`generated_events`) |
| `optimize.py` | the rectangular-cut search of `cut_mode="optimize"`: a numpy port of M. MacKenzie's `optimize_selection_fast.C` (greedy path, randomized paths, local search; even events choose, odd events report), on the Punzi FoM. Knows nothing about Mu2e |
| `tune.py` | the optimize mode's table: the preselection's first track per event, the tuned variables, signal and per-event DIO weights in 0.1 MeV bins, the cosmic rows; the momentum-window scan and the window/cut alternation |
| `cosmic_reference.py` | the fixed cosmic reference table the optimize mode reads (`data/fullsim_cosmic_reference.npz`, made by `scripts/make_fullsim_cosmic_reference.py`), and its normalization |
| `cuts.py` | pyfitter's `Analyze.define_cuts` with a cut set's thresholds (`CUT_SETS`: `run1a_tsda`, the default, and `80_1d`), as track-level masks (`cut_masks`), each set's default cuts and time window, toggling by name, and the cut flow |

The window scan, DIO spectrum and cosmic normalization are imported from
`approx_ce_sensitivity`. The signal and DIO normalization is Production's:
counts for the stopped mu- NPOT gives (`normalization.py`).

## Setup

```bash
source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
pyenv ana
cd /path/to/analysis-mcp
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # shared interactive nodes
```

No mu2e job is run. The analysis reads the ROOT files with uproot.

## Input: CE mixed with pileup

CeMLeadingLogMix1BB EventNtuple files, e.g.
`/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_1-001/root/*/*/nts.*.root`
(about 8k events and 470 MB each). The digitization filter keeps about 41%
of the generated events, so the efficiency's denominator is the events
generated, not the events in the file: each file's `dh.gencount`, looked up
in SAM on its parent mcs file (20000 for the file below). Where SAM cannot
be reached, pass the sum as `n_generated`.

## Running `fullsim_sensitivity`

```python
import glob
from tools.analysis_tools import run_analysis

D = "/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_1-001/root"
result = run_analysis(
    "fullsim_sensitivity",
    output_dir="output/cemix",
    data_files=sorted(glob.glob(f"{D}/*/*/nts.*.root"))[:10],
    timeout_s=3600,                    # ~6 s per 8k-event file
)
print(result.status, result.message)   # message: one-line summary, or the error
result.metadata["sensitivity"]         # every metric is in metadata
```

On `nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root`
alone: 20000 events were generated and 8212 reached the ntuple (acceptance
0.41), and 3010 give a CE passing the cuts in the time window (efficiency
0.15). For 1e18 POT (46.7 conversions at R_mue = 1e-13) the best window is
[103.5, 104.7] MeV/c with S = 4.69, cosmics 1.90 and DIO 0.0009.

### Parameters

| parameter | default | what it does |
|---|---|---|
| `npot`, `mean_pot_per_event`, `cosmic_rate_per_s_per_mev` | as `approx_ce_sensitivity` | NPOT, and the cosmic rate over the on-spill time they imply. |
| `stopped_muons_per_pot` | 0 (compute from SAM) | Stopped mu- per POT: MuBeamCat x MuminusStopsCat efficiency x 1000. Signal and DIO scale with NPOT times this. 0 computes it in SAM from the chain `stop_datasets` gives; a number (7.67e-4 for the MDC2025 chain) skips SAM. |
| `stop_datasets` | `auto` | The stop chain. `auto` traces it from the inputs' SAM ancestry: the MuminusStopsCat dataset the CE were generated from, then the MuBeamCat dataset that came from (for the MDC2025au_best_v1_1 CE mix, `sim.mu2e.MuBeamCat.MDC2025ab.art, sim.mu2e.MuminusStopsCat.MDC2025ac.art`). Or name the datasets, one per stage, for inputs whose ancestry is not in SAM. Each stage's efficiency is events in the dataset over events generated (`event_count` / `dh.gencount`, summed over its files), as `mu2eGenFilterEff` computes it, so no conditions-database table is needed. |
| `rmue` | 1e-13 | R_mue, relative to capture: the expected CE count is NPOT x `stopped_muons_per_pot` x 0.609 x `rmue`. |
| `n_generated` | 0 (look up in SAM) | Events generated to make the inputs: the efficiency's denominator. |
| `upstream_eff` | 1 | An extra efficiency factor for a loss `dh.gencount` does not count. Scales the signal and the DIO. |
| `cut_set` | `run1a_tsda` | The pyfitter cut set: `run1a_tsda` (Run 1A selection, signed d0 in (-60, 94) mm, TSDA veto) or `80_1d`. Sets the thresholds, the default cuts and the default `time_window`. |
| `time_window` | empty (the cut set's) | `t_low,t_high` (ns) a CE's time at the tracker front must fall in: 540,1650 for `run1a_tsda`, 640,1650 for `80_1d`. The momentum window is scanned. |
| `trigger_paths` | `apr_TrkDe_80m70p, cpr_TrkDe_80m70p` | Paths for `or_trigger`: an event passes if any of them fired. With or without the `trig_` prefix. A path the file does not record is an error that lists the ones it does. |
| `enable_cuts`, `disable_cuts` | empty | Comma-separated cut names to add to or drop from the default set. An unknown name is an error that lists the known ones. |
| `trkqual_min`, `trkpid_min` | -1 (the cut set's) | Thresholds of `good_trkqual` and `good_trkpid`: 0.145, 0.574 for `run1a_tsda`; 0.155, 0.54 for `80_1d`. |

### Cuts

A pyfitter cut set (`VERSION_CUTS['run1a_tsda']` or `['80_1d']` in
pyfitter's `config.py`, applied as its `analyze.py` does), in pyfitter's
order: `has_a_track`, `is_good_track`, `has_trk_front_seg`,
`is_reco_electron_or_positron`, `has_downstream`, `charge_selection`,
`or_trigger`, `upstream_veto`, `no_multi_trk_veto`, `good_trkpid`,
`pz_over_pt`, `within_d0`, `st_boundary`, `has_st`, `no_opa`, `no_tsda`,
`good_trkqual`, `has_hits`, `within_t0err`, `no_crv_veto`. `within_d0`
(the signed d0 = maxr - 2|rad| at the tracker front, as Mu2eEvtAna
`Track_t::D0Front`) and `no_tsda` (`trk.tsdainter == 0`) are on for
`run1a_tsda` and off, but available, for `80_1d`. A file without
`trk.tsdainter` is refused when `no_tsda` is on. On the CE mix file above,
`run1a_tsda`'s relative cut efficiencies match pyfitter's `au_run1a_tsda`
cut flow within the file's statistics. Its
fit ranges, `in_mom_range` (100-110 MeV/c) and `within_t0` (540-1650 ns),
are available with `enable_cuts` but off by default: here the momentum
window is scanned and the time window is `time_window`. With every cut on,
the cut flow on the file above is pyfitter's, cut for cut.

Two cuts follow what pyfitter's code does rather than what its comments
say. The upstream and multi-track vetoes ask for `trk.t0`, which pyfitter
never reads, so they time each track by its tracker-front segments (the
first one, and the mean). `good_trkpid` also requires a calorimeter
cluster with energy above 0 in the event. `CUT_DESCRIPTIONS` in `cuts.py`
defines each cut, and the log prints them next to the cut flow.

### Outputs

In `output_dir`:

- `fullsim_sensitivity.log`: the inputs and assumptions, the cut flow, the
  selected events by origin, the efficiency, the best window and the top
  windows scanned.
- `cut_flow.csv`: events with a track passing each cut so far.
- `figures/`: `sig_vs_bkg.png`, `response.png` (reconstructed minus true
  momentum), `mom_vs_time.png`.

| metric | meaning |
|---|---|
| `sensitivity` | S/sqrt(B) in the best window |
| `signal_mom_low_mevc`, `signal_mom_high_mevc` | the momentum window (bin centres, as `approx_ce_sensitivity`) |
| `signal_time_low_ns`, `signal_time_high_ns` | the time window |
| `signal_rate`, `dio_background`, `cosmic_background`, `total_background` | S and B in the window, for `npot` |
| `signal_efficiency`, `signal_efficiency_window` | CE passing cuts and time window per CE generated, overall and in the momentum window |
| `n_signal_window` | reconstructed CE in the window |
| `n_events`, `n_events_processed`, `n_events_selected`, `n_events_counted` | events read, processed by the ntuple job, with a selected track, and counted as signal |
| `n_events_generated`, `acceptance` | events generated for the inputs, and the fraction of them that reached the ntuple |
| `n_files`, `npot`, `upstream_eff`, `cosmic_rate_per_s_per_mev`, `stopped_muons_per_pot`, `rmue` | the inputs behind the numbers |
| `n_stopped_muons`, `n_conversions` | stopped mu- for `npot`, and the CE they give at `rmue` before any efficiency |
| `opt_*` | `cut_mode="optimize"` only, -1 otherwise: see below |

### Tuning the selection: `cut_mode="optimize"`

With `cut_mode="optimize"` the fixed result above is reported unchanged, and
the selection is also tuned on the inputs:

- **Preselection**: the cut set's vetoes and requirements (trigger, upstream
  and multi-track vetoes, a calorimeter cluster, no OPA, no TSDA, CRV) and
  its time window, fixed. The stopping-target cuts are dropped, as in
  M. MacKenzie's `optimize_target_selection_fast`: they assume the standard
  target. The time window is not tuned: it stands against radiative pion
  captures and the beam flash, which this background model does not hold.
- **Tuned**: rectangular cuts on cos(theta) (from tan(dip)), R(max), signed
  d0, TrkQual, TrkPID, t0err and N(active), from the cut set's thresholds,
  and the momentum window, alternating with the cuts.
- **Signal**: the inputs' true CE, as in the fixed mode.
- **DIO**: each CE event, with its own momentum change, stands in for a DIO
  electron: its DIO weight is the theory spectrum integrated over the true
  energies that land in the window. Summed over events this is the fixed
  mode's folded DIO; per event it passes or fails the cuts with its CE.
- **Cosmics**: a fixed reference sample reconstructed like the inputs
  (`cosmic_reference`), flat over 100-110 MeV/c and scaled to the window,
  normalized so that the loose set-81 selection gives the loose cosmic rate
  `approx_ce_sensitivity` uses in box mode.
- **Search**: the macro's greedy path and five randomized paths from no cut,
  the cut set and the previous round as further starts, each refined by a
  local search; on the Punzi FoM `S/(1.28/2 + sqrt(B))`. Cuts and window
  are chosen on even event numbers and every `opt_*` FoM below is on odd
  ones, so the gain is not the search's own optimism.

| metric | what it is |
|---|---|
| `opt_punzi_test`, `opt_punzi_train` | Punzi FoM of the tuned selection on the odd (reported) and even (chosen) events |
| `opt_seed_punzi_test`, `opt_seed_punzi_train` | the same for the cut set's own selection (no stopping-target cuts) in its best window |
| `opt_gain_test` | `opt_punzi_test / opt_seed_punzi_test` |
| `opt_signal_test`, `opt_background_test`, `opt_seed_*` | S and B behind them, for `npot` |
| `opt_window_*_mevc`, `opt_seed_window_*_mevc` | the momentum windows |
| `opt_cut_<var>_low`, `opt_cut_<var>_high` | the tuned cuts; an edge never placed reads as its search range's end |
| `opt_n_ce_rows`, `opt_n_cosmic_rows` | CE and cosmic rows of the table |

The search was checked against the macro itself: on the macro's own
in-memory table for `st37plus6up6dn` (dumped from it), the port finds S/sqrt(B)
2.296 (train) and 2.443 (test) against the macro's 2.299 and 2.451, with the
same cuts within their flat directions.

The reference must be reconstructed with the same code and conditions as the
inputs (TrkQual and TrkPID are MVAs that change between releases). Make it
from the CRY signal-region cosmics, `dts.mu2e.CosmicSignal.MDC2025ap.art`,
run through the inputs' own reconstruction, then:

```bash
python3 scripts/make_fullsim_cosmic_reference.py --files 'nts.*.root' \
    --reco '<code tarball, conditions, run>' --out data/fullsim_cosmic_reference.npz
```

Its sidecar `.json` records how; every optimized result reports it.

### Stopped muons per POT for another campaign

Nothing to set: with `stop_datasets` at `auto`, the chain is traced from
the input files' SAM parents. The CE mix file's ancestry is

```
nts.CeMLeadingLogMix1BB -> mcs -> dig -> dts.CeMLeadingLog.MDC2025ap
  -> sim.MuminusStopsCat.MDC2025ac -> sim.TargetStopsCat -> sim.TargetStops
  -> sim.MuBeamCat.MDC2025ab
```

and each generation follows one file per parent dataset, so it takes a few
SAM lookups however many parents a concatenated file has. Files of one
input dataset are traced once. Inputs that trace to different chains, or a
generation with two MuminusStopsCat datasets, are an error. For inputs
whose ancestry is not in SAM (e.g. a renamed local file), name the chain:
`"sim.mu2e.MuBeamCat.<config>.art, sim.mu2e.MuminusStopsCat.<config>.art"`.

The efficiencies are computed from SAM as Production's
`Scripts/CreateSimEfficiency.sh` computes them with `mu2eGenFilterEff`,
before anything is loaded into the SimEfficiencies2 table, so a chain that
was never put in the database works. The log and `metadata.sim_efficiencies`
give each stage's dataset, file count, passed and generated events.

The same from Python:

```python
from tools.analyses.fullsim import normalization
chain = normalization.stop_chain_of_inputs([nts_file_name])  # SAM file names
stages = normalization.sim_efficiencies(chain)
normalization.stopped_muons_per_pot(stages)   # 7.67e-4
```

A table that is in the database (`dbTool print-run --purpose Sim_best
--version v1_1 --run 1430 --table SimEfficiencies2 --content`) or a
`mu2eGenFilterEff` output file can still be read with
`normalization.parse_sim_efficiencies` and passed to `stopped_muons_per_pot`.

## Using the pieces directly

```python
from tools.analyses.fullsim import cuts, sensitivity
from tools.analyses.fullsim.provenance import generated_events

active = cuts.active_cuts(enable="", disable="has_st")
ev = sensitivity.reduce_file(path, active, cuts.TRKQUAL_MIN, cuts.TRKPID_MIN)
# one entry per selected event: p, t at the tracker front, p_true, origin
ce = ev["origin"] == 168
n_generated = generated_events(path.name)        # SAM dh.gencount
spectra = sensitivity.build_spectra(ev["p"][ce], ev["p_true"][ce],
                                    n_generated, 1.0, 1e18, 1.6e7, 1.282e-5)
```

Read files one at a time (`reduce_file`) and `combine` the results, so the
jagged arrays for many files never sit in memory at once.

## Adding a full-simulation analysis

Read the input with `read_eventntuple`, and add any branches you need to
`BRANCHES` there. Select tracks with `cut_masks` / `apply_cuts`, so every
analysis applies the same cuts. The EventNtuple already pairs each track
with its truth (`trkmcsim`, `trksegsmc`), so no kinematic matching is
needed. Then write a module here that defines a `SPEC`, add
`"fullsim.<module>"` to `_ANALYSIS_MODULES` in `tools/registry.py`, and add
tests. The tests build EventNtuple groups in memory (`_eventntuple` in
`tests/test_tools.py`), because uproot cannot write EventNtuple's nested
branches.

## Tests

`python3 tests/test_tools.py` covers the cuts, the track-pair vetoes, the
trigger OR, origins, reduction, the generated-event lookup (with a fake SAM), stopped muons per
POT and the signal/DIO/cosmic normalization on
synthetic events. When the CeMLeadingLogMix1BB file above is on disk it
also checks pyfitter's cut flow, the measured efficiency, the window and
the acceptance and `upstream_eff` scaling on it (with `n_generated`
given, so it needs no network). The optimize mode's search is tested on toy
tables (the obvious cuts, unplaced edges, half-integer edges, tiny DIO
weights beside large cosmic ones), its DIO weights against the spectrum, the
reference's normalization, and, on the CE mix file, a full run with a toy
reference.
