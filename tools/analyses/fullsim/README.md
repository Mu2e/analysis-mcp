# Full-simulation analyses (`fullsim`)

Analyses of reconstructed EventNtuple files (`EventNtuple/ntuple`) with MC
truth. Today there is one, `fullsim_sensitivity`, the full-simulation
counterpart of `approx_ce_sensitivity`. It computes the same S/sqrt(B) in
the best momentum window, with the same DIO and cosmic backgrounds, but
measures the signal efficiency, shape and detector response from
reconstructed CE mixed with pileup, after pyfitter's cut-set 80 CE-like
cuts.
The repo README.md, under `fullsim_sensitivity`, says how each step is
computed.

| module | what it holds |
|---|---|
| `sensitivity.py` | `fullsim_sensitivity` (the registered `SPEC`): per-file reduction, the signal/DIO/cosmic spectra (`build_spectra`) |
| `eventntuple.py` | the branch set and reader (`read_eventntuple`), trigger paths, surface ids and process codes, and `origin_codes`: what made each event's first track |
| `normalization.py` | stopped mu- per POT from the stop chain's stage efficiencies, computed from their SAM datasets as `CreateSimEfficiency.sh` does (`sim_efficiencies`, `stopped_muons_per_pot`), and the captures/DIO per stopped mu-, as Production's `normalizations.py` |
| `provenance.py` | events generated to make a file: `dh.gencount` of its nearest SAM ancestor that has one (`generated_events`) |
| `cuts.py` | pyfitter's `Analyze.define_cuts` with cut-set `80_1d`'s thresholds, as track-level masks (`cut_masks`), the default set, toggling by name, and the cut flow |

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
| `time_window` | `640,1650` | `t_low,t_high` (ns) a CE's time at the tracker front must fall in. The momentum window is scanned. |
| `trigger_paths` | `apr_TrkDe_80m70p, cpr_TrkDe_80m70p` | Paths for `or_trigger`: an event passes if any of them fired. With or without the `trig_` prefix. A path the file does not record is an error that lists the ones it does. |
| `enable_cuts`, `disable_cuts` | empty | Comma-separated cut names to add to or drop from the default set. An unknown name is an error that lists the known ones. |
| `trkqual_min`, `trkpid_min` | 0.155, 0.54 | Thresholds of `good_trkqual` and `good_trkpid` (cut-set `80_1d`'s). |

### Cuts

pyfitter's cut-set 80 (`VERSION_CUTS['80_1d']` in pyfitter's `config.py`,
applied as its `analyze.py` does), in pyfitter's order: `has_a_track`,
`is_good_track`, `has_trk_front_seg`, `is_reco_electron_or_positron`,
`has_downstream`, `charge_selection`, `or_trigger`, `upstream_veto`,
`no_multi_trk_veto`, `good_trkpid`, `pz_over_pt`, `st_boundary`, `has_st`,
`no_opa`, `good_trkqual`, `has_hits`, `within_t0err`, `no_crv_veto`. Its
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
given, so it needs no network).
