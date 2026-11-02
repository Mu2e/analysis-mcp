"""Tests for the science tools as plain Python — no MCP layer, no mu2e run.

The `pyenv ana` environment has no pytest, so these are bare asserts:

    python3 tests/test_tools.py

The registry tests loop over every registered analysis, so a new analysis is
covered by them as soon as it is added to registry.py.
"""

import contextlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import numpy as np

from tools import ANALYSES, list_analyses, run_analysis
from tools.analyses import approx_ce_sensitivity as sens
from tools.analyses import edep as edep_mod
from tools.analyses.edep import parse_edep_summary
from tools.analyses.fullsim import cuts as fs_cuts
from tools.analyses.fullsim import sensitivity as fs_sens
from tools.analyses.fullsim.eventntuple import (DEFAULT_TRIGGERS, TRIGGER_PREFIX,
                                                origin_codes)
from tools.analyses.count import (CountsError, dataset_description,
                                  dataset_hint, parse_counts,
                                  parse_prescale_filters, saved_rates,
                                  wrong_dataset)
from tools.analyses import count as count_mod
from tools.analyses import muon_stop_rate as stop_mod
from tools.analyses.muon_stop_rate import (PRESCALE_FILTER, job_fcl_text,
                                           parse_species_counts,
                                           stop_collection, stop_rates)
from tools.analyses.stop_materials import combine_tables, material_rates
from tools.analyses import trigger as trig
from tools.analyses.trigger_efficiency import efficiency_metrics
from tools.analyses import trigger_efficiency_ntuple as ntrig
from tools.analyses import trigger_efficiency_ntuple_binned as bntrig
from tools.analyses import selection_efficiency_ntuple as nsel
from tools.analyses import momentum_resolution_ntuple as nres
from tools.analyses import muon_stops as mstops
from tools.analyses import calo_cluster_resolution_ntuple as ncalo
from tools.analyses import variable_comparison_ntuple as vcmp
from tools.analyses import occupancy as occ
from tools.analyses import occupancy_digi as occ_digi
from tools.analyses import occupancy_ntuple as occ_ntuple
from tools.analyses import occupancy_reco as occ_reco
from tools.analyses.trigger_rate import duty_factor, rate_metrics
from tools.analyses.trigger_timing import timing_metrics
from tools.analyses import total_edep_per_pot as dose
from tools.analyses import edep_occupancy as eocc
from tools.analyses import edep_occupancy_combine as eoccc
from tools.mu2e_env import EnvError, Mu2eEnv, configure, current
from tools.mu2e_job import (build_input_args, root_snapshot,
                            validate_input_paths, written_root_files)
from tools.selection import SelectionError, apply_selection, evaluate_expression
from tools.spectrum import Kernel, Spectrum
from tools.spec import FCL_DIR, ParamSpec

# Verbatim shape of the block EdepAna_module.cc prints, with art's usual
# surrounding noise.
SAMPLE_EDEP_STDOUT = """\
%MSG-i ArtReport:  PostBeginJob 09-Sep-2026 11:22:33 CDT
Begin processing the 1st record. run: 1200 subRun: 0 event: 1
EdepAna summary:
  Saw 998.5 events (1000 gen events) --> output rate = 0.9985 events / gen event
  Average calo energy deposition per event: 12.3456 MeV
  Average calo energy deposition per gen event: 12.3271 MeV
  Events with calo Edep > 50 MeV: 42
  Average tracker energy deposition per event: 3.21 MeV
  Average tracker energy deposition per gen event: 3.2052 MeV
Art has completed and will exit with status 0.
"""

# The same, from an EdepAna with the step counts (Offline PR #2067): two more
# blocks after the summary. The tracker's per-event value is -1, as the module
# prints it for a zero denominator.
SAMPLE_EDEP_STEPS_STDOUT = SAMPLE_EDEP_STDOUT.replace(
    "Art has completed", """\
  Calo step counts (E > 0.5 MeV, 500 <= t <= 1e+09 ns):
    all       : total = 19           per event = 0.019029     per gen event = 0.019
    E cut     : total = 16           per event = 0.016024     per gen event = 0.016
    E + t cut : total = 12.5         per event = 0.012519     per gen event = 0.0125
  Tracker step counts (E > 0.0005 MeV, 500 <= t <= 1e+09 ns):
    all       : total = 1.70683e+06  per event = -1           per gen event = 1706.8
    E cut     : total = 16204        per event = -1           per gen event = 16.204
    E + t cut : total = 0            per event = -1           per gen event = -1
Art has completed""")

# What `mu2e --print-description EdepAna` lists, abridged, for each Offline.
EDEP_DESCRIPTION_OLD = """\
    EdepAna
        Primary: <art::InputTag>
        DebugLevel: <int>
"""
EDEP_DESCRIPTION_NEW = EDEP_DESCRIPTION_OLD + """\
        TrkStepEnergyThreshold: <double>
        CaloStepEnergyThreshold: <double>
"""

# edep asks the configured Offline whether EdepAna has the step counts, which
# would start mu2e: no test may, so the answer is stubbed. Tests set it.
_OFFLINE_HAS_STEPS = [True]


def _stub_module_accepts(module_type, parameter, outdir, **_):
    assert (module_type, parameter) == ("EdepAna", "TrkStepEnergyThreshold")
    return _OFFLINE_HAS_STEPS[0]


edep_mod.module_accepts = _stub_module_accepts

# Verbatim shape of what print_counts.fcl prints, with art's usual noise. One
# prescale block per filter the production job ran; only the target-stop one
# applies to a TargetStops file.
SAMPLE_COUNTS_STDOUT = """\
18-Sep-2026 12:04:13 CDT  Opened input file "sim.mmackenz.TargetStops.Run1Bak_local0813111400.001800_00000000.art"

ProductPrint mu2e::PrescaleFilterFraction_PolyStopPrescaleFilter__MuBeamResampler
 Fraction passing filter 0.000800 N Seen 12500 with prescale fraction 0.001000

ProductPrint mu2e::PrescaleFilterFraction_TargetStopPrescaleFilter__MuBeamResampler
 Fraction passing filter 1.000000 N Seen 12500 with prescale fraction 1.000000

ProductPrint mu2e::PrescaleFilterFraction_EarlyPrescaleFilter__MuBeamResampler
 Fraction passing filter 0.034800 N Seen 12500 with prescale fraction 0.033333

Begin processing the 1st record. run: 1800 subRun: 0 event: 3 at 18-Sep-2026 12:04:13 CDT
GenEventCount: 12500 events in run: 1800 subRun: 0
     1 BeginRun records found
     1 Subrun records found
   735 Event records found
GenEventCount total: 12500 events in 1 SubRuns

Art has completed and will exit with status 0.
"""

# stop_counts.fcl's output: the counts above, plus art's TrigReport for the
# two selector paths, in the format of a real MDC2025ba TargetStops job.
SAMPLE_STOP_COUNTS_STDOUT = SAMPLE_COUNTS_STDOUT.replace("""
Art has completed""", """
TrigReport ---------- Event summary -------------
TrigReport Events total = 735 passed = 735 failed = 0
TrigReport ---------- Trigger-path summary ------------
TrigReport    Path ID        Run     Passed     Failed      Error Name
TrigReport          0        735        728          7          0 selectMuminus
TrigReport          1        735          7        728          0 selectMuplus
TrigReport ---------- End-path summary ---------
TrigReport        Run    Success      Error
TrigReport        735        735          0
TrigReport ---------- Modules in path: selectMuminus ------------
TrigReport    Path ID    Visited     Passed     Failed      Error Name
TrigReport          0        735        728          7          0 muminusSelector
TrigReport ---------- Modules in path: selectMuplus ------------
TrigReport    Path ID    Visited     Passed     Failed      Error Name
TrigReport          1        735          7        728          0 muplusSelector
TrigReport ---------- Modules in End-path ----------
TrigReport        Run    Success      Error Name
TrigReport        735        735          0 printModule

Art has completed""")

# The end of a trigger job's stdout: art's TrigReport, in the format of a run
# of three menu paths (cpr_ at prescale 3) over 200 CE digi events.
SAMPLE_TRIGGER_STDOUT = """\
Full event                                                           0.00106055     0.0163061      1.10655      0.0102812     0.077483        200
TrigReport ---------- Event summary -------------
TrigReport Events total = 200 passed = 182 failed = 18

TrigReport ---------- Trigger-path summary ------------
TrigReport    Path ID        Run     Passed     Failed      Error Name
TrigReport        400        200        181         19          0 calo_photon
TrigReport        160        200         40        160          0 cpr_TrkDe_80m70p
TrigReport        210        200        171         29          0 apr_TrkDe_80m70p

TrigReport ---------- End-path summary ---------
TrigReport        Run    Success      Error

TrigReport ---------- Modules in path: apr_TrkDe_80m70p ------------
TrigReport    Path ID    Visited     Passed     Failed      Error Name
TrigReport        210        200        200          0          0 Prefetch

Art has completed and will exit with status 0.
"""

# The table OccupancyAna_module.cc prints at endJob (printf "  %-14s %-28s %8li
# %8li %12.3f %12.3f %8i"), after art's usual noise. A configured collection
# that no event had has found 0 and a mean of 0.
SAMPLE_OCCUPANCY_RECO_STDOUT = """\
Begin processing the 1st record. run: 1470 subRun: 0 event: 1
[OccupancyAna::analyze] Warning: calorecodigis collection CaloRecoDigiMaker not found, its count is set to -1
[OccupancyAna::endJob] Occupancy summary over 50 events:
  collection     tag                             found  missing         mean          rms      max
  strawhits      makeSH                             50        0      779.420      460.005     2312
  combohits      makePH                             50        0      276.100      163.666      859
  calohits       CaloHitMaker                       50        0       56.280       28.993      139
  caloclusters   CaloClusterMaker                   50        0        5.500        2.872       13
  calorecodigis  CaloRecoDigiMaker                   0       50        0.000        0.000        0

Art has completed and will exit with status 0.
"""

SAMPLE_OCCUPANCY_DIGI_STDOUT = """\
[OccupancyAna::endJob] Occupancy summary over 50 events:
  collection     tag                             found  missing         mean          rms      max
  strawdigis     makeSD                             50        0       44.740       30.500       93
  calodigis      CaloDigiMaker                      50        0        0.040        0.280        2
  crvdigis       CrvDigi                            50        0        0.000        0.000        0

Art has completed and will exit with status 0.
"""

# One stdout sample per art_files analysis, so the registry test can exercise
# each parser. Add an entry when adding such an analysis.
# The end of a muon_stops job (fcl/muon_stops.fcl on a TargetStops file):
# GenEventCountReader's totals. The stops themselves are in its ROOT file.
SAMPLE_MUON_STOPS_STDOUT = """\
%MSG-i GenEventCount:  GenEventCountReader:genCountLogger@EndSubRun  27-Oct-2026 09:06:50 CDT run: 1800 subRun: 0
GenEventCount: 12500 events in run: 1800 subRun: 0
%MSG
%MSG-i GenEventCountSummary:  GenEventCountReader:genCountLogger@EndJob  27-Oct-2026 09:06:50 CDT ModuleEndJob
GenEventCount total: 12500 events in 1 SubRuns
%MSG
TrigReport Events total = 735 passed = 735 failed = 0
"""

SAMPLE_STDOUT = {"edep": SAMPLE_EDEP_STDOUT,
                 "count": SAMPLE_COUNTS_STDOUT,
                 "muon_stop_rate": SAMPLE_STOP_COUNTS_STDOUT,
                 "trigger_efficiency": SAMPLE_TRIGGER_STDOUT,
                 "trigger_rate": SAMPLE_TRIGGER_STDOUT,
                 "trigger_timing": SAMPLE_TRIGGER_STDOUT,
                 "muon_stops": SAMPLE_MUON_STOPS_STDOUT,
                 "occupancy_digi": SAMPLE_OCCUPANCY_DIGI_STDOUT,
                 "occupancy_reco": SAMPLE_OCCUPANCY_RECO_STDOUT,
                 "edep_occupancy": SAMPLE_EDEP_STEPS_STDOUT}


# --- the edep parser ---------------------------------------------------------

def test_edep_parses_all_eight_fields():
    metrics = parse_edep_summary(SAMPLE_EDEP_STDOUT)
    assert metrics == {
        "n_events": 998.5,
        "n_gen_events": 1000.0,
        "event_rate": 0.9985,
        "avg_calo_edep_per_event_mev": 12.3456,
        "avg_calo_edep_per_gen_event_mev": 12.3271,
        "n_events_calo_edep_above_50mev": 42.0,
        "avg_trk_edep_per_event_mev": 3.21,
        "avg_trk_edep_per_gen_event_mev": 3.2052,
    }, metrics


def test_edep_parses_scientific_notation():
    stdout = SAMPLE_EDEP_STDOUT.replace("12.3456 MeV", "1.23456e-02 MeV")
    assert parse_edep_summary(stdout)["avg_calo_edep_per_event_mev"] == 0.0123456


def test_edep_parse_returns_none_without_summary():
    assert parse_edep_summary("Art has completed and will exit with status 1.") is None


def test_edep_parse_returns_none_on_truncated_summary():
    truncated = SAMPLE_EDEP_STDOUT.split("Events with calo Edep")[0]
    assert parse_edep_summary(truncated) is None


def test_edep_step_counts_parse_both_blocks_and_keep_minus_one():
    parsed = edep_mod.parse_step_counts(SAMPLE_EDEP_STEPS_STDOUT)
    assert parsed is not None
    metrics, windows = parsed
    assert tuple(metrics) == edep_mod.STEP_METRICS and len(metrics) == 18
    assert metrics["calo_nsteps_total"] == 19.0
    assert metrics["calo_nsteps_ecut_tcut_total"] == 12.5
    assert metrics["calo_nsteps_ecut_tcut_per_gen_event"] == 0.0125
    assert metrics["trk_nsteps_total"] == 1.70683e6
    assert metrics["trk_nsteps_ecut_per_gen_event"] == 16.204
    # -1 is the module's "no denominator" and is passed through
    assert metrics["trk_nsteps_per_event"] == -1.0
    assert metrics["trk_nsteps_ecut_tcut_per_gen_event"] == -1.0
    assert windows == {
        "calo": {"energy_threshold_mev": 0.5, "time_min_ns": 500.0, "time_max_ns": 1e9},
        "trk": {"energy_threshold_mev": 0.0005, "time_min_ns": 500.0, "time_max_ns": 1e9}}
    assert edep_mod.STEP_UNITS["trk_nsteps_ecut_per_event"] == "steps / event"
    # the summary parser is unchanged by the extra blocks
    assert parse_edep_summary(SAMPLE_EDEP_STEPS_STDOUT) == parse_edep_summary(SAMPLE_EDEP_STDOUT)


def test_edep_step_counts_are_absent_from_old_stdout_or_incomplete():
    assert edep_mod.parse_step_counts(SAMPLE_EDEP_STDOUT) is None
    no_tracker = SAMPLE_EDEP_STEPS_STDOUT.split("  Tracker step counts")[0] + "Art has completed\n"
    assert edep_mod.parse_step_counts(no_tracker) is None
    no_row = SAMPLE_EDEP_STEPS_STDOUT.replace("    E cut     : total = 16204", "")
    assert edep_mod.parse_step_counts(no_row) is None


def test_edep_summary_mentions_step_counts_only_when_present():
    base = {name: 1.0 for name in edep_mod.parse_edep_summary(SAMPLE_EDEP_STDOUT)}
    base.update(edep_mod._SELECTION_METRICS and {"n_events_selected": 1.0,
                                                 "selected_per_gen_event": 0.5})
    assert "Steps per event" not in edep_mod.summarize_edep(base)
    full = {**base, **edep_mod.parse_step_counts(SAMPLE_EDEP_STEPS_STDOUT)[0]}
    text = edep_mod.summarize_edep(full)
    assert "calo 0.01252" in text and "tracker -1" in text, text


def test_edep_step_params_validate_with_the_example_fcl_defaults():
    spec = ANALYSES["edep"]
    params = spec.resolve_params({})
    assert (params["trk_step_energy_threshold_mev"], params["trk_step_time_min_ns"],
            params["trk_step_time_max_ns"], params["calo_step_energy_threshold_mev"],
            params["calo_step_time_min_ns"], params["calo_step_time_max_ns"]) == (
                0.0005, 500.0, 1e9, 0.5, 500.0, 1e9)
    assert edep_mod.non_default_step_params(params) == []
    assert edep_mod.non_default_step_params(
        spec.resolve_params({"calo_step_energy_threshold_mev": 1})) == [
            "calo_step_energy_threshold_mev"]
    for bad in ({"trk_step_energy_threshold_mev": -1}, {"calo_step_time_max_ns": 1e13},
                {"calo_step_time_min_ns": "soon"}):
        try:
            spec.resolve_params(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad}: no error")
    inverted = spec.resolve_params({"trk_step_time_min_ns": 900, "trk_step_time_max_ns": 800})
    assert "trk_step_time_min_ns" in edep_mod.step_param_problem(inverted)
    assert edep_mod.step_param_problem(params) is None


def test_edep_job_fcl_appends_the_step_parameters_to_the_shipped_fcl():
    params = ANALYSES["edep"].resolve_params({"trk_step_energy_threshold_mev": 0.001,
                                              "calo_step_time_max_ns": 2e6})
    text = edep_mod.job_fcl_text(params)
    assert text.startswith((FCL_DIR / "edep.fcl").read_text(encoding="utf-8").rstrip())
    # all six, after the base so they land in the analyzer's table
    for key, value in (("TrkStepEnergyThreshold", "0.001"), ("TrkStepTimeMin", "500.0"),
                       ("TrkStepTimeMax", "1000000000.0"),
                       ("CaloStepEnergyThreshold", "0.5"), ("CaloStepTimeMin", "500.0"),
                       ("CaloStepTimeMax", "2000000.0")):
        assert f"physics.analyzers.EDepAna.{key} : {value}\n" in text, key
    assert "StepEnergyThreshold" not in (FCL_DIR / "edep.fcl").read_text(encoding="utf-8"), \
        "the shipped fcl must stay valid for an Offline without the step counts"


def test_module_accepts_reads_the_description_once_per_environment(tmp_dir):
    from tools import mu2e_job
    calls = []

    class Proc:
        returncode, stderr = 0, ""
        stdout = EDEP_DESCRIPTION_OLD

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return Proc

    real = mu2e_job.subprocess.run
    mu2e_job.subprocess.run = fake_run
    mu2e_job._MODULE_DESCRIPTIONS.clear()
    try:
        env = Mu2eEnv(kind="work_area", work_area=Path(tmp_dir))
        out = Path(tmp_dir)
        assert mu2e_job.module_accepts("EdepAna", "Primary", out, env=env)
        assert not mu2e_job.module_accepts("EdepAna", "TrkStepEnergyThreshold", out, env=env)
        Proc.stdout = EDEP_DESCRIPTION_NEW      # cached: not asked again
        assert not mu2e_job.module_accepts("EdepAna", "TrkStepEnergyThreshold", out, env=env)
        assert len(calls) == 1 and "--print-description" in calls[0][2]
        mu2e_job._MODULE_DESCRIPTIONS.clear()
        assert mu2e_job.module_accepts("EdepAna", "TrkStepEnergyThreshold", out, env=env)
        assert not mu2e_job.module_accepts("EdepAna", "StepEnergy", out, env=env)  # a name, not a substring
        Proc.returncode = 1
        mu2e_job._MODULE_DESCRIPTIONS.clear()
        try:
            mu2e_job.module_accepts("EdepAna", "Primary", out, env=env)
            raise AssertionError("no error for a failed probe")
        except mu2e_job.ModuleProbeError:
            pass
    finally:
        mu2e_job.subprocess.run = real
        mu2e_job._MODULE_DESCRIPTIONS.clear()


# --- the shared count parser (count.py) --------------------------------------

def test_counts_parses_events_gen_events_and_the_named_prescale():
    assert parse_counts(SAMPLE_COUNTS_STDOUT, PRESCALE_FILTER) == {
        "n_events": 735.0,
        "n_gen_events": 12500.0,
        "prescale": 1.0,        # the TargetStop filter's, not PolyStop's 0.001
    }


def test_counts_takes_the_prescale_of_whichever_filter_is_named():
    counts = parse_counts(SAMPLE_COUNTS_STDOUT, "PolyStopPrescaleFilter")
    assert counts["prescale"] == 0.001
    assert counts["n_events"] == 735.0      # the file's events, either way


def test_counts_needs_no_prescale_filter_at_all():
    """No filter named: prescale 1, and the blocks that ARE there are ignored.

    The point of `count`: nothing assumes a prescale module exists. The sample
    holds three prescale blocks and none of them may be picked up by accident.
    """
    for unset in (None, ""):
        assert parse_counts(SAMPLE_COUNTS_STDOUT, unset) == {
            "n_events": 735.0,
            "n_gen_events": 12500.0,
            "prescale": 1.0,
        }
    # ... and the counts still parse out of a job that printed no block at all
    bare = SAMPLE_COUNTS_STDOUT[SAMPLE_COUNTS_STDOUT.index("Begin processing"):]
    assert parse_prescale_filters(bare) == {}
    assert parse_counts(bare)["n_events"] == 735.0


def test_counts_never_falls_back_to_no_prescale_for_a_filter_that_is_missing():
    """A named-but-absent filter is an error, not a silent prescale of 1.

    Falling back would report a prescaled file's rate short by exactly the
    prescale, with nothing in the output to show for it.
    """
    without = SAMPLE_COUNTS_STDOUT.replace("PolyStopPrescaleFilter", "OtherFilter")
    try:
        parse_counts(without, "PolyStopPrescaleFilter")
    except CountsError as exc:
        assert "PolyStopPrescaleFilter" in str(exc)
    else:
        raise AssertionError("a named filter that is absent must be reported")


def test_counts_reads_every_prescale_block():
    filters = parse_prescale_filters(SAMPLE_COUNTS_STDOUT)
    assert set(filters) == {"PolyStopPrescaleFilter", "TargetStopPrescaleFilter",
                            "EarlyPrescaleFilter"}
    poly = filters["PolyStopPrescaleFilter"]
    assert poly["prescale"] == 0.001 and poly["fraction_passing"] == 0.0008
    assert poly["n_seen"] == 12500.0 and poly["process"] == "MuBeamResampler"


def test_counts_rejects_output_without_the_target_stop_filter():
    without = SAMPLE_COUNTS_STDOUT.replace("TargetStopPrescaleFilter",
                                           "IPAStopPrescaleFilter")
    try:
        parse_counts(without, PRESCALE_FILTER)
    except CountsError as exc:
        assert "TargetStopPrescaleFilter" in str(exc)
        assert "IPAStopPrescaleFilter" in str(exc)      # names what it did find
    else:
        raise AssertionError("a file without the filter must be reported")


def test_counts_rejects_output_without_the_generated_event_count():
    without = SAMPLE_COUNTS_STDOUT.replace("GenEventCount total: 12500 events", "")
    try:
        parse_counts(without, PRESCALE_FILTER)
    except CountsError as exc:
        assert "generated" in str(exc)
    else:
        raise AssertionError("missing gen-event bookkeeping must be reported")


def test_dataset_description_reads_mu2e_names_and_passes_on_others():
    name = Path("sim.mmackenz.TargetStops.Run1Bak_local0813111400.001800_00000000.art")
    assert dataset_description(name) == "TargetStops"
    assert dataset_description(Path("my_stops.art")) is None   # not a Mu2e name


def test_dataset_hint_comes_from_the_filter_label():
    assert dataset_hint("TargetStopPrescaleFilter") == "targetstop"
    assert dataset_hint("PolyStopPrescaleFilter") == "polystop"
    assert dataset_hint("SomethingElse") == ""      # nothing to check against
    assert dataset_hint(None) == "" and dataset_hint("") == ""   # no filter at all


def test_wrong_dataset_follows_the_filter_it_is_given():
    stem = "Run1Bak_local0813111400.001800_00000000.art"
    target = Path(f"sim.mmackenz.TargetStops.{stem}")
    poly = Path(f"sim.mmackenz.PolyStops.{stem}")
    unnamed = Path("stops.art")
    # A poly-stop file parses fine and carries the target filter's product, so
    # only the name says it is the wrong input.
    assert wrong_dataset([target, unnamed], PRESCALE_FILTER) == []
    assert wrong_dataset([target, poly], PRESCALE_FILTER) == [
        f"sim.mmackenz.PolyStops.{stem} (PolyStops)"
    ]
    # With no filter named nothing is divided out, so no file is the wrong one.
    assert wrong_dataset([target, poly]) == []
    assert wrong_dataset([target, poly], "") == []
    # ... and the check follows the filter: with the poly filter it is the
    # target-stop file that is out of place.
    assert wrong_dataset([target, poly], "PolyStopPrescaleFilter") == [
        f"sim.mmackenz.TargetStops.{stem} (TargetStops)"
    ]
    assert wrong_dataset([target, poly], "OddlyNamedFilter") == []


def test_saved_rates_divide_out_the_prescale():
    counts = {"n_events": 735.0, "n_gen_events": 12500.0, "prescale": 0.5}
    assert abs(saved_rates(counts)["saved_per_gen_event"] - 0.1176) < 1e-6
    # an unprescaled file (prescale 1) is just the ratio
    plain = {"n_events": 735.0, "n_gen_events": 12500.0, "prescale": 1.0}
    assert abs(saved_rates(plain)["saved_per_gen_event"] - 0.0588) < 1e-6


def test_stop_rates_divide_out_the_prescale_and_scale_by_the_upstream_efficiency():
    counts = {"n_events": 735.0, "n_gen_events": 12500.0, "prescale": 0.5}
    rates = stop_rates(counts, upstream_eff=2.0e-3)
    # the prescale kept half the events, so the true rate is twice 735/12500
    assert abs(rates["stops_per_gen_event"] - 0.1176) < 1e-6
    assert abs(rates["stops_per_pot"] - 0.1176 * 2.0e-3) < 1e-9
    # without a mu-/mu+ split there are no per-charge rates to report
    assert "muminus_stops_per_gen_event" not in rates


def test_stop_rates_count_the_mu_minus_and_mu_plus_stops_apart():
    counts = {**parse_counts(SAMPLE_STOP_COUNTS_STDOUT, PRESCALE_FILTER),
              **parse_species_counts(SAMPLE_STOP_COUNTS_STDOUT)}
    assert (counts["n_muminus"], counts["n_muplus"]) == (728.0, 7.0), counts
    rates = stop_rates(counts, upstream_eff=2.0e-3)
    # every stop is still in stops_per_gen_event; the mu- rate is the part a
    # capture normalization wants, and the two charges add up to it
    assert abs(rates["stops_per_gen_event"] - 735 / 12500) < 1e-12
    assert abs(rates["muminus_stops_per_gen_event"] - 728 / 12500) < 1e-12
    assert abs(rates["muplus_stops_per_gen_event"] - 7 / 12500) < 1e-12
    assert abs(rates["muminus_stops_per_pot"] - 728 / 12500 * 2.0e-3) < 1e-15
    # a prescaled file divides both charges by the same prescale
    halved = stop_rates({**counts, "prescale": 0.5}, upstream_eff=1.0)
    assert abs(halved["muminus_stops_per_gen_event"] - 2 * 728 / 12500) < 1e-12


def test_species_counts_come_from_the_path_rows_not_the_module_rows():
    # the per-module rows have the same shape, under the modules' names
    assert parse_species_counts(SAMPLE_STOP_COUNTS_STDOUT) == {
        "n_muminus": 728.0, "n_muplus": 7.0}
    for stdout, words in (
            (SAMPLE_COUNTS_STDOUT, ["no TrigReport row", "selectMuminus"]),
            (SAMPLE_STOP_COUNTS_STDOUT.replace(
                "        7          0 selectMuminus", "        7          2 selectMuminus"),
             ["selectMuminus", "2 event(s) in error"])):
        try:
            parse_species_counts(stdout)
        except CountsError as exc:
            assert all(w in str(exc) for w in words), str(exc)
        else:
            raise AssertionError("an unusable TrigReport was not rejected")


def test_stop_counts_read_the_streams_own_simparticles():
    assert stop_collection("TargetStopPrescaleFilter") == "TargetStopFilter"
    assert stop_collection("PolyStopPrescaleFilter") == "PolyStopFilter"
    text = job_fcl_text("PolyStopFilter")
    shipped = (FCL_DIR / "stop_counts.fcl").read_text()
    assert text.startswith(shipped.rstrip())
    for selector in ("muminusSelector", "muplusSelector"):
        assert f'physics.filters.{selector}.SimParticles : "PolyStopFilter"' in text
        assert selector in shipped
    # the counts the job shares with print_counts.fcl, and the summary the
    # path counts are read from
    for needed in ("printModule", "genCountReader", "eventCount",
                   "wantSummary : true", "selectMuminus", "selectMuplus",
                   "muMinusCaptureAtRest"):
        assert needed in shipped, needed


def _fake_counts_job(stdout, returncode=0):
    """A stand-in for run_mu2e_job that prints `stdout`, recording the fcl."""
    from tools.mu2e_job import JobOutcome
    seen = []

    def fake(*, fcl, input_paths, outdir, single, timeout_s, max_events=None, **_):
        seen.append(fcl)
        outdir.mkdir(parents=True, exist_ok=True)
        return JobOutcome(command="mu2e", returncode=returncode, timed_out=False,
                          stdout=stdout, stderr="", log_path=outdir / "mu2e.log",
                          input_paths=input_paths, input_flag="-s",
                          environment="test", file_list_path=None,
                          written_root_files=[])
    return fake, seen


def test_muon_stop_rate_runs_its_own_fcl_and_reports_the_charges(tmp_dir):
    stops = Path(tmp_dir) / "sim.mu2e.TargetStops.MDC2025ba_x.001430_00000053.art"
    stops.touch()
    fake, seen = _fake_counts_job(SAMPLE_STOP_COUNTS_STDOUT)
    real = count_mod.run_mu2e_job
    count_mod.run_mu2e_job = fake
    try:
        result = run_analysis(analysis="muon_stop_rate", data_file=str(stops),
                              output_dir=str(Path(tmp_dir) / "out"),
                              parameters={"upstream_eff": 2.0e-3})
    finally:
        count_mod.run_mu2e_job = real
    assert result.status == "success", result.message
    meta = result.metadata
    assert seen == [Path(tmp_dir) / "out" / stop_mod.JOB_FCL_NAME], seen
    assert 'SimParticles : "TargetStopFilter"' in seen[0].read_text()
    assert meta["stop_collection"] == "TargetStopFilter"
    assert (meta["n_muminus"], meta["n_muplus"]) == (728.0, 7.0), meta
    assert abs(meta["muminus_stops_per_pot"] - 728 / 12500 * 2.0e-3) < 1e-15
    assert "728 mu-" in result.message, result.message


def test_muon_stop_rate_says_when_the_file_lacks_the_stop_collection(tmp_dir):
    stops = Path(tmp_dir) / "sim.mu2e.TargetStops.v1.001430_00000000.art"
    stops.touch()
    fake, _ = _fake_counts_job(
        "---- ProductNotFound BEGIN\n  getValidHandle: no TargetStopFilter\n", returncode=1)
    real = count_mod.run_mu2e_job
    count_mod.run_mu2e_job = fake
    try:
        result = run_analysis(analysis="muon_stop_rate", data_file=str(stops),
                              output_dir=str(Path(tmp_dir) / "out"),
                              parameters={"upstream_eff": 1.0})
    finally:
        count_mod.run_mu2e_job = real
    assert result.status == "error"
    for words in ("'TargetStopFilter' SimParticle collection", "prescale_filter",
                  "'count'"):
        assert words in result.message, result.message


# --- the spectrum helper -----------------------------------------------------

def test_spectrum_axis_and_total():
    s = Spectrum(np.full(10, 2.0), 0.0, 0.1)
    assert s.nbins == 10 and abs(s.xmax - 1.0) < 1e-12
    assert abs(s.centers()[0] - 0.05) < 1e-12
    assert s.values.sum() == 20.0


def test_spectrum_rebin_merges_groups_and_drops_leftovers():
    s = Spectrum(np.arange(1.0, 11.0), 0.0, 1.0).rebin(3)
    assert s.nbins == 3 and s.width == 3.0 and abs(s.xmax - 9.0) < 1e-12
    assert list(s.values) == [6.0, 15.0, 24.0]   # the 10th bin is dropped


def test_spectrum_regrid_sums_onto_a_coarser_axis():
    fine = Spectrum(np.ones(100), 0.0, 0.1)          # 10 units over [0, 10)
    coarse = fine.regrid(Spectrum(np.zeros(5), 0.0, 1.0))
    assert coarse.nbins == 5 and list(coarse.values) == [10.0] * 5
    assert coarse.values.sum() == 50.0               # half of it fell off the axis


def test_gaussian_kernel_is_normalized_and_centered():
    kernel = Kernel.gaussian(width=0.05, sigma=0.2)
    assert abs(kernel.mass.sum() - 1.0) < 1e-12
    spectrum = kernel.as_spectrum(0.05)
    mpv, fwhm = sens.mpv_fwhm(spectrum)
    assert abs(mpv) < 0.03                                  # centered on zero
    assert abs(fwhm - 2.355 * 0.2) < 0.06                   # FWHM = 2.355 sigma


def test_kernel_from_density_keeps_the_response_total():
    # A response carrying only half the probability (an efficiency folded in),
    # on a coarser grid than the target.
    response = Spectrum(np.full(10, 0.5 / (10 * 0.2)), -1.0, 0.2)
    kernel = Kernel.from_density(response, width=0.05)
    assert abs(kernel.mass.sum() - 0.5) < 1e-9


def test_dio_spectrum_is_a_normalized_density():
    dio = sens.load_dio_spectrum()
    # Density in energy: sum(contents) * bin width == 1
    assert abs(dio.values.sum() * dio.width - 1.0) < 1e-9
    # Falls steeply toward the CE endpoint
    at = lambda e: dio.values[int((e - dio.xmin) / dio.width)]
    assert at(60.0) > at(100.0) > at(104.5)


def test_smear_conserves_total_and_shifts_by_the_response_mean():
    true = Spectrum(np.zeros(200), 0.0, 0.1)
    true.values[100] = 1.0                                  # delta at 10 MeV
    response = Spectrum(np.zeros(200), -5.0, 0.05)          # delta at -2 MeV
    response.values[59] = 1.0 / response.width

    reco = true.smear(Kernel.from_density(response, true.width))
    assert abs(reco.values.sum() - 1.0) < 1e-9              # probability kept
    assert abs(reco.centers()[int(np.argmax(reco.values))] - 8.0) < 0.2


def test_smear_drops_content_pushed_off_the_axis():
    true = Spectrum(np.zeros(100), 0.0, 0.1)
    true.values[10] = 1.0                                   # 1 MeV
    response = Spectrum(np.zeros(100), -5.0, 0.1)
    response.values[9] = 1.0 / response.width               # shift by -4 MeV
    reco = true.smear(Kernel.from_density(response, true.width))
    assert reco.values.sum() < 1e-12                        # left the axis


def test_scan_finds_the_best_window():
    signal, dio, cosmic = (Spectrum(np.zeros(100), 50.0, 1.0, name=n)
                           for n in ("signal", "dio", "cosmic"))
    signal.values[45:55] = 10.0          # a signal bump at ~95-105 MeV
    dio.values[:] = 1.0
    cosmic.values[:] = 1.0
    best, top = sens.scan_signal_box(signal, dio, cosmic)
    assert best["low_mev"] >= 50.0
    # The bump spans bins 46..55 -> centers 95.5..104.5
    assert 94.0 <= best["low_mev"] <= 96.0, best
    assert 104.0 <= best["high_mev"] <= 106.0, best
    assert top[0]["sensitivity"] == best["sensitivity"]


def test_scan_keeps_a_tiny_window_count_against_a_huge_spectrum_total():
    """Regression: window sums must not be prefix-sum differences.

    The DIO spectrum spans ~18 orders of magnitude. Differencing whole-spectrum
    prefix sums to get a count of order 1 loses it completely to float
    cancellation, which silently reported dio_background = 0.
    """
    signal, dio, cosmic = (Spectrum(np.zeros(100), 50.0, 1.0, name=n)
                           for n in ("signal", "dio", "cosmic"))
    signal.values[50:60] = 10.0
    cosmic.values[:] = 1.0
    dio.values[0] = 4.0e17            # enormous, far below the signal window
    dio.values[50:60] = 2.0e-1        # what we must still be able to see

    best, _ = sens.scan_signal_box(signal, dio, cosmic)
    assert best["dio"] > 0.0, "tiny DIO count was lost to cancellation"
    assert abs(best["dio"] - 0.2 * 10) < 1e-6 or best["dio"] > 0.1, best


def test_sensitivity_rejects_a_file_without_the_tree(tmp_dir):
    """A non-EdepAna ROOT file (or a pre-tree one) is reported, not a crash."""
    import uproot
    path = Path(tmp_dir) / "empty.root"
    with uproot.recreate(path) as f:
        f["something_else"] = np.histogram(np.zeros(1), bins=2)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"sig_eff": 0.1})
    assert result.status == "error"
    assert "EDepAna/tree" in result.message


# --- selections and the EdepAna tree ------------------------------------------

def test_selection_takes_python_and_root_spellings_alike():
    x = {"a": np.array([1.0, 5.0, 20.0]), "b": np.array([0.0, 1.0, 2.0])}
    expect = [False, True, False]
    assert apply_selection("a > 2 and a < 10", x, 3).tolist() == expect
    assert apply_selection("a > 2 && a < 10", x, 3).tolist() == expect
    assert apply_selection("2 < a < 10", x, 3).tolist() == expect
    assert apply_selection("!(a <= 2 || a >= 10)", x, 3).tolist() == expect
    assert apply_selection("abs(a - 2*b) > 10", x, 3).tolist() == [False, False, True]
    assert apply_selection("b != 1", x, 3).tolist() == [True, False, True]
    assert apply_selection("", x, 3).all()                  # no cut at all


def test_selection_fails_a_cut_on_a_missing_value():
    x = {"a": np.array([np.nan, 5.0])}
    assert apply_selection("a > 0", x, 2).tolist() == [False, True]
    assert apply_selection("a <= 0", x, 2).tolist() == [False, False]


def test_expression_gives_a_quantity_and_refuses_a_cut():
    x = {"a": np.array([1.0, 2.0]), "b": np.array([0.5, np.nan])}
    assert evaluate_expression("a - b", x, 2)[0] == 0.5
    assert np.isnan(evaluate_expression("a - b", x, 2)[1])
    assert evaluate_expression("3", x, 2).tolist() == [3.0, 3.0]
    for bad, words in (("", "empty"), ("a > 1", "true/false"),
                       ("c", "unknown variable")):
        try:
            evaluate_expression(bad, x, 2)
        except SelectionError as exc:
            assert words in str(exc), (bad, str(exc))
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def test_selection_refuses_what_it_should_not_run():
    x = {"a": np.array([1.0, 2.0])}
    for bad, words in (("a > ", "does not parse"),
                       ("c > 1", "unknown variable"),
                       ("a.__class__ > 1", "not allowed"),
                       ("__import__('os') > 1", "unknown variable"),
                       ("a + 1", "true/false")):
        try:
            apply_selection(bad, x, 2)
        except SelectionError as exc:
            assert words in str(exc), (bad, str(exc))
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def _edep_branches():
    """Three events: one with no primary, one that never reached the tracker
    front, one that did — raw branches as uproot hands them over."""
    def per_event(*values):
        out = np.empty(len(values), dtype=object)
        out[:] = [np.asarray(v, dtype=np.float32) for v in values]
        return out
    branches = {
        "nprimaries": np.array([0, 1, 2]),
        "event_calo_edep": np.array([0.0, 30.0, 90.0]),
        "event_calo_edep_vis": np.array([0.0, 25.0, 80.0]),
        "event_trk_edep": np.array([0.0, 0.1, 0.2]),
        "weight": np.ones(3), "run": np.ones(3), "subrun": np.ones(3),
        "event": np.arange(3), "ngen": np.array([5, 5, 5]),
    }
    first = {"primary_start_x": -3904.0 + 30.0, "primary_start_y": 40.0,
             "primary_start_z": 5500.0, "primary_start_px": 0.0,
             "primary_start_py": 0.0, "primary_start_pz": 104.0,
             "primary_start_e": 104.97, "primary_start_m": 0.511,
             "primary_start_pdg": 11, "primary_calo_edep": 80.0,
             "primary_calo_edep_vis": 75.0}
    for name, value in first.items():
        branches[name] = per_event([], [value], [value, 0.0])
    branches["primary_trk_front_p"] = per_event([], [0.0], [103.0, 0.0])
    branches["primary_trk_front_energy"] = per_event([], [0.0], [103.5, 0.0])
    return branches


def test_edep_variables_take_the_first_primary_and_mark_what_is_missing():
    v = edep_mod.edep_variables(_edep_branches())
    assert np.isnan(v["primary_start_e"][0])                # no primary
    assert v["primary_start_e"][2] == np.float32(104.97)    # first primary's
    assert v["has_trk_front"].tolist() == [False, False, True]
    assert np.isnan(v["primary_trk_front_energy"][1])       # never got there
    assert abs(v["primary_trk_front_energy_diff"][2] - (103.5 - 104.97)) < 1e-4
    assert abs(v["primary_start_r"][1] - 50.0) < 1e-3       # detector frame
    assert set(v) == set(edep_mod.EDEP_VARIABLES)
    # every advertised variable is usable in a selection
    for name in edep_mod.EDEP_VARIABLES:
        edep_mod.select_events(v, f"{name} == {name}")


def test_edep_selection_counts_weighted_events_per_gen_event():
    v = edep_mod.edep_variables(_edep_branches())
    assert edep_mod.selected_metrics(v, "event_calo_edep_vis > 50", 10.0) == {
        "n_events_selected": 1.0, "selected_per_gen_event": 0.1}
    assert edep_mod.selected_metrics(v, "", 10.0)["n_events_selected"] == 3.0


def test_edep_optional_step_branches_are_variables_only_where_present(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=20)
    old = edep_mod.read_edep_tree(path)
    assert set(old) == set(edep_mod.EDEP_VARIABLES)       # no step columns
    new_path = Path(tmp_dir) / "nts.owner.edep.steps.root"
    _write_edep_tree(new_path, n=20, with_steps=True)
    new = edep_mod.read_edep_tree(new_path)
    assert set(new) == set(edep_mod.EDEP_VARIABLES) | set(edep_mod.OPTIONAL_EDEP_VARIABLES)
    assert new["event_trk_nsteps_ecut_tcut"].tolist() == list(range(20))
    mask = edep_mod.select_events(new, "event_trk_nsteps_ecut_tcut >= 15")
    assert mask.sum() == 5
    # a mixed set of files has them for neither
    assert set(edep_mod.read_edep_tree([path, new_path])) == set(edep_mod.EDEP_VARIABLES)
    # a selection naming one on an old file is a plain error, not a KeyError
    try:
        edep_mod.select_events(old, "event_trk_nsteps > 3")
        raise AssertionError("no error")
    except edep_mod.EdepTreeError as exc:
        assert "event_trk_nsteps" in str(exc) and "PR #2067" in str(exc), exc
    # a name that is merely unknown still gets the selection's own message
    try:
        edep_mod.select_events(old, "no_such_thing > 3")
        raise AssertionError("no error")
    except SelectionError:
        pass


def _write_edep_tree(path: Path, n: int = 4000, ngen_total: int | None = None,
                     with_steps: bool = False) -> None:
    """A CE-like EdepAna tree: ~105 MeV electrons losing a little on the way.

    `ngen` is EdepAna's running generated-event count, raised at each subrun;
    with `ngen_total` it climbs to that over the file, else it is n throughout.
    """
    import awkward as ak
    import uproot
    rng = np.random.default_rng(1)
    e0 = np.full(n, 104.97, dtype=np.float32)
    front = (e0 - rng.exponential(0.6, n)).astype(np.float32)
    calo = rng.uniform(0.0, 100.0, n).astype(np.float32)
    def one(values):
        return ak.unflatten(values, np.ones(n, dtype=np.int64))
    tree = {
        "nprimaries": np.ones(n, dtype=np.int32),
        "event_calo_edep": calo, "event_calo_edep_vis": calo,
        "event_trk_edep": np.zeros(n, dtype=np.float32),
        "weight": np.ones(n, dtype=np.float32),
        "run": np.ones(n, dtype=np.int32), "subrun": np.ones(n, dtype=np.int32),
        "event": np.arange(n, dtype=np.int32),
        "ngen": (np.full(n, n, dtype=np.int64) if ngen_total is None else
                 np.ceil(np.arange(1, n + 1) * ngen_total / n).astype(np.int64)),
        "primary_trk_front_p": one(front), "primary_trk_front_energy": one(front),
        "primary_start_e": one(e0), "primary_start_pdg": one(np.full(n, 11, np.int32)),
    }
    if with_steps:      # the int branches of Offline PR #2067: i for every cut
        for det in ("calo", "trk"):
            for cut in ("", "_ecut", "_ecut_tcut"):
                tree[f"event_{det}_nsteps{cut}"] = np.arange(n, dtype=np.int32)
    zeros = np.zeros(n, dtype=np.float32)
    for name in ("x", "y", "z", "px", "py", "pz", "m"):
        tree[f"primary_start_{name}"] = one(zeros)
    for name in ("primary_calo_edep", "primary_calo_edep_vis"):
        tree[name] = one(calo)
    # mktree, not assignment: this uproot writes a dict as an RNTuple
    types = {name: ("var * " + str(values.layout.content.dtype)
                    if isinstance(values, ak.Array) else values.dtype)
             for name, values in tree.items()}
    with uproot.recreate(path) as f:
        f.mktree("EDepAna/tree", types).extend(tree)


def test_signal_is_r_mue_times_the_captures_per_stop(tmp_dir):
    """R_mue is per muon capture, so the CEs per stopped muon are R_mue times
    the fraction of stops that capture, not R_mue divided by it."""
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"sig_eff": 1e-3})
    assert result.status == "success", result.message
    assert result.metadata["signal_br"] == 1.0e-13 * 0.609, result.metadata["signal_br"]
    log = (Path(tmp_dir) / "approx_ce_sensitivity.log").read_text()
    assert "R_mue = 1e-13" in log and "1e-9" not in log, log


def test_sensitivity_runs_on_the_tree_with_the_selection_it_is_given(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path)
    results = {}
    for cut in ("event_calo_edep_vis > 10", "event_calo_edep_vis > 50"):
        result = run_analysis(analysis="approx_ce_sensitivity",
                              data_file=str(path), output_dir=tmp_dir,
                              parameters={"sig_eff": 0.5, "selection": cut})
        assert result.status == "success", result.message
        results[cut] = result.metadata
    loose, tight = results.values()
    # the shape normalization is per selected event, so the selection matters
    assert loose["n_events_selected"] > tight["n_events_selected"] > 0
    assert loose["selection"] == sens.DEFAULT_SELECTION

    nothing = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                           output_dir=tmp_dir,
                           parameters={"sig_eff": 0.5,
                                       "selection": "event_calo_edep_vis > 1000"})
    assert nothing.status == "error" and "0 of 4000" in nothing.message
    typo = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                        output_dir=tmp_dir,
                        parameters={"sig_eff": 0.5, "selection": "calo > 10"})
    assert typo.status == "error" and "unknown variable" in typo.message


# --- approx_ce_sensitivity: the efficiency from the stopping rate --------------

def test_generated_events_is_each_files_last_running_count(tmp_dir):
    a, b = Path(tmp_dir) / "a.root", Path(tmp_dir) / "b.root"
    _write_edep_tree(a, n=400, ngen_total=1000)
    _write_edep_tree(b, n=300, ngen_total=900)
    assert edep_mod.generated_events(a) == 1000
    assert edep_mod.generated_events([a, b]) == 1900


def test_sensitivity_takes_exactly_one_of_sig_eff_and_stops_per_pot(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path)
    for params in ({}, {"sig_eff": 0.1, "stops_per_pot": 1e-3}):
        result = run_analysis(analysis="approx_ce_sensitivity",
                              data_file=str(path), output_dir=tmp_dir,
                              parameters=params)
        assert result.status == "error", params
        assert "one of sig_eff or stops_per_pot" in result.message, result.message
    declared = list_analyses().metadata["analyses"]["approx_ce_sensitivity"]["parameters"]
    assert not declared["sig_eff"]["required"]
    assert not declared["stops_per_pot"]["required"]


def test_stops_per_pot_is_sig_eff_times_the_files_ce_acceptance(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=4000, ngen_total=8000)     # acceptance 0.5
    by_stops = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                            output_dir=str(Path(tmp_dir) / "stops"),
                            parameters={"stops_per_pot": 2e-3, "selection": ""})
    by_eff = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=str(Path(tmp_dir) / "eff"),
                          parameters={"sig_eff": 1e-3, "selection": ""})
    assert by_stops.status == "success", by_stops.message
    assert by_eff.status == "success", by_eff.message
    meta = by_stops.metadata
    assert meta["sensitivity"] == by_eff.metadata["sensitivity"]
    assert (meta["sig_eff"], meta["ce_acceptance"], meta["n_gen_events"],
            meta["stops_per_pot"]) == (1e-3, 0.5, 8000.0, 2e-3), meta


def test_signal_efficiency_is_stops_per_pot_times_selected_per_generated():
    variables = {"weight": np.ones(400)}
    mask = np.arange(400) < 300
    assert sens.signal_efficiency(variables, mask, 1000.0, 2e-3) == (2e-3 * 0.3, 0.3)


def test_signal_efficiency_refuses_a_selection_no_event_passes():
    try:
        sens.signal_efficiency({"weight": np.ones(10)}, np.zeros(10, bool),
                               100.0, 2e-3)
    except sens.SensitivityError as exc:
        assert "no event" in str(exc) and "selection" in str(exc), exc
    else:
        raise AssertionError("an acceptance of 0 was accepted")


def test_signal_efficiency_refuses_an_acceptance_above_one():
    try:
        sens.signal_efficiency({"weight": np.ones(400)}, np.ones(400, bool),
                               100.0, 2e-3)
    except sens.SensitivityError as exc:
        assert "acceptance" in str(exc) and "4" in str(exc), exc
    else:
        raise AssertionError("an acceptance of 4 was accepted")


def test_signal_efficiency_refuses_an_empty_file_for_what_it_is():
    try:
        sens.signal_efficiency({"weight": np.ones(0)}, np.ones(0, bool),
                               0.0, 2e-3)
    except sens.SensitivityError as exc:
        assert "no events" in str(exc), exc
    else:
        raise AssertionError("an empty file was accepted")


def test_generated_events_refuses_a_merged_file(tmp_dir):
    """ngen restarts with each job, so a falling count means several edep
    outputs were merged into one file: their totals cannot be told apart."""
    import uproot
    path = Path(tmp_dir) / "merged.root"
    with uproot.recreate(path) as f:
        f.mktree("EDepAna/tree", {"ngen": np.int64}).extend(
            {"ngen": np.array([1000, 2000, 2000, 500, 1500], dtype=np.int64)})
    try:
        edep_mod.generated_events(path)
    except edep_mod.EdepTreeError as exc:
        assert "merged" in str(exc), exc
    else:
        raise AssertionError("a merged file was accepted")


def test_sensitivity_refuses_a_nan_efficiency_input(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=100, ngen_total=200)
    for params in ({"sig_eff": 0.1, "stops_per_pot": "nan"},
                   {"sig_eff": "nan", "stops_per_pot": 2e-3}):
        result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                              output_dir=tmp_dir, parameters=params)
        assert result.status == "error", params
        assert "nan" in result.message.lower(), result.message


def test_stops_per_pot_needs_a_generated_count(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=100, ngen_total=0)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"stops_per_pot": 2e-3})
    assert result.status == "error"
    assert "generated" in result.message, result.message


def test_ce_acceptance_counts_only_events_passing_the_selection(tmp_dir):
    """The selection sets the signal's size, not only its shape: a tighter
    cut must lower the efficiency stops_per_pot works out."""
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=4000, ngen_total=8000)
    cut = "event_calo_edep_vis > 50"
    passing = float((edep_mod.read_edep_tree(path)["event_calo_edep_vis"] > 50).sum())
    acceptance = passing / 8000.0
    assert 0.0 < acceptance < 0.45, acceptance       # the cut removes events
    by_stops = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                            output_dir=str(Path(tmp_dir) / "stops"),
                            parameters={"stops_per_pot": 2e-3, "selection": cut})
    by_eff = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=str(Path(tmp_dir) / "eff"),
                          parameters={"sig_eff": 2e-3 * acceptance, "selection": cut})
    assert by_stops.status == "success", by_stops.message
    assert by_eff.status == "success", by_eff.message
    meta = by_stops.metadata
    assert (meta["ce_acceptance"], meta["sig_eff"]) == (acceptance, 2e-3 * acceptance), meta
    assert meta["sensitivity"] == by_eff.metadata["sensitivity"]


def _fake_edep_job(tmp_dir, stdout=None):
    """A stand-in for run_mu2e_job: writes an EdepAna tree where the job
    would and reports the summary block, recording the max_events it got."""
    from tools.mu2e_job import JobOutcome
    class Calls(list):
        fcl: list = []      # the fcl each job was given
    calls = Calls()
    calls_fcl = calls.fcl = []

    def fake(*, fcl, input_paths, outdir, single, timeout_s, max_events=None, **_):
        calls.append(max_events)
        outdir.mkdir(parents=True, exist_ok=True)
        root = outdir / "nts.owner.edep.test.root"
        _write_edep_tree(root, n=100, with_steps=_OFFLINE_HAS_STEPS[0])
        calls_fcl.append(fcl)
        return JobOutcome(command="mu2e", returncode=0, timed_out=False,
                          stdout=stdout or (SAMPLE_EDEP_STEPS_STDOUT if _OFFLINE_HAS_STEPS[0]
                                            else SAMPLE_EDEP_STDOUT), stderr="",
                          log_path=outdir / "mu2e.log", input_paths=input_paths,
                          input_flag="-s", environment="test", file_list_path=None,
                          written_root_files=[str(root)])
    return fake, calls


def test_edep_records_max_events_next_to_its_file(tmp_dir):
    from tools.spec import RunContext
    fake, calls = _fake_edep_job(tmp_dir)
    real = edep_mod.run_mu2e_job
    edep_mod.run_mu2e_job = fake
    try:
        for max_events in (100, None):
            outdir = Path(tmp_dir) / f"run{max_events}"
            outcome = edep_mod.run(RunContext(
                input_paths=[Path(tmp_dir) / "in.art"], outdir=outdir,
                params=ANALYSES["edep"].resolve_params({"selection": ""}),
                timeout_s=60, max_events=max_events))
            assert outcome.error is None, outcome.error
            root = Path(outcome.files[0])
            assert edep_mod.read_run_record(root) == {"max_events": max_events}
            # the record rides next to the file; it is not an output to chain
            assert outcome.files == [str(root)], outcome.files
    finally:
        edep_mod.run_mu2e_job = real
    assert calls == [100, None], calls


def _run_edep_with_fake_job(tmp_dir, has_steps, parameters=None, stdout=None):
    from tools.spec import RunContext
    fake, calls = _fake_edep_job(tmp_dir, stdout)
    real, _OFFLINE_HAS_STEPS[0] = edep_mod.run_mu2e_job, has_steps
    edep_mod.run_mu2e_job = fake
    try:
        params = ANALYSES["edep"].resolve_params(parameters)
        outcome = edep_mod.run(RunContext(
            input_paths=[Path(tmp_dir) / "in.art"], outdir=Path(tmp_dir) / "out",
            params=params, timeout_s=60))
    finally:
        edep_mod.run_mu2e_job, _OFFLINE_HAS_STEPS[0] = real, True
    return outcome, calls.fcl


def test_edep_with_the_step_counts_runs_a_per_run_fcl_and_reports_them(tmp_dir):
    outcome, fcls = _run_edep_with_fake_job(tmp_dir, True,
                                            {"calo_step_energy_threshold_mev": 0.5})
    assert outcome.error is None, outcome.error
    assert fcls == [Path(tmp_dir) / "out" / "edep_job.fcl"]
    assert "CaloStepEnergyThreshold : 0.5" in fcls[0].read_text()
    assert outcome.metrics["trk_nsteps_ecut_per_gen_event"] == 16.204
    assert outcome.metrics["trk_nsteps_per_event"] == -1.0
    assert outcome.extra["step_count_windows"]["calo"]["energy_threshold_mev"] == 0.5
    assert outcome.extra["job_fcl"] == str(fcls[0])
    # every metric the spec declares is there, in order
    assert tuple(outcome.metrics) == ANALYSES["edep"].metrics


def test_edep_on_an_old_offline_runs_the_shipped_fcl_and_leaves_steps_out(tmp_dir):
    outcome, fcls = _run_edep_with_fake_job(tmp_dir, False)
    assert outcome.error is None, outcome.error
    assert fcls == [FCL_DIR / "edep.fcl"]
    assert not (Path(tmp_dir) / "out" / "edep_job.fcl").exists()
    assert not any("nsteps" in name for name in outcome.metrics)
    assert outcome.metrics["n_events_calo_edep_above_50mev"] == 42.0
    assert "PR #2067" in outcome.extra["step_counts"]
    assert "step_count_windows" not in outcome.extra


def test_edep_on_an_old_offline_refuses_a_changed_step_parameter(tmp_dir):
    outcome, fcls = _run_edep_with_fake_job(
        tmp_dir, False, {"trk_step_time_min_ns": 100, "calo_step_energy_threshold_mev": 1})
    assert fcls == [], "no job may run"
    assert "predates the step counts" in outcome.error and "PR #2067" in outcome.error
    assert "trk_step_time_min_ns" in outcome.error
    assert "calo_step_energy_threshold_mev" in outcome.error


def test_edep_on_an_old_offline_refuses_a_step_selection_before_running(tmp_dir):
    outcome, fcls = _run_edep_with_fake_job(
        tmp_dir, False, {"selection": "event_trk_nsteps_ecut_tcut > 100"})
    assert fcls == [] and "event_trk_nsteps_ecut_tcut" in outcome.error, outcome.error
    outcome, fcls = _run_edep_with_fake_job(
        tmp_dir, True, {"selection": "event_trk_nsteps_ecut_tcut > 100"})
    assert outcome.error is None and len(fcls) == 1, outcome.error


def test_edep_with_steps_refuses_an_empty_window_and_missing_blocks(tmp_dir):
    outcome, fcls = _run_edep_with_fake_job(
        tmp_dir, True, {"calo_step_time_min_ns": 900, "calo_step_time_max_ns": 800})
    assert fcls == [] and "calo_step_time_min_ns" in outcome.error
    outcome, _ = _run_edep_with_fake_job(tmp_dir, True, stdout=SAMPLE_EDEP_STDOUT)
    assert "step-count blocks not found" in outcome.error


def test_edep_reports_a_failed_probe_plainly(tmp_dir):
    from tools.mu2e_job import ModuleProbeError
    from tools.spec import RunContext

    def broken(*args, **kwargs):
        raise ModuleProbeError("mu2e --print-description EdepAna failed")
    real = edep_mod.module_accepts
    edep_mod.module_accepts = broken
    try:
        outcome = edep_mod.run(RunContext(
            input_paths=[Path(tmp_dir) / "in.art"], outdir=Path(tmp_dir),
            params=ANALYSES["edep"].resolve_params({}), timeout_s=60))
    finally:
        edep_mod.module_accepts = real
    assert "step counts" in outcome.error and "failed" in outcome.error


def test_stops_per_pot_refuses_a_file_from_a_max_events_run(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=4000, ngen_total=8000)
    edep_mod.write_run_record(path, max_events=100)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"stops_per_pot": 2e-3})
    assert result.status == "error"
    assert "max_events" in result.message, result.message


def test_stops_per_pot_says_when_it_cannot_confirm_a_full_run(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=4000, ngen_total=8000)
    unrecorded = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                              output_dir=str(Path(tmp_dir) / "a"),
                              parameters={"stops_per_pot": 2e-3})
    assert unrecorded.status == "success", unrecorded.message
    assert unrecorded.metadata["edep_run_recorded"] is False
    log = (Path(tmp_dir) / "a" / "approx_ce_sensitivity.log").read_text()
    assert "cannot confirm" in log, log
    edep_mod.write_run_record(path, max_events=None)
    recorded = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                            output_dir=str(Path(tmp_dir) / "b"),
                            parameters={"stops_per_pot": 2e-3})
    assert recorded.status == "success", recorded.message
    assert recorded.metadata["edep_run_recorded"] is True
    log = (Path(tmp_dir) / "b" / "approx_ce_sensitivity.log").read_text()
    assert "cannot confirm" not in log, log


def test_stops_per_pot_is_at_most_one(tmp_dir):
    declared = list_analyses().metadata["analyses"]["approx_ce_sensitivity"]["parameters"]
    assert declared["stops_per_pot"]["maximum"] == 1.0
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=100, ngen_total=200)
    result = run_analysis(analysis="approx_ce_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"stops_per_pot": 1.5})
    assert result.status == "error"
    assert "above the maximum" in result.message, result.message


# --- the trigger ---------------------------------------------------------------

def test_trigger_paths_take_prescales_and_default_to_one():
    assert trig.parse_trigger_paths(
        "calo_photon:1, cpr_TrkDe_80m70p:10 apr_TrkDe_80m70p") == [
        ("calo_photon", 1), ("cpr_TrkDe_80m70p", 10), ("apr_TrkDe_80m70p", 1)]


def test_trigger_paths_refuse_what_cannot_be_run():
    for bad, words in (("", "no trigger paths"),
                       ("apr_TrkDe:0", "at least 1"),
                       ("apr_TrkDe:x", "whole number"),
                       ("apr_TrkDe:2.5", "whole number"),
                       ("apr_TrkDe, apr_TrkDe:2", "listed twice"),
                       ('bad"name', "does not name"),
                       ("1abc", "does not name")):
        try:
            trig.parse_trigger_paths(bad)
        except trig.TriggerError as exc:
            assert words in str(exc), (bad, str(exc))
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def test_prescale_module_follows_the_menu_generator():
    assert trig.prescale_module("cpr_TrkDe_80m70p") == "CprTrkDe80m70pPS"
    assert trig.prescale_module("apr_TrkDe_80m70p_D0200") == "AprTrkDe80m70pD0200PS"
    assert trig.prescale_module("calo_photon") == "CaloPhotonPS"


def test_prescale_module_names_every_path_in_the_published_menu():
    """Against the generated menu itself, where it is on disk."""
    gen = Path("/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay/build/"
               "al9-prof-e29-p107/mu2e-trig-config/gen")
    menu, ps = gen / "trig_physMenu_OnSpill.fcl", gen / "trig_physMenuPSConfig_OnSpill.fcl"
    if not (menu.exists() and ps.exists()):
        return
    import re
    # tpr_/mpr_ paths are in the menu but not meant for the real trigger
    paths = [p for p in re.findall(r'"\d+:(\w+)"', menu.read_text())
             if not p.startswith(("tpr_", "mpr_"))]
    labels = set(re.findall(r"^\s*(\w+PS):", ps.read_text(), re.M))
    assert paths and {trig.prescale_module(p) for p in paths} <= labels


def test_job_fcl_sets_the_paths_and_their_prescales(tmp_dir):
    base = Path(tmp_dir) / "base.fcl"
    base.write_text('#include "mu2e-trig-config/test/timingTest.fcl"\n')
    text = trig.job_fcl_text([("apr_TrkDe_80m70p", 1), ("cpr_TrkDe_80m70p", 10)], base)
    assert text.startswith('#include "mu2e-trig-config/test/timingTest.fcl"')
    assert 'physics.trigger_paths : [ "apr_TrkDe_80m70p", "cpr_TrkDe_80m70p" ]' in text
    assert ("physics.filters.CprTrkDe80m70pPS.eventModeConfig : [ "
            "{ eventMode: OnSpill prescale: 10 }, "
            "{ eventMode: OffSpill prescale: 10 } ]") in text
    # the shipped base is the one runs use
    assert trig.FCL.exists() and "timingTest.fcl" in trig.FCL.read_text()


def test_job_fcl_takes_a_base_overrides_and_leading_paths(tmp_dir):
    base = Path(tmp_dir) / "base.fcl"
    base.write_text('#include "mu2e-trig-config/test/timingTest.fcl"\n'
                    'services.TimeTracker.dbOutput.overwrite : true\n')
    text = trig.job_fcl_text(
        [("apr_TrkDe_80m70p", 2)], base,
        base_fcl="trigger/reduced_tracker/redigi_trigger_tuned.fcl",
        overrides='#include "trigger/reduced_tracker/helix_tuning.fcl"\n'
                  "physics.producers.X.minNHit : 7",
        leading_paths=["TriggerablePath"])
    assert text.startswith('#include "trigger/reduced_tracker/redigi_trigger_tuned.fcl"')
    assert "timingTest" not in text and "TimeTracker.dbOutput.overwrite" in text
    # overrides come after the base and before the paths, which always win
    assert text.index("helix_tuning.fcl") < text.index("physics.trigger_paths")
    assert 'physics.trigger_paths : [ "TriggerablePath", "apr_TrkDe_80m70p" ]' in text
    assert "AprTrkDe80m70pPS.eventModeConfig" in text and "prescale: 2" in text
    plain = trig.job_fcl_text([("apr_TrkDe_80m70p", 1)], base)
    assert "fcl_overrides" not in plain and "TriggerablePath" not in plain


def test_base_fcl_and_leading_paths_refuse_what_cannot_be_included():
    assert trig.parse_base_fcl("") == ""
    assert trig.parse_base_fcl(" a/b_c.fcl ") == "a/b_c.fcl"
    for bad in ("/abs/x.fcl", "../x.fcl", 'x".fcl', "x.txt", "a b.fcl"):
        try:
            trig.parse_base_fcl(bad)
        except trig.TriggerError:
            continue
        raise AssertionError(f"accepted {bad!r}")
    assert trig.parse_leading_paths("A, B C") == ["A", "B", "C"]
    try:
        trig.parse_leading_paths("a;b")
    except trig.TriggerError:
        pass
    else:
        raise AssertionError("accepted a;b")


def test_trig_report_parses_events_and_every_path():
    report = trig.parse_trig_report(SAMPLE_TRIGGER_STDOUT)
    assert (report.n_events, report.n_passed) == (200, 182)
    assert list(report.paths) == ["calo_photon", "cpr_TrkDe_80m70p",
                                  "apr_TrkDe_80m70p"]
    assert report.paths["cpr_TrkDe_80m70p"] == trig.PathCounts(200, 40, 160, 0)
    # the per-module blocks after it are not mistaken for paths
    assert "Prefetch" not in report.paths
    assert trig.parse_trig_report("Art has completed and will exit with status 1.") is None


def test_binomial_is_never_exactly_certain():
    p, err = trig.binomial(0, 150)
    assert p == 0.0 and 0.0 < err < 0.02
    p, err = trig.binomial(150, 150)
    assert p == 1.0 and 0.0 < err < 0.02
    p, err = trig.binomial(500, 1000)
    assert abs(err - (0.25 / 1000) ** 0.5) < 1e-3       # the usual, mid-range
    assert trig.binomial(0, 0) == (0.0, 0.0)


def test_rate_scales_the_accept_fraction_by_the_duty_cycled_event_rate():
    report = trig.TrigReport(n_events=1000, n_passed=10)
    metrics = rate_metrics(report, 5.0e5, 0.3)
    assert metrics["accept_fraction"] == 0.01
    assert abs(metrics["onspill_rate_hz"] - 5000.0) < 1e-9
    assert abs(metrics["event_rate_hz"] - 1.5e5) < 1e-6
    assert abs(metrics["rate_hz"] - 1500.0) < 1e-9          # averaged over the cycle
    assert abs(metrics["rate_hz_err"] - metrics["accept_fraction_err"] * 1.5e5) < 1e-9
    assert metrics["duty_factor"] == 0.3


def test_duty_factor_comes_from_the_batch_mode_unless_given():
    assert duty_factor("1BB", 0.0) == 0.322
    assert duty_factor("2bb", 0.0) == 0.246
    assert duty_factor("2BB", 0.5) == 0.5                   # given: overrides
    try:
        duty_factor("3BB", 0.0)
    except trig.TriggerError as exc:
        assert "1BB" in str(exc) and "duty_factor" in str(exc)
    else:
        raise AssertionError("an unknown batch mode should be refused")
    params = ANALYSES["trigger_rate"].resolve_params({"trigger_paths": "apr_TrkDe_80m70p"})
    assert params["batch_mode"] == "1BB" and params["duty_factor"] == 0.0


def _write_timing_db(path: Path) -> None:
    """Event 7 is the slow first one, then three ordinary events. Each runs a
    2 ms Prefetch, then module A (cpr) and B (apr) for its processing time;
    module C (apr, behind a filter) runs only on events 9 and 10, for 4 ms.
    TimeTracker's whole-event time adds 1 ms of framework on top."""
    import sqlite3
    with sqlite3.connect(path) as con:
        con.execute("CREATE TABLE TimeEvent(Run, SubRun, Event, Time)")
        con.execute("CREATE TABLE TimeModule(Run, SubRun, Event, Path, "
                    "ModuleLabel, ModuleType, Time)")
        for event, t in ((7, 1.0), (8, 0.010), (9, 0.020), (10, 0.030)):
            c = 0.004 if event in (9, 10) else 0.0
            con.execute("INSERT INTO TimeEvent VALUES (1, 0, ?, ?)",
                        (event, t + c + 0.002 + 0.001))
            con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, 'apr', "
                        "'Prefetch', 'PrefetchDAQData', 0.002)", (event,))
            con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, 'cpr', 'A', "
                        "'TypeA', ?)", (event, t * 0.75))
            con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, 'apr', 'B', "
                        "'TypeB', ?)", (event, t * 0.25))
            if c:
                con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, 'apr', "
                            "'C', 'TypeC', ?)", (event, c))


def test_timing_leaves_out_the_data_fetch_and_the_warm_up_events(tmp_dir):
    db = Path(tmp_dir) / "triggerTiming.db"
    _write_timing_db(db)
    timed = trig.read_timing_db(db, skip_events=1)
    assert timed.n_skipped == 1
    # processing only: no Prefetch, no framework time
    assert np.allclose(timed.times, [0.010, 0.024, 0.034])
    assert np.allclose(timed.fetch_times, 0.002)
    assert np.allclose(timed.full_event_times, [0.013, 0.027, 0.037])
    assert abs(timed.per_path["cpr"] - 0.015) < 1e-12     # 0.75 * mean 0.020
    assert abs(timed.per_path["apr"] - (0.005 + 0.008 / 3)) < 1e-12  # no Prefetch
    everything = trig.read_timing_db(db, skip_events=0)
    assert everything.times.size == 4 and everything.times[0] == 1.0
    metrics = timing_metrics(timed.times)
    assert abs(metrics["mean_time_ms"] - 68.0 / 3) < 1e-9
    assert abs(metrics["median_time_ms"] - 24.0) < 1e-9
    assert abs(metrics["max_time_ms"] - 34.0) < 1e-9
    try:
        trig.read_timing_db(Path(tmp_dir) / "missing.db", 1)
    except trig.TriggerError as exc:
        assert "no timing database" in str(exc)
    else:
        raise AssertionError("a missing database should be reported")


def test_ignored_modules_are_left_out_of_the_event_and_path_times(tmp_dir):
    db = Path(tmp_dir) / "triggerTiming.db"
    _write_timing_db(db)
    base = trig.read_timing_db(db, skip_events=1)
    timed = trig.read_timing_db(db, skip_events=1,
                                ignore=trig.parse_ignore_modules("C, TypeB"))
    # B (by type) and C (by label) removed: only A's 75% is left
    assert np.allclose(timed.times, [0.0075, 0.015, 0.0225])
    assert np.allclose(timed.ignored_times, base.times - timed.times)
    assert "apr" not in timed.per_path or abs(timed.per_path["apr"]) < 1e-12
    rows = {r["label"]: r for r in timed.module_table()}
    assert rows["B"]["counted"] is False and rows["C"]["counted"] is False
    assert rows["A"]["counted"] is True
    assert not base.ignored_times.any()
    assert trig.parse_ignore_modules("") == frozenset()
    try:
        trig.parse_ignore_modules("a;b")
    except trig.TriggerError:
        pass
    else:
        raise AssertionError("accepted a;b")


def test_module_timing_counts_the_events_each_module_ran_on(tmp_dir):
    db = Path(tmp_dir) / "triggerTiming.db"
    _write_timing_db(db)
    table = trig.read_timing_db(db, skip_events=1).module_table()
    # every module, in the order they first ran, the fetch included but flagged
    assert [(r["path"], r["label"]) for r in table] == [
        ("apr", "Prefetch"), ("cpr", "A"), ("apr", "B"), ("apr", "C")]
    rows = {r["label"]: r for r in table}
    assert rows["Prefetch"]["counted"] is False and rows["A"]["counted"] is True
    # C sat behind a filter: it ran on 2 of the 3 timed events
    assert rows["A"]["n_seen"] == 3 and rows["C"]["n_seen"] == 2
    assert abs(rows["C"]["seen_fraction"] - 2 / 3) < 1e-12
    assert abs(rows["C"]["mean_ms"] - 4.0) < 1e-9          # per event it ran on
    assert abs(rows["C"]["ms_per_event"] - 8.0 / 3) < 1e-9  # per timed event
    assert abs(rows["A"]["max_ms"] - 22.5) < 1e-9 and rows["A"]["median_ms"] == 15.0
    # the warm-up event is not counted as a run of anything
    assert rows["Prefetch"]["n_seen"] == 3


def test_a_shared_module_is_added_up_over_the_paths_that_ran_it(tmp_dir):
    """CaloHit runs once per event, charged to whichever path got there first."""
    import sqlite3
    db = Path(tmp_dir) / "triggerTiming.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE TimeEvent(Run, SubRun, Event, Time)")
        con.execute("CREATE TABLE TimeModule(Run, SubRun, Event, Path, "
                    "ModuleLabel, ModuleType, Time)")
        for event in range(1, 6):
            con.execute("INSERT INTO TimeEvent VALUES (1, 0, ?, 0.01)", (event,))
            first = "calo_photon" if event % 2 == 0 else "apr"
            con.execute("INSERT INTO TimeModule VALUES (1, 0, ?, ?, 'CaloHit', "
                        "'CaloHitMakerFast', 0.001)", (event, first))
    rows = trig.read_timing_db(db, skip_events=1).module_table()
    split = {r["path"]: r for r in rows}
    assert split["calo_photon"]["n_seen"] == 2 and split["apr"]["n_seen"] == 2
    for row in rows:
        assert row["n_seen_all_paths"] == 4
        assert abs(row["ms_per_event_all_paths"] - 1.0) < 1e-12


def test_code_files_are_found_down_the_backing_chain(tmp_dir):
    root = Path(tmp_dir)
    area, musing, offline = root / "area", root / "musing", root / "offline"
    for d in (area, musing, offline):
        d.mkdir()
    (area / "backing").symlink_to(musing)
    (musing / "backing").symlink_to(offline)
    target = musing / "mu2e-trig-config" / "ci"
    target.mkdir(parents=True)
    (target / "data_files.txt").write_text("/data/a.art\n\n# note\n/data/b.art\n")

    env = Mu2eEnv.for_work_area(area)
    found = env.find_code_file("mu2e-trig-config/ci/data_files.txt")
    assert found == area / "backing" / "mu2e-trig-config/ci/data_files.txt"
    assert env.find_code_file("nowhere/at/all.txt") is None

    before = current()
    try:
        configure(env)
        assert trig.default_pileup_inputs() == [Path("/data/a.art"), Path("/data/b.art")]
        configure(Mu2eEnv.for_work_area(offline))
        try:
            trig.default_pileup_inputs()
        except trig.TriggerError as exc:
            assert "pass data_file" in str(exc)
        else:
            raise AssertionError("no CI file list should be reported")
    finally:
        configure(None)
        assert current().describe() == before.describe()


def test_a_musing_is_searched_from_its_published_directory():
    env = Mu2eEnv.for_musing("SimJob MDC2025ay")
    found = env.find_code_file(trig.DEFAULT_PILEUP_LIST)
    musing_dir = Path("/cvmfs/mu2e.opensciencegrid.org/Musings/SimJob/MDC2025ay")
    if musing_dir.is_dir():
        assert found == musing_dir / trig.DEFAULT_PILEUP_LIST


def test_only_analyses_with_default_inputs_run_without_data_files(tmp_dir):
    result = run_analysis(analysis="trigger_efficiency", output_dir=tmp_dir,
                          parameters={"trigger_paths": "apr_TrkDe_80m70p"})
    assert result.status == "error" and "exactly one of" in result.message
    listed = list_analyses().metadata["analyses"]
    assert "default_inputs" in listed["trigger_rate"]
    assert "default_inputs" in listed["trigger_timing"]
    assert "default_inputs" not in listed["trigger_efficiency"]


# --- trigger efficiency from EventNtuples -------------------------------------

# A CE EventNtuple with trig_<path> branches; tests using it are skipped where
# it is not on disk.
NTUPLE_FILE = Path("/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/"
                   "MDC2025au_best_v1_1-001/root/a8/50/nts.mu2e.CeMLeadingLogMix1BB."
                   "MDC2025au_best_v1_1-001.001430_00000000.root")


def test_ntuple_trigger_names_drop_the_prefix_and_refuse_typos():
    assert ntrig.parse_trigger_names("apr_TrkDe_80m70p, trig_cpr_TrkDe_80m70p") == [
        "apr_TrkDe_80m70p", "cpr_TrkDe_80m70p"]
    for bad in ("", "apr_TrkDe, trig_apr_TrkDe", "bad-name"):
        try:
            ntrig.parse_trigger_names(bad)
        except ntrig.NtupleError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def _ntuple_arrays():
    """Two events in the EventNtuple's layout. Event 0: a downstream e- that
    crosses TT_Front twice (upstream leg first), at 104 MeV/c, plus a track
    with no segments at all. Event 1: an upstream-going e+."""
    import awkward as ak

    def seg(sid, px, py, pz, time, momerr=0.2):
        return {"mom": {"fCoordinates": {"fX": px, "fY": py, "fZ": pz}},
                "time": time, "momerr": momerr, "sid": sid}

    def pars(d0, t0err):
        return {"d0": d0, "maxr": 600.0, "tanDip": 0.7, "rad": 250.0,
                "t0": 900.0, "t0err": t0err}

    good = [seg(0, 0.0, 30.0, -80.0, 850.0),        # upstream-going TT_Front leg
            seg(0, 0.0, 80.0, 66.4529, 900.0),       # downstream TT_Front: p = 104
            seg(1, 0.0, 79.0, 66.0, 902.0)]          # TT_Mid, pz > 0
    upstream = [seg(1, 0.0, 70.0, -60.0, 700.0)]
    tracks = {
        "trk.status": [[1, 1], [1]], "trk.goodfit": [[1, 0], [1]],
        "trk.pdg": [[11, 11], [-11]], "trk.nactive": [[30, 5], [25]],
        "trk.chisq": [[40.0, 9.0], [30.0]], "trk.ndof": [[30, 3], [20]],
        "trkqual.result": [[0.9, 0.1], [0.8]], "trkpid.result": [[0.7, 0.0], [0.6]],
        "trkcalohit.did": [[12, -1], [-1]], "trkcalohit.edep": [[95.0, 0.0], [0.0]],
        "trkcalohit.dt": [[1.5, 0.0], [0.0]],
        "trksegs": [[good, []], [upstream]],
        "trksegpars_lh": [[[pars(10.0, 9.0), pars(20.0, 9.0), pars(30.0, 0.5)], []],
                          [[pars(40.0, 0.4)]]],
        "event": [5, 6], "run": [1, 1], "subrun": [0, 0],
    }
    for name in ntrig._TRK_FIELDS:
        tracks.setdefault(f"trk.{name}", [[0, 0], [0]])
    return {name: ak.Array(values) for name, values in tracks.items()}


def test_ntuple_track_variables_take_the_downstream_front_and_the_mid_fit():
    v = ntrig.track_variables(_ntuple_arrays())
    assert v["_event"].tolist() == [0, 0, 1]
    assert abs(v["p_front"][0] - 104.0) < 0.01               # not the upstream leg
    assert v["t_front"][0] == 900.0 and v["downstream"].tolist() == [True, False, False]
    assert abs(v["tandip_front"][0] - 66.4529 / 80.0) < 1e-6
    assert v["d0"][0] == 30.0 and v["t0err"][0] == 0.5       # the TT_Mid entry
    assert np.isnan(v["p_front"][1]) and np.isnan(v["d0"][1])  # no segments
    assert np.isnan(v["p_front"][2]) and v["d0"][2] == 40.0  # upstream: no front
    assert v["has_calo"].tolist() == [True, False, False]
    assert v["calo_edep"][0] == 95.0 and np.isnan(v["calo_edep"][1])
    assert v["charge"].tolist() == [-1.0, -1.0, 1.0]
    assert v["ntrk"].tolist() == [2.0, 2.0, 1.0] and v["event"].tolist() == [5, 5, 6]
    assert set(v) - {"_event"} == set(ntrig.TRACK_VARIABLES)
    for name in ntrig.TRACK_VARIABLES:            # every advertised one is usable
        apply_selection(f"{name} == {name}", v, v["_event"].size)


def test_ntuple_efficiency_is_over_events_with_a_selected_track():
    v = ntrig.track_variables(_ntuple_arrays())
    triggers = {"a": np.array([True, True]), "b": np.array([False, True])}
    counts = ntrig.efficiency_counts(v, triggers, 2, ntrig.DEFAULT_SELECTION)
    assert counts["n_selected"] == 1 and counts["n_tracks_selected"] == 1
    assert counts["n_triggered"] == 1 and counts["per_path"] == {"a": 1, "b": 0}
    everything = ntrig.efficiency_counts(v, triggers, 2, "")
    assert everything["n_selected"] == 2 and everything["per_path"]["b"] == 1
    metrics = ntrig.efficiency_metrics(counts)
    assert metrics["efficiency"] == 1.0 and metrics["efficiency_err"] > 0.0


def _ntuple_mc_arrays():
    """_ntuple_arrays with MC truth: the e- track's particle crosses TT_Front
    downstream at 105 MeV/c, the step stored after a later TT_Mid step and an
    earlier upstream TT_Front one; the other two tracks have no true steps."""
    import awkward as ak

    def step(sid, pz, time, pt=0.0):
        return {"sid": sid, "time": time,
                "mom": {"fCoordinates": {"fX": 0.0, "fY": pt, "fZ": pz}}}

    def sim(pdg, code, p):
        return {"pdg": pdg, "startCode": code, "gen": 0, "time": 600.0,
                "nhits": 30, "nactive": 28,
                "mom": {"fCoordinates": {"fX": 0.0, "fY": 0.0, "fZ": p}}}

    arrays = _ntuple_arrays()
    arrays["trksegsmc"] = ak.Array([
        [[step(1, 60.0, 905.0, 80.0), step(0, -60.0, 850.0, 80.0),
          step(0, 63.0, 901.0, 84.0), step(0, 70.0, 990.0, 10.0)], []], [[]]])
    arrays["trkmcsim"] = ak.Array([[[sim(11, 168, 104.9), sim(13, 0, 50.0)], []],
                                   [[sim(-11, 179, 70.0)]]])
    arrays[ntrig.NPOT_BRANCH] = ak.Array([3.0e7, 1.5e7])
    return arrays


def test_ntuple_mc_variables_take_the_earliest_downstream_front_crossing():
    v = ntrig.track_variables(_ntuple_mc_arrays())
    assert abs(v["mc_p_front"][0] - 105.0) < 1e-4 and v["mc_t_front"][0] == 901.0
    assert abs(v["mc_cos_front"][0] - 63.0 / 105.0) < 1e-6
    assert v["mc_p_mid"][0] == 100.0 and v["mc_has_front"].tolist() == [True, False, False]
    assert np.isnan(v["mc_p_front"][1]) and np.isnan(v["mc_pdg"][1])  # no MC
    assert v["mc_pdg"].tolist()[::2] == [11, -11] and v["mc_start_code"][2] == 179
    assert abs(v["mc_p_start"][0] - 104.9) < 1e-4
    assert v["mc_npot"].tolist() == [3.0e7, 3.0e7, 1.5e7]     # per event
    assert set(v) - {"_event"} == set(ntrig.TRACK_VARIABLES) | set(ntrig.MC_VARIABLES)
    for name in ntrig.MC_VARIABLES:
        apply_selection(f"{name} == {name}", v, v["_event"].size)


def test_ntuple_mc_variables_on_data_say_there_is_no_truth():
    v = ntrig.track_variables(_ntuple_arrays())
    ntrig.check_mc_names(v, "p_front > 80")                   # no MC asked for
    try:
        ntrig.check_mc_names(v, "p_front > 80", "mc_npot / 1e6")
    except ntrig.NtupleError as exc:
        assert "mc_npot" in str(exc) and "no " in str(exc)
    else:
        raise AssertionError("mc_npot on a file without MC should be refused")


def test_th1_stores_the_binned_mean_and_rms(tmp_dir):
    import uproot
    edges = np.array([0.0, 1.0, 2.0, 4.0])
    counts = np.array([1.0, 2.0, 1.0])
    path = Path(tmp_dir) / "h.root"
    with uproot.recreate(path) as f:
        f["h"] = bntrig._th1(counts, edges, "h", "x")
    with uproot.open(path) as f:
        h = f["h"]
        sumw, sumwx, sumwx2 = (h.member(m) for m in ("fTsumw", "fTsumwx", "fTsumwx2"))
    centers = np.array([0.5, 1.5, 3.0])
    mean = (counts * centers).sum() / counts.sum()
    assert np.isclose(sumwx / sumw, mean)
    assert np.isclose(sumwx2 / sumw - mean ** 2,
                      (counts * (centers - mean) ** 2).sum() / counts.sum())


def test_binned_ntuple_edges_are_uniform_or_explicit():
    assert bntrig.bin_edges(3, 80, 110, "").tolist() == [80, 90, 100, 110]
    assert bntrig.bin_edges(3, 80, 110, "0, 1 5").tolist() == [0, 1, 5]
    for args in ((3, 110, 80, ""), (2.5, 0, 1, ""), (3, 0, 1, "1, 1, 2"),
                 (3, 0, 1, "5"), (3, 0, 1, "a, b")):
        try:
            bntrig.bin_edges(*args)
        except ntrig.NtupleError:
            pass
        else:
            raise AssertionError(f"{args} should have been refused")


def test_binned_ntuple_enters_each_event_once_at_its_first_selected_track():
    variables = {"_event": np.array([0, 0, 1, 2, 3]),
                 "p": np.array([50.0, 95.0, 105.0, 101.0, 120.0]),
                 "q": np.array([1.0, 1.0, 1.0, np.nan, 1.0])}
    selected, value, n_tracks = bntrig.event_values(variables, 5, "p > 90", "p * q")
    assert selected.tolist() == [True, True, True, True, False] and n_tracks == 4
    assert value[:2].tolist() == [95.0, 105.0] and np.isnan(value[2])
    triggers = {"a": np.array([True, False, True, True, True]),
                "b": np.array([False, True, False, False, False])}
    edges = np.array([90.0, 100.0, 104.0])
    counts = bntrig.binned_counts(selected, value, triggers, edges)
    assert counts["selected"].tolist() == [1, 0] and counts["n_overflow"] == 2
    assert counts["n_undefined"] == 1 and counts["n_underflow"] == 0
    assert counts["n_selected"] == 4 and counts["n_triggered"] == 4
    both = bntrig.add_counts(counts, counts)
    assert both["selected"].tolist() == [2, 0] and both["per_path"] == {"a": 6, "b": 2}
    rows = bntrig.bin_rows(both, edges)
    assert rows[0]["efficiency"] == 1.0 and rows[0]["paths"]["b"]["passed"] == 0
    metrics = bntrig.binned_metrics(both, 8, 2)
    assert tuple(metrics) == bntrig.METRICS and metrics["n_in_range"] == 2.0


def test_ntuple_trigger_efficiency_on_a_real_file(tmp_dir):
    if not NTUPLE_FILE.exists():
        return
    result = run_analysis(analysis="trigger_efficiency_ntuple",
                          data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                          parameters={"trigger_paths": "apr_TrkDe_80m70p, "
                                                       "trig_cpr_TrkDe_80m70p"})
    assert result.status == "success", result.message
    meta = result.metadata
    assert meta["n_events"] == 8212 and 0 < meta["n_selected"] < meta["n_events"]
    assert meta["n_triggered"] <= meta["n_selected"]
    for row in meta["paths"].values():                       # OR >= each path
        assert row["passed"] <= meta["n_triggered"]
    missing = run_analysis(analysis="trigger_efficiency_ntuple",
                           data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                           parameters={"trigger_paths": "not_a_path"})
    assert missing.status == "error" and "It has: " in missing.message


def test_selection_efficiency_counts_every_ntuple_event_by_default():
    v = ntrig.track_variables(_ntuple_arrays())
    # a third event with no track at all still counts in the denominator
    counts = nsel.efficiency_counts(v, 3, ntrig.DEFAULT_SELECTION)
    assert counts["n_denominator"] == 3 and counts["n_tracks_denominator"] == 3
    assert counts["n_passed"] == 1 and counts["n_tracks_passed"] == 1
    metrics = nsel.efficiency_metrics(counts)
    assert abs(metrics["efficiency"] - 1 / 3) < 1e-12 and metrics["efficiency_err"] > 0
    assert tuple(metrics) == nsel.METRICS


def test_selection_efficiency_needs_one_track_to_pass_both_cuts():
    v = ntrig.track_variables(_ntuple_arrays())
    counts = nsel.efficiency_counts(v, 3, "pdg == 11", denominator="nactive >= 25")
    assert counts["n_denominator"] == 2 and counts["n_passed"] == 1
    # event 0's short track passes the denominator and its good track the
    # selection: not the same track, so the event does not pass
    split = nsel.efficiency_counts(v, 3, "p_front > 100", denominator="nactive < 10")
    assert split["n_denominator"] == 1 and split["n_passed"] == 0


def test_selection_efficiency_refuses_a_bad_cut_and_needs_a_selection(tmp_dir):
    v = ntrig.track_variables(_ntuple_arrays())
    try:
        nsel.efficiency_counts(v, 2, "p_front > 100", denominator="no_such_var > 1")
    except SelectionError as exc:
        assert "no_such_var" in str(exc)
    else:
        raise AssertionError("an unknown variable should have been refused")
    result = run_analysis(analysis="selection_efficiency_ntuple",
                          data_file="/a.root", output_dir=tmp_dir)
    # not just "selection" in the message: the analysis name contains it too
    assert result.status == "error", result.message
    assert "missing required parameter(s)" in result.message, result.message
    assert result.message.endswith(": selection"), result.message


def test_selection_efficiency_checks_each_cut_on_its_own():
    # Combined with the denominator, a bare quantity would coerce to a bool;
    # on its own it is refused, so it is refused here too.
    v = ntrig.track_variables(_ntuple_arrays())
    for selection, denominator in (("p_front", "nactive >= 25"),
                                   ("pdg == 11", "p_front")):
        try:
            nsel.efficiency_counts(v, 3, selection, denominator=denominator)
        except SelectionError:
            pass
        else:
            raise AssertionError(f"{selection!r} / {denominator!r} was accepted")


def test_run_analysis_keeps_its_positional_arguments(tmp_dir):
    # parameters is the seventh positional argument, as before reference
    # inputs; they are keyword-only.
    result = run_analysis("selection_efficiency_ntuple", tmp_dir, "/a.root",
                          None, None, None, {})
    assert result.status == "error", result.message
    assert "missing required parameter(s)" in result.message, result.message
    import inspect
    kinds = {name: p.kind for name, p in inspect.signature(run_analysis).parameters.items()}
    assert all(kinds[name] == inspect.Parameter.KEYWORD_ONLY for name in
               ("reference_file", "reference_files", "reference_dataset"))


def test_selection_efficiency_on_a_real_file(tmp_dir):
    if not NTUPLE_FILE.exists():
        return
    result = run_analysis(analysis="selection_efficiency_ntuple",
                          data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                          parameters={"selection": ntrig.DEFAULT_SELECTION})
    assert result.status == "success", result.message
    meta = result.metadata
    # the same events trigger_efficiency_ntuple selects (README: 7178 of 8212)
    assert meta["n_events"] == 8212 and meta["n_denominator"] == 8212
    assert meta["n_passed"] == 7178
    narrower = run_analysis(analysis="selection_efficiency_ntuple",
                            data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                            parameters={"selection": "100 < p_front < 110",
                                        "denominator": ntrig.DEFAULT_SELECTION})
    assert narrower.status == "success", narrower.message
    assert narrower.metadata["n_denominator"] == 7178
    assert 0 < narrower.metadata["n_passed"] < 7178


def test_resolution_and_energy_loss_come_from_the_shared_mc_variables():
    # the e- track: reconstructed at 104, true 105 at TT_Front, created at 104.9
    v = ntrig.track_variables(_ntuple_mc_arrays())
    r = nres.response_variables(v)
    assert abs(r["resolution"][0] - (104.0 - 105.0)) < 1e-3
    assert abs(r["energy_loss"][0] - (104.9 - 105.0)) < 1e-4
    assert np.isnan(r["resolution"][1:]).all() and np.isnan(r["energy_loss"][1:]).all()
    assert set(r) == set(nres.RESPONSE_VARIABLES)
    variables = {**v, **r}
    for name in nres.RESPONSE_VARIABLES:           # every advertised one is usable
        nres.select_tracks(variables, f"{name} == {name}")
    assert nres.select_tracks(variables, "mc_start_code == 168 and resolution < 0"
                              ).tolist() == [True, False, False]


def test_resolution_sigma68_is_the_gaussian_sigma_and_ignores_the_tail():
    rng = np.random.default_rng(1)
    core = rng.normal(0.0, 0.2, 100_000)
    s = nres.summary(core, "x")
    assert abs(s["x_sigma68"] - 0.2) < 0.004 and abs(s["x_median"]) < 0.004
    assert abs(s["x_rms"] - 0.2) < 0.004
    # a 2% tail far out moves the rms a lot and sigma68 hardly at all
    tailed = np.concatenate([core, np.full(2_000, -20.0), [np.nan]])
    t = nres.summary(tailed, "x")
    assert t["x_rms"] > 2.0 and abs(t["x_sigma68"] - 0.2) < 0.01


def test_resolution_writes_histograms_and_refuses_bad_binning(tmp_dir):
    import uproot
    params = {name: p.default for name, p in
              ((p.name, p) for p in nres.SPEC.parameters)}
    nres._check_binning(params)
    for bad in ({"resolution_max": -5.0}, {"energy_loss_bin_width": 1e-6},
                {"resolution_bin_width": 100.0}):
        try:
            nres._check_binning({**params, **bad})
        except nres.ResolutionError as exc:
            assert next(iter(bad)) in str(exc) or "resolution_max" in str(exc)
        else:
            raise AssertionError(f"{bad} should have been refused")
    res = np.array([-0.1, 0.0, 0.02, 9.0])                    # 9.0 off the axis
    eloss = np.array([0.5, 0.7, 1.5, 2.0])
    metrics = nres.resolution_metrics(
        {"n_events": 3, "n_events_selected": 3, "n_tracks_selected": 4}, res, eloss)
    assert tuple(metrics) == nres.METRICS and metrics["n_tracks_with_truth"] == 4
    hists = {"resolution": nres.histogram(res, -5.0, 3.0, 0.05, "resolution"),
             "energy_loss": nres.histogram(eloss, 0.0, 10.0, 0.1, "energy_loss")}
    assert hists["resolution"].values.size == 160 and hists["resolution"].values.sum() == 3
    # A width that does not divide the range keeps the requested endpoints.
    uneven = nres.histogram(res, -5.0, 3.0, 3.0, "resolution")
    assert uneven.nbins == 3 and uneven.edges()[-1] == 3.0
    files = nres._write_outputs(Path(tmp_dir), hists, metrics, "")
    assert [Path(f).name for f in files] == ["momentum_resolution.root",
                                             "resolution.png", "energy_loss.png"]
    with uproot.open(files[0]) as f:
        assert f["energy_loss"].values().tolist() == hists["energy_loss"].values.tolist()


def test_resolution_says_a_file_without_truth_needs_simulation(tmp_dir):
    import uproot
    path = Path(tmp_dir) / "data.root"
    with uproot.recreate(path) as f:
        f.mkdir("EventNtuple").mktree("ntuple", {"event": "int32"}).extend(
            {"event": np.arange(3, dtype=np.int32)})
    result = run_analysis(analysis="momentum_resolution_ntuple",
                          data_file=str(path), output_dir=tmp_dir)
    assert result.status == "error" and "MC truth" in result.message, result.message


def test_resolution_on_a_real_file(tmp_dir):
    if not NTUPLE_FILE.exists():
        return
    result = run_analysis(analysis="momentum_resolution_ntuple",
                          data_file=str(NTUPLE_FILE), output_dir=tmp_dir)
    assert result.status == "success", result.message
    meta = result.metadata
    # the default selection is trigger_efficiency_ntuple's: the same 7178 events
    assert meta["n_events_selected"] == 7178
    assert meta["n_tracks_with_truth"] <= meta["n_tracks_selected"]
    assert 0.1 < meta["resolution_sigma68"] < 0.5 and abs(meta["resolution_median"]) < 0.1
    assert 0.0 < meta["energy_loss_median"] < meta["energy_loss_mean"]  # a long tail
    assert all(Path(f).exists() for f in result.files) and len(result.files) == 3
    ce = run_analysis(analysis="momentum_resolution_ntuple",
                      data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                      parameters={"selection": ntrig.DEFAULT_SELECTION
                                  + " and mc_start_code == 168"})
    assert ce.status == "success", ce.message
    assert ce.metadata["n_tracks_with_truth"] <= meta["n_tracks_with_truth"]
def test_binned_ntuple_trigger_efficiency_on_a_real_file(tmp_dir):
    if not NTUPLE_FILE.exists():
        return
    paths = "apr_TrkDe_80m70p, cpr_TrkDe_80m70p"
    overall = run_analysis(analysis="trigger_efficiency_ntuple",
                           data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                           parameters={"trigger_paths": paths})
    for variable, low, high in (("p_front", 80, 110), ("mc_npot / 1e6", 0, 50)):
        result = run_analysis(analysis="trigger_efficiency_ntuple_binned",
                              data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                              parameters={"trigger_paths": paths, "variable": variable,
                                          "low": low, "high": high})
        assert result.status == "success", result.message
        meta = result.metadata
        for key in ("n_selected", "n_triggered", "efficiency"):   # same events
            assert meta[key] == overall.metadata[key], key
        assert sum(r["n_selected"] for r in meta["bins"]) == meta["n_in_range"]
        assert (meta["n_in_range"] + meta["n_underflow"] + meta["n_overflow"]
                + meta["n_undefined"]) == meta["n_selected"]
        assert len(result.files) == 3 and all(Path(f).exists() for f in result.files)


def _calo_arrays():
    """Two events of caloclusters/caloclustersmc. Event 0: a CE cluster
    (100 MeV reconstructed, 101 true, 98 of it from the CE, which first
    deposited at 700 ns; reconstructed at 702) and a pileup one. Event 1: a
    cluster with no truth at all."""
    import awkward as ak
    leaves = {
        "caloclusters.energyDep_": [[100.0, 20.0], [15.0]],
        "caloclusters.energyDepErr_": [[2.0, 1.0], [1.0]],
        "caloclusters.time_": [[702.0, 900.0], [500.0]],
        "caloclusters.timeErr_": [[0.3, 0.5], [0.5]],
        "caloclusters.diskID_": [[0, 1], [0]],
        "caloclusters.size_": [[5, 2], [1]],
        "caloclusters.isSplit_": [[0, 0], [0]],
        "caloclusters.cog_.fCoordinates.fX": [[300.0, 0.0], [400.0]],
        "caloclusters.cog_.fCoordinates.fY": [[400.0, 500.0], [0.0]],
        "caloclustersmc.nsim": [[2, 1], [0]],
        "caloclustersmc.etot": [[101.0, 21.0], [0.0]],
        "caloclustersmc.eprimary": [[98.0, 21.0], [0.0]],
        "caloclustersmc.tprimary": [[700.0, 899.0], [0.0]],
        "caloclustersmc.prel._rel": [[0, -1], [-1]],
        "event": [7, 8], "run": [1, 1], "subrun": [0, 0],
    }
    return {name: ak.Array(values) for name, values in leaves.items()}


def test_calo_cluster_variables_compare_with_the_truth_and_mark_what_is_missing():
    v = ncalo.cluster_variables(_calo_arrays())
    assert v["_event"].tolist() == [0, 0, 1] and v["nclusters"].tolist() == [2, 2, 1]
    assert v["de"][0] == -1.0 and v["dt"][0] == 2.0 and v["r"][0] == 500.0
    assert abs(v["de_frac"][0] + 1 / 101) < 1e-12 and abs(v["mc_purity"][0] - 98 / 101) < 1e-12
    assert v["de"][1] == -1.0 and v["mc_rel"][1] == -1        # pileup: truth, no relation
    for name in ("mc_etot", "mc_tprimary", "mc_rel", "de", "dt", "de_frac"):
        assert np.isnan(v[name][2]), name                     # nsim 0: no truth
    assert set(v) - {"_event"} == set(ncalo.CLUSTER_VARIABLES)
    for name in ncalo.CLUSTER_VARIABLES:           # every advertised one is usable
        ncalo.select_clusters(v, f"{name} == {name}")
    assert ncalo.select_clusters(v, ncalo.DEFAULT_SELECTION).tolist() == [True, False, False]


def test_calo_resolution_writes_its_outputs_and_refuses_bad_binning(tmp_dir):
    import uproot
    params = {p.name: p.default for p in ncalo.SPEC.parameters}
    ncalo._check_binning(params)
    for bad in ({"dt_max": -6.0}, {"de_bin_width": 1e-6}, {"de_bin_width": 1000.0}):
        try:
            ncalo._check_binning({**params, **bad})
        except ncalo.CaloResolutionError as exc:
            assert next(iter(bad)).split("_")[0] in str(exc)
        else:
            raise AssertionError(f"{bad} should have been refused")
    de = np.array([-1.0, 0.0, 0.5, -30.0])                    # -30 off the axis
    dt = np.array([1.8, 2.0, 2.2, 1.9])
    metrics = ncalo.resolution_metrics(
        {"n_events": 2, "n_clusters": 5, "n_clusters_selected": 4}, de, de / 100.0, dt)
    assert tuple(metrics) == ncalo.METRICS and metrics["n_clusters_with_truth"] == 4
    assert metrics["dt_median"] == 1.95 and abs(metrics["de_frac_median"] + 0.005) < 1e-12
    hists = {name: ncalo.histogram(values, params[f"{name}_min"], params[f"{name}_max"],
                                   params[f"{name}_bin_width"], name)
             for name, values in (("de", de), ("dt", dt))}
    assert hists["de"].values.sum() == 3 and hists["dt"].values.size == 150
    files = ncalo._write_outputs(Path(tmp_dir), hists, metrics, ncalo.DEFAULT_SELECTION)
    assert [Path(f).name for f in files] == ["calo_cluster_resolution.root",
                                             "cluster_de.png", "cluster_dt.png"]
    with uproot.open(files[0]) as f:
        assert f["dt"].values().tolist() == hists["dt"].values.tolist()


def test_calo_resolution_says_a_file_without_truth_needs_simulation(tmp_dir):
    import uproot
    path = Path(tmp_dir) / "data.root"
    with uproot.recreate(path) as f:
        f.mkdir("EventNtuple").mktree("ntuple", {"event": "int32"}).extend(
            {"event": np.arange(3, dtype=np.int32)})
    result = run_analysis(analysis="calo_cluster_resolution_ntuple",
                          data_file=str(path), output_dir=tmp_dir)
    assert result.status == "error" and "MC truth" in result.message, result.message


def test_calo_resolution_on_a_real_file(tmp_dir):
    if not NTUPLE_FILE.exists():
        return
    result = run_analysis(analysis="calo_cluster_resolution_ntuple",
                          data_file=str(NTUPLE_FILE), output_dir=tmp_dir,
                          parameters={"selection": "mc_rel == 0 and energy > 50"})
    assert result.status == "success", result.message
    meta = result.metadata
    assert meta["n_events"] == 8212 and meta["n_clusters"] == 56159
    assert meta["n_clusters_selected"] == meta["n_clusters_with_truth"] == 6491
    assert 0.5 < meta["de_sigma68"] < 3.0 and abs(meta["de_median"]) < 1.0
    assert 0.2 < meta["dt_sigma68"] < 1.5 and 1.0 < meta["dt_median"] < 3.0
    assert len(result.files) == 3 and all(Path(f).exists() for f in result.files)


def test_calo_clusters_read_without_truth_refuse_only_the_mc_names():
    arrays = {k: v for k, v in _calo_arrays().items() if not k.startswith("caloclustersmc")}
    v = ncalo.cluster_variables(arrays, with_truth=False)
    assert not set(ncalo.TRUTH_VARIABLES) & set(v) and v["energy"].tolist() == [100, 20, 15]
    ncalo.check_truth_names(v, "energy > 10", "time")
    try:
        ncalo.check_truth_names(v, "energy > 10", "dt")
    except ntrig.NtupleError as exc:
        assert "dt" in str(exc) and "caloclustersmc" in str(exc)
    else:
        raise AssertionError("dt on a file without truth should have been refused")


# --- variable_comparison_ntuple ----------------------------------------------------

def _comparison_sample(events, objects):
    """A Sample from event numbers (run 1, subrun 0) and (event index, value)
    objects, every object selected."""
    keys = vcmp.event_keys(np.ones(len(events)), np.zeros(len(events)), events)
    variables = {"_event": np.array([e for e, _ in objects], dtype=np.int64),
                 "x": np.array([x for _, x in objects], dtype=np.float64)}
    return vcmp.sample_from_objects(variables, keys, "", "x")


def test_comparison_takes_each_events_first_selected_object():
    variables = {"_event": np.array([0, 0, 2]), "x": np.array([5.0, 6.0, 7.0]),
                 "q": np.array([0.0, 1.0, 1.0])}
    sample = vcmp.sample_from_objects(variables, vcmp.event_keys([1] * 3, [0] * 3, [1, 2, 3]),
                                      "q > 0", "x * 2")
    assert sample.values.tolist() == [12.0, 14.0]
    assert sample.selected.tolist() == [True, False, True]
    assert sample.first[0] == 12.0 and np.isnan(sample.first[1])


def test_comparison_compares_a_true_false_variable_as_0_1():
    variables = {"_event": np.array([0, 1, 2]), "has_front": np.array([True, False, True])}
    keys = vcmp.event_keys([1] * 3, [0] * 3, [1, 2, 3])
    sample = vcmp.sample_from_objects(variables, keys, "", "has_front")
    assert sample.values.tolist() == [1.0, 0.0, 1.0]
    # A cut is still not a quantity.
    try:
        vcmp.sample_from_objects(variables, keys, "", "has_front and has_front")
    except vcmp.SelectionError:
        pass
    else:
        raise AssertionError("a condition was accepted as a quantity")


def test_comparison_ratio_shows_bins_only_the_reference_has():
    shown, ratio, error = vcmp.ratio_points(np.array([4.0, 0.0, 3.0]),
                                            np.array([4.0, 5.0, 0.0]), 1.0, 1.0)
    assert shown.tolist() == [True, True, False]
    assert ratio[:2].tolist() == [1.0, 0.0]
    assert np.isclose(error[0], np.sqrt(1 / 4 + 1 / 4)) and error[1] == 0.0


def test_comparison_matches_events_and_counts_what_changed():
    # Events 1-5 in both. New: 1 same, 2 shifted, 3 gained, 5 gone; 6 only new.
    new = _comparison_sample([1, 2, 3, 4, 6], [(0, 10.0), (1, 21.0), (2, 30.0),
                                               (3, 40.0), (4, 60.0)])
    ref = _comparison_sample([1, 2, 3, 4, 5], [(0, 10.0), (1, 20.0), (3, 40.0),
                                               (4, 50.0)])
    m = vcmp.match(new, ref, "auto")
    assert m["matched"] and m["n_common_events"] == 4
    assert m["n_only_new"] == 1 and m["n_only_ref"] == 1
    metrics, diff = vcmp.matched_metrics(new, ref, m["i_new"], m["i_ref"])
    assert metrics["n_events_matched"] == 4 and metrics["n_events_both_selected"] == 3
    assert metrics["n_events_gained"] == 1 and metrics["n_events_lost"] == 0
    assert metrics["n_events_changed"] == 1 and sorted(diff.tolist()) == [0.0, 0.0, 1.0]
    dist = vcmp.distribution_metrics(new, ref)
    assert dist["n_selected_new"] == 5 and dist["mean_shift"] == 32.2 - 30.0


def test_comparison_matches_only_samples_that_share_most_events():
    new = _comparison_sample([1, 2, 3, 4], [(i, 1.0) for i in range(4)])
    ref = _comparison_sample([4, 5, 6, 7], [(i, 1.0) for i in range(4)])
    assert not vcmp.match(new, ref, "auto")["matched"]       # 1 of 4 by chance
    assert not vcmp.match(new, new, "no")["matched"]
    assert vcmp.match(new, ref, "yes")["matched"]
    other = _comparison_sample([9], [(0, 1.0)])
    try:
        vcmp.match(new, other, "yes")
    except vcmp.ComparisonError as exc:
        assert "not the same events" in str(exc)
    else:
        raise AssertionError("'yes' with no shared event should have been refused")


def test_comparison_does_not_match_repeated_event_ids():
    # Event 2 twice in the new sample: intersect1d would pair only one of them.
    new = _comparison_sample([1, 2, 2, 3], [(i, 1.0) for i in range(4)])
    ref = _comparison_sample([1, 2, 3], [(i, 1.0) for i in range(3)])
    m = vcmp.match(new, ref, "auto")
    assert not m["matched"] and m["n_duplicate_ids_new"] == 1
    assert m["i_new"].size == 0
    try:
        vcmp.match(new, ref, "yes")
    except vcmp.ComparisonError as exc:
        assert "match_events='no'" in str(exc)
    else:
        raise AssertionError("'yes' with repeated event ids should have been refused")


def test_comparison_ranges_and_bins():
    assert vcmp.parse_range("") is None and vcmp.parse_range("80, 110") == (80.0, 110.0)
    for bad in ("80", "110, 80", "a, b"):
        try:
            vcmp.parse_range(bad)
        except vcmp.ComparisonError:
            pass
        else:
            raise AssertionError(f"range {bad!r} should have been refused")
    assert vcmp.auto_range(np.full(10, 3.0)) == (2.5, 3.5)
    assert vcmp.difference_range(np.zeros(5)) == (-1.0, 1.0)
    assert vcmp.difference_range(np.array([0.0, 0.0, -0.2, 0.1])) [1] < 0.21


def _comparison_file(path, events, values):
    """An EventNtuple/ntuple with run/subrun/event and one value per event."""
    import uproot
    with uproot.recreate(path) as f:
        f.mkdir("EventNtuple").mktree("ntuple", {"run": "int32", "subrun": "int32",
                                                 "event": "int32", "x": "float64"}).extend(
            {"run": np.ones(len(events), np.int32), "subrun": np.zeros(len(events), np.int32),
             "event": np.asarray(events, np.int32), "x": np.asarray(values, np.float64)})


def _read_x(path):
    """A Level reader for _comparison_file: one object per event."""
    import uproot
    x = uproot.open(path)["EventNtuple/ntuple"]["x"].array(library="np")
    return {"_event": np.arange(x.size), "x": x}, x.size


def test_comparison_runs_end_to_end_and_writes_its_outputs(tmp_dir):
    new, ref = Path(tmp_dir) / "new.root", Path(tmp_dir) / "ref.root"
    _comparison_file(new, range(100), np.arange(100) * 0.1 + 0.05)
    _comparison_file(ref, range(100), np.arange(100) * 0.1)
    saved = vcmp.LEVELS["track"]
    vcmp.LEVELS["track"] = vcmp.Level("track", {"x": "x"}, _read_x, lambda *a: None)
    try:
        result = run_analysis(analysis="variable_comparison_ntuple", data_file=str(new),
                              reference_file=str(ref), output_dir=tmp_dir,
                              parameters={"variable": "x", "n_bins": 20, "range": "0, 10"})
        assert result.status == "success", result.message
        meta = result.metadata
        assert meta["n_events_matched"] == 100 and meta["n_events_changed"] == 100
        assert abs(meta["diff_median"] - 0.05) < 1e-9 and meta["reference_files"] == [str(ref)]
        assert "against 1 reference file(s)" in result.message
        names = [Path(f).name for f in result.files]
        assert names == ["variable_comparison.root", "variable_comparison.png",
                         "variable_difference.png"], names
        import uproot
        with uproot.open(result.files[0]) as f:
            assert f["new"].values().sum() == 100 and f["diff"].values().size == 21
        result = run_analysis(analysis="variable_comparison_ntuple", data_file=str(new),
                              reference_file=str(ref), output_dir=tmp_dir,
                              parameters={"variable": "x", "match_events": "no"})
        assert result.status == "success" and result.metadata["n_events_matched"] == 0
        assert np.isnan(result.metadata["diff_median"]) and len(result.files) == 2
        import uproot
        with uproot.open(result.files[0]) as f:
            h = f["new"]
            mean = h.member("fTsumwx") / h.member("fTsumw")
            assert abs(mean - result.metadata["mean_new"]) < 0.1, mean
        # The reference loses every object (new 9.95 only passes): still compared.
        result = run_analysis(analysis="variable_comparison_ntuple", data_file=str(new),
                              reference_file=str(ref), output_dir=tmp_dir,
                              parameters={"variable": "x", "selection": "x > 9.92",
                                          "normalize": "area"})
        assert result.status == "success", result.message
        meta = result.metadata
        assert meta["n_selected_new"] == 1 and meta["n_selected_ref"] == 0
        assert np.isnan(meta["mean_ref"]) and np.isnan(meta["ks_pvalue"])
        assert meta["n_events_gained"] == 1 and meta["n_events_lost"] == 0
        result = run_analysis(analysis="variable_comparison_ntuple", data_file=str(new),
                              reference_file=str(ref), output_dir=tmp_dir,
                              parameters={"variable": "x", "selection": "x > 1e9"})
        assert result.status == "error" and "either sample" in result.message, result.message
    finally:
        vcmp.LEVELS["track"] = saved


def test_reference_inputs_go_only_to_an_analysis_that_compares(tmp_dir):
    result = run_analysis(analysis="variable_comparison_ntuple", data_file="/a.root",
                          output_dir=tmp_dir, parameters={"variable": "p_front"})
    assert result.status == "error" and "reference_file" in result.message
    result = run_analysis(analysis="variable_comparison_ntuple", data_file="/a.root",
                          reference_file="/b.root", reference_files=["/c.root"],
                          output_dir=tmp_dir, parameters={"variable": "p_front"})
    assert result.status == "error" and "exactly one" in result.message
    result = run_analysis(analysis="calo_cluster_resolution_ntuple", data_file="/a.root",
                          reference_file="/b.root", output_dir=tmp_dir)
    assert result.status == "error" and "takes_reference" in result.message
    catalogue = list_analyses().metadata["analyses"]
    assert catalogue["variable_comparison_ntuple"]["takes_reference"] is True
    assert "takes_reference" not in catalogue["calo_cluster_resolution_ntuple"]


def test_comparison_of_a_real_file_with_itself_finds_no_change(tmp_dir):
    if not NTUPLE_FILE.exists():
        return
    for level, variable in (("track", "p_front"), ("cluster", "energy")):
        result = run_analysis(analysis="variable_comparison_ntuple",
                              data_file=str(NTUPLE_FILE), reference_file=str(NTUPLE_FILE),
                              output_dir=tmp_dir,
                              parameters={"level": level, "variable": variable})
        assert result.status == "success", result.message
        meta = result.metadata
        assert meta["n_events_matched"] == 8212 and meta["n_events_changed"] == 0
        assert meta["n_events_gained"] == meta["n_events_lost"] == 0
        assert meta["mean_shift"] == 0 and meta["ks_pvalue"] == 1.0


# --- muon_stops ----------------------------------------------------------------

def _stop_records(stops):
    """StoppedParticlesDumper rows, as the job's leaf-list branch reads:
    (x, y, z, time) per stop, all mu-."""
    dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("time", "f4"),
                      ("tauNormalized", "f4"), ("pdg", "i4"), ("origin", "i4"),
                      ("term", "i4")])
    return np.array([(x, y, z, t, -1.0, 13, 56, 32) for x, y, z, t in stops], dtype=dtype)


def _vd_records(steps):
    """StepPointMCDumper (VDHit) rows: (VD id, pdg, pz, |p|) per step."""
    dtype = np.dtype([(n, "f4") for n in ("x", "y", "z", "time", "px", "py", "pz",
                                          "pmag", "ek", "charge")]
                     + [("pdgId", "i4"), ("particleId", "u4"), ("volumeCopy", "u4")])
    return np.array([(0, 0, 0, 0, 0, 0, pz, p, 0, -1, pdg, 1, vd)
                     for vd, pdg, pz, p in steps], dtype=dtype)


def test_muon_stops_fcl_points_both_dumpers_at_the_stop_filter():
    text = mstops.job_fcl_text("PolyStopFilter")
    assert text.startswith(mstops.FCL.read_text())          # the shipped fcl, then
    assert text.rstrip().endswith(
        'physics.analyzers.vdDumper.hitsInputTag : "PolyStopFilter:virtualdetector"')
    assert 'stoppedMuonDumper.inputCollection : "PolyStopFilter"' in text
    for bad in ("", "Target Stop", 'X"; physics.an: [] #'):
        try:
            mstops.job_fcl_text(bad)
        except mstops.MuonStopsError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def test_muon_stops_reads_the_generated_count_from_stdout():
    assert mstops.parse_gen_count(SAMPLE_MUON_STOPS_STDOUT) == 12500
    assert mstops.parse_gen_count("TrigReport Events total = 3") is None


def test_muon_stops_profile_measures_radius_from_the_ds_axis():
    # two stops 50 mm off the DS axis (x = -3904), one on it
    stops = mstops.stop_variables(_stop_records([
        (-3904.0 + 30.0, 40.0, 5000.0, 200.0), (-3904.0 - 50.0, 0.0, 5100.0, 300.0),
        (-3904.0, 0.0, 5300.0, 700.0)]))
    assert np.allclose(stops["r"], [50.0, 50.0, 0.0])
    vd = mstops.vd_columns(_vd_records([
        (9, 13, 20.0, 30.0),       # mu- entering ST_In: counted
        (9, 13, -5.0, 10.0),       # the same muon scattered back out: not
        (9, 11, 30.0, 40.0),       # an electron: not
        (8, 13, 50.0, 60.0),       # Coll5_Out: not
        (9, 13, 25.0, 34.0)]))
    p = mstops.st_in_momenta(vd)
    assert p.tolist() == [30.0, 34.0]
    m = mstops.stop_metrics(stops, p, 1000)
    assert tuple(m) == mstops.METRICS and m["n_stops"] == 3 and m["n_gen_events"] == 1000
    assert abs(m["stop_z_mean"] - 15400 / 3) < 1e-3
    assert m["stop_z_median"] == 5100.0 and abs(m["stop_r_mean"] - 100 / 3) < 1e-4
    assert m["stop_time_median"] == 300.0 and m["n_st_in"] == 2
    assert m["p_st_in_mean"] == 32.0 and m["p_st_in_rms"] == 2.0
    h = mstops.histogram(stops["z"], 5.0, "z")             # edges on multiples of 5
    assert h.xmin == 5000.0 and h.edges()[-1] == 5305.0 and h.values.sum() == 3


def _fake_stops_job(stdout=SAMPLE_MUON_STOPS_STDOUT, returncode=0):
    from tools.mu2e_job import JobOutcome
    calls = []

    def fake(*, fcl, input_paths, outdir, single, timeout_s, max_events=None, **_):
        calls.append(Path(fcl).read_text())
        root = outdir / "nts.owner.muonstops.test.root"
        root.write_bytes(b"")                    # read_dumps is faked too
        return JobOutcome(command="mu2e", returncode=returncode, timed_out=False,
                          stdout=stdout, stderr="", log_path=outdir / "mu2e.log",
                          input_paths=input_paths, input_flag="-s",
                          environment="test", file_list_path=None,
                          written_root_files=[str(root)])
    return fake, calls


def test_muon_stops_run_writes_its_profile_and_says_what_went_wrong(tmp_dir):
    from tools.spec import RunContext
    real_job, real_read = mstops.run_mu2e_job, mstops.read_dumps
    stops = _stop_records([(-3874.0, 40.0, 5000.0 + 10 * i, 200.0 + i) for i in range(20)])
    steps = _vd_records([(9, 13, 20.0, 25.0 + i) for i in range(20)])

    def context(name, **params):
        return RunContext(input_paths=[Path(tmp_dir) / "in.art"],
                          outdir=Path(tmp_dir) / name, timeout_s=60,
                          params={"stop_filter": "TargetStopFilter", **params})
    try:
        mstops.run_mu2e_job, calls = _fake_stops_job()
        mstops.read_dumps = lambda path: (stops, steps)
        outcome = mstops.run(context("ok"))
        assert outcome.error is None, outcome.error
        assert outcome.metrics["n_stops"] == 20 and outcome.metrics["n_gen_events"] == 12500
        assert outcome.metrics["n_st_in"] == 20 and outcome.extra["stops_by_pdg"] == {13: 20}
        names = [Path(f).name for f in outcome.files]
        assert names[1:] == ["muon_stops_hists.root", "stop_z.png", "stop_r.png",
                             "stop_time.png", "p_st_in.png", "stop_r_vs_z.png"], names
        assert 'inputCollection : "TargetStopFilter"' in calls[0]  # the per-run fcl
        import uproot
        with uproot.open(outcome.files[1]) as f:
            assert f["stop_r_vs_z"].values().sum() == 20 and f["p_st_in"].values().sum() == 20

        mstops.read_dumps = lambda path: (stops[:0], steps[:0])
        empty = mstops.run(context("empty"))
        assert empty.error and "no stopped particle" in empty.error, empty.error

        mstops.run_mu2e_job, _ = _fake_stops_job(stdout="no bookkeeping here")
        assert "GenEventCount" in mstops.run(context("nogen")).error

        mstops.run_mu2e_job, _ = _fake_stops_job(returncode=1)
        failed = mstops.run(context("failed"))
        assert "stop_filter" in failed.error and "PolyStopFilter" in failed.error
    finally:
        mstops.run_mu2e_job, mstops.read_dumps = real_job, real_read


# --- stop_materials ----------------------------------------------------------

# A MuBeam-stage ntuple with TargetMuonFinder/, PolyMuonFinder/ and
# IPAMuonFinder/stopmat; the test using it is skipped where it is not on disk.
STOPMAT_FILE = Path("/exp/mu2e/app/users/mmackenz/mu2eopt/"
                    "nts.mmackenz.mubeam.Run1Bak_local0818120248.001800_00000000.root")


def test_material_rates_divide_by_n_gen_and_sort_most_first():
    rows = material_rates([("Al", 30.0, 30.0 ** 0.5), ("Steel", 70.0, 70.0 ** 0.5),
                           ("Ti", 0.0, 0.0)], n_gen_events=1000.0)
    assert [r["material"] for r in rows] == ["Steel", "Al", "Ti"]
    assert abs(rows[0]["stops_per_gen_event"] - 0.07) < 1e-12
    assert abs(rows[1]["stops_per_gen_event_err"] - 30.0 ** 0.5 / 1000.0) < 1e-12
    assert abs(rows[0]["fraction"] - 0.7) < 1e-12 and rows[2]["fraction"] == 0.0


def test_combine_tables_matches_materials_by_name_not_bin():
    """Each file labels its axis in its own order; a missing material is 0."""
    combined = dict((name, (stops, err)) for name, stops, err in combine_tables([
        [("Al", 3.0, 3.0), ("Steel", 4.0, 4.0)],
        [("Steel", 12.0, 3.0), ("Ti", 1.0, 1.0)],
    ]))
    assert combined["Steel"] == (16.0, 5.0)       # errors add in quadrature
    assert combined["Al"] == (3.0, 3.0) and combined["Ti"] == (1.0, 1.0)


def test_stop_materials_names_the_modules_it_could_have_read(tmp_dir):
    """A module without a stopmat is an error listing the ones that have one."""
    import uproot
    path = Path(tmp_dir) / "stops.root"
    with uproot.recreate(path) as f:
        f["PolyMuonFinder/stopmat"] = np.histogram(np.zeros(1), bins=2)
    result = run_analysis(analysis="stop_materials", data_file=str(path),
                          output_dir=tmp_dir, parameters={"n_gen_events": 10})
    assert result.status == "error"
    assert "TargetMuonFinder/stopmat" in result.message
    assert "PolyMuonFinder" in result.message


def test_stop_materials_needs_n_gen_events(tmp_dir):
    result = run_analysis(analysis="stop_materials", data_file="/a.root",
                          output_dir=tmp_dir)
    assert result.status == "error" and "n_gen_events" in result.message


def test_stop_materials_reads_the_labelled_bins_of_a_real_file(tmp_dir):
    if not STOPMAT_FILE.exists():
        print(f"     (skipped: {STOPMAT_FILE} not on disk)")
        return
    result = run_analysis(analysis="stop_materials", data_file=str(STOPMAT_FILE),
                          output_dir=tmp_dir, parameters={"n_gen_events": 1e6})
    assert result.status == "success", result.message
    meta = result.metadata
    assert meta["stop_module"] == "TargetMuonFinder"
    materials = {row["material"]: row for row in meta["materials"]}
    assert "StoppingTarget_Al" in materials
    # every stop is in a named bin, and the total is what the rates add up to
    assert meta["unnamed_stops"] == 0.0
    assert meta["n_stops"] == meta["hist_entries"]
    total = sum(row["stops_per_gen_event"] for row in meta["materials"])
    assert abs(total - meta["stops_per_gen_event"]) < 1e-12
    assert abs(meta["stops_per_gen_event"] - meta["n_stops"] / 1e6) < 1e-15

    poly = run_analysis(analysis="stop_materials", data_file=str(STOPMAT_FILE),
                        output_dir=tmp_dir,
                        parameters={"n_gen_events": 1e6,
                                    "stop_module": "PolyMuonFinder"})
    assert poly.status == "success", poly.message
    assert poly.metadata["hist_path"] == "PolyMuonFinder/stopmat"

    # the same file twice: twice the stops over twice the generated events
    both = run_analysis(analysis="stop_materials",
                        data_files=[str(STOPMAT_FILE)] * 2, output_dir=tmp_dir,
                        parameters={"n_gen_events": 2e6})
    assert both.status == "success", both.message
    assert both.metadata["n_stops"] == 2 * meta["n_stops"]
    assert abs(both.metadata["stops_per_gen_event"]
               - meta["stops_per_gen_event"]) < 1e-15
    assert [f["n_stops"] for f in both.metadata["per_file"]] == [meta["n_stops"]] * 2
    doubled = {row["material"]: row for row in both.metadata["materials"]}
    al = materials["StoppingTarget_Al"]
    assert doubled["StoppingTarget_Al"]["stops"] == 2 * al["stops"]
    assert abs(doubled["StoppingTarget_Al"]["stops_err"]
               - 2 ** 0.5 * al["stops_err"]) < 1e-9


def test_stop_materials_names_the_file_in_a_list_that_lacks_the_histogram(tmp_dir):
    import uproot
    bad = Path(tmp_dir) / "bad.root"
    with uproot.recreate(bad) as f:
        f["other"] = np.histogram(np.zeros(1), bins=2)
    if not STOPMAT_FILE.exists():
        print(f"     (skipped: {STOPMAT_FILE} not on disk)")
        return
    result = run_analysis(analysis="stop_materials",
                          data_files=[str(STOPMAT_FILE), str(bad)],
                          output_dir=tmp_dir, parameters={"n_gen_events": 10})
    assert result.status == "error"
    assert str(bad) in result.message and str(STOPMAT_FILE) not in result.message


# --- fullsim_sensitivity -----------------------------------------------------

def _seg(sid, time=900.0, p=(0.0, 85.0, 60.0), t0err=0.5, maxr=500.0, d0=50.0,
         tandip=0.7):
    """One track segment and its loop-helix parameters. The default momentum
    is CE-like: |p| = 104 MeV/c, p_z/p_T = 0.71."""
    return ({"sid": sid, "time": time,
             "mom": {"fCoordinates": {"fX": p[0], "fY": p[1], "fZ": p[2]}}},
            {"t0err": t0err, "maxr": maxr, "d0": d0, "tanDip": tandip})


def _track(pdg=11, nactive=30, qual=0.9, pid=0.9, status=1, goodfit=1, segs=None,
           sims=((168, 0, 10.0),)):
    """A track passing every default cut unless told otherwise: segments at
    the tracker front and middle, an ST boundary and the ST foils. `sims` are
    (startCode, gen, rho) for the track's particle, then its ancestors."""
    if segs is None:
        segs = [_seg(0), _seg(1), _seg(100), _seg(104)]
    return {"pdg": pdg, "nactive": nactive, "qual": qual, "pid": pid,
            "status": status, "goodfit": goodfit, "segs": segs,
            "sims": [{"startCode": code, "gen": gen,
                      "pos": {"fCoordinates": {"fX": rho, "fY": 0.0}},
                      "mom": {"fCoordinates": {"fX": 0.0, "fY": 84.0, "fZ": 63.0}}}
                     for code, gen, rho in sims]}


def _eventntuple(events):
    """EventNtuple groups, as read_eventntuple returns them, from events of
    {"tracks": [...], "crv": [(time, PEs, nHits, start, end)], "trig": 0/1,
    "calo": [cluster energies]}. "trig" sets every default trigger path; a
    dict {path: 0/1} sets each. "calo" defaults to one 50 MeV cluster.
    Every event carries a far-off coincidence so the CRV arrays have a type."""
    import awkward as ak
    far = (-5000.0, 1.0, 1, -5000.0, -4990.0)
    crv = [ev.get("crv", []) + [far] for ev in events]
    tracks = [ev["tracks"] for ev in events]
    return {
        "evt": ak.Array([{"run": 1, "subrun": 0, "event": i,
                          **{TRIGGER_PREFIX + name: trig.get(name, 0)
                             if isinstance(trig := ev.get("trig", 1), dict) else trig
                             for name in DEFAULT_TRIGGERS}}
                         for i, ev in enumerate(events)]),
        "crv": ak.Array([{f"crvcoincs.{field}": [c[k] for c in coincs]
                          for k, field in enumerate(("time", "PEs", "nHits",
                                                     "timeStart", "timeEnd"))}
                         for coincs in crv]),
        "calo": ak.Array([{"caloclusters.energyDep_": [float(e) for e in
                                                       ev.get("calo", [50.0])]}
                          for ev in events]),
        "trk": ak.Array([{"trk.pdg": [t["pdg"] for t in trks],
                          "trk.nactive": [t["nactive"] for t in trks],
                          "trk.status": [t["status"] for t in trks],
                          "trk.goodfit": [t["goodfit"] for t in trks],
                          "trkqual.result": [t["qual"] for t in trks],
                          "trkpid.result": [t["pid"] for t in trks]}
                         for trks in tracks]),
        "trkfit": ak.Array([{"trksegs": [[s for s, _ in t["segs"]] for t in trks],
                             "trksegpars_lh": [[lh for _, lh in t["segs"]] for t in trks]}
                            for trks in tracks]),
        "trkmc": ak.Array([{"trkmcsim": [t["sims"] for t in trks]} for trks in tracks]),
    }


def _failed_cuts(event, active=fs_cuts.DEFAULT_CUTS, sign="minus", track=0):
    masks = fs_cuts.cut_masks(_eventntuple([event]), sign)
    return sorted(name for name in active if not masks[name][0][track])


def test_fullsim_a_ce_like_track_passes_every_default_cut():
    assert _failed_cuts({"tracks": [_track()]}) == []


def test_fullsim_each_default_cut_rejects_what_it_should():
    front, mid, boundary, foils = _seg(0), _seg(1), _seg(100), _seg(104)
    cases = [
        (["is_good_track"], {"tracks": [_track(status=-1)]}),
        (["is_good_track"], {"tracks": [_track(goodfit=0)]}),
        # without a front segment there is no direction or tan(dip) either
        (["has_downstream", "has_trk_front_seg", "pz_over_pt"],
         {"tracks": [_track(segs=[mid, boundary, foils])]}),
        (["charge_selection", "is_reco_electron_or_positron"], {"tracks": [_track(pdg=13)]}),
        (["has_downstream"],
         {"tracks": [_track(segs=[_seg(0, p=(0, 85, -60)), mid, boundary, foils])]}),
        (["charge_selection"], {"tracks": [_track(pdg=-11)]}),
        (["or_trigger"], {"tracks": [_track()], "trig": 0}),
        (["good_trkpid"], {"tracks": [_track(pid=0.5)]}),
        (["good_trkpid"], {"tracks": [_track()], "calo": []}),
        (["pz_over_pt"],
         {"tracks": [_track(segs=[_seg(0, tandip=0.9), mid, boundary, foils])]}),
        (["st_boundary"], {"tracks": [_track(segs=[front, mid, foils])]}),
        (["has_st"], {"tracks": [_track(segs=[front, mid, boundary])]}),
        (["no_opa"], {"tracks": [_track(segs=[front, mid, boundary, foils, _seg(95)])]}),
        (["good_trkqual"], {"tracks": [_track(qual=0.1)]}),
        (["has_hits"], {"tracks": [_track(nactive=19)]}),
        # t0err counts at the tracker middle, not the front
        (["within_t0err"],
         {"tracks": [_track(segs=[front, _seg(1, t0err=1.0), boundary, foils])]}),
        ([], {"tracks": [_track(segs=[_seg(0, t0err=1.0), mid, boundary, foils])]}),
        # the CRV veto is asymmetric: a coincidence 50 ns before the track
        # vetoes it, one 50 ns after does not
        (["no_crv_veto"], {"tracks": [_track()], "crv": [(850.0, 5.0, 3, 800.0, 900.0)]}),
        ([], {"tracks": [_track()], "crv": [(950.0, 5.0, 3, 900.0, 1000.0)]}),
    ]
    for expected, event in cases:
        assert _failed_cuts(event) == sorted(expected), (expected, _failed_cuts(event))


def test_fullsim_vetoes_pair_tracks_in_time():
    def at(time, pz=60.0, pdg=11):
        return _track(pdg=pdg, segs=[_seg(0, time=time, p=(0, 85, pz)), _seg(1),
                                     _seg(100), _seg(104)])
    # an upstream track 40-110 ns before a downstream one is its reflection
    assert _failed_cuts({"tracks": [at(900.0), at(830.0, pz=-60)]}) == ["upstream_veto"]
    assert _failed_cuts({"tracks": [at(900.0), at(870.0, pz=-60)]}) == []
    # two downstream e+- within 150 ns veto each other, for sign minus only
    pair = {"tracks": [at(900.0), at(1000.0, pdg=-11)]}
    assert _failed_cuts(pair) == ["no_multi_trk_veto"]
    assert _failed_cuts(pair, track=1) == ["charge_selection", "no_multi_trk_veto"]
    assert _failed_cuts({"tracks": [at(900.0), at(1100.0)]}) == []
    assert _failed_cuts(pair, sign="plus") == ["charge_selection"]


def test_fullsim_trigger_passes_on_any_of_the_paths():
    apr, cpr = DEFAULT_TRIGGERS
    assert _failed_cuts({"tracks": [_track()], "trig": {apr: 1}}) == []
    assert _failed_cuts({"tracks": [_track()], "trig": {cpr: 1}}) == []
    assert _failed_cuts({"tracks": [_track()], "trig": {}}) == ["or_trigger"]
    # only the paths asked for count
    data = _eventntuple([{"tracks": [_track()], "trig": {apr: 1}}])
    masks = fs_cuts.cut_masks(data, "minus", triggers=(cpr,))
    assert not masks["or_trigger"][0][0]
    from tools.analyses.fullsim.eventntuple import EventNtupleError, parse_trigger_paths
    assert parse_trigger_paths(f"trig_{apr}, {cpr} {apr}") == [apr, cpr]
    for bad in ("", " , ", "bad-name"):
        try:
            parse_trigger_paths(bad)
        except EventNtupleError:
            pass
        else:
            raise AssertionError(f"{bad!r} should have been refused")


def test_fullsim_a_segment_cut_applies_only_at_its_surface():
    # a late time at the ST does not matter; at the front it does
    st_late = {"tracks": [_track(segs=[_seg(0), _seg(1), _seg(100), _seg(104, time=50.0)])]}
    front_early = {"tracks": [_track(segs=[_seg(0, time=50.0), _seg(1), _seg(100),
                                           _seg(104)])]}
    active = fs_cuts.DEFAULT_CUTS + ("within_t0",)
    assert _failed_cuts(st_late, active) == []
    assert _failed_cuts(front_early, active) == ["within_t0"]


def test_fullsim_active_cuts_toggle_by_name_and_refuse_unknown_ones():
    active = fs_cuts.active_cuts("within_t0", "has_st, no_opa")
    assert "within_t0" in active and "has_st" not in active and "no_opa" not in active
    assert "in_mom_range" not in active and active[0] == "has_a_track"
    # the cut flow follows pyfitter's order whatever order they are named in
    assert active == [n for n in fs_cuts.CUT_DESCRIPTIONS if n in active]
    try:
        fs_cuts.active_cuts("not_a_cut", "")
    except fs_cuts.CutError as exc:
        assert "Known:" in str(exc), exc
    else:
        raise AssertionError("not_a_cut was accepted")


def test_fullsim_origin_is_the_first_classified_particle_in_the_chain():
    events = [
        {"tracks": [_track(sims=[(168, 0, 10.0)])]},              # CE
        {"tracks": [_track(sims=[(167, 0, 10.0)])]},              # CE endpoint
        {"tracks": [_track(sims=[(166, 0, 30.0)])]},              # DIO on the ST
        {"tracks": [_track(sims=[(166, 0, 200.0)])]},             # DIO on the IPA
        {"tracks": [_track(sims=[(170, 0, 30.0)])]},              # leading-log DIO
        {"tracks": [_track(sims=[(12, 0, 0.0), (171, 0, 0.0)])]},  # conversion of an iRMC photon
        {"tracks": [_track(sims=[(12, 44, 0.0)])]},                # cosmic
        {"tracks": [_track(sims=[(12, 0, 0.0), (13, 0, 0.0)])]},   # nothing known
    ]
    origin = origin_codes(_eventntuple(events)["trkmc"]["trkmcsim"])
    assert origin.tolist() == [168, 168, 166, 0, 166, 171, -1, -2]


def test_fullsim_reduce_counts_the_first_selected_track_of_each_event():
    data = _eventntuple([
        {"tracks": [_track()]},
        # the first track fails, so the second (a DIO, 200 ns later so the
        # multi-track veto leaves it) represents the event
        {"tracks": [_track(qual=0.0),
                    _track(segs=[_seg(0, time=1100.0, p=(0, 80, 60)), _seg(1), _seg(100),
                                 _seg(104)], sims=[(166, 0, 20.0)])]},
        {"tracks": [_track(nactive=5)]},                            # cut away
        # selected, but with two front segments, so not counted (a reflected
        # track would be, but pz_over_pt removes those)
        {"tracks": [_track(segs=[_seg(0), _seg(0, time=950.0), _seg(1), _seg(100),
                                 _seg(104)])]},
        # likewise when the second front segment's time is NaN: it still counts
        {"tracks": [_track(segs=[_seg(0), _seg(0, time=np.nan), _seg(1), _seg(100),
                                 _seg(104)])]},
    ])
    data["n_processed"] = 7          # the job processed more than it kept
    active = fs_cuts.active_cuts("", "")
    reduced = fs_sens.reduce_data(data, active, fs_cuts.TRKQUAL_MIN, fs_cuts.TRKPID_MIN)
    assert reduced["n_events"] == 5 and reduced["n_selected"] == 4
    assert reduced["n_processed"] == 7
    assert reduced["flow"][-1] == 4 and len(reduced["flow"]) == len(active)
    assert reduced["origin"].tolist() == [168, 166]
    assert np.allclose(reduced["p"], [np.hypot(85, 60), 100.0])
    assert np.allclose(reduced["p_true"], [105.0, 105.0])
    assert reduced["event"].tolist() == [0, 1]


def test_fullsim_rates_follow_production_normalization():
    """Signal = NPOT x stops/POT x captures/stop x R_mue x efficiency; the DIO
    is the theory spectrum for NPOT x stops/POT x DIO/stop, folded with the
    measured response, which carries the same efficiency; cosmics are flat
    per MeV/c over the on-spill time."""
    from tools.analyses.fullsim import normalization as norm
    from tools.analyses import approx_ce_sensitivity as fast
    if not fast.DIO_TABLE.exists():
        print("     (skipped: DIO table not on disk)")
        return
    rng = np.random.default_rng(1)
    p_true = np.full(400, 104.97)
    p_reco = p_true - rng.exponential(0.3, p_true.size) + rng.normal(0, 0.15, p_true.size)
    spectra = fs_sens.build_spectra(p_reco, p_true, n_generated=1000,
                                    upstream_eff=0.5, npot=1e18,
                                    mean_pot_per_event=1.6e7,
                                    cosmic_rate_per_s_per_mev=1e-5,
                                    stopped_muons_per_pot=1e-3, rmue=1e-13)
    eff = 400 / 1000 * 0.5
    stops = 1e18 * 1e-3
    assert np.isclose(spectra["efficiency"], eff)
    assert np.isclose(spectra["stopped_muons"], stops)
    assert np.isclose(spectra["signal"].values.sum(), stops * 0.609 * 1e-13 * eff)
    assert np.isclose(spectra["response"].values.sum() * spectra["response"].width, eff)
    # the response only moves DIO a little, so the folded total is the
    # theory total times the efficiency
    dio_total = norm.DIO_PER_STOPPED_MUON * stops * eff
    assert np.isclose(spectra["dio"].values.sum(), dio_total, rtol=1e-3)
    width = spectra["signal"].width
    onspill = 1e18 / 1.6e7 * fast.ONSPILL_SECONDS_PER_EVENT
    assert np.allclose(spectra["cosmic"].values, 1e-5 * onspill * width)
    # the scan is approx_ce_sensitivity's, and finds the peak
    best, _ = fast.scan_signal_box(spectra["signal"], spectra["dio"], spectra["cosmic"])
    assert 103.5 < best["low_mev"] < best["high_mev"] < 105.5


# SimEfficiencies2 for Sim_best v1_1, run 1430, as dbTool prints it.
SIM_EFFICIENCIES_TABLE = """\
MuBeamCat,213816,100000000,0.00213816
EleBeamCat,5532579,100000000,0.05532579
MuminusStopsCat,1435092,4000000000,0.000358773
MuplusStopsCat,7578,4000000000,0.0000018945
IPAStopsCat,22519,3584800000,0.00000628180093729078
PiTotalLifeimeWeight_filter,0,0,38536.77997060446
"""


def test_fullsim_stopped_muons_per_pot_follow_the_sim_chain():
    from tools.analyses.fullsim import normalization as norm
    table = norm.parse_sim_efficiencies(SIM_EFFICIENCIES_TABLE)
    # Production's normalizations.py: MuBeamCat x MuminusStopsCat x 1000
    assert np.isclose(norm.stopped_muons_per_pot(table),
                      0.00213816 * 0.000358773 * 1000)
    assert np.isclose(norm.STOPPED_MUONS_PER_POT, norm.stopped_muons_per_pot(table))
    assert np.isclose(norm.CAPTURES_PER_STOPPED_MUON + norm.DIO_PER_STOPPED_MUON, 1.0)
    del table["MuminusStopsCat"]
    try:
        norm.stopped_muons_per_pot(table)
    except norm.NormalizationError as exc:
        assert "MuminusStopsCat" in str(exc), exc
    else:
        raise AssertionError("a table without MuminusStopsCat was accepted")


# The MDC2025 stop chain's files as SAM has them (2026-10-07): MuBeamCat's
# four, one with event_count left out as SAM does for a zero, and
# MuminusStopsCat's one.
MUBEAM_FILES = [("sim.mu2e.MuBeamCat.MDC2025ab.001430_00020851.art", 25000000, 53513),
                ("sim.mu2e.MuBeamCat.MDC2025ab.001430_00032620.art", 25000000, 53834),
                ("sim.mu2e.MuBeamCat.MDC2025ab.001430_00020001.art", 25000000, 53262),
                ("sim.mu2e.MuBeamCat.MDC2025ab.001430_00020000.art", 25000000, 53207)]
MUSTOPS_FILES = [("sim.mu2e.MuminusStopsCat.MDC2025ac.001430_00000000.art",
                  4000000000, 1435092)]


def _fake_sam(datasets):
    """list_files and fetch over {dataset: [(file, gencount, event_count)]};
    an event_count of None is left out of the metadata."""
    records = {}
    for files in datasets.values():
        for name, gencount, count in files:
            records[name] = {"file_name": name, "dh.gencount": gencount}
            if count is not None:
                records[name]["event_count"] = count
    return ((lambda ds: [f[0] for f in datasets.get(ds, [])]),
            (lambda names: [records[n] for n in names]))


def test_fullsim_stage_efficiencies_come_from_the_datasets_without_the_db():
    """As CreateSimEfficiency.sh (mu2eGenFilterEff): events in the dataset
    over events generated, summed over its files, so the MDC2025 chain gives
    the SimEfficiencies2 rows without reading them."""
    from tools.analyses.fullsim import normalization as norm
    beam, stops = norm.STOP_CHAIN_DATASETS
    list_files, fetch = _fake_sam({beam: MUBEAM_FILES, stops: MUSTOPS_FILES})
    stages = norm.sim_efficiencies(norm.STOP_CHAIN_DATASETS, list_files, fetch)
    assert stages == norm.STOP_CHAIN_MDC2025
    table = norm.parse_sim_efficiencies(SIM_EFFICIENCIES_TABLE)
    for stage in norm.STOP_CHAIN:
        assert np.isclose(stages[stage].efficiency, table[stage]), stage
    assert np.isclose(norm.stopped_muons_per_pot(stages), norm.STOPPED_MUONS_PER_POT)
    # a missing event_count is zero events, not an error
    zero = [MUBEAM_FILES[0][:2] + (None,)] + MUBEAM_FILES[1:]
    list_files, fetch = _fake_sam({beam: zero})
    assert norm.dataset_efficiency(beam, list_files, fetch).passed == 213816 - 53513


def test_fullsim_stage_efficiency_errors_say_what_is_wrong():
    from tools.analyses.fullsim import normalization as norm
    beam, stops = norm.STOP_CHAIN_DATASETS
    no_gencount = [(MUBEAM_FILES[0][0], 0, 5)]
    def unreachable(_):
        raise OSError("network down")
    cases = [
        (["MuBeamCat"], _fake_sam({}), "not a dataset name"),
        (["sim.mu2e.MuBeamCat.MDC2025ab.art with availability x"], _fake_sam({}),
         "not a dataset name"),
        ([beam], _fake_sam({}), "no files"),
        ([beam], _fake_sam({beam: no_gencount}), "no dh.gencount"),
        ([beam], (unreachable, None), "network down"),
        ([beam, "sim.mu2e.MuBeamCat.MDC2026a.art"], _fake_sam({beam: MUBEAM_FILES}),
         "two datasets"),
    ]
    for datasets, (list_files, fetch), needle in cases:
        try:
            norm.sim_efficiencies(datasets, list_files, fetch)
        except norm.NormalizationError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"{needle}: no error")
    # one stage only cannot give stops per POT
    stages = norm.sim_efficiencies([beam], *_fake_sam({beam: MUBEAM_FILES}))
    try:
        norm.stopped_muons_per_pot(stages)
    except norm.NormalizationError as exc:
        assert "MuminusStopsCat" in str(exc), exc
    else:
        raise AssertionError("a chain without MuminusStopsCat was accepted")


def _ce_ancestry(config="MDC2025", stops="MDC2025ac", beam="MDC2025ab", seq="001430_00000000"):
    """A SAM catalog shaped like the CE mix file's real ancestry: nts -> mcs
    -> dig (many dts parents) -> dts -> MuminusStopsCat -> TargetStopsCat ->
    TargetStops -> MuBeamCat. Returns (nts file name, catalog)."""
    nts = f"nts.mu2e.CeMLeadingLogMix1BB.{config}au_best_v1_1-001.{seq}.root"
    mcs = f"mcs.mu2e.CeMLeadingLogMix1BB.{config}au_best_v1_1.{seq}.art"
    dig = f"dig.mu2e.CeMLeadingLogMix1BB.{config}au_best_v1_3.{seq}.art"
    dts = [f"dts.mu2e.CeMLeadingLog.{config}ap.001430_{i:08d}.art" for i in range(3)]
    mustops = f"sim.mu2e.MuminusStopsCat.{stops}.001430_00000000.art"
    tscat = f"sim.mu2e.TargetStopsCat.{stops}.001430_00000000.art"
    ts = [f"sim.mu2e.TargetStops.{stops}.001430_{i:08d}.art" for i in range(3)]
    mubeam = f"sim.mu2e.MuBeamCat.{beam}.001430_00020001.art"
    beam_parents = [f"sim.mu2e.Beam.{beam}.001430_{i:08d}.art" for i in range(3)]
    def parents(*names):
        return {"parents": [{"file_name": n} for n in names]}
    catalog = {nts: parents(mcs), mcs: parents(dig), dig: parents(*dts),
               **{d: parents(mustops) for d in dts},
               mustops: parents(tscat), tscat: parents(*ts),
               **{t: parents(mubeam) for t in ts},
               mubeam: parents(*beam_parents)}
    return nts, catalog


def test_fullsim_stop_chain_is_traced_from_the_inputs_ancestry():
    from tools.analyses.fullsim import normalization as norm
    nts, catalog = _ce_ancestry()
    fetched = []
    def fetch(name):
        fetched.append(name)
        return catalog[name]
    # the CE mix's chain is the one CreateSimEfficiency.sh names for MDC2025
    assert norm.stop_chain_of(nts, fetch) == norm.STOP_CHAIN_DATASETS
    # one file per parent dataset is followed, not every parent
    assert len(fetched) == 7, fetched
    # files of one dataset are traced once
    other = nts.replace("00000000.root", "00000001.root")
    catalog[other] = catalog[nts]
    fetched.clear()
    assert norm.stop_chain_of_inputs([nts, other], fetch) == norm.STOP_CHAIN_DATASETS
    assert len(fetched) == 7, fetched
    # a later iteration is traced to its own chain
    nts26, catalog26 = _ce_ancestry("MDC2026", "MDC2026c", "MDC2026b")
    assert norm.stop_chain_of(nts26, catalog26.__getitem__) == (
        "sim.mu2e.MuBeamCat.MDC2026b.art", "sim.mu2e.MuminusStopsCat.MDC2026c.art")


def test_fullsim_stop_chain_tracing_errors_say_what_is_wrong():
    from tools.analyses.fullsim import normalization as norm
    nts, catalog = _ce_ancestry()
    nts26, catalog26 = _ce_ancestry("MDC2026", "MDC2026c", "MDC2026b")
    both = {**catalog, **catalog26}
    # a dts parent from a second stops dataset makes the stops ambiguous
    ambiguous = dict(catalog)
    dts0 = catalog[catalog[catalog[nts]["parents"][0]["file_name"]]["parents"][0]["file_name"]]
    first_dts = dts0["parents"][0]["file_name"]
    ambiguous[first_dts] = {"parents": [
        {"file_name": "sim.mu2e.MuminusStopsCat.MDC2025ac.001430_00000000.art"},
        {"file_name": "sim.mu2e.MuminusStopsCat.MDC2025zz.001430_00000000.art"}]}
    orphan = {nts: {"parents": []}}
    cases = [
        ([nts, nts26], both.__getitem__, "different stop chains"),
        ([nts], ambiguous.__getitem__, "more than one MuminusStopsCat"),
        ([nts], orphan.__getitem__, "no MuminusStopsCat dataset"),
        ([nts], {}.__getitem__, "cannot read the SAM metadata"),
        (["my_ce_ntuple.root"], catalog.__getitem__, "not a SAM file name"),
    ]
    for names, fetch, needle in cases:
        try:
            norm.stop_chain_of_inputs(names, fetch)
        except norm.NormalizationError as exc:
            assert needle in str(exc), (needle, str(exc))
        else:
            raise AssertionError(f"{needle}: no error")


def test_fullsim_generated_events_come_from_the_nearest_ancestor_with_gencount():
    from tools.analyses.fullsim import provenance
    nts = "nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root"
    mcs = "mcs.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1.001430_00000000.art"
    catalog = {nts: {"parents": [{"file_name": mcs}]},
               mcs: {"dh.gencount": 20000, "event_count": 8212,
                     "parents": [{"file_name": "dig.x"}]}}
    assert provenance.generated_events(nts, fetch=catalog.__getitem__) == 20000
    # no ancestor with a count, or no SAM: an error that says which file
    orphan = {nts: {"parents": []}}
    for fetch, needle in ((orphan.__getitem__, "no dh.gencount"),
                          ({}.__getitem__, "cannot read the SAM metadata")):
        try:
            provenance.generated_events(nts, fetch=fetch)
        except provenance.ProvenanceError as exc:
            assert needle in str(exc) and nts in str(exc), exc
        else:
            raise AssertionError(f"{needle}: no error")


def test_fullsim_run_reports_a_file_that_is_not_an_eventntuple(tmp_dir):
    path = Path(tmp_dir) / "nts.owner.edep.test.root"
    _write_edep_tree(path, n=10)
    result = run_analysis(analysis="fullsim_sensitivity", data_file=str(path),
                          output_dir=tmp_dir)
    assert result.status == "error" and "EventNtuple/ntuple" in result.message
    window = run_analysis(analysis="fullsim_sensitivity", data_file=str(path),
                          output_dir=tmp_dir, parameters={"time_window": "1650,640"})
    assert window.status == "error" and "time_window" in window.message


# A CeMLeadingLogMix1BB EventNtuple file: CE- mixed with pileup; skipped
# where not on disk.
CE_MIX_FILE = Path("/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/"
                   "MDC2025au_best_v1_1-001/root/a8/50/nts.mu2e.CeMLeadingLogMix1BB."
                   "MDC2025au_best_v1_1-001.001430_00000000.root")


def test_fullsim_measures_the_efficiency_on_ce_mix(tmp_dir):
    if not CE_MIX_FILE.exists():
        print("     (skipped: CE mix file not on disk)")
        return
    # the file's parent mcs file has dh.gencount 20000 in SAM, and the stop
    # chain gives STOPPED_MUONS_PER_POT; both passed here so the test needs
    # no network
    from tools.analyses.fullsim import normalization as norm
    generated = {"n_generated": 20000,
                 "stopped_muons_per_pot": norm.STOPPED_MUONS_PER_POT}
    result = run_analysis(analysis="fullsim_sensitivity", data_file=str(CE_MIX_FILE),
                          output_dir=tmp_dir, parameters=generated)
    assert result.status == "success", result.message
    md = result.metadata
    assert md["n_events_processed"] == 8212 and md["n_events_generated"] == 20000
    assert np.isclose(md["acceptance"], 8212 / 20000)
    assert (md["n_events_selected"], md["n_events_counted"]) == (4181, 3010)
    # the efficiency counts against what was generated, so it carries the
    # digitization filter's acceptance
    assert np.isclose(md["signal_efficiency"], 3010 / 20000)
    assert np.isclose(md["signal_rate"], 4.686, rtol=1e-3)
    # the cut flow is pyfitter's for cut-set 80_1d on this file
    assert list(md["cut_flow"].values())[-1] == 4181
    assert md["cut_flow"]["no_multi_trk_veto"] == 6909 and md["cut_flow"]["pz_over_pt"] == 4550
    assert np.allclose((md["signal_mom_low_mevc"], md["signal_mom_high_mevc"]), (103.5, 104.7))
    # 1e18 POT x 7.67e-4 stopped mu-/POT x 0.609 captures x R_mue 1e-13
    assert np.isclose(md["n_conversions"], 46.717, rtol=1e-4)
    assert md["total_background"] == md["dio_background"] + md["cosmic_background"]
    # a pileup DIO track that passes the cuts is not signal
    assert md["selected_by_origin"] == {"DIO": 1, "CE-": 4180}
    # upstream_eff scales the signal and the DIO alike, not the cosmics
    half = run_analysis(analysis="fullsim_sensitivity", data_file=str(CE_MIX_FILE),
                        output_dir=tmp_dir, parameters={**generated, "upstream_eff": 0.5})
    hm = half.metadata
    assert np.isclose(hm["signal_efficiency"], 0.5 * 3010 / 20000)
    assert hm["cosmic_rate_per_s_per_mev"] == md["cosmic_rate_per_s_per_mev"]
    bad = run_analysis(analysis="fullsim_sensitivity", data_file=str(CE_MIX_FILE),
                       output_dir=tmp_dir, parameters={**generated, "trigger_paths": "apr_Nope"})
    assert bad.status == "error" and "trig_apr_Nope" in bad.message
    few = run_analysis(analysis="fullsim_sensitivity", data_file=str(CE_MIX_FILE),
                       output_dir=tmp_dir, parameters={**generated, "n_generated": 100})
    assert few.status == "error" and "n_generated 100" in few.message


# --- occupancy (OccupancyAna: digi and reco from a job, ntuple from a file) ---------

REAL_OCCUPANCY_NTUPLE = Path(
    "/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeEndpointMix1BB-KL/Run1Baw_best_v1_5/"
    "root/e8/33/nts.mu2e.CeEndpointMix1BB-KL.Run1Baw_best_v1_5.001470_00000010.root")


def test_occupancy_summary_table_parses_rows_found_missing_and_stats():
    summary = occ.parse_summary(SAMPLE_OCCUPANCY_RECO_STDOUT)
    assert summary.n_events == 50 and list(summary.rows) == [
        "strawhits", "combohits", "calohits", "caloclusters", "calorecodigis"]
    row = summary.rows["strawhits"]
    assert (row.tag, row.found, row.missing, row.mean, row.rms, row.max) == (
        "makeSH", 50, 0, 779.42, 460.005, 2312)
    assert summary.rows["calorecodigis"].missing == 50
    assert occ.parse_summary("Art has completed and will exit with status 0.") is None


def test_occupancy_table_metrics_use_minus_one_for_skipped_and_missing():
    names = occ_reco.COLLECTIONS
    metrics = occ.table_metrics(names, occ.parse_summary(SAMPLE_OCCUPANCY_RECO_STDOUT))
    assert metrics["strawhits_mean"] == 779.42 and metrics["strawhits_max"] == 2312.0
    assert metrics["strawhits_events_found"] == 50.0
    # configured but in no event: no statistics, the missing events counted
    assert metrics["calorecodigis_mean"] == -1.0
    assert metrics["calorecodigis_events_found"] == 0.0
    assert metrics["calorecodigis_events_missing"] == 50.0
    # not configured at all: everything -1
    digi = occ.parse_summary(SAMPLE_OCCUPANCY_DIGI_STDOUT)
    skipped = occ.table_metrics(("strawdigis", "crvpulses"), digi)
    assert skipped["crvpulses_events_found"] == -1.0 == skipped["crvpulses_events_missing"]
    assert tuple(occ.table_metrics(names, occ.parse_summary(SAMPLE_OCCUPANCY_RECO_STDOUT))
                 ) == occ.table_metric_names(names)


def test_occupancy_warns_of_a_collection_missing_from_every_event():
    summary = occ.parse_summary(SAMPLE_OCCUPANCY_RECO_STDOUT)
    (warning,) = occ.warnings(occ_reco.COLLECTIONS, summary)
    assert "calorecodigis (CaloRecoDigiMaker)" in warning and "tag" in warning
    assert not occ.warnings(occ_digi.COLLECTIONS,
                            occ.parse_summary(SAMPLE_OCCUPANCY_DIGI_STDOUT))
    metrics = {m: 1.0 for m in ANALYSES["occupancy_reco"].metrics}
    metrics.update(calohits_events_found=0.0, caloclusters_events_found=-1.0)
    text = ANALYSES["occupancy_reco"].summarize(metrics)
    assert "calohits MISSING from every event" in text and "caloclusters" not in text


def test_occupancy_defaults_are_the_production_labels_and_reco_leaves_digis_off():
    digi = {p.name: p.default for p in ANALYSES["occupancy_digi"].parameters}
    assert digi == {"strawdigis_tag": "makeSD", "calodigis_tag": "CaloDigiMaker",
                    "crvdigis_tag": "CrvDigi"}
    reco = {p.name: p.default for p in ANALYSES["occupancy_reco"].parameters}
    assert reco["strawhits_tag"] == "makeSH" and reco["combohits_tag"] == "makePH"
    assert reco["calohits_tag"] == "CaloHitMaker"
    assert reco["caloclusters_tag"] == "CaloClusterMaker"
    assert reco["calorecodigis_tag"] == ""                  # a SelectReco subset
    assert not {"strawdigis_tag", "crvdigis_tag", "crvpulses_tag"} & set(reco)
    # the window defaults to everything
    assert (reco["hit_time_min"], reco["hit_time_max"], reco["calo_hit_energy_min"]
            ) == (-1e9, 1e9, 0.0)
    assert all(p.allow_empty for p in ANALYSES["occupancy_digi"].parameters)


def test_occupancy_job_fcl_appends_the_tags_and_window_to_the_shipped_text():
    text = occ.job_fcl_text({"strawhits": "makeSH", "calohits": "CaloHitMaker:x:P"},
                            (700.0, 1695.0, 10.0))
    assert text.startswith(occ.FCL.read_text(encoding="utf-8").rstrip())
    assert "physics.analyzers.OccupancyAna.StrawHitCollection : \"makeSH\"" in text
    assert "CaloHitCollection : \"CaloHitMaker:x:P\"" in text
    assert "ComboHitCollection" not in text.split("set by tools")[1]    # skipped
    assert "HitTimeMin : 700\n" in text and "HitTimeMax : 1695\n" in text
    assert "CaloHitEnergyMin : 10\n" in text
    assert "HitTime" not in occ.job_fcl_text({"strawdigis": "makeSD"}).split("set by tools")[1]
    # the shipped fcl itself configures no collection: the run decides them
    assert "Collection" not in "".join(
        line for line in occ.FCL.read_text().splitlines() if not line.startswith("#"))


def test_occupancy_tags_must_be_input_tags_and_one_must_be_left():
    params = {"strawhits_tag": "makeSH", "combohits_tag": " ", "calohits_tag": "A:b:C"}
    assert occ.parse_tags(params, occ_reco.COLLECTIONS) == {
        "strawhits": "makeSH", "calohits": "A:b:C"}
    for bad in ('make SH', 'a"b', "x:y:z:w", "a;b"):
        try:
            occ.parse_tags({"strawhits_tag": bad}, occ_reco.COLLECTIONS)
        except occ.OccupancyError as exc:
            assert "strawhits_tag" in str(exc) and "input tag" in str(exc)
        else:
            raise AssertionError(f"{bad!r} was accepted as a tag")
    try:
        occ.parse_tags({n + "_tag": "" for n in occ_digi.COLLECTIONS}, occ_digi.COLLECTIONS)
    except occ.OccupancyError as exc:
        assert "every collection is skipped" in str(exc) and "crvdigis_tag" in str(exc)
    else:
        raise AssertionError("an all-skipped request was accepted")


def test_occupancy_parameters_validate():
    spec = ANALYSES["occupancy_reco"]
    assert spec.resolve_params({"strawhits_tag": ""})["strawhits_tag"] == ""
    for bad in ({"calo_hit_energy_min": -1}, {"hit_time_min": "soon"},
                {"strawhits_tag": 5}, {"crvdigis_tag": "CrvDigi"}):
        try:
            spec.resolve_params(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad} was accepted")


def _write_occupancy_tree(path, events, ndisk=2, nplane=36):
    """A synthetic OccupancyAna/tree: `events` rows of (strawhits, calohits,
    plane counts, disk counts, windowed straw hits); -1 where skipped."""
    import uproot

    n = len(events)
    with uproot.recreate(path) as f:
        tree = f.mktree("OccupancyAna/tree", {
            "nstrawhits": "i4", "ncalohits": "i4", "nstrawhits_sel": "i4",
            "ncalohits_sel": "i4", "ncombohit_straws": "i4",
            "strawhits_plane": np.dtype(("i4", (nplane,))),
            "calohits_disk": np.dtype(("i4", (ndisk,)))})
        planes = np.full((n, nplane), -1, dtype=np.int32)
        disks = np.full((n, ndisk), -1, dtype=np.int32)
        for i, (nsh, nch, plane0, disk0) in enumerate(events):
            if nsh >= 0:
                planes[i] = 1
                planes[i, 0] = plane0
            if nch >= 0:
                disks[i] = (disk0, nch - disk0)
        tree.extend({
            "nstrawhits": np.array([e[0] for e in events], dtype=np.int32),
            "ncalohits": np.array([e[1] for e in events], dtype=np.int32),
            "nstrawhits_sel": np.array([e[0] // 2 if e[0] >= 0 else -1 for e in events],
                                       dtype=np.int32),
            "ncalohits_sel": np.array([e[1] // 2 if e[1] >= 0 else -1 for e in events],
                                      dtype=np.int32),
            "ncombohit_straws": np.full(n, -1, dtype=np.int32),
            "strawhits_plane": planes, "calohits_disk": disks})


def test_occupancy_tree_gives_plane_disk_and_windowed_means_over_events_that_have_them(tmp_dir):
    path = Path(tmp_dir) / "occ.root"
    # the second event lacks the straw hit collection, the third the calo hits
    _write_occupancy_tree(path, [(100, 10, 9, 6), (-1, 20, 0, 12), (50, -1, 3, 0)])
    tree = occ.read_tree([path])
    assert tree["strawhits_plane"].shape == (3, 36)
    metrics, extra = occ.tree_metrics(("strawhits", "combohits", "calohits"), tree)
    assert metrics["strawhits_busiest_plane_mean"] == 6.0       # plane 0: (9 + 3) / 2
    assert extra["busiest_plane"]["strawhits"] == 0
    assert metrics["strawhits_sel_mean"] == 37.5                # (50 + 25) / 2
    assert metrics["calohits_disk0_mean"] == 9.0 and metrics["calohits_disk1_mean"] == 6.0
    assert metrics["calohits_sel_mean"] == 7.5
    assert metrics["combohits_sel_mean"] == -1.0                # no such branch
    assert metrics["combohit_straws_mean"] == -1.0              # -1 in every event
    # a collection that is not counted stays -1
    assert occ.tree_metrics(("calohits",), tree)[0].get("strawhits_sel_mean") is None
    assert tuple(occ.tree_metrics(occ_reco.COLLECTIONS, tree)[0]) == \
        occ.tree_metric_names(occ_reco.COLLECTIONS)


@contextlib.contextmanager
def _fake_occupancy_job(stdout, stderr="", returncode=0, events=((100, 10, 9, 6),)):
    """Replace run_mu2e_job in occupancy.py with one that prints `stdout` and
    writes a synthetic tree; returns the fcl text the job was given."""
    from tools.mu2e_job import JobOutcome

    seen: dict = {}

    def fake(*, fcl, input_paths, outdir, single, timeout_s, max_events=None, **_):
        seen["fcl"] = Path(fcl).read_text()
        seen["max_events"] = max_events
        _write_occupancy_tree(outdir / "nts.owner.occupancy.version.sequencer.root",
                              list(events))
        return JobOutcome(
            command="mu2e", returncode=returncode, timed_out=False, stdout=stdout,
            stderr=stderr, log_path=outdir / "mu2e.log", input_paths=input_paths,
            input_flag="-s", environment="fake", file_list_path=None,
            written_root_files=[str(outdir / "nts.owner.occupancy.version.sequencer.root")])

    original = occ.run_mu2e_job
    occ.run_mu2e_job = fake
    try:
        yield seen
    finally:
        occ.run_mu2e_job = original


def _run_occupancy(name, tmp_dir, stdout, params=None, **kwargs):
    from tools.spec import RunContext

    spec = ANALYSES[name]
    out = Path(tmp_dir) / "out"
    out.mkdir(exist_ok=True)
    with _fake_occupancy_job(stdout, **kwargs) as seen:
        outcome = spec.run(RunContext(
            input_paths=[Path("/a.art")], outdir=out,
            params=spec.resolve_params(params), timeout_s=60, max_events=5))
    return outcome, seen


def test_occupancy_reco_runs_a_per_run_fcl_and_reports_every_declared_metric(tmp_dir):
    outcome, seen = _run_occupancy(
        "occupancy_reco", tmp_dir, SAMPLE_OCCUPANCY_RECO_STDOUT,
        {"hit_time_min": 700, "calorecodigis_tag": "CaloRecoDigiMaker"})
    assert outcome.error is None, outcome.error
    assert "HitTimeMin : 700" in seen["fcl"] and seen["max_events"] == 5
    assert "CaloRecoDigiCollection : \"CaloRecoDigiMaker\"" in seen["fcl"]
    assert outcome.extra["job_fcl"] == str(Path(tmp_dir) / "out" / "occupancy_job.fcl")
    assert tuple(outcome.metrics) == ANALYSES["occupancy_reco"].metrics
    assert outcome.metrics["n_events"] == 50.0
    assert outcome.metrics["strawhits_mean"] == 779.42
    assert outcome.metrics["strawhits_busiest_plane_mean"] == 9.0
    assert outcome.metrics["calohits_sel_mean"] == 5.0
    assert "calorecodigis" in outcome.extra["warnings"][0]      # missing in every event


def test_occupancy_digi_marks_skipped_collections_minus_one(tmp_dir):
    # the job prints no row for a collection that has no tag
    stdout = "\n".join(line for line in SAMPLE_OCCUPANCY_DIGI_STDOUT.splitlines()
                       if "crvdigis" not in line)
    outcome, seen = _run_occupancy("occupancy_digi", tmp_dir, stdout,
                                   {"crvdigis_tag": ""})
    assert outcome.error is None, outcome.error
    assert "CrvDigiCollection" not in seen["fcl"].split("set by tools")[1]
    assert "HitTime" not in seen["fcl"].split("set by tools")[1]     # digis have no window
    assert outcome.extra["skipped"] == ["crvdigis"]
    assert outcome.metrics["crvdigis_mean"] == -1.0
    assert outcome.metrics["crvdigis_events_found"] == -1.0
    assert outcome.metrics["strawdigis_mean"] == 44.74


def test_occupancy_with_every_collection_skipped_never_starts_a_job(tmp_dir):
    outcome, seen = _run_occupancy(
        "occupancy_digi", tmp_dir, "", {"strawdigis_tag": "", "calodigis_tag": "",
                                        "crvdigis_tag": ""})
    assert "every collection is skipped" in outcome.error and "fcl" not in seen
    outcome, seen = _run_occupancy("occupancy_reco", tmp_dir, "",
                                   {"hit_time_min": 5, "hit_time_max": 1})
    assert "hit_time_min" in outcome.error and "fcl" not in seen


def test_occupancy_on_an_offline_without_the_module_says_it_needs_the_pr(tmp_dir):
    art = ('Library specification "OccupancyAna" does not correspond to any library '
           'in CET_PLUGIN_PATH of type "module"')
    outcome, _ = _run_occupancy("occupancy_reco", tmp_dir, "", stderr=art, returncode=9)
    assert "OccupancyAna needs an Offline that includes PR #2070" in outcome.error
    assert "Traceback" not in outcome.error
    # any other failure is reported as itself
    outcome, _ = _run_occupancy("occupancy_reco", tmp_dir, "", stderr="boom", returncode=1)
    assert outcome.error == "mu2e exited 1"
    outcome, _ = _run_occupancy("occupancy_reco", tmp_dir, "Art has completed")
    assert "summary table not found" in outcome.error


def _write_event_ntuple(path, *, hitcount=True, crv=True, calo=True, tracks=True,
                        n=3):
    """A synthetic EventNtuple/ntuple with only the groups asked for. The
    branches are flat and dotted, which is one of the layouts read."""
    import awkward as ak
    import uproot

    types, data = {}, {}
    if hitcount:
        for leaf, values in (("nsd", [100, 200, 300]), ("nesel", [90, 190, 290]),
                             ("nrsel", [80, 180, 280]), ("ntsel", [100, 200, 300]),
                             ("nbkg", [0, 0, 3])):
            types[f"hitcount.{leaf}"] = "i4"
            data[f"hitcount.{leaf}"] = np.array(values, dtype=np.int32)
    if calo:
        types.update({"calohits.time_": "var * float32", "calohits.eDep_": "var * float32",
                      "caloclusters.time_": "var * float32"})
        data.update({
            "calohits.time_": ak.Array([[100., 900., 1500.], [], [800., 850.]]),
            "calohits.eDep_": ak.Array([[1., 20., 30.], [], [5., 15.]]),
            "caloclusters.time_": ak.Array([[900.], [], [800., 850.]])})
    if crv:
        types.update({"crvcoincs.time": "var * float32", "crvsummary.nHitCounters": "i4",
                      "crvsummary.totalPEs": "i4"})
        data.update({"crvcoincs.time": ak.Array([[1.], [], [2., 3., 4.]]),
                     "crvsummary.nHitCounters": np.array([1, 0, 4], dtype=np.int32),
                     "crvsummary.totalPEs": np.array([10, 0, 40], dtype=np.int32)})
    if tracks:
        types["tcnt.ntrk"] = "i4"
        data["tcnt.ntrk"] = np.array([0, 1, 1], dtype=np.int32)
    with uproot.recreate(path) as f:
        f.mktree("EventNtuple/ntuple", types).extend(data)


def test_occupancy_ntuple_counts_jagged_calo_hits_and_the_window(tmp_dir):
    path = Path(tmp_dir) / "full.root"
    _write_event_ntuple(path)
    result = run_analysis(analysis="occupancy_ntuple", data_file=str(path),
                          output_dir=tmp_dir,
                          parameters={"hit_time_min": 700, "hit_time_max": 1000,
                                      "calo_hit_energy_min": 10})
    assert result.status == "success", result.message
    meta = result.metadata
    assert meta["n_events"] == 3
    assert meta["strawdigis_mean"] == 200.0 and meta["strawdigis_max"] == 300.0
    assert abs(meta["strawdigis_rms"] - np.std([100, 200, 300])) < 1e-9
    assert meta["strawhits_bkg_max"] == 3.0
    assert meta["calohits_mean"] == 5 / 3 and meta["calohits_max"] == 3.0   # sizes of [3, 0, 2]
    assert meta["caloclusters_mean"] == 1.0
    # window 700-1000 ns and >= 10 MeV: the 20 MeV hit and the 15 MeV one
    assert meta["calohits_sel_mean"] == 2 / 3
    assert meta["crvcoincs_mean"] == 4 / 3 and meta["crv_total_pes_max"] == 40.0
    assert meta["crv_hit_counters_mean"] == 5 / 3 and meta["ntracks_mean"] == 2 / 3
    assert "missing" not in meta
    assert set(ANALYSES["occupancy_ntuple"].metrics) <= set(meta)


def test_occupancy_ntuple_reports_missing_groups_and_errors_only_if_all_are_missing(tmp_dir):
    path = Path(tmp_dir) / "partial.root"
    _write_event_ntuple(path, crv=False, calo=False)
    result = run_analysis(analysis="occupancy_ntuple", data_file=str(path),
                          output_dir=tmp_dir)
    assert result.status == "success", result.message
    meta = result.metadata
    assert {"calohits", "caloclusters", "crvcoincs", "crv_hit_counters",
            "crv_total_pes", "calohits_sel"} == set(meta["missing"])
    assert meta["calohits_mean"] == -1.0 and meta["calohits_sel_mean"] == -1.0
    assert meta["crvcoincs_max"] == -1.0 and meta["strawdigis_mean"] == 200.0
    # the default window is everything
    assert meta["window"]["hit_time_max"] == 1e9
    empty = Path(tmp_dir) / "none.root"
    # a tree with a single unrelated branch
    import uproot
    with uproot.recreate(empty) as f:
        f.mktree("EventNtuple/ntuple", {"evtinfo.event": "i4"}).extend(
            {"evtinfo.event": np.array([1, 2], dtype=np.int32)})
    result = run_analysis(analysis="occupancy_ntuple", data_file=str(empty),
                          output_dir=tmp_dir)
    assert result.status == "error" and "none of the occupancy branches" in result.message
    not_ntuple = Path(tmp_dir) / "other.root"
    with uproot.recreate(not_ntuple) as f:
        f.mktree("other", {"x": "i4"}).extend({"x": np.array([1], dtype=np.int32)})
    result = run_analysis(analysis="occupancy_ntuple", data_file=str(not_ntuple),
                          output_dir=tmp_dir)
    assert result.status == "error" and "no EventNtuple/ntuple" in result.message


def test_occupancy_ntuple_combines_files_and_drops_a_group_missing_from_one(tmp_dir):
    a, b = Path(tmp_dir) / "a.root", Path(tmp_dir) / "b.root"
    _write_event_ntuple(a)
    _write_event_ntuple(b, crv=False)
    result = run_analysis(analysis="occupancy_ntuple", data_files=[str(a), str(b)],
                          output_dir=tmp_dir)
    assert result.status == "success", result.message
    assert result.metadata["n_events"] == 6
    assert result.metadata["strawdigis_mean"] == 200.0
    assert result.metadata["crvcoincs_mean"] == -1.0        # not in every file
    assert "crvcoincs" in result.metadata["missing"]
    assert len(result.metadata["per_file_events"]) == 2
    both = run_analysis(analysis="occupancy_ntuple", data_files=[str(a), str(a)],
                        output_dir=tmp_dir)
    assert both.metadata["calohits_mean"] == 5 / 3


def test_occupancy_ntuple_on_a_real_file(tmp_dir):
    from tools import catalog

    if (not catalog.CLIENT.exists(REAL_OCCUPANCY_NTUPLE)
            or catalog.CLIENT.locality(REAL_OCCUPANCY_NTUPLE) not in catalog.ONLINE):
        return
    result = run_analysis(analysis="occupancy_ntuple", data_file=str(REAL_OCCUPANCY_NTUPLE),
                          output_dir=tmp_dir)
    assert result.status == "success", result.message
    meta = result.metadata
    assert meta["n_events"] == 622 and "missing" not in meta
    assert 700 < meta["strawdigis_mean"] < 800 and meta["strawhits_bkg_max"] == 0.0
    assert 50 < meta["calohits_mean"] < 60 and 5 < meta["caloclusters_mean"] < 6
    assert meta["calohits_sel_mean"] == meta["calohits_mean"]     # full window


# --- total_edep_per_pot (reads no files) -------------------------------------

DOSE_PARAMS = {"ebeam_edep_per_gen_event_mev": 6.89476e-06,
               "mubeam_edep_per_gen_event_mev": 1.334e-07,
               "mustop_edep_per_gen_event_mev": 2.84332e-04,
               "stops_per_pot": 1.1514376428e-03}


def test_total_edep_per_pot_normalizes_each_source_and_adds_them():
    params = dict(DOSE_PARAMS, pot_per_ebeam_event=dose.POT_PER_EBEAM_EVENT,
                  mubeam_events_per_pot=dose.MUBEAM_EVENTS_PER_POT)
    m = dose.per_pot(params)
    ebeam = 6.89476e-06 / 11.536718606512062
    mubeam = 1.334e-07 * 0.01278168
    mustop = 2.84332e-04 * 1.1514376428e-03
    assert abs(m["ebeam_edep_per_pot_mev"] - ebeam) < 1e-18
    assert abs(m["mubeam_edep_per_pot_mev"] - mubeam) < 1e-18
    assert abs(m["mustop_edep_per_pot_mev"] - mustop) < 1e-18
    assert abs(m["total_edep_per_pot_mev"] - (ebeam + mubeam + mustop)) < 1e-18
    assert abs(sum(m[f"{s}_fraction"] for s in dose.SOURCES) - 1.0) < 1e-12
    assert tuple(m) == dose.SPEC.metrics


def test_total_edep_per_pot_with_nothing_deposited_has_zero_shares():
    params = {k: 0.0 for k in DOSE_PARAMS}
    params.update(pot_per_ebeam_event=1.0, mubeam_events_per_pot=1.0)
    m = dose.per_pot(params)
    assert m["total_edep_per_pot_mev"] == 0.0
    assert all(m[f"{s}_fraction"] == 0.0 for s in dose.SOURCES)


def test_a_parameters_analysis_runs_without_input_files(tmp_dir):
    result = run_analysis(analysis="total_edep_per_pot", output_dir=tmp_dir,
                          parameters=DOSE_PARAMS)
    assert result.status == "success", result.message
    assert result.metadata["n_input_files"] == 0
    assert result.metadata["input_kind"] == "parameters"
    assert result.metadata["total_edep_per_pot_mev"] > 0
    assert "MeV/POT" in result.message


def test_a_parameters_analysis_refuses_input_files(tmp_dir):
    path = Path(tmp_dir) / "x.root"
    path.write_text("")
    for extra in ({"data_file": str(path)}, {"data_files": [str(path)]},
                  {"max_events": 10}):
        result = run_analysis(analysis="total_edep_per_pot",
                              output_dir=tmp_dir, parameters=DOSE_PARAMS,
                              **extra)
        assert result.status == "error", extra
        assert "reads no input files" in result.message, result.message


def test_a_parameters_analysis_needs_its_required_parameters(tmp_dir):
    params = dict(DOSE_PARAMS)
    del params["stops_per_pot"]
    result = run_analysis(analysis="total_edep_per_pot", output_dir=tmp_dir,
                          parameters=params)
    assert result.status == "error"
    assert "stops_per_pot" in result.message, result.message


def test_list_analyses_describes_the_parameters_kind():
    entry = list_analyses().metadata["analyses"]["total_edep_per_pot"]
    assert entry["input_kind"] == "parameters"
    assert entry["takes_data_files"] is False
    assert entry["parameters"]["pot_per_ebeam_event"]["default"] == \
        dose.POT_PER_EBEAM_EVENT


# --- the registry (loops over every analysis) --------------------------------

# --- edep_occupancy and edep_occupancy_combine ---------------------------------

_NCRY_PER_RBIN = [64, 78, 84, 106, 104, 102, 116, 20]


def _occ_hist(values, edges=None):
    values = np.asarray(values, dtype=float)
    if edges is None:
        edges = [np.arange(n + 1.0) for n in values.shape]
    return eocc.Hist(values, np.abs(values), edges)


def _occ_hists(straw=((3, 2, 40), 10.0), others=2.0, crystals=((10, 20.0),),
               radial_hits=None):
    """Per-event occupancy histograms with a known hot straw (and a row whose
    other five panels hold `others` hits), hot crystals and a radial profile."""
    hits = np.zeros(eocc.N_PLANES * eocc.N_PANELS * eocc.N_STRAWS)
    (plane, panel, istraw), value = straw
    for p in range(eocc.N_PANELS):
        hits[(plane * eocc.N_PANELS + p) * eocc.N_STRAWS + istraw] = (
            value if p == panel else others)
    hits[5] = 1.0                                     # a quiet straw elsewhere
    chits = np.zeros(1348)
    for cid, v in crystals:
        chits[cid] = v
    ncry = np.array(_NCRY_PER_RBIN, dtype=float)
    r_edges = np.linspace(385.0, 665.0, 9)
    h = {"trk_straw_hits": _occ_hist(hits), "trk_straw_occ": _occ_hist(3 * hits),
         "trk_straw_edep": _occ_hist(0.001 * 3 * hits),
         "calo_crystal_hits": _occ_hist(chits), "calo_crystal_occ": _occ_hist(2 * chits),
         "calo_crystal_edep": _occ_hist(5 * chits)}
    for det in ("trk", "calo"):
        for kind in ("occ", "edep", "ecut"):
            h[f"{det}_time_folded_{kind}"] = _occ_hist(
                np.ones(100), [np.linspace(0, 1695.0, 101)])
        for kind in ("occ", "edep"):
            h[f"{det}_rz_{kind}"] = _occ_hist(np.ones((4, 3)),
                                              [np.arange(5.0), np.arange(4.0)])
    for d in range(2):
        rh = np.array(radial_hits if radial_hits is not None
                      else [1, 1, 1, 1, 1, 6, 1, 1], dtype=float) * ncry * 0.1
        h[f"calo_disk{d}_ncrystals_r"] = _occ_hist(ncry, [r_edges])
        h[f"calo_disk{d}_rcrys_hits"] = _occ_hist(rh, [r_edges])
        h[f"calo_disk{d}_rcrys_occ"] = _occ_hist(2 * rh, [r_edges])
        h[f"calo_disk{d}_rcrys_edep"] = _occ_hist(5 * rh, [r_edges])
    return h


def _write_raw_occupancy(path, hists, n_events):
    """An EdepAna file as the Offline module writes it: occupancy/ hists per event."""
    import uproot
    with uproot.recreate(path) as f:
        for name, hist in hists.items():
            eocc._write_hist(f, f"{eocc.OCCUPANCY_DIR}/{name}", hist)
        eocc._write_hist(f, f"{eocc.OCCUPANCY_DIR}/nevents", _occ_hist([n_events], [np.array([0.0, 1.0])]))


def _run_edep_occupancy(tmp_dir, hists=None, n_events=1000.0, parameters=None,
                        has_steps=True, name="comp"):
    """edep_occupancy with edep's job replaced by a file the test wrote."""
    from tools.spec import RunContext
    outdir = Path(tmp_dir) / name
    outdir.mkdir(parents=True, exist_ok=True)
    raw = outdir / "nts.owner.edep.test.root"
    if hists is not None:
        _write_raw_occupancy(raw, hists, n_events)
    else:
        import uproot
        with uproot.recreate(raw) as f:
            f["other"] = "not an EdepAna file"
    seen = {}

    def fake_edep(context):
        seen["params"] = context.params
        return RunOutcome(metrics={"n_events": n_events}, files=[str(raw)],
                          extra={"selection": "", "job_fcl": "x"})
    from tools.spec import RunOutcome
    saved = edep_mod.run, edep_mod.module_accepts
    edep_mod.run, edep_mod.module_accepts = fake_edep, lambda *a, **k: has_steps
    try:
        params = ANALYSES["edep_occupancy"].resolve_params(parameters)
        outcome = eocc.run(RunContext(input_paths=[Path(tmp_dir) / "in.art"], outdir=outdir,
                                      params=params, timeout_s=60))
    finally:
        edep_mod.run, edep_mod.module_accepts = saved
    return outcome, seen


def test_edep_occupancy_scales_to_microbunch_and_finds_the_peaks(tmp_dir):
    outcome, seen = _run_edep_occupancy(
        tmp_dir, _occ_hists(crystals=((10, 20.0), (700, 6.0))), n_events=1000.0,
        parameters={"events_per_microbunch": 100.0, "label": "MuBeamFlash"})
    assert outcome.error is None, outcome.error
    m = outcome.metrics
    assert tuple(m) == ANALYSES["edep_occupancy"].metrics
    assert seen["params"]["selection"] == ""            # edep's job, no selection
    k = 100.0 / 1000.0                                   # per event -> per microbunch
    assert m["events_per_microbunch"] == 100.0 and m["n_events_analyzed"] == 1000.0
    khz = lambda x: x * k / 1695.0 * 1e6
    assert abs(m["trk_peak_straw_hit_rate_khz"] - khz(10.0)) < 1e-9
    assert (m["trk_peak_straw_plane"], m["trk_peak_straw_panel"],
            m["trk_peak_straw_straw"]) == (3.0, 2.0, 40.0)
    # the row averages the hot straw with the five other panels
    assert abs(m["trk_peak_row_hit_rate_khz"] - khz((10.0 + 5 * 2.0) / 6)) < 1e-9
    assert (m["trk_peak_row_plane"], m["trk_peak_row_straw"]) == (3.0, 40.0)
    assert m["trk_peak_row_radius_mm"] == 380.0 + 3.125 * 40
    # mean over all straws: (10 + 5*2 + 1) hits over 20736 straws
    assert abs(m["trk_mean_straw_hit_rate_khz"] - khz(21.0 / 20736)) < 1e-12
    assert abs(m["trk_hits_per_ub"] - 21.0 * k) < 1e-9
    assert abs(m["trk_steps_per_ub"] - 63.0 * k) < 1e-9
    assert abs(m["trk_peak_straw_edep_rate_mev_per_s"]
               - 0.001 * 3 * 10.0 * k / 1695.0 * 1e9) < 1e-6
    # statistical error: variance of a bin is its raw content, scaled by k^2
    assert abs(m["trk_peak_straw_hit_rate_khz_err"]
               - math.sqrt(10.0 * k * k) / 1695.0 * 1e6) < 1e-9
    assert m["calo_peak_crystal_id"] == 10.0 and m["calo_peak_crystal_disk"] == 0.0
    assert abs(m["calo_peak_crystal_hit_rate_khz"] - khz(20.0)) < 1e-9
    assert abs(m["calo_hits_per_ub"] - 26.0 * k) < 1e-9
    # radial: bin 5 is six times busier per crystal; rate per crystal = hits / crystals
    per_crystal = 6 * 0.1
    assert abs(m["calo_disk0_peak_radial_hit_rate_khz"] - khz(per_crystal)) < 1e-9
    assert m["calo_disk1_peak_radial_hit_r_mm"] == 0.5 * (385.0 + 35 * 5 + 385.0 + 35 * 6)
    # artifacts: the scaled file is re-readable, and the plots exist
    scaled = Path(outcome.files[1])
    assert scaled.name == eocc.SCALED_FILE
    hists, meta = eocc.read_scaled_file(scaled)
    assert meta["label"] == "MuBeamFlash" and meta["n_events"] == 1000.0
    assert abs(hists["trk_straw_hits"].values.max() - 10.0 * k) < 1e-12
    assert abs(hists["trk_straw_hits"].variances.max() - 10.0 * k * k) < 1e-12
    assert [Path(f).name for f in outcome.files[2:]] == list(eocc.PLOT_FILES)
    assert all(Path(f).stat().st_size > 1000 for f in outcome.files[2:])
    assert "peak_channels" in outcome.extra and "independent" in outcome.extra["error_note"]


def test_edep_occupancy_defaults_to_per_event(tmp_dir):
    outcome, _ = _run_edep_occupancy(tmp_dir, _occ_hists(), n_events=1000.0)
    assert outcome.error is None, outcome.error
    # events_per_microbunch = 1: the numbers are per event (21 hits over 1000 events)
    assert outcome.metrics["events_per_microbunch"] == 1.0
    assert abs(outcome.metrics["trk_hits_per_ub"] - 21.0 / 1000.0) < 1e-12


def test_edep_occupancy_is_unchanged_by_hadd_of_several_jobs(tmp_dir):
    # hadd of 3 jobs triples every histogram, the crystal counts included
    one, _ = _run_edep_occupancy(tmp_dir, _occ_hists(), n_events=1000.0, name="one")
    tripled = {n: h.scaled(3.0) for n, h in _occ_hists().items()}
    three, _ = _run_edep_occupancy(tmp_dir, tripled, n_events=3000.0, name="three")
    assert one.error is None and three.error is None, (one.error, three.error)
    for key in ("calo_disk0_peak_radial_hit_rate_khz", "trk_peak_row_hit_rate_khz",
                "calo_peak_crystal_hit_rate_khz", "trk_hits_per_ub"):
        assert abs(one.metrics[key] - three.metrics[key]) < 1e-9 * max(1.0, one.metrics[key]), key


def test_edep_occupancy_says_what_an_old_offline_needs(tmp_dir):
    outcome, _ = _run_edep_occupancy(tmp_dir, _occ_hists(), has_steps=False)
    assert "occupancy histograms" in outcome.error and "edepana-occupancy" in outcome.error
    # a new enough EdepAna without the occupancy directory, or with an old one
    outcome, _ = _run_edep_occupancy(tmp_dir, None)
    assert "no EDepAna/occupancy directory" in outcome.error
    assert "occupancy histograms" in outcome.error
    old = _occ_hists()
    del old["calo_crystal_hits"]
    outcome, _ = _run_edep_occupancy(tmp_dir, old, name="old")
    assert "lacks calo_crystal_hits" in outcome.error


def test_edep_occupancy_warns_when_the_microbunch_differs_from_the_folding(tmp_dir):
    outcome, _ = _run_edep_occupancy(tmp_dir, _occ_hists(),
                                     parameters={"microbunch_ns": 1000.0})
    assert outcome.error is None
    assert "folded" in outcome.extra["warnings"][0]


def test_edep_occupancy_through_the_real_edep_job_reports_a_file_without_histograms(tmp_dir):
    """edep.run itself (fake job, a tree file with no occupancy/) is what runs."""
    from tools.spec import RunContext
    fake, calls = _fake_edep_job(tmp_dir)
    real = edep_mod.run_mu2e_job, edep_mod.module_accepts
    edep_mod.run_mu2e_job, edep_mod.module_accepts = fake, lambda *a, **k: True
    try:
        outcome = eocc.run(RunContext(
            input_paths=[Path(tmp_dir) / "in.art"], outdir=Path(tmp_dir) / "o",
            params=ANALYSES["edep_occupancy"].resolve_params(
                {"trk_step_time_min_ns": 400.0}), timeout_s=60))
    finally:
        edep_mod.run_mu2e_job, edep_mod.module_accepts = real
    assert len(calls) == 1 and "TrkStepTimeMin : 400.0" in calls.fcl[0].read_text()
    assert "no EDepAna/occupancy directory" in outcome.error, outcome.error


def test_tracker_rows_average_the_six_panels():
    hits = np.zeros(eocc.N_PLANES * eocc.N_PANELS * eocc.N_STRAWS)
    for panel in range(6):
        hits[(7 * 6 + panel) * 96 + 12] = panel + 1.0
    rows = eocc.tracker_rows(_occ_hist(hits))
    assert rows.values.shape == (36, 96) and rows.values[7, 12] == 3.5
    assert abs(rows.variances[7, 12] - 21.0 / 36) < 1e-12


def _combine_inputs(tmp_dir):
    """Two components peaking in different channels, as scaled files."""
    paths = []
    for name, kwargs, n_per_ub in (
            ("A", dict(straw=((3, 2, 40), 10.0), others=0.0, crystals=((10, 20.0),)), 50.0),
            ("B", dict(straw=((9, 1, 7), 8.0), others=1.0, crystals=((10, 5.0), (700, 9.0))), 20.0)):
        outcome, _ = _run_edep_occupancy(
            tmp_dir, _occ_hists(**kwargs), n_events=100.0, name=name,
            parameters={"events_per_microbunch": n_per_ub, "label": name})
        assert outcome.error is None, outcome.error
        paths.append(Path(outcome.files[1]))
    return paths


def test_combine_sums_channels_so_the_peak_is_not_the_sum_of_peaks(tmp_dir):
    a, b = _combine_inputs(tmp_dir)
    result = run_analysis(analysis="edep_occupancy_combine", data_files=[str(a), str(b)],
                          output_dir=str(Path(tmp_dir) / "tot"))
    assert result.status == "success", result.message
    m = result.metadata
    ka, kb = 50.0 / 100.0, 20.0 / 100.0
    # both components run on the same straw only if their hot straws coincide; here
    # they differ, so the total peak is the larger single straw, not 10*ka + 8*kb
    straw_a, straw_b = 10.0 * ka, 8.0 * kb
    assert abs(m["trk_peak_straw_hit_rate_khz"] - straw_a / 1695.0 * 1e6) < 1e-9
    assert m["trk_peak_straw_hit_rate_khz"] < (straw_a + straw_b) / 1695.0 * 1e6
    assert (m["trk_peak_straw_plane"], m["trk_peak_straw_straw"]) == (3.0, 40.0)
    # crystal 10 holds 20*ka + 5*kb in the sum
    assert m["calo_peak_crystal_id"] == 10.0
    assert abs(m["calo_peak_crystal_hit_rate_khz"]
               - (20.0 * ka + 5.0 * kb) / 1695.0 * 1e6) < 1e-9
    assert m["n_components"] == 2.0 and m["events_per_microbunch"] == 70.0
    assert tuple(k for k in m if k in ANALYSES["edep_occupancy_combine"].metrics) \
        == ANALYSES["edep_occupancy_combine"].metrics
    # shares of the crystal where the sum peaks
    comp = m["components"]
    sa = comp["A"]["calo_peak_crystal_hit_share"]
    sb = comp["B"]["calo_peak_crystal_hit_share"]
    assert abs(sa + sb - 1.0) < 1e-12 and abs(sa - 20.0 * ka / (20.0 * ka + 5.0 * kb)) < 1e-12
    assert comp["A"]["trk_peak_straw_hit_share"] == 1.0
    assert comp["B"]["trk_peak_straw_hit_share"] == 0.0
    assert abs(comp["A"]["trk_straw_hits"] - 11.0 * ka) < 1e-9   # hot straw + the quiet one
    # the total file can itself be combined with another component
    total = Path(result.files[0])
    assert total.name == eoccc.TOTAL_FILE
    again = run_analysis(analysis="edep_occupancy_combine",
                         data_files=[str(total), str(a)], output_dir=str(Path(tmp_dir) / "t2"))
    assert again.status == "success", again.message


def test_combine_refuses_mismatched_cuts_and_raw_edep_files(tmp_dir):
    a, _ = _combine_inputs(tmp_dir)
    c, _ = _run_edep_occupancy(tmp_dir, _occ_hists(), name="C",
                               parameters={"trk_step_energy_threshold_mev": 0.002})
    result = run_analysis(analysis="edep_occupancy_combine",
                          data_files=[str(a), c.files[1]], output_dir=str(Path(tmp_dir) / "x"))
    assert result.status == "error" and "step_cuts" in result.message
    raw = Path(tmp_dir) / "C" / "nts.owner.edep.test.root"
    result = run_analysis(analysis="edep_occupancy_combine",
                          data_files=[str(a), str(raw)], output_dir=str(Path(tmp_dir) / "y"))
    assert result.status == "error" and "per-microbunch occupancy file" in result.message


def test_edep_occupancy_documents_the_dose_samples():
    entry = list_analyses().metadata["analyses"]["edep_occupancy"]
    text = entry["parameters"]["events_per_microbunch"]["description"]
    assert "Early" in text and "DetStepFilter" in text and "3746.4" in text
    assert entry["parameters"]["events_per_microbunch"]["default"] == 1.0
    assert "takes_data_files" in entry and entry["takes_data_files"] is True
    comb = list_analyses().metadata["analyses"]["edep_occupancy_combine"]
    assert comb["takes_data_files"] is True and comb["produced_by"] == ["edep_occupancy"]


def test_registry_includes_every_analysis():
    assert {"edep", "count", "muon_stop_rate", "approx_ce_sensitivity",
            "stop_materials", "fullsim_sensitivity"} <= set(ANALYSES)


def test_every_spec_is_self_consistent():
    for name, spec in ANALYSES.items():
        assert spec.name == name, f"{name}: SPEC.name is {spec.name}"
        assert spec.description.strip(), f"{name}: needs a description"
        assert spec.metrics, f"{name}: needs at least one metric"
        assert callable(spec.run) and callable(spec.summarize)
        assert spec.input_kind in ("art_files", "root_file", "parameters"), \
            spec.input_kind
        for metric in (spec.units or {}):
            assert metric in spec.metrics, f"{name}: unit for unknown '{metric}'"
        for param in spec.parameters:
            assert param.description.strip(), f"{name}: {param.name} needs a description"
        # art_files analyses run an fcl shipped in this repo's fcl/;
        # root_file ones must not claim one at all
        if spec.input_kind == "art_files":
            assert spec.fcl is not None and spec.fcl.parent == FCL_DIR, name
            assert current().missing_fcl(spec.fcl) is None, \
                f"{name}: {current().missing_fcl(spec.fcl)}"
        else:
            assert spec.fcl is None, f"{name}: {spec.input_kind} analysis should have no fcl"
            # its input comes from another analysis here, or from elsewhere
            # (a production job's ntuple) — either way, say which
            assert spec.produced_by or spec.input_hint.strip(), \
                f"{name}: say what produces its input"


def test_every_summarizer_handles_its_own_metrics():
    """summarize() must cope with exactly the metric names the spec declares."""
    for name, spec in ANALYSES.items():
        metrics = {metric: 1.0 for metric in spec.metrics}
        assert spec.summarize(metrics).strip(), f"{name}: empty summary"


def test_every_art_analysis_parser_matches_its_declared_metrics():
    for name, spec in ANALYSES.items():
        if spec.input_kind != "art_files":
            continue
        sample = SAMPLE_STDOUT.get(name)
        assert sample is not None, f"{name}: add a sample stdout to SAMPLE_STDOUT"
        if name == "edep":
            # the summary block, then what the selection adds from the tree
            # (the step counts, absent on an older Offline, come last)
            assert (tuple(parse_edep_summary(sample)) + edep_mod._SELECTION_METRICS
                    + tuple(edep_mod.parse_step_counts(SAMPLE_EDEP_STEPS_STDOUT)[0])
                    == spec.metrics)
        elif name == "count":
            assert tuple(saved_rates(parse_counts(sample))) == spec.metrics
        elif name == "muon_stop_rate":
            parsed = stop_rates({**parse_counts(sample, PRESCALE_FILTER),
                                 **parse_species_counts(sample)}, upstream_eff=1.0)
            assert tuple(parsed) == spec.metrics
        elif name == "trigger_efficiency":
            report = trig.parse_trig_report(sample)
            assert tuple(efficiency_metrics(report)) == spec.metrics
        elif name == "trigger_rate":
            report = trig.parse_trig_report(sample)
            assert tuple(rate_metrics(report, 1.0, 0.5)) == spec.metrics
        elif name == "muon_stops":
            # the generated count from stdout, the rest from its ROOT file
            stops = mstops.stop_variables(_stop_records([(-3874.0, 40.0, 5400.0, 300.0)]))
            n_gen = mstops.parse_gen_count(sample)
            assert tuple(mstops.stop_metrics(stops, np.array([30.0]), n_gen)) == spec.metrics
        elif name == "trigger_timing":
            # its numbers come from the timing database, not stdout
            assert tuple(timing_metrics(np.array([0.01, 0.02]))) == spec.metrics
        elif name == "edep_occupancy":
            # EdepAna's stdout is parsed by edep; the metrics come from its histograms
            hists = _occ_hists()
            meta = {"microbunch_ns": 1695.0, "n_events": 10.0, "events_per_microbunch": 1.0}
            assert tuple(eocc.metrics_from(hists, meta)[0]) == spec.metrics
        elif name in ("occupancy_digi", "occupancy_reco"):
            # the table's metrics, then what the per-event tree adds
            names = (occ_digi if name == "occupancy_digi" else occ_reco).COLLECTIONS
            table = occ.table_metrics(names, occ.parse_summary(sample))
            assert ("n_events", *table, *occ.tree_metric_names(names)) == spec.metrics
        else:
            raise AssertionError(f"{name}: add its parser to this test")


def test_list_analyses_reports_every_registered_analysis():
    result = list_analyses()
    assert result.status == "success"
    catalogue = result.metadata["analyses"]
    assert set(catalogue) == set(ANALYSES)

    edep = catalogue["edep"]
    assert edep["input_kind"] == "art_files"
    assert edep["fcl"] == str(FCL_DIR / "edep.fcl")
    assert edep["fcl_exists"] is True
    assert edep["units"]["avg_calo_edep_per_event_mev"] == "MeV"

    stops = catalogue["muon_stop_rate"]
    assert stops["input_kind"] == "art_files"
    assert stops["fcl"] == str(FCL_DIR / "stop_counts.fcl")
    assert stops["fcl_exists"] is True
    assert stops["parameters"]["upstream_eff"]["required"] is True
    assert stops["units"]["stops_per_pot"] == "stops / POT"

    counts = catalogue["count"]
    assert counts["input_kind"] == "art_files"
    assert counts["fcl"] == str(FCL_DIR / "print_counts.fcl")
    assert counts["fcl_exists"] is True
    # the whole point of `count`: naming a prescale filter is optional
    assert counts["parameters"]["prescale_filter"]["required"] is False
    assert counts["parameters"]["prescale_filter"]["default"] == ""
    assert counts["units"]["saved_per_gen_event"] == "events / generated event"

    ce = catalogue["approx_ce_sensitivity"]
    assert ce["input_kind"] == "root_file"
    assert ce["produced_by"] == ["edep"]          # chaining is discoverable
    # one of the two, checked when it runs: 0 means not given
    assert ce["parameters"]["sig_eff"]["required"] is False
    assert ce["parameters"]["stops_per_pot"]["required"] is False
    assert ce["parameters"]["npot"]["required"] is False
    cosmic = ce["parameters"]["cosmic_rate_per_s_per_mev"]
    assert cosmic["required"] is False
    assert cosmic["default"] == sens.COSMIC_RATE_PER_SECOND_PER_MEV
    # the normalization a result was built on is reported with it
    assert {"npot", "cosmic_rate_per_s_per_mev"} <= set(ce["metrics"])
    assert ce["units"]["npot"] == "POT"
    assert "fcl" not in ce

    mats = catalogue["stop_materials"]
    assert mats["input_kind"] == "root_file"
    assert mats["parameters"]["n_gen_events"]["required"] is True
    assert mats["parameters"]["stop_module"]["default"] == "TargetMuonFinder"
    # a root_file analysis says whether it takes data_files; art ones always do
    assert mats["takes_data_files"] is True
    assert ce["takes_data_files"] is False
    assert edep["takes_data_files"] is True


# --- parameter handling ------------------------------------------------------

def test_params_resolve_defaults_and_validate():
    spec = ANALYSES["approx_ce_sensitivity"]
    resolved = spec.resolve_params({"sig_eff": 0.25})
    assert resolved["sig_eff"] == 0.25
    assert resolved["npot"] == sens.NPOT           # default filled in


def test_missing_required_parameter_is_reported(tmp_dir):
    result = run_analysis(analysis="trigger_efficiency_ntuple",
                          data_file="/nonexistent/x.root", output_dir=tmp_dir)
    assert result.status == "error"
    assert "trigger_paths" in result.message and "missing" in result.message


def test_out_of_range_parameter_is_reported(tmp_dir):
    result = run_analysis(analysis="approx_ce_sensitivity",
                          data_file="/nonexistent/x.root", output_dir=tmp_dir,
                          parameters={"sig_eff": 5.0})
    assert result.status == "error"
    assert "sig_eff" in result.message and "maximum" in result.message


def test_unknown_parameter_is_reported(tmp_dir):
    result = run_analysis(analysis="approx_ce_sensitivity",
                          data_file="/nonexistent/x.root", output_dir=tmp_dir,
                          parameters={"sig_eff": 0.1, "wat": 1.0})
    assert result.status == "error"
    assert "wat" in result.message and "unknown" in result.message


def test_cosmic_rate_defaults_to_the_modules_assumption():
    spec = ANALYSES["approx_ce_sensitivity"]
    params = spec.resolve_params({"sig_eff": 0.1})
    assert params["cosmic_rate_per_s_per_mev"] == sens.COSMIC_RATE_PER_SECOND_PER_MEV
    # and a supplied value is what reaches the run
    params = spec.resolve_params({"sig_eff": 0.1, "cosmic_rate_per_s_per_mev": 5.0e-3})
    assert params["cosmic_rate_per_s_per_mev"] == 5.0e-3
    try:
        spec.resolve_params({"sig_eff": 0.1, "cosmic_rate_per_s_per_mev": -1.0})
    except ValueError as exc:
        assert "cosmic_rate_per_s_per_mev" in str(exc)
    else:
        raise AssertionError("a negative rate should be rejected")


def test_text_param_defaults_and_validates():
    spec = ANALYSES["muon_stop_rate"]
    # left out, it falls back to the target-stop stream
    assert (spec.resolve_params({"upstream_eff": 0.012})["prescale_filter"]
            == "TargetStopPrescaleFilter")
    assert (spec.resolve_params({"upstream_eff": 0.012,
                                 "prescale_filter": "PolyStopPrescaleFilter"})
            ["prescale_filter"] == "PolyStopPrescaleFilter")
    for bad in ("", "   ", 7):
        try:
            spec.resolve_params({"upstream_eff": 0.012, "prescale_filter": bad})
        except ValueError as exc:
            assert "prescale_filter" in str(exc)
        else:
            raise AssertionError(f"{bad!r} should not be a valid label")


def test_optional_text_param_takes_the_empty_string_as_an_answer():
    """count's prescale_filter: "" means "no such filter", not a typo.

    muon_stop_rate's same-named knob (above) still refuses it — the difference
    is ParamSpec.allow_empty, not the name.
    """
    spec = ANALYSES["count"]
    assert spec.resolve_params(None)["prescale_filter"] == ""
    assert spec.resolve_params({"prescale_filter": ""})["prescale_filter"] == ""
    assert spec.resolve_params({"prescale_filter": "  "})["prescale_filter"] == ""
    assert (spec.resolve_params({"prescale_filter": "PolyStopPrescaleFilter"})
            ["prescale_filter"] == "PolyStopPrescaleFilter")
    # allow_empty is about blank text, not about text: a number is still wrong
    try:
        spec.resolve_params({"prescale_filter": 7})
    except ValueError as exc:
        assert "prescale_filter" in str(exc)
    else:
        raise AssertionError("a non-string label should be rejected")


def test_param_spec_rejects_non_numbers():
    param = ParamSpec(name="p", description="d", default=1.0)
    try:
        param.check("not a number")
    except ValueError as exc:
        assert "must be a number" in str(exc)
        return
    raise AssertionError("a non-numeric parameter was accepted")


def test_param_spec_rejects_non_finite_numbers():
    param = ParamSpec(name="p", description="d", default=1.0)
    for bad in ("nan", "inf", float("-inf")):
        try:
            param.check(bad)
        except ValueError as exc:
            assert "finite" in str(exc)
        else:
            raise AssertionError(f"{bad!r} should have been refused")


# --- input handling (-s / -S) ------------------------------------------------

def test_single_input_uses_dash_s_and_no_filelist(tmp_dir):
    flag, arg, file_list = build_input_args(
        [Path("/data/one.art")], Path(tmp_dir), single=True
    )
    assert (flag, arg) == ("-s", "/data/one.art")
    assert file_list is None
    assert not (Path(tmp_dir) / "filelist.txt").exists()


def test_multiple_inputs_write_one_path_per_line_for_dash_S(tmp_dir):
    outdir = Path(tmp_dir)
    paths = [Path("/data/a.art"), Path("/data/b.art"), Path("/data/c.art")]
    flag, arg, file_list = build_input_args(paths, outdir, single=False)

    assert flag == "-S"
    assert file_list == outdir / "filelist.txt"
    assert arg == str(file_list)
    # mu2e wants one absolute path per line, trailing newline included.
    assert file_list.read_text() == "/data/a.art\n/data/b.art\n/data/c.art\n"


# --- where Offline comes from ------------------------------------------------

def test_musing_is_taken_as_a_name_and_a_version():
    for spelling in ("SimJob MDC2025au", "SimJob/MDC2025au"):
        env = Mu2eEnv.for_musing(spelling)
        assert env.musing == ("SimJob", "MDC2025au")
        assert env.describe() == "musing SimJob MDC2025au"
    # and the version is not optional
    for bad in ("SimJob", "SimJob MDC2025au extra"):
        try:
            Mu2eEnv.for_musing(bad)
        except EnvError as exc:
            assert "SimJob MDC2025au" in str(exc)      # shows the spelling wanted
        else:
            raise AssertionError(f"{bad!r} should not be a Musing")


def test_a_musing_sets_itself_up_and_leaves_the_fcl_to_art(tmp_dir):
    env = Mu2eEnv.for_musing("SimJob MDC2025au")
    commands = env.setup_commands(Path(tmp_dir))
    assert commands[-1] == "muse setup SimJob MDC2025au"
    assert any("setupmu2e-art.sh" in c for c in commands)
    # No directory of ours to resolve against: art finds it on FHICL_FILE_PATH,
    # so the relative path is passed through and cannot be pre-checked.
    fcl = Path("Mu2eOptAna/fcl/edep.fcl")
    assert env.base_dir(Path(tmp_dir)) is None
    assert env.resolve_fcl(fcl, Path(tmp_dir)) == fcl
    assert env.missing_fcl(fcl, Path(tmp_dir)) is None


def test_a_shipped_fcl_is_used_as_is_and_checked_under_a_musing(tmp_dir):
    env = Mu2eEnv.for_musing("SimJob MDC2025au")
    fcl = FCL_DIR / "edep.fcl"
    assert env.resolve_fcl(fcl, Path(tmp_dir)) == fcl
    assert env.missing_fcl(fcl, Path(tmp_dir)) is None
    assert "not found" in env.missing_fcl(FCL_DIR / "nope.fcl", Path(tmp_dir))


def test_a_work_area_is_set_up_in_place_and_resolves_its_own_fcl(tmp_dir):
    area = Path(tmp_dir)
    (area / "Mu2eOptAna" / "fcl").mkdir(parents=True)
    (area / "Mu2eOptAna" / "fcl" / "edep.fcl").write_text("# fcl")

    env = Mu2eEnv.for_work_area(area)
    assert env.setup_commands(area) == [
        f"cd {area}",
        "source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh",
        "muse setup",
    ]
    assert env.resolve_fcl(Path("Mu2eOptAna/fcl/edep.fcl")) == area / "Mu2eOptAna/fcl/edep.fcl"
    assert env.missing_fcl(Path("Mu2eOptAna/fcl/edep.fcl")) is None
    # a missing one is caught before any job starts
    assert "not found" in env.missing_fcl(Path("Mu2eOptAna/fcl/nope.fcl"))


def test_a_work_area_that_is_not_a_directory_is_rejected(tmp_dir):
    try:
        Mu2eEnv.for_work_area(Path(tmp_dir) / "nowhere")
    except EnvError as exc:
        assert "not a directory" in str(exc)
    else:
        raise AssertionError("a missing work area should be rejected")


def test_a_tarball_is_unpacked_once_beside_the_job_then_set_up(tmp_dir):
    tarball = Path(tmp_dir) / "code.tar"
    tarball.write_bytes(b"not really a tarball, only its path is used here")
    job = Path(tmp_dir) / "job"

    env = Mu2eEnv.for_tarball(tarball)
    assert env.unpack_dir(job) == job / "code"          # self-contained by default
    commands = env.setup_commands(job)
    unpack, enter = commands[0], commands[1]
    assert str(tarball) in unpack and "tar -xf" in unpack
    assert f"{job / 'code'}.unpacked" in unpack         # the marker that makes it once
    assert str(job / "code") in enter
    assert commands[-1] == "muse setup"

    # a shared unpack directory is used as given, and can be pre-checked
    shared = Mu2eEnv.for_tarball(tarball, code_dir=Path(tmp_dir) / "shared")
    assert shared.unpack_dir(job) == Path(tmp_dir) / "shared"
    assert shared.base_dir() == Path(tmp_dir) / "shared"
    # nothing unpacked yet, so nothing can be said about the fcl
    assert shared.missing_fcl(Path("Mu2eOptAna/fcl/edep.fcl")) is None

    sub = Mu2eEnv.for_tarball(tarball, code_dir=Path(tmp_dir) / "shared",
                              code_subdir="Code")
    assert sub.base_dir() == Path(tmp_dir) / "shared" / "Code"


def test_configured_environment_is_what_analyses_see(tmp_dir):
    before = current()
    try:
        configure(Mu2eEnv.for_musing("SimJob MDC2025au"))
        catalogue = list_analyses()
        assert "musing SimJob MDC2025au" in catalogue.message
        assert catalogue.metadata["environment"] == "musing SimJob MDC2025au"
        edep = catalogue.metadata["analyses"]["edep"]
        # shipped with the server, so the same file under any environment
        assert edep["fcl"] == str(FCL_DIR / "edep.fcl")
        assert edep["fcl_exists"] is True
    finally:
        configure(None)
        assert current().describe() == before.describe()


def test_written_root_files_reports_a_rerun_that_overwrote_its_output(tmp_dir):
    """Rerunning into the same directory must still report the job's output."""
    outdir = Path(tmp_dir)
    stale = outdir / "nts.owner.edep.Run1B.001800_00000000.root"
    stale.write_bytes(b"from an earlier run")
    before = root_snapshot(outdir)

    # nothing touched yet
    assert written_root_files(outdir, before) == []

    # the job overwrites the file it wrote last time, and adds another
    stale.write_bytes(b"from this run, a different size")
    fresh = outdir / "nts.owner.edep.Run1B.001801_00000000.root"
    fresh.write_bytes(b"new this time")
    assert written_root_files(outdir, before) == sorted([str(stale), str(fresh)])


def test_written_root_files_ignores_files_the_job_left_alone(tmp_dir):
    outdir = Path(tmp_dir)
    untouched = outdir / "someone_elses.root"
    untouched.write_bytes(b"not ours")
    before = root_snapshot(outdir)
    (outdir / "notes.txt").write_text("not a ROOT file")   # nor is this
    assert written_root_files(outdir, before) == []


def test_validate_input_paths_flags_only_bad_ones(tmp_dir):
    good = Path(tmp_dir) / "good.art"
    good.touch()
    problems = validate_input_paths([good, Path("rel.art"), Path("/nope/x.art")])
    assert len(problems) == 2
    assert any("not an absolute path" in p and "rel.art" in p for p in problems)
    assert any("does not exist" in p and "x.art" in p for p in problems)


# --- catalogued inputs: SAM definitions, metacat datasets, file names ------

# Two files and the directories SAM records for them (2026-10-12): a
# production nts file on tape, and a user dts file on scratch.
SAM_TAPE_FILE = ("nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001."
                 "001430_00000000.root",
                 "/pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/"
                 "MDC2025au_best_v1_1-001/root/a8/50")
SAM_SCRATCH_FILE = ("dts.mmackenz.CeEndpoint.MDC2025ax.001430_00000000.art",
                    "/pnfs/mu2e/scratch/datasets/usr-sim/dts/mmackenz/"
                    "CeEndpoint/MDC2025ax/art/a4/0e")


class FakeCatalogs:
    """SAM, metacat and dCache from dictionaries; records what was asked."""

    def __init__(self, sam_defs=None, metacat=None, locations=None,
                 on_disk=(), locality=None, down=False):
        self.sam_defs, self.metacat = sam_defs or {}, metacat or {}
        self.locations = locations or {}
        self.on_disk = {Path(p) for p in on_disk}
        self.localities = locality or {}
        self.down, self.asked = down, []

    def _ask(self, what):
        self.asked.append(what)
        if self.down:
            import urllib.error
            raise urllib.error.URLError("connection refused")

    def sam_definition_files(self, defname):
        self._ask(("sam", defname))
        return self.sam_defs.get(defname)

    def metacat_dataset_files(self, did):
        self._ask(("metacat", did))
        return self.metacat.get(did)

    def sam_locations(self, names):
        self._ask(("locations", tuple(names)))
        return {n: [Path(p) for p in self.locations[n]]
                for n in names if n in self.locations}

    def exists(self, path):
        return Path(path) in self.on_disk or Path(path).exists()

    def locality(self, path):
        return self.localities.get(Path(path))


def _expect_catalog_error(needles, **kwargs):
    from tools.catalog import CatalogError, resolve_inputs
    try:
        resolve_inputs(**kwargs)
    except CatalogError as exc:
        for needle in needles:
            assert needle in str(exc), (needle, str(exc))
        return str(exc)
    raise AssertionError(f"{needles}: no error")


def test_standard_paths_follow_the_layout_sam_records():
    from tools.catalog import standard_paths
    for name, directory in (SAM_TAPE_FILE, SAM_SCRATCH_FILE):
        assert Path(directory) / name in standard_paths(name), name
    # disk first, tape last: the order copies are preferred in
    paths = standard_paths(SAM_SCRATCH_FILE[0])
    assert [str(p).split("/")[3] for p in paths] == ["persistent", "scratch", "tape"]
    for name in ("my_ntuple.root", "xyz.mu2e.a.b.0.art", "ext.mu2e.a.b.0.art"):
        assert standard_paths(name) == [], name


def test_metacat_query_output_is_read_as_a_json_text_sequence():
    from tools.catalog import _json_records
    text = '\x1e{"name": "a"}\n\x1e{"name": "b"}\n'
    assert [r["name"] for r in _json_records(text)] == ["a", "b"]
    assert _json_records('[{"name": "c"}]') == [{"name": "c"}]


def test_a_sam_definition_resolves_to_its_files_first_by_name():
    from tools.catalog import resolve_inputs, standard_paths
    names = [f"dig.mu2e.X.v1.001430_0000000{i}.art" for i in (2, 0, 1)]
    disk = {n: standard_paths(n)[0] for n in names}
    fake = FakeCatalogs(sam_defs={"dig.mu2e.X.v1.art": names},
                        on_disk=disk.values())
    resolved = resolve_inputs(dataset="dig.mu2e.X.v1.art", max_files=2,
                              client=fake)
    assert resolved.paths == [disk[n] for n in sorted(names)[:2]], resolved.paths
    assert resolved.source == {"dataset": "dig.mu2e.X.v1.art", "catalog": "sam",
                               "n_files_in_dataset": 3, "locations": {"disk": 2},
                               "n_catalogued_files": 2}, resolved.source
    # found as a SAM definition, so metacat is never asked
    assert not any(kind == "metacat" for kind, _ in fake.asked), fake.asked


def test_a_dataset_not_in_sam_is_looked_up_in_metacat():
    from tools.catalog import resolve_inputs, standard_paths
    name = SAM_SCRATCH_FILE[0]
    scratch = standard_paths(name)[1]
    # a bare name goes to its owner's namespace; SAM has no record of the
    # file, so the standard layout finds it
    fake = FakeCatalogs(metacat={"mmackenz:dts.mmackenz.CeEndpoint.MDC2025ax.art": [name]},
                        on_disk=[scratch])
    for dataset in ("dts.mmackenz.CeEndpoint.MDC2025ax.art",
                    "mmackenz:dts.mmackenz.CeEndpoint.MDC2025ax.art"):
        fake.asked.clear()
        resolved = resolve_inputs(dataset=dataset, client=fake)
        assert resolved.paths == [scratch], resolved.paths
        assert resolved.source["catalog"] == "metacat"
        assert resolved.source["locations"] == {"scratch": 1}
        # a DID is a metacat name: SAM's definitions are not asked
        asked_sam = ("sam", dataset) in fake.asked
        assert asked_sam == (":" not in dataset), fake.asked


def test_a_registered_file_name_is_located_and_a_disk_copy_wins():
    from tools.catalog import resolve_inputs
    name, tape_dir = SAM_TAPE_FILE
    tape = Path(tape_dir) / name
    persistent = Path(tape_dir.replace("/tape/", "/persistent/datasets/")) / name
    fake = FakeCatalogs(locations={name: [tape, persistent]},
                        on_disk=[tape, persistent])
    absolute = "/some/where/else.art"
    # bare, as a DID, and next to a path that is passed through untouched
    resolved = resolve_inputs([name, f"mu2e:{name}", absolute], client=fake)
    assert resolved.paths == [persistent, persistent, Path(absolute)], resolved.paths
    assert resolved.source == {"locations": {"disk": 2}, "n_catalogued_files": 2}
    # with only the tape copy, it is used, being staged
    fake = FakeCatalogs(locations={name: [tape]}, on_disk=[tape],
                        locality={tape: "ONLINE_AND_NEARLINE"})
    assert resolve_inputs([name], client=fake).paths == [tape]
    # paths alone never reach a catalog
    fake = FakeCatalogs(down=True)
    assert resolve_inputs([absolute], client=fake).paths == [Path(absolute)]
    assert fake.asked == []


def test_unstaged_tape_files_are_refused_with_the_command_that_stages_them():
    from tools.catalog import standard_paths
    names = [f"mcs.mu2e.X.v1.001430_0000000{i}.art" for i in range(2)]
    tape = {n: standard_paths(n)[2] for n in names}
    locality = {tape[names[0]]: "ONLINE", tape[names[1]]: "NEARLINE"}
    fake = FakeCatalogs(sam_defs={"mcs.mu2e.X.v1.art": names},
                        metacat={"mu2e:mcs.mu2e.Y.v1.art": names},
                        on_disk=tape.values(), locality=locality)
    _expect_catalog_error(
        ["1 of 2", "not staged", "samweb prestage-dataset --defname=mcs.mu2e.X.v1.art"],
        dataset="mcs.mu2e.X.v1.art", client=fake)
    _expect_catalog_error(["mdh prestage-files mu2e:mcs.mu2e.Y.v1.art"],
                          dataset="mu2e:mcs.mu2e.Y.v1.art", client=fake)
    _expect_catalog_error([f"mdh prestage-files {names[1]}"],
                          entries=names, client=fake)
    # a locality dCache does not report is not held against the file
    fake.localities = {}
    from tools.catalog import resolve_inputs
    assert len(resolve_inputs(entries=names, client=fake).paths) == 2


def test_catalog_trusts_a_fallback_bundle_when_the_default_store_is_empty(tmp_dir):
    """A relocated conda Python (CVMFS ana 2.7.0) can have no CA at all
    outside pyenv; the catalogs then fall back to a bundle on disk."""
    import ssl
    from tools import catalog
    bundle = Path(tmp_dir) / "cacert.pem"
    loaded = []

    class EmptyStore(ssl.SSLContext):
        def cert_store_stats(self):
            return {"x509": 0, "crl": 0, "x509_ca": 0}

        def load_verify_locations(self, cafile=None, **_):
            loaded.append(cafile)

    real = (catalog.ssl.create_default_context, catalog.CA_BUNDLES, dict(catalog.os.environ))
    catalog.ssl.create_default_context = lambda: EmptyStore(ssl.PROTOCOL_TLS_CLIENT)
    catalog.CA_BUNDLES = (Path(tmp_dir) / "missing.pem", bundle)
    try:
        for name in ("SSL_CERT_FILE", "SSL_CERT_DIR"):
            catalog.os.environ.pop(name, None)
        bundle.write_text("")
        catalog.ssl_context.cache_clear()
        catalog.ssl_context()
        # the first bundle that exists (certifi, if installed, comes second)
        assert loaded and loaded[0] in (str(bundle),) + tuple(
            str(b) for b in catalog.ca_bundles()[1:2]), loaded
        # a store the caller chose through the environment is left alone
        loaded.clear()
        catalog.os.environ["SSL_CERT_FILE"] = str(bundle)
        catalog.ssl_context.cache_clear()
        catalog.ssl_context()
        assert loaded == [], loaded
    finally:
        catalog.ssl.create_default_context, catalog.CA_BUNDLES = real[:2]
        catalog.os.environ.clear()
        catalog.os.environ.update(real[2])
        catalog.ssl_context.cache_clear()


def test_catalog_says_what_to_do_about_an_unverified_certificate():
    import ssl
    import urllib.error
    from tools.catalog import network_problem
    bad = urllib.error.URLError(ssl.SSLCertVerificationError(
        "certificate verify failed: unable to get local issuer certificate"))
    assert "SSL_CERT_FILE" in network_problem(bad)
    assert network_problem(OSError("connection refused")) == "connection refused"


def test_catalogued_input_errors_say_what_is_wrong():
    from tools.catalog import MAX_DATASET_FILES, resolve_inputs, standard_paths
    _expect_catalog_error(
        ["no such dataset", "SAM definition nts.me.A.b.root",
         "metacat dataset me:nts.me.A.b.root"],
        dataset="nts.me.A.b.root", client=FakeCatalogs())
    _expect_catalog_error(["has no files"], dataset="x.mu2e.e.art",
                          client=FakeCatalogs(sam_defs={"x.mu2e.e.art": []}))
    big = [f"sim.mu2e.B.v.{i:06d}_00000000.art" for i in range(MAX_DATASET_FILES + 1)]
    fake = FakeCatalogs(sam_defs={"sim.mu2e.B.v.art": big})
    _expect_catalog_error(["max_files", str(len(big))],
                          dataset="sim.mu2e.B.v.art", client=fake)
    first = standard_paths(big[0])[0]
    fake.on_disk = {first}
    assert resolve_inputs(dataset="sim.mu2e.B.v.art", max_files=1,
                          client=fake).paths == [first]
    name = "nts.mu2e.Nowhere.v1.001430_00000000.root"
    _expect_catalog_error(["1 of 1", name, "no copy on disk", "/pnfs/mu2e/tape/"],
                          entries=[name], client=FakeCatalogs())
    _expect_catalog_error(["my_ntuple.root", "neither registered", "absolute path"],
                          entries=["my_ntuple.root"], client=FakeCatalogs())
    _expect_catalog_error(["cannot list dataset", "connection refused"],
                          dataset="a.mu2e.b.art", client=FakeCatalogs(down=True))
    _expect_catalog_error(["cannot look up", "SAM"],
                          entries=[name], client=FakeCatalogs(down=True))


def test_run_analysis_runs_over_a_dataset_it_resolved(tmp_dir):
    from tools import catalog
    names = [f"dig.mu2e.X.v1.001430_0000000{i}.art" for i in range(3)]
    copies = {n: Path(tmp_dir) / n for n in names}
    for path in copies.values():
        path.touch()
    fake_catalog = FakeCatalogs(sam_defs={"dig.mu2e.X.v1.art": names},
                                locations={n: [p] for n, p in copies.items()})
    fake_job, _ = _fake_edep_job(tmp_dir)
    seen = []

    def job(**kwargs):
        seen.append((kwargs["input_paths"], kwargs["single"]))
        return fake_job(**kwargs)

    real_client, real_job = catalog.CLIENT, edep_mod.run_mu2e_job
    catalog.CLIENT, edep_mod.run_mu2e_job = fake_catalog, job
    try:
        result = run_analysis(analysis="edep", dataset="dig.mu2e.X.v1.art",
                              max_files=2, output_dir=str(Path(tmp_dir) / "out"))
        one = run_analysis(analysis="edep", data_file=names[2],
                           output_dir=str(Path(tmp_dir) / "one"))
    finally:
        catalog.CLIENT, edep_mod.run_mu2e_job = real_client, real_job
    assert result.status == "success", result.message
    # the job saw the resolved paths, and as a file list
    assert seen[0] == ([copies[n] for n in names[:2]], False), seen
    meta = result.metadata
    assert meta["data_files"] == [str(copies[n]) for n in names[:2]]
    assert meta["inputs_from"]["dataset"] == "dig.mu2e.X.v1.art"
    assert meta["inputs_from"]["n_files_in_dataset"] == 3
    # a registered name passed as data_file is still one -s file
    assert one.status == "success", one.message
    assert seen[1] == ([copies[names[2]]], True), seen


def test_run_analysis_reports_dataset_misuse(tmp_dir):
    from tools import catalog
    names = ["nts.mu2e.S.v1.001430_00000000.root", "nts.mu2e.S.v1.001430_00000001.root"]
    for name in names:
        (Path(tmp_dir) / name).touch()
    fake = FakeCatalogs(sam_defs={"nts.mu2e.S.v1.root": names},
                        locations={n: [Path(tmp_dir) / n] for n in names})
    real = catalog.CLIENT
    catalog.CLIENT = fake
    try:
        cases = [
            (dict(analysis="edep", dataset="d.art", data_file="/a.art"),
             "not data_file and dataset"),
            (dict(analysis="edep", data_files=["/a.art"], max_files=1),
             "max_files applies only to a dataset"),
            (dict(analysis="edep", dataset="no.such.def.art"), "no such dataset"),
            (dict(analysis="approx_ce_sensitivity", dataset="nts.mu2e.S.v1.root",
                  parameters={"sig_eff": 0.1}), "max_files=1"),
        ]
        for kwargs, needle in cases:
            result = run_analysis(output_dir=tmp_dir, **kwargs)
            assert result.status == "error", kwargs
            assert needle in result.message, (needle, result.message)
    finally:
        catalog.CLIENT = real


# --- run_analysis argument validation (no job, no computation) ---------------

def test_relative_data_file_is_rejected(tmp_dir):
    result = run_analysis(analysis="edep", data_file="some/relative.art",
                          output_dir=tmp_dir)
    assert result.status == "error"
    assert "absolute" in result.message


def test_missing_data_file_is_rejected(tmp_dir):
    result = run_analysis(analysis="edep", data_file="/nonexistent/nope.art",
                          output_dir=tmp_dir)
    assert result.status == "error"
    assert "does not exist" in result.message


def test_one_bad_path_in_data_files_is_reported(tmp_dir):
    good = Path(tmp_dir) / "good.art"
    good.touch()
    result = run_analysis(analysis="edep",
                          data_files=[str(good), "/nonexistent/nope.art"],
                          output_dir=tmp_dir)
    assert result.status == "error"
    assert "does not exist" in result.message and "nope.art" in result.message
    assert "good.art" not in result.message  # the valid one isn't flagged


def test_neither_input_is_rejected_without_default_inputs(tmp_dir):
    result = run_analysis(analysis="edep", output_dir=tmp_dir)
    assert result.status == "error"
    assert "exactly one" in result.message


def test_both_inputs_at_once_are_rejected(tmp_dir):
    result = run_analysis(analysis="edep", data_file="/a.art",
                          data_files=["/b.art"], output_dir=tmp_dir)
    assert result.status == "error"
    assert "not data_file and data_files" in result.message


def test_root_file_analysis_rejects_a_file_list(tmp_dir):
    result = run_analysis(analysis="approx_ce_sensitivity",
                          data_files=["/a.root", "/b.root"], output_dir=tmp_dir,
                          parameters={"sig_eff": 0.1})
    assert result.status == "error"
    assert "single ROOT file" in result.message


def test_root_file_analysis_rejects_max_events(tmp_dir):
    result = run_analysis(analysis="approx_ce_sensitivity",
                          data_file="/a.root", output_dir=tmp_dir,
                          parameters={"sig_eff": 0.1}, max_events=10)
    assert result.status == "error"
    assert "max_events" in result.message


def test_unknown_analysis_is_rejected(tmp_dir):
    """The Literal enum from the registry must reject unknown names."""
    try:
        run_analysis(analysis="not_an_analysis", data_file="/a.art",
                     output_dir=tmp_dir)
    except Exception:
        return  # pydantic rejected it, as intended
    raise AssertionError("unknown analysis name was not rejected")


def main() -> int:
    import tempfile
    import traceback

    failures = 0
    for name, func in sorted(globals().items()):
        if not name.startswith("test_") or not callable(func):
            continue
        # A fresh directory per test — tests that write filelist.txt must not
        # be visible to the ones asserting it is absent.
        with tempfile.TemporaryDirectory() as tmp_dir:
            kwargs = {"tmp_dir": tmp_dir} if "tmp_dir" in func.__code__.co_varnames else {}
            try:
                func(**kwargs)
                print(f"ok   {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc or '(no message)'}")
            except Exception:
                failures += 1
                print(f"ERROR {name}:\n{traceback.format_exc()}")
    print(f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
