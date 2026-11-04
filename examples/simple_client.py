#!/usr/bin/env python3
"""A minimal MCP client: start the server, list the analyses, run one (or two).

What it does, in the order an agent would:

  1. connect to the server and list its tools
  2. `list_analyses`  -- the catalogue: input kinds, metrics, parameters
  3. `run_analysis`   -- `edep` over the art file you name, if you name one
  4. `run_analysis`   -- `muon_stop_rate`, if you pass --stops-file: another
                         analysis over its own input, so it takes its own file
  5. `run_analysis`   -- `stop_materials`, if you pass --stopmat-file: stops
                         per material from a stop finder's stopmat histogram
  6. `run_analysis`   -- the trigger, only if you pass --trigger:
                         `trigger_rate` and `trigger_timing` on pileup, and
                         `trigger_efficiency` on --trigger-signal-file
  7. `run_analysis`   -- `trigger_efficiency_ntuple`, if you pass
                         --trigger-ntuple: the trigger efficiency read from
                         EventNtuple trig_<path> branches, for a track
                         selection -- no mu2e job
  8. `run_analysis`   -- `edep_occupancy` once per pileup component you pass
                         with --occupancy, then `edep_occupancy_combine` over
                         their occupancy_per_microbunch.root files: the peak
                         straw, straw-row, crystal and radial calo rates of
                         the total pileup, and each component's share
  9. `run_analysis`   -- `approx_ce_sensitivity` over the nts.*.root step 3
                         wrote, which is what `produced_by` is for

Environment (the `ana` python already has the mcp SDK; no installs needed):

    source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh
    pyenv ana 2.7.0

Run it -- the client starts the server itself over stdio. The path may be
relative or use ~; it is resolved here, because the server takes only
absolute paths (it runs the job from its own working directory):

    python3 examples/simple_client.py ../dts.mmackenz.CeEndpoint.....art

With the stopping rate too, which needs a target-stop sim file of its own:

    python3 examples/simple_client.py ../dts.mmackenz.CeEndpoint.....art \
        --stops-file ../sim.mmackenz.TargetStops.....art

Stops per material, from the ntuple of a job that ran the stop finders. It
needs the generated events that file is equivalent to, and leaves out the art
file so no mu2e job runs at all:

    python3 examples/simple_client.py \
        --stopmat-file ../nts.mmackenz.mubeam.Run1Bak_local0818120248.001800_00000000.root \
        --n-gen-events 1e5 [--stop-module PolyMuonFinder]

Several such files are combined into one table; --n-gen-events is then the
total for all of them:

    python3 examples/simple_client.py --stopmat-file ../nts.*.root \
        --n-gen-events 4e5

The trigger analyses, which run only when asked for. Rate and timing run
over the server's default pileup sample (mu2e-trig-config's CI files) unless
--trigger-pileup-file names others; the efficiency needs a signal digi file:

    python3 examples/simple_client.py --trigger \
        [--trigger-signal-file /pnfs/.../dig.mu2e.CeEndpointOnSpill.....art] \
        [--trigger-paths 'cpr_TrkDe_80m70p:1, apr_TrkDe_80m70p:1'] \
        [--max-events 1000]

The trigger efficiency from EventNtuple file(s), which needs no mu2e job:
the decisions are read from the ntuple's trig_<path> branches, for the events
with a track passing the server's default selection (a converged downstream
e- fit, p > 80 MeV/c at the tracker front, >= 15 active hits, chi2/dof < 5)
or --ntuple-selection. --trigger-paths applies, without the prescales:

    python3 examples/simple_client.py \
        --trigger-ntuple /pnfs/mu2e/tape/phy-nts/nts/mu2e/CeMLeadingLogMix1BB/MDC2025au_best_v1_1-001/root/a8/50/nts.mu2e.CeMLeadingLogMix1BB.MDC2025au_best_v1_1-001.001430_00000000.root \
        [--ntuple-selection 'status == 1 and pdg == 11 and downstream and 100 < p_front < 110']

Pileup occupancy and peak channel rates, one --occupancy per component:
LABEL[:EVENTS_PER_MICROBUNCH] then its dts file(s). The Run 1A events per
microbunch of the MDC2025ac components are built in (see RUN1A_EVENTS_PER_UB),
so a known label needs no number. With two or more components the totals are
combined channel by channel. It needs an Offline whose EdepAna has the
occupancy histograms (Offline PR #2097, merged 2026-11-03), so pass --musing
with a Musing released after it, or --work-area on Offline main. --max-events is fine here (the averages use the events analyzed), but a
peak straw from few events is biased high; the row and radial peaks less so:

    python3 examples/simple_client.py --musing 'SimJob <Musing with #2097>' \
        --occupancy MuStopPileupCat /pnfs/.../dts.mu2e.MuStopPileupCat.MDC2025ac.....art \
        --occupancy MuBeamFlashCat  /pnfs/.../dts.mu2e.MuBeamFlashCat.MDC2025ac.....art \
        --occupancy EleBeamFlashCat /pnfs/.../dts.mu2e.EleBeamFlashCat.MDC2025ac.....art \
        [--max-events 2000]

A quick smoke test that does not wait for a full mu2e job (per-gen-event
metrics are meaningless with --max-events, so skip the chained sensitivity):

    python3 examples/simple_client.py path/to/some.art \
        --max-events 200 --no-chain

Against a server you started by hand, which is the same code over HTTP:

    python3 -m analysis_mcp_server --transport streamable-http --port 8000 \
        --work-area /exp/mu2e/app/users/mmackenz/mu2eopt &
    python3 examples/simple_client.py path/to/some.art \
        --url http://127.0.0.1:8000/mcp

Where Offline comes from is the server's own setting, and this client starts
the server with --work-area pointing at the local muse area above. Pass
--musing 'SimJob MDC2025au' or --code-tarball <file> to set up a published
Musing or a code tarball instead; with --url the running server's setting
applies and these are ignored.

There is no default input file on purpose: `edep` runs over whatever art file
you point it at, and `approx_ce_sensitivity` only means anything for a CE
signal sample (it needs events leaving >10 MeV in the calorimeter).
"""

import argparse
import asyncio
import json
import sys
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

REPO = Path(__file__).resolve().parent.parent

# The muse work area these examples were written against, backed by SimJob
# MDC2025ay (Offline v13_39_00, the first with EdepAna, which `edep` needs).
# --musing or --code-tarball point the server somewhere else instead.
WORK_AREA = "/exp/mu2e/app/users/mmackenz/mu2eopt"

# The trigger paths --trigger runs by default: the calorimeter-seeded and
# agnostic track paths. The menu's TPR and MPR paths are left out: they are
# not meant for the real trigger.
DEFAULT_TRIGGER_PATHS = "cpr_TrkDe_80m70p:1, apr_TrkDe_80m70p:1"

# dts events per microbunch of each MDC2025ac pileup component at Run 1A
# intensity (1.58e7 POT/microbunch, SimEfficiencies2 Sim_best v1_1), as in
# edep_occupancy's own parameter help. --occupancy LABEL:VALUE overrides.
RUN1A_EVENTS_PER_UB = {
    "MuStopPileupCat": 226.57,
    "MuBeamFlashCat": 173.25,
    "EleBeamFlashCat": 92.78,
    "EarlyMuBeamFlashCat": 276.6,
    "EarlyEleBeamFlashCat": 3746.4,
}

# The peak metrics printed for the occupancy runs, rather than all ~60.
OCCUPANCY_METRICS = (
    "n_events_analyzed", "events_per_microbunch",
    "trk_hits_per_ub", "calo_hits_per_ub",
    "trk_mean_straw_hit_rate_khz",
    "trk_peak_straw_hit_rate_khz", "trk_peak_straw_hit_rate_khz_err",
    "trk_peak_straw_plane", "trk_peak_straw_panel", "trk_peak_straw_straw",
    "trk_peak_row_hit_rate_khz", "trk_peak_row_hit_rate_khz_err",
    "trk_peak_row_plane", "trk_peak_row_straw", "trk_peak_row_radius_mm",
    "calo_peak_crystal_hit_rate_khz", "calo_peak_crystal_hit_rate_khz_err",
    "calo_peak_crystal_id", "calo_peak_crystal_disk",
    "calo_disk0_peak_radial_hit_rate_khz", "calo_disk0_peak_radial_hit_r_mm",
    "calo_disk1_peak_radial_hit_rate_khz", "calo_disk1_peak_radial_hit_r_mm",
    "n_components",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drive the Mu2e analysis MCP server over one art file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data_file", nargs="?",
        help="Path to the input art file for edep (and, chained from it, "
             "approx_ce_sensitivity). Optional when --stops-file or "
             "--stopmat-file names something else to run. Every path given "
             "here, relative or with ~, is resolved against the current "
             "directory before the call: the server only takes absolute "
             "paths, since it runs the job from its own working directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO / "output" / "example"),
        help="Where the runs write job output, logs and figures.",
    )
    parser.add_argument(
        "--url",
        help="Connect to an already-running streamable-http server "
             "(e.g. http://127.0.0.1:8000/mcp). Without it the client starts "
             "the server itself over stdio. That server carries its own "
             "Offline setup, so the code options below do not apply to it.",
    )
    code = parser.add_mutually_exclusive_group()
    code.add_argument(
        "--work-area", default=WORK_AREA,
        help="Muse work area the server sets up in.",
    )
    code.add_argument(
        "--musing", metavar="'NAME VERSION'",
        help="Set up a published Musing instead, e.g. 'SimJob MDC2025ay'. "
             "edep needs EdepAna, in Offline from v13_39_00 (MDC2025ay).",
    )
    code.add_argument(
        "--code-tarball",
        help="Set up an unpacked code tarball instead.",
    )
    parser.add_argument(
        "--code-dir",
        help="Where --code-tarball is unpacked. Worth setting to reuse one "
             "unpacking across runs; the default unpacks beside each job.",
    )
    parser.add_argument(
        "--code-subdir",
        help="Directory inside the unpacked tarball to run muse setup in.",
    )
    parser.add_argument(
        "--max-events", type=int,
        help="Cap events in the mu2e job for a quick check. Any 'per gen "
             "event' metric is meaningless when this is set.",
    )
    parser.add_argument(
        "--sig-eff", type=float, default=2.5e-4,
        help="Signal efficiency handed to approx_ce_sensitivity.",
    )
    parser.add_argument(
        "--stops-file",
        help="A target-stop sim file (sim.*.TargetStops.*.art). Given one, "
             "the client also runs muon_stop_rate over it.",
    )
    parser.add_argument(
        "--upstream-eff", type=float, default=0.012,
        help="Efficiency of everything upstream of the stopping target "
             "(generated events per POT), handed to muon_stop_rate.",
    )
    parser.add_argument(
        "--prescale-filter",
        help="Prescale filter label for muon_stop_rate. Left out, the server "
             "falls back to its default, the target-stop stream.",
    )
    parser.add_argument(
        "--stopmat-file", nargs="+", metavar="FILE",
        help="ROOT file(s) (nts.*.root) of a job that ran the stop finders. "
             "Given any, the client also runs stop_materials, combining "
             "several into one table.",
    )
    parser.add_argument(
        "--n-gen-events", type=float,
        help="Generated events the --stopmat-file(s) are equivalent to, in "
             "total, handed to stop_materials. Required with --stopmat-file.",
    )
    parser.add_argument(
        "--stop-module",
        help="Stop finder whose <module>/stopmat to read, for stop_materials. "
             "Left out, the server falls back to its default, "
             "TargetMuonFinder.",
    )
    parser.add_argument(
        "--trigger", action="store_true",
        help="Also run the trigger analyses: trigger_rate and trigger_timing "
             "on pileup, and trigger_efficiency if --trigger-signal-file is "
             "given. Off by default; each is a real mu2e job.",
    )
    parser.add_argument(
        "--trigger-paths", default=DEFAULT_TRIGGER_PATHS,
        help="Trigger paths and prescales for --trigger, 'path:prescale' "
             "entries separated by commas.",
    )
    parser.add_argument(
        "--batch-mode", choices=("1BB", "2BB"), default="1BB",
        help="Proton batches per cycle for trigger_rate, which sets the "
             "spill duty factor (1BB 0.322, 2BB 0.246).",
    )
    parser.add_argument(
        "--trigger-signal-file", nargs="+", metavar="FILE",
        help="Signal digi art file(s) for trigger_efficiency, with --trigger.",
    )
    parser.add_argument(
        "--trigger-pileup-file", nargs="+", metavar="FILE",
        help="Pileup art file(s) for trigger_rate and trigger_timing. Left "
             "out, the server uses its default, mu2e-trig-config's CI files.",
    )
    parser.add_argument(
        "--trigger-ntuple", nargs="+", metavar="FILE",
        help="EventNtuple file(s) with trig_<path> branches. Given any, the "
             "client also runs trigger_efficiency_ntuple over them for "
             "--trigger-paths (prescales dropped), combining several files.",
    )
    parser.add_argument(
        "--ntuple-selection",
        help="Track selection for --trigger-ntuple. Left out, the server's "
             "default applies.",
    )
    parser.add_argument(
        "--occupancy", nargs="+", action="append",
        metavar="LABEL[:EVENTS_PER_UB] FILE",
        help="A pileup component for edep_occupancy: its label, optionally "
             "':' and its dts events per microbunch, then its dts art "
             "file(s). Repeat once per component; two or more are then "
             "combined with edep_occupancy_combine. Labels in "
             f"{', '.join(RUN1A_EVENTS_PER_UB)} default to their Run 1A "
             "events per microbunch.",
    )
    parser.add_argument(
        "--timeout-s", type=int, default=1800,
        help="Kill the mu2e job after this many seconds.",
    )
    parser.add_argument(
        "--no-chain", action="store_true",
        help="Stop after edep instead of feeding its ROOT file to "
             "approx_ce_sensitivity.",
    )
    args = parser.parse_args()
    if not (args.data_file or args.stops_file or args.stopmat_file
            or args.trigger or args.trigger_ntuple or args.occupancy):
        parser.error("name something to run: an art file for edep, "
                     "--stops-file, --stopmat-file, --trigger, "
                     "--trigger-ntuple or --occupancy")
    components = []
    for entry in args.occupancy or []:
        if len(entry) < 2:
            parser.error(f"--occupancy {entry[0]}: give the label, then at "
                         "least one dts file")
        label, _, value = entry[0].partition(":")
        if value:
            try:
                per_ub = float(value)
            except ValueError:
                parser.error(f"--occupancy {entry[0]}: '{value}' is not a number")
        elif label in RUN1A_EVENTS_PER_UB:
            per_ub = RUN1A_EVENTS_PER_UB[label]
        else:
            parser.error(f"--occupancy {label}: not a known component, so "
                         f"give its events per microbunch as {label}:VALUE")
        components.append((label, per_ub, entry[1:]))
    if len({label for label, _, _ in components}) != len(components):
        parser.error("--occupancy labels must differ: each names one component")
    args.occupancy = components
    if args.ntuple_selection is not None and not args.trigger_ntuple:
        parser.error("--ntuple-selection needs --trigger-ntuple")
    if (args.trigger_signal_file or args.trigger_pileup_file) and not args.trigger:
        parser.error("--trigger-signal-file and --trigger-pileup-file need "
                     "--trigger")
    if args.stopmat_file and args.n_gen_events is None:
        parser.error("--stopmat-file needs --n-gen-events")
    return args


def payload(result: Any) -> dict[str, Any]:
    """The ArtifactResult a tool returned, as a plain dict.

    FastMCP sends the model back as structured content and, for older
    clients, as a JSON text block; accept either.
    """
    if getattr(result, "structuredContent", None):
        return result.structuredContent
    for block in result.content:
        if getattr(block, "type", None) == "text":
            return json.loads(block.text)
    raise RuntimeError(f"tool returned nothing usable: {result!r}")


def show_catalogue(analyses: dict[str, Any]) -> None:
    for name, entry in analyses.items():
        print(f"\n  {name}  [{entry['input_kind']}]")
        print(f"    {entry['description']}")
        if entry.get("produced_by"):
            print(f"    input comes from: {', '.join(entry['produced_by'])}")
        units = entry.get("units", {})
        metrics = ", ".join(
            f"{m} ({units[m]})" if m in units else m for m in entry["metrics"]
        )
        print(f"    metrics: {metrics}")
        for param, info in entry.get("parameters", {}).items():
            need = "required" if info["required"] else f"default {info['default']}"
            print(f"    param {param} [{need}]: {info['description']}")


def show_result(result: dict[str, Any], metrics: list[str]) -> None:
    """Print the tool's message, the metrics it advertises, and its files."""
    meta = result["metadata"]
    print(f"  status : {result['status']}")
    print(f"  message: {result['message']}")
    if result["status"] != "success":
        tail = meta.get("stdout_tail")
        if tail:
            print("  --- last lines of the job log ---")
            for line in tail.splitlines():
                print(f"  | {line}")
        print(f"  log    : {meta.get('log_path', '(none)')}")
        return
    for name in metrics:
        if name in meta:
            print(f"  {name:38s} {meta[name]:g}")
    for path in result["files"]:
        print(f"  wrote  : {path}")
    print(f"  log    : {meta.get('log_path', '(none)')}")


def files_arg(paths: list[Path]) -> dict[str, Any]:
    """One file as data_file, several as data_files, as for any analysis."""
    if len(paths) == 1:
        return {"data_file": str(paths[0])}
    return {"data_files": [str(p) for p in paths]}


def show_trigger_paths(result: dict[str, Any], value: str) -> None:
    """The per-path block the efficiency and rate return in metadata.paths."""
    if result["status"] != "success":
        return
    print(f"  {'path':<28} {'prescale':>8} {'passed':>7} {value:>24}")
    for name, row in result["metadata"]["paths"].items():
        # an ntuple's decisions were made already: no prescale to show
        prescale = row.get("prescale", "-")
        print(f"  {name:<28} {prescale:>8} {row['passed']:>7} "
              f"{row[value]:>12.4g} +- {row[value + '_err']:<9.2g}")


def code_args(args: argparse.Namespace) -> list[str]:
    """The server flags saying where Offline comes from: one of the three."""
    if args.musing:
        return ["--musing", args.musing]
    if args.code_tarball:
        return ["--code-tarball", args.code_tarball,
                *(["--code-dir", args.code_dir] if args.code_dir else []),
                *(["--code-subdir", args.code_subdir] if args.code_subdir else [])]
    return ["--work-area", args.work_area]


async def connect(stack: AsyncExitStack, args: argparse.Namespace) -> ClientSession:
    """Open a session, either against `--url` or a server we start ourselves."""
    if args.url:
        print(f"Connecting to {args.url} ...")
        read, write, _ = await stack.enter_async_context(
            streamablehttp_client(args.url)
        )
    else:
        server_args = ["-m", "analysis_mcp_server", "--transport", "stdio",
                       *code_args(args)]
        print(f"Starting the server over stdio: {' '.join(code_args(args))} ...")
        server = StdioServerParameters(
            command=sys.executable,
            args=server_args,
            cwd=str(REPO),
        )
        read, write = await stack.enter_async_context(stdio_client(server))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return session


async def main() -> int:
    args = parse_args()
    outdir = Path(args.output_dir).expanduser().resolve()

    # The tool rejects relative paths, and rightly so -- the mu2e job runs
    # in output_dir, not here. Resolve ours so a relative argument works.
    def absolute(path: str | None) -> Path | None:
        return Path(path).expanduser().resolve() if path else None

    data_file = absolute(args.data_file)
    stops_file = absolute(args.stops_file)
    stopmat_files = [absolute(f) for f in args.stopmat_file or []]
    signal_files = [absolute(f) for f in args.trigger_signal_file or []]
    pileup_files = [absolute(f) for f in args.trigger_pileup_file or []]
    ntuple_files = [absolute(f) for f in args.trigger_ntuple or []]
    occupancy = [(label, per_ub, [absolute(f) for f in files])
                 for label, per_ub, files in args.occupancy]
    for path in (data_file, stops_file, *stopmat_files, *signal_files,
                 *pileup_files, *ntuple_files,
                 *(f for _, _, files in occupancy for f in files)):
        if path is not None and not path.exists():
            print(f"No such input file: {path}", file=sys.stderr)
            return 2

    async with AsyncExitStack() as stack:
        session = await connect(stack, args)

        tools = await session.list_tools()
        print(f"Tools: {', '.join(tool.name for tool in tools.tools)}")

        # 1. What can this server do?
        print("\n=== list_analyses ===")
        catalogue = payload(await session.call_tool("list_analyses", {}))
        print(catalogue["message"])
        print(f"Offline environment: {catalogue['metadata']['environment']}")
        analyses = catalogue["metadata"]["analyses"]
        show_catalogue(analyses)

        # 2. Energy deposition over the art file.
        edep = None
        if data_file is not None:
            print(f"\n=== run_analysis: edep on {data_file} ===")
            print("(a real mu2e job -- this can take a while)")
            edep = payload(await session.call_tool("run_analysis", {
                "analysis": "edep",
                "data_file": str(data_file),
                "output_dir": str(outdir / "edep"),
                "timeout_s": args.timeout_s,
                **({"max_events": args.max_events} if args.max_events else {}),
            }))
            show_result(edep, analyses["edep"]["metrics"])
            if edep["status"] != "success":
                return 1

        # 3. A second analysis, over its own input. Its prescale_filter
        #    parameter is optional, so it is passed only when you set one and
        #    the server's default -- the target-stop stream -- runs otherwise.
        if stops_file is not None:
            print(f"\n=== run_analysis: muon_stop_rate on {stops_file} ===")
            parameters: dict[str, Any] = {"upstream_eff": args.upstream_eff}
            if args.prescale_filter:
                parameters["prescale_filter"] = args.prescale_filter
            stops = payload(await session.call_tool("run_analysis", {
                "analysis": "muon_stop_rate",
                "data_file": str(stops_file),
                "output_dir": str(outdir / "muon_stop_rate"),
                "parameters": parameters,
                "timeout_s": args.timeout_s,
            }))
            show_result(stops, analyses["muon_stop_rate"]["metrics"])
            if stops["status"] == "success":
                meta = stops["metadata"]
                print(f"  prescale filter used: {meta['prescale_filter']} "
                      f"(of {', '.join(sorted(meta['prescale_filters']))})")

        # 4. Stops per material. The ntuple has no generated-event count of
        #    its own, so n_gen_events always comes from you; stop_module is
        #    optional and passed only when set.
        failed = False
        if stopmat_files:
            names = (str(stopmat_files[0]) if len(stopmat_files) == 1
                     else f"{len(stopmat_files)} files")
            print(f"\n=== run_analysis: stop_materials on {names} ===")
            parameters = {"n_gen_events": args.n_gen_events}
            if args.stop_module:
                parameters["stop_module"] = args.stop_module
            mats = payload(await session.call_tool("run_analysis", {
                "analysis": "stop_materials",
                # one file as data_file, several as data_files, as for any
                # analysis; the server combines the list into one table
                **({"data_file": str(stopmat_files[0])} if len(stopmat_files) == 1
                   else {"data_files": [str(f) for f in stopmat_files]}),
                "output_dir": str(outdir / "stop_materials"),
                "parameters": parameters,
            }))
            show_result(mats, analyses["stop_materials"]["metrics"])
            if mats["status"] == "success":
                if len(stopmat_files) > 1:
                    for entry in mats["metadata"]["per_file"]:
                        print(f"  {entry['n_stops']:>8g} stops in {entry['file']}")
                print(f"  {'material':<24} {'stops':>8} {'stops / gen event':>26}"
                      f" {'fraction':>9}")
                for row in mats["metadata"]["materials"]:
                    print(f"  {row['material']:<24} {row['stops']:>8g} "
                          f"{row['stops_per_gen_event']:>12.4e} +- "
                          f"{row['stops_per_gen_event_err']:<9.2e} "
                          f"{row['fraction']:>9.4f}")
            else:
                failed = True

        # 5. The trigger, only when asked for: the same paths and prescales
        #    through all three. Rate and timing take no input file unless you
        #    name one -- the server then runs its default pileup sample.
        if args.trigger:
            common: dict[str, Any] = {
                "parameters": {"trigger_paths": args.trigger_paths},
                "timeout_s": args.timeout_s,
                **({"max_events": args.max_events} if args.max_events else {}),
            }
            pileup = files_arg(pileup_files) if pileup_files else {}
            where = (f"{len(pileup_files)} pileup file(s)" if pileup_files
                     else "the server's default pileup sample")
            if signal_files:
                print(f"\n=== run_analysis: trigger_efficiency on "
                      f"{len(signal_files)} signal file(s) ===")
                eff = payload(await session.call_tool("run_analysis", {
                    "analysis": "trigger_efficiency", **files_arg(signal_files),
                    "output_dir": str(outdir / "trigger_efficiency"), **common,
                }))
                show_result(eff, analyses["trigger_efficiency"]["metrics"])
                show_trigger_paths(eff, "efficiency")
                failed |= eff["status"] != "success"

            print(f"\n=== run_analysis: trigger_rate on {where} ===")
            rate = payload(await session.call_tool("run_analysis", {
                "analysis": "trigger_rate", **pileup,
                "output_dir": str(outdir / "trigger_rate"),
                **common,
                "parameters": {**common["parameters"],
                               "batch_mode": args.batch_mode},
            }))
            show_result(rate, analyses["trigger_rate"]["metrics"])
            show_trigger_paths(rate, "rate_hz")
            failed |= rate["status"] != "success"

            print(f"\n=== run_analysis: trigger_timing on {where} ===")
            timing = payload(await session.call_tool("run_analysis", {
                "analysis": "trigger_timing", **pileup,
                "output_dir": str(outdir / "trigger_timing"), **common,
            }))
            show_result(timing, analyses["trigger_timing"]["metrics"])
            if timing["status"] == "success":
                meta = timing["metadata"]
                print(f"  left out: first {meta['skipped_events']} event(s); "
                      f"data fetch {meta['fetch_time_mean_ms']:.3g} ms/event "
                      f"(whole event {meta['full_event_time_mean_ms']:.3g} ms)")
                for path, ms in meta["path_time_ms"].items():
                    print(f"  {path:<28} {ms:9.3g} ms/event")
                # Every module in the order it ran. n_seen falls along a path
                # as its filters (and prescale) stop events; a module shared
                # by paths is charged to the first one that ran it.
                print(f"\n  {'path:module':<52} {'n_seen':>7} {'ms/run':>8} "
                      f"{'ms/event':>9}")
                for row in meta["modules"]:
                    name = f"{row['path']}:{row['label']}"
                    note = "" if row["counted"] else "  (fetch, not counted)"
                    print(f"  {name:<52} {row['n_seen']:>7} "
                          f"{row['mean_ms']:>8.3g} {row['ms_per_event']:>9.3g}{note}")
            failed |= timing["status"] != "success"

        # 6. The trigger efficiency from EventNtuples: the decisions the
        #    production trigger made, read from trig_<path> branches, so the
        #    paths are names only -- prescales do not apply after the fact.
        #    The selection is passed only when set, else the server's default.
        if ntuple_files:
            names = ", ".join(p.split(":")[0].strip()
                              for p in args.trigger_paths.split(","))
            what = (str(ntuple_files[0]) if len(ntuple_files) == 1
                    else f"{len(ntuple_files)} files")
            print(f"\n=== run_analysis: trigger_efficiency_ntuple on {what} ===")
            parameters = {"trigger_paths": names}
            if args.ntuple_selection is not None:
                parameters["selection"] = args.ntuple_selection
            ntuple_eff = payload(await session.call_tool("run_analysis", {
                "analysis": "trigger_efficiency_ntuple",
                **files_arg(ntuple_files),
                "output_dir": str(outdir / "trigger_efficiency_ntuple"),
                "parameters": parameters,
            }))
            show_result(ntuple_eff, analyses["trigger_efficiency_ntuple"]["metrics"])
            if ntuple_eff["status"] == "success":
                print(f"  selection: {ntuple_eff['metadata']['selection']}")
            show_trigger_paths(ntuple_eff, "efficiency")
            failed |= ntuple_eff["status"] != "success"

        # 7. Pileup occupancy: one edep_occupancy per component, each scaled
        #    to a microbunch by its own events_per_microbunch, then the
        #    combine adds their per-channel histograms. The peak of the sum
        #    is not the sum of the peaks, so the metrics are never added.
        scaled_files: list[str] = []
        for label, per_ub, files in occupancy:
            print(f"\n=== run_analysis: edep_occupancy on {label} "
                  f"({len(files)} file(s), {per_ub:g} events / microbunch) ===")
            occ = payload(await session.call_tool("run_analysis", {
                "analysis": "edep_occupancy", **files_arg(files),
                "output_dir": str(outdir / "edep_occupancy" / label),
                "parameters": {"events_per_microbunch": per_ub, "label": label},
                "timeout_s": args.timeout_s,
                **({"max_events": args.max_events} if args.max_events else {}),
            }))
            show_result(occ, OCCUPANCY_METRICS)
            for warning in occ["metadata"].get("warnings", []):
                print(f"  warning: {warning}")
            if occ["status"] != "success":
                failed = True
                continue
            scaled_files += [f for f in occ["files"]
                             if Path(f).name == "occupancy_per_microbunch.root"]

        if len(scaled_files) > 1:
            print(f"\n=== run_analysis: edep_occupancy_combine over "
                  f"{len(scaled_files)} components ===")
            total = payload(await session.call_tool("run_analysis", {
                "analysis": "edep_occupancy_combine",
                "data_files": scaled_files,
                "output_dir": str(outdir / "edep_occupancy" / "total"),
            }))
            show_result(total, OCCUPANCY_METRICS)
            if total["status"] == "success":
                # each component's rate in the channel where the SUM peaks
                shares = (("trk_peak_straw_hit", "straw"),
                          ("trk_peak_row_hit", "row"),
                          ("calo_peak_crystal_hit", "crystal"))
                print(f"\n  {'component':<22}" + "".join(
                    f" {name + ' kHz (share)':>24}" for _, name in shares))
                for label, row in total["metadata"]["components"].items():
                    print(f"  {label:<22}" + "".join(
                        f" {row[tag + '_rate_khz']:>14.4g} ({row[tag + '_share']:5.1%})"
                        for tag, _ in shares))
            failed |= total["status"] != "success"

        if edep is None or args.no_chain:
            return 1 if failed else 0

        # 8. Chain: the nts.*.root edep wrote is what the sensitivity reads.
        ntuples = [f for f in edep["files"] if Path(f).name.startswith("nts.")]
        if not ntuples:
            print("\nNo nts.*.root in edep's output, nothing to chain.")
            return 1

        print(f"\n=== run_analysis: approx_ce_sensitivity on {ntuples[0]} ===")
        sens = payload(await session.call_tool("run_analysis", {
            "analysis": "approx_ce_sensitivity",
            "data_file": ntuples[0],
            "output_dir": str(outdir / "sensitivity"),
            "parameters": {"sig_eff": args.sig_eff},
        }))
        show_result(sens, analyses["approx_ce_sensitivity"]["metrics"])
        return 0 if sens["status"] == "success" and not failed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
