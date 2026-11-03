"""The MCP tool functions: discover analyses, then run one.

Plain Python — no MCP imports. The type hints, Field constraints, and
docstrings below become the MCP tool schema that agents see. Both tools are
driven by `registry.ANALYSES`, so adding an analysis there extends the
`analysis` enum and the list_analyses output automatically.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, validate_call

from .catalog import CatalogError, resolve_inputs
from .mu2e_env import current as current_env
from .mu2e_job import validate_input_paths
from .registry import ANALYSES, ANALYSIS_NAMES
from .spec import ArtifactResult, RunContext

# Built from the registry, so the schema's enum grows with it. Literal accepts
# a tuple of values at runtime, which is what lets this stay dynamic.
AnalysisName = Literal[ANALYSIS_NAMES]  # type: ignore[valid-type]


@validate_call
def list_analyses() -> ArtifactResult:
    """List the analyses run_analysis can run, with their inputs and metrics.

    Use this tool first to discover valid `analysis` names, what each measures,
    which metric names it reports (and their units), which parameters it takes,
    and what its input file must be. It also names the Offline environment the
    server runs mu2e jobs in — a muse work area, a Musing, or a code tarball —
    which is fixed when the server starts. `input_kind` says what to feed it:

      "art_files"  mu2e art file(s) — pass data_file, data_files or a
                   dataset, or none where `default_inputs` says what is used
                   instead
      "root_file"  a ROOT file, written by an earlier analysis (see
                   `produced_by`) or by a production job (see
                   `input_hint`) — pass data_file, or data_files where
                   `takes_data_files` is true
      "parameters" no file at all — pass only `parameters`, usually
                   earlier analyses' metrics (see `produced_by`)

    Chaining: an analysis whose `produced_by` names another should be given a
    ROOT file from that one's `files` output.

    Comparisons: an analysis with `takes_reference` compares its input (the
    new sample) with a baseline passed as reference_file, reference_files
    or reference_dataset; with `reference_optional` too, the baseline may be
    left out.
    """
    env = current_env()
    catalogue = {name: spec.describe() for name, spec in sorted(ANALYSES.items())}
    missing = [
        name for name, entry in catalogue.items()
        if entry.get("fcl_exists") is False
    ]
    message = (f"{len(catalogue)} analyses available: {', '.join(catalogue)}. "
               f"mu2e jobs run against {env.describe()}.")
    if missing:
        message += f" WARNING: fcl file missing for {', '.join(missing)}."
    return ArtifactResult(
        status="success",
        files=[],
        message=message,
        metadata={"analyses": catalogue, "environment": env.describe()},
    )


@validate_call
def run_analysis(
    analysis: AnalysisName,
    output_dir: Annotated[str, Field(min_length=1)],
    data_file: Annotated[str | None, Field(min_length=1)] = None,
    data_files: Annotated[list[str] | None, Field(min_length=1)] = None,
    dataset: Annotated[str | None, Field(min_length=1)] = None,
    max_files: Annotated[int | None, Field(ge=1)] = None,
    parameters: dict[str, Any] | None = None,
    max_events: Annotated[int | None, Field(ge=1)] = None,
    timeout_s: Annotated[int, Field(ge=1, le=7200)] = 900,
    # Keyword-only and last, so a call passing parameters, max_events or
    # timeout_s by position binds them as it did before these were added.
    *,
    reference_file: Annotated[str | None, Field(min_length=1)] = None,
    reference_files: Annotated[list[str] | None, Field(min_length=1)] = None,
    reference_dataset: Annotated[str | None, Field(min_length=1)] = None,
) -> ArtifactResult:
    """Run one analysis over its input file(s) and return its metrics.

    Call `list_analyses` first if you are unsure of the `analysis` name, the
    parameters it needs, or which metrics it reports; the metrics land in
    `metadata` under the names listed there.

    What happens depends on the analysis' `input_kind`:

    * "art_files" runs a real mu2e job (`mu2e -c <fcl> -s <file>`, or `-S` with
      a generated file list for several inputs). Expect seconds to many
      minutes. Metrics cover the whole input set as ONE job, not one result per
      file — run the tool once per file if you need per-file numbers.
    * "root_file" runs a Python computation over a ROOT file, which is fast.
      Feed it a path from the producing analysis' `files` output, or the
      production file `input_hint` describes. One whose `takes_data_files`
      is true also takes several, and combines them into one result.

    Args:
        analysis: Which analysis to run (see list_analyses), e.g. "edep".
        output_dir: Directory the analysis runs in and writes to — job output,
            logs, figures. Created if missing, and safe to reuse: a rerun
            overwrites the previous run's output rather than failing. Use a
            fresh directory per run only when you want the outputs kept apart.
        data_file: One input file: an absolute path, or the name of a file
            registered in SAM or metacat (e.g. "nts.mu2e.CeMLeadingLogMix1BB.
            MDC2025au_best_v1_1-001.001430_00000000.root", or as a DID
            "mu2e:<name>"), which is located in dCache for you. Pass exactly
            one of data_file, data_files or dataset — or none, for an
            analysis that lists `default_inputs` (e.g. the trigger rate and
            timing default to mu2e-trig-config's pileup sample).
        data_files: Several input files, each an absolute path or a
            registered file name as for data_file, analyzed together into
            one result: one mu2e job for "art_files" analyses, combined
            histograms for a "root_file" analysis that takes them (see
            `takes_data_files` in list_analyses).
        dataset: A catalogued dataset whose files are the input, located in
            dCache: a SAM definition name (e.g. "nts.mu2e.
            CeMLeadingLogMix1BB.MDC2025au_best_v1_1.root") or a metacat
            dataset DID ("<namespace>:<name>"). A bare name not defined in
            SAM is looked up in metacat under its owner's namespace. Disk
            copies are preferred to tape, and a run whose tape files are not
            staged is refused with the command that stages them. The files
            run together as with data_files.
        max_files: With dataset or reference_dataset only: use the first
            this many files of each, by name. Needed for a dataset over 2000
            files; useful for a quick check.
        parameters: Analysis-specific physics knobs, e.g.
            {"sig_eff": 2.5e-4} for approx_ce_sensitivity. list_analyses reports
            each analysis' parameters, defaults, and which are required.
        max_events: Process at most this many events (mu2e --nevts). Only for
            "art_files" analyses; useful for a quick check before a full run.
            CAUTION: generated-event counts come from the input's subrun
            bookkeeping and cover the whole file either way, so any
            "per gen event" metric is meaningless when this is set.
        timeout_s: Kill the job if it runs longer than this many seconds.
            Raise it when passing many files.
        reference_file: For an analysis with `takes_reference` (see
            list_analyses) only, and required there unless it also lists
            `reference_optional`: the baseline the input
            is compared with, e.g. the ntuple before a code change when
            data_file is the one after. Given as for data_file.
        reference_files: Several baseline files, as for data_files.
        reference_dataset: A catalogued baseline dataset, as for dataset.
    """
    spec = ANALYSES[analysis]

    def fail(message: str, **extra: Any) -> ArtifactResult:
        return ArtifactResult(
            status="error", files=[], message=f"{analysis}: {message}",
            metadata={"analysis": analysis, **extra},
        )

    given = {name: value for name, value in (("data_file", data_file),
                                             ("data_files", data_files),
                                             ("dataset", dataset))
             if value is not None}
    if len(given) > 1:
        return fail(f"pass one of data_file (one input file), data_files (a "
                    f"list of input files) or dataset, not {' and '.join(given)}.",
                    **given)
    reference = {name: value for name, value in (
        ("reference_file", reference_file), ("reference_files", reference_files),
        ("reference_dataset", reference_dataset)) if value is not None}
    if reference and not spec.takes_reference:
        return fail(f"compares nothing, so {' and '.join(reference)} does not "
                    "apply: only an analysis whose takes_reference is true (see "
                    "list_analyses) takes a reference.", **reference)
    if spec.takes_reference and (len(reference) > 1 or (
            not reference and not spec.reference_optional)):
        return fail("compares its input with a baseline: pass exactly one of "
                    "reference_file, reference_files or reference_dataset (the "
                    "baseline, e.g. before a change), with data_file(s) or "
                    "dataset as the new sample.", **reference)
    if spec.input_kind == "parameters":
        if given or max_events is not None:
            return fail("reads no input files; pass only parameters. "
                        f"{spec.input_hint}", **given)
    elif max_files is not None and dataset is None and reference_dataset is None:
        return fail("max_files applies only to a dataset; pick the files "
                    "yourself with data_files.")
    if not given and spec.input_kind != "parameters":
        if spec.default_inputs is None:
            return fail("pass exactly one of data_file (one input file), "
                        "data_files (a list of input files) or dataset.")
        try:
            data_files = [str(p) for p in spec.default_inputs()]
        except ValueError as exc:
            return fail(str(exc))

    if spec.input_kind == "root_file":
        if data_files is not None and not spec.combines_files:
            return fail(
                f"takes a single ROOT file: pass data_file, not data_files. "
                f"{spec.input_hint}"
            )
        if max_events is not None:
            return fail("does not run a mu2e job, so max_events does not apply.")

    # Physics knobs are validated against the spec, which names the offender.
    try:
        params = spec.resolve_params(parameters)
    except ValueError as exc:
        return fail(str(exc), parameters=parameters)

    # Catalogued names and datasets become paths here, so everything after
    # sees paths alone, as if the caller had passed them.
    if spec.input_kind == "parameters":
        resolved = None
        paths = []
    else:
        try:
            resolved = resolve_inputs(
                [data_file] if data_file is not None else data_files,
                dataset=dataset, max_files=max_files)
        except CatalogError as exc:
            return fail(str(exc), **given)
        paths = resolved.paths
    if len(paths) > 1 and not spec.takes_file_list:
        return fail(f"takes a single ROOT file, but dataset {dataset} has "
                    f"{len(paths)}: pass max_files=1, or one file as data_file. "
                    f"{spec.input_hint}", dataset=dataset)
    problems = validate_input_paths(paths)
    if problems:
        return fail("invalid input file(s): " + "; ".join(problems),
                    data_files=[str(p) for p in paths])

    reference_paths, reference_source = [], {}
    if reference:
        try:
            resolved_reference = resolve_inputs(
                [reference_file] if reference_file is not None else reference_files,
                dataset=reference_dataset, max_files=max_files)
        except CatalogError as exc:
            return fail(f"reference: {exc}", **reference)
        reference_paths = resolved_reference.paths
        reference_source = resolved_reference.source
        problems = validate_input_paths(reference_paths)
        if problems:
            return fail("invalid reference file(s): " + "; ".join(problems),
                        reference_files=[str(p) for p in reference_paths])

    env = current_env()
    outdir = Path(output_dir).expanduser().resolve()
    # Only code already on disk can be checked up front; for a Musing the
    # answer comes from art when the job runs.
    if spec.fcl is not None and (problem := env.missing_fcl(spec.fcl, outdir)):
        return fail(problem, fcl=str(spec.fcl), environment=env.describe())

    outcome = spec.run(RunContext(
        input_paths=paths,
        outdir=outdir,
        params=params,
        timeout_s=timeout_s,
        max_events=max_events,
        wants_file_list=data_file is None,
        reference_paths=reference_paths,
    ))

    # Describes the run in every result, so a caller chaining analyses can tell
    # which inputs and assumptions a number came from without re-reading logs.
    metadata: dict[str, Any] = {
        "analysis": analysis,
        "input_kind": spec.input_kind,
        "data_files": [str(p) for p in paths],
        "n_input_files": len(paths),
        **({"inputs_from": resolved.source}
           if resolved is not None and resolved.source else {}),
        **({"reference_files": [str(p) for p in reference_paths],
            "n_reference_files": len(reference_paths)} if reference_paths else {}),
        **({"reference_from": reference_source} if reference_source else {}),
        **({"parameters": params} if params else {}),
        **({"max_events": max_events} if max_events is not None else {}),
        **({"fcl": str(spec.fcl), "environment": env.describe()}
           if spec.fcl is not None else {}),
        **outcome.extra,
    }
    if outcome.log_path is not None:
        metadata["log_path"] = str(outcome.log_path)

    if outcome.error is not None or outcome.metrics is None:
        error = outcome.error or "analysis produced no metrics"
        where = f" See {outcome.log_path} for details." if outcome.log_path else ""
        return ArtifactResult(
            status="error", files=outcome.files,
            message=f"{analysis}: {error}.{where}", metadata=metadata,
        )

    return ArtifactResult(
        status="success",
        files=outcome.files,
        message=(f"{analysis}: " if spec.input_kind == "parameters"
                 else f"{analysis} over {len(paths)} input file(s)"
                      + (f" against {len(reference_paths)} reference file(s)"
                         if reference_paths else "") + ": ")
                + spec.summarize(outcome.metrics),
        metadata={**metadata, **outcome.metrics},
    )
