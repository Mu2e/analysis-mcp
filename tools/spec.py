"""What an analysis has to provide, and the result contract every tool returns.

An analysis is one entry in the registry. Two kinds exist so far, and the
difference is only what it consumes and what its runner does:

  input_kind="art_files"  a mu2e job over art file(s) whose stdout is parsed
                          (e.g. "edep" runs edep.fcl)
  input_kind="root_file"  Python computation over a ROOT file produced by an
                          earlier analysis (e.g. "approx_ce_sensitivity" reads
                          the nts.*.root that "edep" writes)
  input_kind="parameters" Python computation over its parameters alone, no
                          file (e.g. "total_edep_per_pot" adds earlier
                          analyses' metrics, passed as parameters)

Everything generic — argument validation, parameter checking, assembling the
result — lives in analysis_tools.py and works for both. Adding an analysis
means writing a module in `analyses/` with a SPEC and listing it in
registry.py.
"""

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel

from .mu2e_env import current as current_env

# The fcl the analyses here run ships with the server, so it is the same file
# whichever Offline environment a job sets up.
FCL_DIR = Path(__file__).resolve().parent.parent / "fcl"

InputKind = Literal["art_files", "root_file", "parameters"]
ParamKind = Literal["number", "text"]


class ArtifactResult(BaseModel):
    """Uniform result contract returned by every tool."""

    status: Literal["success", "error"]
    files: list[str]
    message: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ParamSpec:
    """A knob an analysis takes, reported by list_analyses.

    `default=None` means the caller must supply it. Most knobs are numbers
    with optional bounds; `kind="text"` is for the ones that name something in
    the job's output, like a filter's module label.

    A text knob is normally required to say something, so "" is refused as a
    typo. `allow_empty=True` makes "" a meaningful answer instead — "there is
    no such thing in this job" — for a knob that names something optional,
    like `count`'s prescale filter. With `default=""` that is also the knob's
    default, so the caller can simply not mention it.
    """

    name: str
    description: str
    default: float | int | str | None = None
    minimum: float | None = None
    maximum: float | None = None
    kind: ParamKind = "number"
    # text only: "" means "not set" rather than an empty value to reject.
    allow_empty: bool = False

    @property
    def required(self) -> bool:
        return self.default is None

    def describe(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "required": self.required,
            "default": self.default,
            "kind": self.kind,
            "minimum": self.minimum,
            "maximum": self.maximum,
        }

    def check(self, value: Any) -> float | str:
        """Validate one supplied value, as the number or text it should be."""
        if self.kind == "text":
            if not isinstance(value, str):
                raise ValueError(f"parameter '{self.name}' must be a string, "
                                 f"got {value!r}")
            if not value.strip():
                if self.allow_empty:
                    return ""
                raise ValueError(f"parameter '{self.name}' must be a non-empty "
                                 f"string, got {value!r}")
            return value
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"parameter '{self.name}' must be a number, got {value!r}")
        # float() takes "nan" and "inf", which no range check would catch.
        if not math.isfinite(number):
            raise ValueError(f"parameter '{self.name}' must be a finite number, "
                             f"got {value!r}")
        if self.minimum is not None and number < self.minimum:
            raise ValueError(f"parameter '{self.name}'={number:g} is below the "
                             f"minimum {self.minimum:g}")
        if self.maximum is not None and number > self.maximum:
            raise ValueError(f"parameter '{self.name}'={number:g} is above the "
                             f"maximum {self.maximum:g}")
        return number


@dataclass
class RunContext:
    """Everything a runner needs to do one job."""

    input_paths: list[Path]
    outdir: Path
    params: dict[str, float | str]
    timeout_s: int
    max_events: int | None = None
    # True when the caller passed data_files (a list) rather than data_file, so
    # a one-element list still goes through mu2e's -S file-list path.
    wants_file_list: bool = False
    # The baseline sample, for an analysis that takes_reference: what
    # input_paths (the new sample) is compared against. Empty otherwise.
    reference_paths: list[Path] = field(default_factory=list)

    @property
    def input_path(self) -> Path:
        """The single input, for root_file analyses that take only one."""
        return self.input_paths[0]


@dataclass
class RunOutcome:
    """What a runner produced. `error` set means it did not work."""

    metrics: dict[str, float] | None = None
    files: list[str] = field(default_factory=list)
    log_path: Path | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# A runner does the work described by a RunContext.
Runner = Callable[[RunContext], RunOutcome]

# Formats a one-line human summary from parsed metrics.
Summarizer = Callable[[dict[str, float]], str]


@dataclass(frozen=True)
class AnalysisSpec:
    """One runnable analysis.

    Attributes:
        name: Short key agents pass as run_analysis(analysis=...).
        description: One line, shown by list_analyses.
        input_kind: What it consumes — art file(s), a ROOT file, or only its
            parameters.
        metrics: Metric names it reports, in reporting order.
        run: Does the work (see Runner).
        summarize: Builds the human-readable one-line message.
        units: Metric name -> unit, for metrics that have one.
        parameters: Physics knobs the caller may/must supply.
        input_hint: What the input file(s) must contain.
        produced_by: Analyses whose output feeds this one, for chaining.
        combines_files: root_file analyses only — takes data_files too, and
            combines the files into one result. art_files analyses always
            take a list, as one mu2e -S job.
        default_inputs: Supplies the input when no data_file(s) are passed.
        default_inputs_hint: What those default inputs are, for agents.
        takes_reference: Compares its input with a second, baseline sample,
            which the caller must pass as reference_file(s) or
            reference_dataset.
        reference_optional: With takes_reference, the baseline may be left
            out: the analysis runs on its input alone, and compares only
            when given one.
    """

    name: str
    description: str
    input_kind: InputKind
    metrics: tuple[str, ...]
    run: Runner
    summarize: Summarizer
    units: dict[str, str] | None = None
    parameters: tuple[ParamSpec, ...] = ()
    input_hint: str = ""
    produced_by: tuple[str, ...] = ()
    # art_files analyses only: absolute (FCL_DIR / "edep.fcl"), or relative to
    # the configured code and resolved per environment by list_analyses.
    fcl: Path | None = None
    combines_files: bool = False
    # Inputs used when the caller passes no data_file(s): returns the paths,
    # or raises ValueError saying why there are none. Described for agents by
    # default_inputs_hint.
    default_inputs: Callable[[], list[Path]] | None = None
    default_inputs_hint: str = ""
    takes_reference: bool = False
    reference_optional: bool = False

    @property
    def takes_file_list(self) -> bool:
        return self.input_kind == "art_files" or self.combines_files

    def describe(self) -> dict[str, Any]:
        """The registry entry as list_analyses reports it."""
        entry: dict[str, Any] = {
            "description": self.description,
            "input_kind": self.input_kind,
            "metrics": list(self.metrics),
            "units": dict(self.units or {}),
            "parameters": {p.name: p.describe() for p in self.parameters},
            "input_hint": self.input_hint,
            "takes_data_files": self.takes_file_list,
        }
        if self.produced_by:
            entry["produced_by"] = list(self.produced_by)
        if self.default_inputs is not None:
            entry["default_inputs"] = self.default_inputs_hint
        if self.takes_reference:
            entry["takes_reference"] = True
            if self.reference_optional:
                entry["reference_optional"] = True
        if self.fcl is not None:
            env = current_env()
            entry["fcl"] = str(env.resolve_fcl(self.fcl))
            # None where only art can tell: a relative fcl under a Musing comes
            # off FHICL_FILE_PATH, and a per-job tarball is not unpacked yet.
            known = self.fcl.is_absolute() or env.base_dir() is not None
            entry["fcl_exists"] = (env.missing_fcl(self.fcl) is None
                                   if known else None)
        return entry

    def resolve_params(self, supplied: dict[str, Any] | None) -> dict[str, float | str]:
        """Merge supplied parameters over the defaults, validating them.

        Raises ValueError naming the offender for unknown, missing, or
        out-of-range values, so run_analysis can report it verbatim.
        """
        supplied = dict(supplied or {})
        known = {p.name: p for p in self.parameters}
        unknown = sorted(set(supplied) - set(known))
        if unknown:
            raise ValueError(
                f"unknown parameter(s) for '{self.name}': {', '.join(unknown)}. "
                f"Known: {', '.join(known) or '(none)'}"
            )
        resolved: dict[str, float | str] = {}
        missing = []
        for name, param in known.items():
            if name in supplied:
                resolved[name] = param.check(supplied[name])
            elif param.required:
                missing.append(name)
            else:
                resolved[name] = param.check(param.default)
        if missing:
            raise ValueError(
                f"missing required parameter(s) for '{self.name}': "
                f"{', '.join(missing)}"
            )
        return resolved
