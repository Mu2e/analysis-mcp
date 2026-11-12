"""Running a mu2e job — the part every analysis shares.

No analysis-specific knowledge lives here: callers supply an fcl and get back
the job's output. `mu2e` exists only after the Offline environment is set up,
so every job runs in a fresh bash that sets it up first, rather than assuming
the server process inherited anything. Which code that is — a muse work area,
a Musing, or a code tarball — is `mu2e_env`'s business; this module asks it
for the commands and for where the fcl lives.
"""

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .mu2e_env import Mu2eEnv, current as current_env


@dataclass
class JobOutcome:
    """What a finished (or timed-out) mu2e job left behind."""

    command: str
    returncode: int | None  # None when the job was killed on timeout
    timed_out: bool
    stdout: str
    stderr: str
    log_path: Path
    input_paths: list[Path]
    input_flag: str  # "-s" or "-S"
    environment: str  # which Offline setup ran, as mu2e_env describes it
    file_list_path: Path | None
    written_root_files: list[str] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return self.timed_out or self.returncode != 0

    def stdout_tail(self, lines: int = 20) -> str:
        return "\n".join((self.stdout + self.stderr).splitlines()[-lines:])


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else value.decode("utf-8", "replace")


def build_input_args(
    paths: list[Path], outdir: Path, *, single: bool
) -> tuple[str, str, Path | None]:
    """Pick mu2e's input flag: -s for one file, -S for a written file list.

    Returns (flag, argument, file_list_path or None). For the -S case the list
    is written to outdir/filelist.txt, one absolute path per line — the format
    mu2e expects.
    """
    if single:
        return "-s", str(paths[0]), None
    file_list_path = outdir / "filelist.txt"
    file_list_path.write_text("".join(f"{p}\n" for p in paths), encoding="utf-8")
    return "-S", str(file_list_path), file_list_path


def root_snapshot(outdir: Path) -> dict[str, tuple[float, int]]:
    """Name -> (mtime, size) for every ROOT file in outdir.

    Taken before and after a job so its output is recognized by having been
    *written*, not by being absent beforehand. Running twice into the same
    directory overwrites the previous run's files, and a job whose output was
    overwritten still produced it — comparing names alone would report nothing
    the second time and leave an analysis chained onto it with no input.
    """
    snapshot: dict[str, tuple[float, int]] = {}
    for path in outdir.glob("*.root"):
        try:
            stat = path.stat()
        except OSError:      # vanished between the glob and the stat
            continue
        snapshot[path.name] = (stat.st_mtime, stat.st_size)
    return snapshot


def written_root_files(outdir: Path, before: dict[str, tuple[float, int]]) -> list[str]:
    """The ROOT files in outdir that a job created or rewrote since `before`."""
    return sorted(
        str(outdir / name)
        for name, stamp in root_snapshot(outdir).items()
        if before.get(name) != stamp
    )


def validate_input_paths(paths: list[Path]) -> list[str]:
    """Return one complaint per unusable input path; empty means all good."""
    return [
        f"not an absolute path: {p}" if not p.is_absolute() else f"does not exist: {p}"
        for p in paths
        if not p.is_absolute() or not p.exists()
    ]


def run_mu2e_job(
    *,
    fcl: Path,
    input_paths: list[Path],
    outdir: Path,
    single: bool,
    timeout_s: int,
    max_events: int | None = None,
    log_name: str = "mu2e.log",
    env: Mu2eEnv | None = None,
    extra_search_dirs: tuple[Path, ...] = (),
) -> JobOutcome:
    """Run `mu2e -c <fcl> -s|-S <input>` in outdir and capture everything.

    Writes the combined stdout/stderr to outdir/<log_name> and reports the
    ROOT files the job wrote there (TFileService output), whether they were
    new or overwrote a previous run's — outputs, logs and file lists are all
    simply overwritten, so a directory can be reused.

    `env` says where Offline comes from; the configured one is used by
    default. `fcl` may be relative, in which case it is resolved against the
    environment's code when that is a directory on disk, and otherwise handed
    to art to find on FHICL_FILE_PATH.

    `extra_search_dirs` are appended to MU2E_SEARCH_PATH, where Mu2e looks up
    geometry and other config files (it never takes an absolute path, and the
    environment's path holds only the code and DataFiles). Appended, so the
    code's own files still win any name clash.
    """
    env = env or current_env()
    outdir.mkdir(parents=True, exist_ok=True)
    log_path = outdir / log_name
    before = root_snapshot(outdir)

    input_flag, input_arg, file_list_path = build_input_args(
        input_paths, outdir, single=single
    )
    nevts = f" --nevts {max_events}" if max_events is not None else ""

    # $1 = input (art file or file list), $2 = output dir — passed as bash
    # positional parameters so paths never need shell quoting.
    script = " && ".join([
        *env.setup_commands(outdir),
        *(f"export MU2E_SEARCH_PATH=\"$MU2E_SEARCH_PATH\":{shlex.quote(str(d))}"
          for d in extra_search_dirs),
        'cd "$2"',
        f'mu2e -c "{env.resolve_fcl(fcl, outdir)}" {input_flag} "$1"{nevts}',
    ])
    try:
        proc = subprocess.run(
            ["bash", "-lc", script, "_", input_arg, str(outdir)],
            cwd=outdir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        stdout, stderr, returncode, timed_out = (
            proc.stdout, proc.stderr, proc.returncode, False,
        )
    except subprocess.TimeoutExpired as exc:
        # In text mode these are usually str, but bytes on some versions.
        stdout, stderr = _as_text(exc.stdout), _as_text(exc.stderr)
        returncode, timed_out = None, True

    inputs_note = "\n".join(f"  {p}" for p in input_paths)
    log_path.write_text(
        f"# {env.describe()}\n"
        f"$ {script}\n({len(input_paths)} input file(s), {input_flag})\n"
        f"{inputs_note}\n\n--- stdout ---\n{stdout}\n\n--- stderr ---\n{stderr}\n",
        encoding="utf-8",
    )

    return JobOutcome(
        command=script,
        returncode=returncode,
        timed_out=timed_out,
        stdout=stdout,
        stderr=stderr,
        log_path=log_path,
        input_paths=list(input_paths),
        input_flag=input_flag,
        environment=env.describe(),
        file_list_path=file_list_path,
        written_root_files=written_root_files(outdir, before),
    )


# --- what a module in the configured Offline accepts ---------------------------

class ModuleProbeError(RuntimeError):
    """`mu2e --print-description` could not describe a module, worded for the caller."""


# (environment, module type) -> the description text. The environment is a
# property of the deployment (see mu2e_env), so one probe per process answers
# for every later job; a failed probe is not cached.
_MODULE_DESCRIPTIONS: dict[tuple[Mu2eEnv, str], str] = {}


def module_description(module_type: str, outdir: Path, *, env: Mu2eEnv | None = None,
                       timeout_s: int = 300) -> str:
    """art's description of `module_type` (`mu2e --print-description`) in `env`.

    Asks the code that will run the job, so it knows what the configured
    Offline's module accepts without guessing from a release name.
    """
    env = env or current_env()
    key = (env, module_type)
    if key not in _MODULE_DESCRIPTIONS:
        outdir.mkdir(parents=True, exist_ok=True)
        script = " && ".join([*env.setup_commands(outdir), 'cd "$2"',
                              'mu2e --print-description "$1"'])
        try:
            proc = subprocess.run(["bash", "-lc", script, "_", module_type, str(outdir)],
                                  cwd=outdir, capture_output=True, text=True,
                                  timeout=timeout_s)
        except subprocess.TimeoutExpired:
            raise ModuleProbeError(f"mu2e --print-description {module_type} "
                                   f"timed out after {timeout_s}s") from None
        if proc.returncode != 0 or not proc.stdout.strip():
            tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-5:])
            raise ModuleProbeError(f"mu2e --print-description {module_type} failed "
                                   f"in {env.describe()} (exit {proc.returncode}): {tail}")
        _MODULE_DESCRIPTIONS[key] = proc.stdout
    return _MODULE_DESCRIPTIONS[key]


def module_accepts(module_type: str, parameter: str, outdir: Path, *,
                   env: Mu2eEnv | None = None) -> bool:
    """Whether `module_type` in the configured Offline declares `parameter`.

    Generic on purpose: it lets an analysis pass a parameter that only newer
    Offline has, without this module knowing what it means. Raises
    ModuleProbeError if the module cannot be described.
    """
    text = module_description(module_type, outdir, env=env)
    return re.search(rf"^\s*{re.escape(parameter)}\s*:", text, re.MULTILINE) is not None
