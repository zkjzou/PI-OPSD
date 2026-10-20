"""Recipe resolver: turn an arm id into the exact ``KEY=VALUE`` environment a launcher runs with.

Usage::

    python -m opsd.recipes list [--bench B] [--table T] [--status S] [--long]
    python -m opsd.recipes show <arm_id> [--overlay NAME]... [KEY=VAL ...]
    python -m opsd.recipes resolve <arm_id> [--overlay NAME]... [KEY=VAL ...] [--hermetic] [--out FILE]
    python -m opsd.recipes validate [--bench B] [--check-files]
    python -m opsd.recipes provenance <resolved.env> --out <run_dir>/provenance.json

Recipes tree (``opsd/recipes/``; override with ``--recipes-dir`` or ``OPSD_RECIPES_DIR``)::

    common/overlays/<name>.env           optional run-shape overlays (smoke_4gpu, 1node_8gpu, ...)
    <bench>/base.env                     run-file defaults that never vary across paper arms
    <bench>/students/<student>.env       per-student settings (4bi, 4bt, 30bi)
    <bench>/objectives/<objective>.env   per-objective settings (opsd, grpo, opsd_grpo, rlsd, ...)
    <bench>/pis.yaml                     pi_id -> per-student hint file (+ ref_seq_cap, content_sha256)
    <bench>/arms.yaml                    TABLES that generate arm ids (see below)
    <bench>/overlays/<name>.env          optional bench-specific refinement of a common overlay

Merge order, later wins (each key keeps the full chain for ``show``)::

    base < student < objective < pi < table env < table extra < seed < overlay(s) < CLI KEY=VAL

``.env`` format: ``KEY=VALUE`` lines, full-line ``#`` comments, optional ``export`` prefix, optional
matching outer quotes. No shell expansion except ``${OPSD_DATA_ROOT}`` and ``${RUN_DIR}`` (also accepted
as ``$OPSD_DATA_ROOT`` / ``$RUN_DIR``); any other ``$NAME`` / ``${...}`` / ``$(...)`` is an error, so the
resolved file means the same thing wherever it is sourced. Inline comments are rejected (a value that
silently ends in ``# note`` is worse than an error).

Arm ids: ``<bench>/<table>/<student>/<pi>[/<key>=<value>...][/seed=<n>]``. ``seed=<n>`` is the 1-based
replicate index; the trainer seed it maps to lives in the table.

``arms.yaml`` (all scalars are read as STRINGS, so ``1e-4``, ``0.50`` and ``true`` keep their spelling)::

    seed_var: TRAINER_SEED           # optional, file default for the env var a seed maps to
    keys:                            # optional, arm-id key -> env var (or a value map), shared by tables
      lambda: OPSD_MIX_LAMBDA
      teacher: {var: OPSD_TEACHER, values: {h10: {OPSD_TEACHER: hard, OPSD_TEACHER_SYNC: "10"}}}
    defaults: {status: done}         # optional, merged into every table entry
    tables:                          # (or a bare top-level list)
      - table: opsd_grpo
        students: [4bi]
        pis: [glm_ao, glm_chk]       # or "all" = every pi in pis.yaml
        objective: opsd_grpo
        env: {EPOCHS: "2"}
        extra:                       # cartesian product, in the order written
          lambda: [0.5, 0.9]         #   env var from `keys`, or {var: X, values: [...]}, or
                                     #   {values: {v1: {ENV: x}, v2: "y"}} (value -> env dict / var value)
        seeds: {1: 42, 2: 43}        # or [1, 2, 3] (index only) or {1: {TRAINER_SEED: 42, ...}}
        exclude: ["4bt/glm_aat"]     # optional fnmatch patterns on the id (or the part after <table>/)
        status: done                 # done | in_progress | planned (planned is refused by resolve/show)
        paper: "tab:opsd_grpo"       # cell reference or null

A (student, pi) pair whose pi has no hint for that student (and no ``default``) is skipped when a table
is expanded; ``validate`` lists the skips. Placeholders ``{bench} {table} {student} {pi} {seed}`` and the
table's extra keys are substituted in every value.

``pis.yaml``::

    glm_ao:
      author: glm-5.2
      analyzer: null
      paper_row: Action+Obs.
      env: {}                        # optional, applied for every student
      4bi: {hint: alfworld/hints/glm-5.2/hints_traj_nothink_cap24k.json, ref_seq_cap: 65536,
            content_sha256: <hex>}   # or just the hint path as a string
      default: {...}                 # optional fallback for students without their own entry

``hint`` is relative to ``$OPSD_DATA_ROOT`` and becomes ``OPSD_HINTS_PATH`` (``hint_var`` overrides the
name); ``ref_seq_cap`` becomes ``OPSD_REF_SEQ_CAP``; ``content_sha256`` becomes
``OPSD_HINTS_CONTENT_SHA256`` (checked by preflight, recorded in provenance.json). Overriding the hint path
on the command line clears that declared hash unless ``OPSD_HINTS_CONTENT_SHA256=...`` is given too.
``pi: none`` needs no entry.

Location keys come from CLI > ambient environment > recipe > default: ``OPSD_DATA_ROOT`` (default
``${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data``), ``RUN_ROOT`` (default ``./runs``), ``RUN_NAME`` (default
derived from the arm id, plus ``__<overlay>`` for every overlay that sets at least one key, so a smoke run
never shares -- and later resumes into -- the full run's directory: ``alfworld/main/4bi/glm_ao --overlay
smoke_4gpu`` -> ``alfworld__main__4bi__glm_ao__smoke_4gpu``; a no-op marker overlay such as ``1node_8gpu``
keeps the arm's name). ``RUN_DIR`` is ``$RUN_ROOT/$RUN_NAME`` unless given on the CLI. Relative
``OPSD_DATA_ROOT`` / ``RUN_ROOT`` / ``RUN_DIR`` values are made absolute against the current working
directory (the launchers resolve in the directory the user launched from), and so are relative paths
given as ``KEY=VAL`` (``OPSD_HINTS_PATH``, ``TASK_IDS_PATH``, ``DEV_TASK_IDS_PATH``, ``EVAL_HINT_DIR``,
``EVAL_HINT_TASK_IDS_PATH``, ``HINTS_DIR``, the pi's ``hint_var``, ``RESUME_PATH``, ``EVAL_CKPT_DIR``,
``TAU2_MANIFEST_DIR``, ``PRESERVE_TEMPLATE``, ``OPSD_ROLLOUT_DUMP_DIR``, ``OPSD_EMA_DIR``; ``MODEL_PATH`` only
when it names an existing path, so a Hugging Face id is kept), because the launchers run the trainer from
the repository root. The resolved environment also carries ``OPSD_ARM_ID`` and ``OPSD_BENCH``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import difflib
import fnmatch
import hashlib
import itertools
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

from opsd import _repo

STATUSES = ("done", "in_progress", "planned")
DEFAULT_SEED_VAR = "TRAINER_SEED"
HINT_VAR = "OPSD_HINTS_PATH"
REF_CAP_VAR = "OPSD_REF_SEQ_CAP"
HINT_SHA_VAR = "OPSD_HINTS_CONTENT_SHA256"
LOCATION_KEYS = ("OPSD_DATA_ROOT", "RUN_ROOT", "RUN_NAME", "RUN_DIR")
# Per-run files/dirs: a RELATIVE value given on the command line is made absolute against the current working
# directory (the directory the user launched from), so it keeps pointing at the same place after the launcher
# changes into the repository root to run the trainer.
INPUT_PATH_KEYS = (
    "OPSD_HINTS_PATH",
    "TASK_IDS_PATH",
    "DEV_TASK_IDS_PATH",
    "EVAL_HINT_DIR",
    "EVAL_HINT_TASK_IDS_PATH",
    "HINTS_DIR",
    "RESUME_PATH",
    "EVAL_CKPT_DIR",
    "TAU2_MANIFEST_DIR",
    "PRESERVE_TEMPLATE",
    "OPSD_ROLLOUT_DUMP_DIR",
    "OPSD_EMA_DIR",
)
# Values that are either a local path or a Hugging Face model id: a relative value is made absolute only when it
# names an existing path (from the current working directory), so `Qwen/Qwen3-4B-Instruct-2507` stays a hub id.
LOCAL_OR_HUB_KEYS = ("MODEL_PATH",)
DERIVED_KEYS = ("OPSD_ARM_ID", "OPSD_BENCH")
NULLS = ("", "~", "null", "None", "none")

# Ambient OPSD_* names that are user/infrastructure settings, not recipe knobs: --hermetic accepts them.
HERMETIC_ALLOWED = {
    "OPSD_DATA_ROOT",
    "OPSD_RECIPES_DIR",
    "OPSD_NVML_LIB_DIR",
    "OPSD_PYTHON",
    "OPSD_PREFLIGHT",
    "OPSD_CLEANUP_GRACE_S",
    "OPSD_DRY_RUN",
    "OPSD_RESOLVED_ENV",
}
HERMETIC_ALLOWED_PREFIXES = (
    "OPSD_PREFLIGHT_",
    "OPSD_BOOTSTRAP_",
    "OPSD_RAY_",
    "OPSD_PI_ANALYZER_",
    "OPSD_ONLINE_HINTS_ANALYZER_",
)

_KEY_RX = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_LINE_RX = re.compile(r"^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$")
_COMPONENT_RX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")
_EXTRA_KEY_RX = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_EXTRA_VAL_RX = re.compile(r"^[A-Za-z0-9_.+-]+$")
_REF_RX = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


class RecipeError(Exception):
    """A recipe file, arm id or override is invalid. The message names the file/arm and the fix."""


# ----------------------------------------------------------------------------------------------- paths
def repo_root(start: Optional[Path] = None) -> Path:
    """The repository root: the directory holding ``pyproject.toml``, ``opsd/__init__.py`` and
    ``opsd/skyrl_ext/__init__.py`` (see ``opsd._repo``)."""
    try:
        return _repo.repo_root(start)
    except FileNotFoundError as e:
        raise RecipeError(f"could not find the repository root: {e}") from None


def default_recipes_dir() -> Path:
    """``$OPSD_RECIPES_DIR`` if set, else the shipped ``opsd/recipes`` (found from the package itself)."""
    env = os.environ.get("OPSD_RECIPES_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _repo.recipes_dir()


def default_data_root(environ: Mapping[str, str]) -> str:
    base = environ.get("XDG_CACHE_HOME") or os.path.join(environ.get("HOME") or os.path.expanduser("~"), ".cache")
    return os.path.join(base, "opsd", "data")


# ------------------------------------------------------------------------------------------------ yaml
class _StrLoader(yaml.SafeLoader):
    """SafeLoader that keeps every scalar a string (only ``<<`` merge keys are interpreted) and rejects
    duplicate mapping keys, which PyYAML otherwise drops silently."""


_StrLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in rs if tag == "tag:yaml.org,2002:merge"]
    for ch, rs in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


def _construct_mapping(loader, node, deep=False):
    # Duplicates are checked among the keys written in THIS mapping only: a key that overrides one merged in
    # with ``<<: *anchor`` is the YAML merge-key semantics, not a duplicate.
    seen = set()
    for key_node, _ in node.value:
        if key_node.tag == "tag:yaml.org,2002:merge":
            continue
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise RecipeError(f"duplicate key {key!r} at line {key_node.start_mark.line + 1}")
        seen.add(key)
    loader.flatten_mapping(node)  # merged pairs first (earlier aliases win), then this mapping's own keys
    out = {}
    for key_node, value_node in node.value:
        out[loader.construct_object(key_node, deep=deep)] = loader.construct_object(value_node, deep=deep)
    return out


_StrLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_yaml(path: Path):
    try:
        with open(path) as f:
            return yaml.load(f, Loader=_StrLoader)
    except RecipeError as e:
        raise RecipeError(f"{path}: {e}") from None
    except yaml.YAMLError as e:
        raise RecipeError(f"{path}: invalid YAML: {e}") from None


def _none(v):
    return None if v is None or (isinstance(v, str) and v.strip() in NULLS) else v


def _as_list(v, what: str) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(x, str) for x in v):
        return list(v)
    raise RecipeError(f"{what}: expected a string or a list of strings, got {v!r}")


def _as_env(v, what: str) -> Dict[str, str]:
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise RecipeError(f"{what}: expected a mapping KEY: VALUE, got {v!r}")
    out = {}
    for k, val in v.items():
        if not isinstance(k, str) or not _KEY_RX.match(k):
            raise RecipeError(f"{what}: invalid env var name {k!r}")
        if isinstance(val, (dict, list)):
            raise RecipeError(f"{what}: value of {k} must be a scalar, got {val!r}")
        out[k] = "" if val is None else str(val)
    return out


# ------------------------------------------------------------------------------------------------- env
def _unquote(val: str) -> Tuple[str, bool]:
    if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
        return val[1:-1], True
    return val, False


def parse_env_file(path: Path) -> Dict[str, Tuple[str, int]]:
    """``{KEY: (value, line_no)}`` from a ``.env`` file (format in the module docstring)."""
    out: Dict[str, Tuple[str, int]] = {}
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE_RX.match(line)
        if not m:
            raise RecipeError(f"{path}:{lineno}: expected KEY=VALUE, got {raw!r}")
        key, val = m.group(1), m.group(2).strip()
        val, quoted = _unquote(val)
        if not quoted and re.search(r"\s#", val):
            raise RecipeError(
                f"{path}:{lineno}: inline comments are not supported ({raw!r}); " "put the comment on its own line"
            )
        if key in out:
            raise RecipeError(f"{path}:{lineno}: duplicate key {key} (first at line {out[key][1]})")
        out[key] = (val, lineno)
    return out


def format_env(env: Mapping[str, str], quote: bool = True) -> str:
    """Sorted ``KEY=VALUE`` lines; values are shell-quoted only when needed, so the file can be sourced."""
    lines = [f"{k}={shlex.quote(env[k]) if quote and env[k] != '' else env[k]}" for k in sorted(env)]
    return "\n".join(lines) + ("\n" if lines else "")


def read_resolved_env(path: Path) -> Dict[str, str]:
    """Inverse of :func:`format_env` (shell-quoted values are unquoted with shlex)."""
    out = {}
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE_RX.match(line)
        if not m:
            raise RecipeError(f"{path}:{lineno}: expected KEY=VALUE, got {raw!r}")
        parts = shlex.split(m.group(2)) if m.group(2) else [""]
        out[m.group(1)] = parts[0] if len(parts) == 1 else " ".join(parts)
    return out


# ---------------------------------------------------------------------------------------------- arm id
@dataclass(frozen=True)
class ArmId:
    bench: str
    table: str
    student: str
    pi: str
    extras: Tuple[Tuple[str, str], ...] = ()
    seed: Optional[str] = None

    def __str__(self) -> str:
        parts = [self.bench, self.table, self.student, self.pi]
        parts += [f"{k}={v}" for k, v in self.extras]
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return "/".join(parts)

    @property
    def run_name(self) -> str:
        """Filesystem/hydra-safe default run name: ``/`` -> ``__``, ``=`` -> ``-``."""
        name = str(self).replace("/", "__").replace("=", "-")
        return re.sub(r"[^A-Za-z0-9_.+-]", "-", name)


def parse_arm_id(text: str) -> ArmId:
    parts = text.strip().strip("/").split("/")
    if len(parts) < 4:
        raise RecipeError(f"arm id {text!r}: expected <bench>/<table>/<student>/<pi>[/<key>=<value>...][/seed=<n>]")
    head, tail = parts[:4], parts[4:]
    for name, comp in zip(("bench", "table", "student", "pi"), head):
        if not _COMPONENT_RX.match(comp) or "=" in comp:
            raise RecipeError(f"arm id {text!r}: invalid {name} component {comp!r}")
    extras: List[Tuple[str, str]] = []
    seed = None
    for i, comp in enumerate(tail):
        k, sep, v = comp.partition("=")
        if not sep or not _EXTRA_KEY_RX.match(k) or not _EXTRA_VAL_RX.match(v):
            raise RecipeError(f"arm id {text!r}: component {comp!r} is not <key>=<value>")
        if k == "seed":
            if i != len(tail) - 1:
                raise RecipeError(f"arm id {text!r}: seed=<n> must be the last component")
            seed = v
        else:
            if any(k == e for e, _ in extras):
                raise RecipeError(f"arm id {text!r}: key {k!r} appears twice")
            extras.append((k, v))
    return ArmId(*head, extras=tuple(extras), seed=seed)


# --------------------------------------------------------------------------------------------- recipes
@dataclass
class ExtraValue:
    text: str
    env: Dict[str, str]


@dataclass
class ExtraAxis:
    key: str
    values: List[ExtraValue]


@dataclass
class SeedSpec:
    index: str
    env: Dict[str, str]


@dataclass
class TableEntry:
    table: str
    students: List[str]
    pis: List[str]
    objective: Optional[str]
    env: Dict[str, str]
    extra: List[ExtraAxis]
    seeds: Optional[List[SeedSpec]]
    exclude: List[str]
    status: str
    paper: Optional[str]
    source: str
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class PiStudent:
    hint: Optional[str]
    hint_var: str
    ref_seq_cap: Optional[str]
    content_sha256: Optional[str]
    env: Dict[str, str]


@dataclass
class Pi:
    pi_id: str
    students: Dict[str, PiStudent]
    default: Optional[PiStudent]
    env: Dict[str, str]
    meta: Dict[str, object]

    def for_student(self, student: str) -> Optional[PiStudent]:
        return self.students.get(student) or self.default


@dataclass
class Arm:
    arm_id: ArmId
    entry: TableEntry
    extra_values: List[Tuple[str, ExtraValue]]
    seed: Optional[SeedSpec]

    @property
    def status(self) -> str:
        return self.entry.status


@dataclass
class Bench:
    name: str
    root: Path
    pis: Dict[str, Pi]
    entries: List[TableEntry]
    arms: Dict[str, Arm]
    skipped: List[str]


_STUDENT_ENTRY_KEYS = {"hint", "hint_var", "ref_seq_cap", "content_sha256", "env"}


def _parse_pi_student(v, what: str) -> PiStudent:
    if isinstance(v, str):
        return PiStudent(hint=_none(v), hint_var=HINT_VAR, ref_seq_cap=None, content_sha256=None, env={})
    if not isinstance(v, dict):
        raise RecipeError(f"{what}: expected a hint path or a mapping, got {v!r}")
    hint_var = v.get("hint_var") or HINT_VAR
    if not _KEY_RX.match(hint_var):
        raise RecipeError(f"{what}: invalid hint_var {hint_var!r}")
    return PiStudent(
        hint=_none(v.get("hint")),
        hint_var=hint_var,
        ref_seq_cap=_none(v.get("ref_seq_cap")),
        content_sha256=_none(v.get("content_sha256")),
        env=_as_env(v.get("env"), f"{what}.env"),
    )


def load_pis(path: Path, students: Optional[Iterable[str]] = None) -> Dict[str, Pi]:
    """Parse pis.yaml. A key is a per-student entry iff it names a student (``students/<s>.env``)."""
    students = set(students or ())
    if not path.exists():
        return {}
    raw = load_yaml(path) or {}
    if isinstance(raw, dict) and set(raw) == {"pis"} and isinstance(raw["pis"], dict):
        raw = raw["pis"]
    if not isinstance(raw, dict):
        raise RecipeError(f"{path}: expected a mapping pi_id -> {{student: {{hint: ...}}, author, ...}}")
    out = {}
    for pi_id, spec in raw.items():
        what = f"{path.name}:{pi_id}"
        if not _COMPONENT_RX.match(str(pi_id)):
            raise RecipeError(f"{what}: invalid pi id")
        spec = spec or {}
        if not isinstance(spec, dict):
            raise RecipeError(f"{what}: expected a mapping, got {spec!r}")
        per_student, meta = {}, {}
        default = None
        for k, v in spec.items():
            if k == "default":
                default = _parse_pi_student(v, f"{what}.default")
            elif k == "env":
                continue
            elif k in students:
                per_student[k] = _parse_pi_student(v, f"{what}.{k}")
            elif isinstance(v, dict) and set(v) & _STUDENT_ENTRY_KEYS:
                raise RecipeError(
                    f"{what}.{k}: looks like a per-student hint entry but there is no "
                    f"students/{k}.env (known students: {sorted(students)})"
                )
            else:
                meta[k] = _none(v) if isinstance(v, str) else v
        out[pi_id] = Pi(pi_id, per_student, default, _as_env(spec.get("env"), f"{what}.env"), meta)
    return out


def _axis_spec(key: str, spec, keys: Mapping[str, object], what: str) -> Tuple[str, Optional[str], Dict[str, object]]:
    """-> (id_key, env var or None, value map) for one ``extra`` entry."""
    values_map: Dict[str, object] = {}
    var: Optional[str] = None
    id_key = key
    # A bare upper-case key is an env var name; its id spelling comes from a `keys` alias if one exists.
    if _KEY_RX.match(key) and key.isupper():
        var = key
        for alias, target in keys.items():
            tvar = target if isinstance(target, str) else (target or {}).get("var")
            if tvar == key:
                id_key = alias
                break
    else:
        target = keys.get(key)
        if isinstance(target, str):
            var = target
        elif isinstance(target, dict):
            var = target.get("var")
            values_map.update(target.get("values") or {})
    if isinstance(spec, dict):
        var = spec.get("var", var)
        vals = spec.get("values")
        if isinstance(vals, dict):
            values_map.update(vals)
            vals = list(vals)
    else:
        vals = spec
    vals = _as_list(vals, f"{what}.extra.{key}")
    if not vals:
        raise RecipeError(f"{what}: extra.{key} has no values")
    if var is not None and not _KEY_RX.match(var):
        raise RecipeError(f"{what}: extra.{key} maps to an invalid env var {var!r}")
    if not _EXTRA_KEY_RX.match(id_key) or id_key == "seed":
        raise RecipeError(f"{what}: extra key {id_key!r} cannot be used in an arm id")
    return id_key, var, {"__values__": vals, **values_map}


def _parse_axis(key: str, spec, keys: Mapping[str, object], what: str) -> ExtraAxis:
    id_key, var, vmap = _axis_spec(key, spec, keys, what)
    out = []
    for v in vmap.pop("__values__"):
        if not _EXTRA_VAL_RX.match(v):
            raise RecipeError(f"{what}: extra.{key} value {v!r} cannot be used in an arm id")
        mapped = vmap.get(v)
        if isinstance(mapped, dict):
            env = _as_env(mapped, f"{what}.extra.{key}.{v}")
        elif isinstance(mapped, str):
            if var is None:
                raise RecipeError(f"{what}: extra.{key}.{v} maps to {mapped!r} but no env var is known")
            env = {var: mapped}
        elif var is not None:
            env = {var: v}
        else:
            raise RecipeError(
                f"{what}: extra.{key}: no env var for arm-id key {key!r}. Add `keys: {{{key}: "
                "ENV_VAR}}`, write `{var: ENV_VAR, values: [...]}`, or map each value to an env dict"
            )
        out.append(ExtraValue(v, env))
    return ExtraAxis(id_key, out)


def _parse_seeds(spec, seed_var: str, what: str) -> Optional[List[SeedSpec]]:
    spec = _none(spec) if isinstance(spec, str) else spec
    if spec is None:
        return None
    if isinstance(spec, (str, list)):
        return [SeedSpec(i, {}) for i in _as_list(spec, f"{what}.seeds")]
    if not isinstance(spec, dict):
        raise RecipeError(f"{what}: seeds must be a list of indices or a mapping index -> trainer seed")
    out = []
    for idx, v in spec.items():
        if not str(idx).isdigit():
            raise RecipeError(f"{what}: seed index {idx!r} must be a positive integer (1-based replicate)")
        if isinstance(v, dict):
            env = _as_env(v, f"{what}.seeds.{idx}")
        elif _none(v) is None:
            env = {}
        else:
            env = {seed_var: str(v)}
        out.append(SeedSpec(str(idx), env))
    return out


_ENTRY_KEYS = {
    "table",
    "students",
    "pis",
    "objective",
    "env",
    "extra",
    "seeds",
    "seed_var",
    "exclude",
    "status",
    "paper",
    "keys",
}


def _parse_entry(
    raw,
    i: int,
    path: Path,
    file_keys: Mapping[str, object],
    file_seed_var: str,
    defaults: Mapping[str, object],
    pis: Mapping[str, Pi],
) -> TableEntry:
    if not isinstance(raw, dict):
        raise RecipeError(f"{path}: table entry #{i + 1} must be a mapping")
    raw = {**defaults, **raw}
    table = raw.get("table")
    what = f"{path.parent.name}/{path.name}[{i + 1}:{table}]"
    if not isinstance(table, str) or not _COMPONENT_RX.match(table):
        raise RecipeError(f"{what}: missing or invalid `table`")
    status = raw.get("status")
    if status not in STATUSES:
        raise RecipeError(f"{what}: status must be one of {'|'.join(STATUSES)}, got {status!r}")
    students = _as_list(raw.get("students"), f"{what}.students")
    if not students:
        raise RecipeError(f"{what}: `students` is empty")
    pis_spec = raw.get("pis")
    if pis_spec in ("all", "*"):
        pi_list = list(pis)
    else:
        pi_list = _as_list(pis_spec, f"{what}.pis")
    if not pi_list:
        raise RecipeError(f"{what}: `pis` is empty")
    for p in pi_list:
        if p != "none" and p not in pis:
            close = difflib.get_close_matches(p, list(pis), n=3)
            raise RecipeError(
                f"{what}: pi {p!r} is not defined in pis.yaml" + (f" (did you mean {close}?)" if close else "")
            )
    keys = dict(file_keys)
    keys.update(raw.get("keys") or {})
    seed_var = raw.get("seed_var") or file_seed_var
    extra_raw = raw.get("extra") or {}
    if not isinstance(extra_raw, dict):
        raise RecipeError(f"{what}: `extra` must be a mapping key -> values")
    extra = [_parse_axis(k, v, keys, what) for k, v in extra_raw.items()]
    if len({a.key for a in extra}) != len(extra):
        raise RecipeError(f"{what}: two extra axes map to the same arm-id key")
    objective = _none(raw.get("objective"))
    meta = {k: v for k, v in raw.items() if k not in _ENTRY_KEYS}
    return TableEntry(
        table=table,
        students=students,
        pis=pi_list,
        objective=objective,
        env=_as_env(raw.get("env"), f"{what}.env"),
        extra=extra,
        seeds=_parse_seeds(raw.get("seeds"), seed_var, what),
        exclude=_as_list(raw.get("exclude"), f"{what}.exclude"),
        status=status,
        paper=_none(raw.get("paper")),
        source=what,
        meta=meta,
    )


def _expand_entry(bench: str, entry: TableEntry, pis: Mapping[str, Pi]) -> Tuple[List[Arm], List[str]]:
    arms, skipped = [], []
    axes = [[(a.key, v) for v in a.values] for a in entry.extra]
    seeds: List[Optional[SeedSpec]] = list(entry.seeds) if entry.seeds else [None]
    for student, pi in itertools.product(entry.students, entry.pis):
        if pi != "none" and pis[pi].for_student(student) is None:
            skipped.append(f"{bench}/{entry.table}/{student}/{pi} ({entry.source}: pis.yaml has no {student} entry)")
            continue
        for combo in itertools.product(*axes) if axes else [()]:
            for seed in seeds:
                aid = ArmId(
                    bench,
                    entry.table,
                    student,
                    pi,
                    tuple((k, v.text) for k, v in combo),
                    None if seed is None else seed.index,
                )
                text = str(aid)
                rest = text.split("/", 2)[2]
                if any(fnmatch.fnmatch(text, pat) or fnmatch.fnmatch(rest, pat) for pat in entry.exclude):
                    continue
                arms.append(Arm(aid, entry, list(combo), seed))
    return arms, skipped


def load_bench(name: str, recipes_dir: Optional[Path] = None) -> Bench:
    recipes_dir = Path(recipes_dir) if recipes_dir else default_recipes_dir()
    root = recipes_dir / name
    arms_path = root / "arms.yaml"
    if not arms_path.is_file():
        avail = list_benches(recipes_dir)
        raise RecipeError(f"no recipes for bench {name!r} ({arms_path} missing); available: {avail}")
    students = sorted(p.stem for p in (root / "students").glob("*.env")) if (root / "students").is_dir() else []
    pis = load_pis(root / "pis.yaml", students)
    raw = load_yaml(arms_path)
    file_keys, seed_var, defaults = {}, DEFAULT_SEED_VAR, {}
    if isinstance(raw, dict):
        file_keys = raw.get("keys") or {}
        seed_var = raw.get("seed_var") or DEFAULT_SEED_VAR
        defaults = raw.get("defaults") or {}
        tables = raw.get("tables", raw.get("arms"))
    else:
        tables = raw
    if not isinstance(tables, list):
        raise RecipeError(f"{arms_path}: expected a list of table entries (top-level or under `tables:`)")
    if not isinstance(file_keys, dict) or not isinstance(defaults, dict):
        raise RecipeError(f"{arms_path}: `keys` and `defaults` must be mappings")
    entries = [_parse_entry(t, i, arms_path, file_keys, seed_var, defaults, pis) for i, t in enumerate(tables)]
    arms: Dict[str, Arm] = {}
    skipped: List[str] = []
    for e in entries:
        got, skip = _expand_entry(name, e, pis)
        skipped += skip
        for a in got:
            key = str(a.arm_id)
            if key in arms:
                raise RecipeError(
                    f"{arms_path}: arm {key} is generated twice ({arms[key].entry.source} and {e.source})"
                )
            arms[key] = a
    return Bench(name, root, pis, entries, arms, skipped)


def list_benches(recipes_dir: Optional[Path] = None) -> List[str]:
    recipes_dir = Path(recipes_dir) if recipes_dir else default_recipes_dir()
    if not recipes_dir.is_dir():
        return []
    return sorted(p.name for p in recipes_dir.iterdir() if (p / "arms.yaml").is_file())


# ------------------------------------------------------------------------------------------- resolving
@dataclass
class Layer:
    name: str
    values: Dict[str, str]


@dataclass
class Resolved:
    arm: Arm
    env: Dict[str, str]
    provenance: Dict[str, List[Tuple[str, str]]]
    layers: List[str]
    warnings: List[str]


def _env_file_layer(path: Path, label_root: Path) -> Layer:
    rel = path.relative_to(label_root) if path.is_relative_to(label_root) else path
    return Layer(str(rel), {k: v for k, (v, _) in parse_env_file(path).items()})


def _overlay_layers(bench: Bench, names: Sequence[str], recipes_dir: Path) -> List[Layer]:
    layers = []
    for name in names:
        p = Path(name).expanduser()
        if p.suffix == ".env" and p.is_file():
            layers.append(Layer(f"overlay:{p}", {k: v for k, (v, _) in parse_env_file(p).items()}))
            continue
        found = False
        for cand in (recipes_dir / "common" / "overlays" / f"{name}.env", bench.root / "overlays" / f"{name}.env"):
            if cand.is_file():
                lay = _env_file_layer(cand, recipes_dir)
                lay.name = f"overlay:{lay.name}"
                layers.append(lay)
                found = True
        if not found:
            avail = sorted(
                {
                    q.stem
                    for d in (recipes_dir / "common" / "overlays", bench.root / "overlays")
                    if d.is_dir()
                    for q in d.glob("*.env")
                }
            )
            raise RecipeError(f"unknown overlay {name!r}; available: {avail}")
    return layers


def overlay_run_name_suffix(names: Sequence[str]) -> str:
    """``__<name>`` per overlay (the file stem for an overlay given as a ``.env`` path), in order, without
    repeats. Callers pass only the overlays that set at least one key (see :func:`resolve`)."""
    out: List[str] = []
    for name in names:
        p = Path(name)
        slug = re.sub(r"[^A-Za-z0-9_.+-]", "-", p.stem if p.suffix == ".env" else name)
        if slug and slug not in out:
            out.append(slug)
    return "".join(f"__{s}" for s in out)


def _absolute_input_paths(overrides: Mapping[str, str], extra_keys: Iterable[str] = ()) -> Dict[str, str]:
    out = dict(overrides)
    for k in set(INPUT_PATH_KEYS) | set(LOCAL_OR_HUB_KEYS) | set(extra_keys):
        v = out.get(k)
        if v and v not in NULLS and not v.startswith(("$", "/")) and "://" not in v:
            path = os.path.expanduser(v)
            if k in LOCAL_OR_HUB_KEYS and not os.path.exists(path):
                continue
            out[k] = os.path.abspath(path)
    return out


def parse_overrides(items: Iterable[str]) -> Dict[str, str]:
    out = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not _KEY_RX.match(k):
            raise RecipeError(f"override {it!r}: expected KEY=VALUE")
        out[k] = v
    return out


def _template(value: str, fields: Mapping[str, str]) -> str:
    if "{" not in value:
        return value
    rx = re.compile(r"(?<!\$)\{(" + "|".join(re.escape(k) for k in fields) + r")\}")
    return rx.sub(lambda m: fields[m.group(1)], value)


def _expand(value: str, ctx: Mapping[str, str], where: str) -> str:
    if "$(" in value or "`" in value:
        raise RecipeError(f"{where}: command substitution is not allowed in recipes ({value!r})")

    def sub(m):
        name = m.group(1) or m.group(2)
        if name not in ctx:
            raise RecipeError(
                f"{where}: ${name} is not allowed in recipes (only ${{OPSD_DATA_ROOT}} and "
                f"${{RUN_DIR}} are expanded); write the literal value ({value!r})"
            )
        return ctx[name]

    out = _REF_RX.sub(sub, value)
    if "${" in out:
        raise RecipeError(f"{where}: unsupported shell expansion in {value!r}")
    return out


def resolve(
    arm_id: str,
    overlays: Sequence[str] = (),
    overrides: Optional[Mapping[str, str]] = None,
    recipes_dir: Optional[Path] = None,
    environ: Optional[Mapping[str, str]] = None,
    allow_planned: bool = False,
) -> Resolved:
    """Resolve ``arm_id`` to its full environment. Raises :class:`RecipeError` on any problem."""
    recipes_dir = Path(recipes_dir) if recipes_dir else default_recipes_dir()
    environ = dict(os.environ if environ is None else environ)
    overrides = dict(overrides or {})
    aid = parse_arm_id(arm_id)
    bench = load_bench(aid.bench, recipes_dir)
    arm = bench.arms.get(str(aid))
    if arm is None:
        close = difflib.get_close_matches(str(aid), list(bench.arms), n=5, cutoff=0.6)
        hint = f"; closest: {close}" if close else ""
        skipped = [s for s in bench.skipped if s.startswith(f"{aid.bench}/{aid.table}/{aid.student}/{aid.pi} ")]
        if skipped:
            hint += f"; skipped while expanding: {skipped[0]}"
        raise RecipeError(f"arm {aid} is not generated by {bench.root / 'arms.yaml'}{hint}")
    warnings: List[str] = []
    if arm.status == "planned" and not allow_planned:
        raise RecipeError(
            f"arm {aid} has status: planned ({arm.entry.source}); it was never run, so there is "
            "nothing to reproduce. Refusing (use --allow-planned to inspect it anyway)."
        )
    if arm.status == "in_progress":
        warnings.append(
            f"arm {aid} is in_progress ({arm.entry.source}): its paper/result numbers may be " "partial or missing"
        )

    layers: List[Layer] = (
        [_env_file_layer(bench.root / "base.env", recipes_dir)] if (bench.root / "base.env").is_file() else []
    )
    if not layers:
        warnings.append(f"{bench.root / 'base.env'} not found; no bench defaults applied")
    student_env = bench.root / "students" / f"{aid.student}.env"
    if not student_env.is_file():
        raise RecipeError(f"arm {aid}: missing {student_env}")
    layers.append(_env_file_layer(student_env, recipes_dir))
    if arm.entry.objective:
        obj = bench.root / "objectives" / f"{arm.entry.objective}.env"
        if not obj.is_file():
            raise RecipeError(f"arm {aid}: objective {arm.entry.objective!r} -> missing {obj}")
        layers.append(_env_file_layer(obj, recipes_dir))
    hint_vars = {HINT_VAR}
    if aid.pi != "none":
        pi = bench.pis[aid.pi]
        ps = pi.for_student(aid.student)
        hint_vars.add(ps.hint_var)
        vals = dict(pi.env)
        vals.update(ps.env)
        if ps.hint:
            hint = ps.hint
            if not (hint.startswith("$") or os.path.isabs(hint)):
                hint = "${OPSD_DATA_ROOT}/" + hint.lstrip("/")
            vals[ps.hint_var] = hint
        if ps.ref_seq_cap:
            vals[REF_CAP_VAR] = ps.ref_seq_cap
        if ps.content_sha256:
            vals[HINT_SHA_VAR] = ps.content_sha256
        which = aid.student if aid.student in pi.students else "default"
        layers.append(Layer(f"{aid.bench}/pis.yaml:{aid.pi}[{which}]", vals))
    if arm.entry.env:
        layers.append(Layer(f"{arm.entry.source}.env", dict(arm.entry.env)))
    for key, val in arm.extra_values:
        layers.append(Layer(f"{arm.entry.source}.extra.{key}={val.text}", dict(val.env)))
    if arm.seed is not None and arm.seed.env:
        layers.append(Layer(f"{arm.entry.source}.seeds[{arm.seed.index}]", dict(arm.seed.env)))
    shaping_overlays = []  # overlays that set something: they get their own default RUN_NAME
    for name in overlays:
        ov = _overlay_layers(bench, [name], recipes_dir)
        if any(lay.values for lay in ov):
            shaping_overlays.append(name)
        layers += ov
    if overrides:
        layers.append(Layer("cli", _absolute_input_paths(overrides, hint_vars)))

    env: Dict[str, str] = {}
    prov: Dict[str, List[Tuple[str, str]]] = {}
    for lay in layers:
        for k, v in lay.values.items():
            env[k] = v
            prov.setdefault(k, []).append((lay.name, v))

    # A hint file given on the command line is not the published file the recipe's content hash describes:
    # clear the declared hash (unless the CLI sets one too), or preflight would refuse the user's own file.
    moved = sorted(v for v in hint_vars if v in overrides)
    if moved and env.get(HINT_SHA_VAR) and HINT_SHA_VAR not in overrides:
        env[HINT_SHA_VAR] = ""
        prov.setdefault(HINT_SHA_VAR, []).append((f"cleared: {', '.join(moved)} set on the command line", ""))
        warnings.append(
            f"{', '.join(moved)} is overridden on the command line, so the recipe's {HINT_SHA_VAR} (the hash of "
            f"the published hint file) is cleared; pass {HINT_SHA_VAR}=<sha256> to have preflight check your file"
        )

    # Location keys: CLI > ambient > recipe > default (RUN_DIR: CLI > RUN_ROOT/RUN_NAME).
    def locate(key: str, default: str) -> str:
        if key in overrides:
            return overrides[key]
        if environ.get(key):
            if key in env and env[key] != environ[key]:
                prov.setdefault(key, []).append(("env", environ[key]))
            elif key not in env:
                prov.setdefault(key, []).append(("env", environ[key]))
            return environ[key]
        if key in env:
            return env[key]
        prov.setdefault(key, []).append(("default", default))
        return default

    data_root = os.path.abspath(os.path.expanduser(locate("OPSD_DATA_ROOT", default_data_root(environ))))
    run_root = os.path.abspath(os.path.expanduser(locate("RUN_ROOT", "./runs")))
    run_name = locate("RUN_NAME", aid.run_name + overlay_run_name_suffix(shaping_overlays))
    if "RUN_DIR" in overrides:
        run_dir = os.path.abspath(os.path.expanduser(overrides["RUN_DIR"]))
    else:
        run_dir = os.path.join(run_root, run_name)
        if environ.get("RUN_DIR") and os.path.abspath(environ["RUN_DIR"]) != run_dir:
            warnings.append(f"ambient RUN_DIR={environ['RUN_DIR']} ignored; RUN_DIR is $RUN_ROOT/$RUN_NAME = {run_dir}")
        prov.setdefault("RUN_DIR", []).append(("derived", run_dir))
    env.update(OPSD_DATA_ROOT=data_root, RUN_ROOT=run_root, RUN_NAME=run_name, RUN_DIR=run_dir)

    fields = {"bench": aid.bench, "table": aid.table, "student": aid.student, "pi": aid.pi, "seed": aid.seed or ""}
    fields.update(dict(aid.extras))
    ctx = {"OPSD_DATA_ROOT": data_root, "RUN_DIR": run_dir}
    for k in list(env):
        if k in ("OPSD_DATA_ROOT", "RUN_DIR"):
            continue
        src = prov[k][-1][0] if k in prov else "?"
        env[k] = _expand(_template(env[k], fields), ctx, f"{k} (from {src})")
    if "/" in env["RUN_NAME"] or env["RUN_NAME"] in ("", ".", ".."):
        raise RecipeError(f"RUN_NAME={env['RUN_NAME']!r} must be a single path component")
    env["OPSD_ARM_ID"] = str(aid)
    env["OPSD_BENCH"] = aid.bench
    prov.setdefault("OPSD_ARM_ID", []).append(("derived", str(aid)))
    prov.setdefault("OPSD_BENCH", []).append(("derived", aid.bench))
    return Resolved(arm=arm, env=env, provenance=prov, layers=[lay.name for lay in layers], warnings=warnings)


def hermetic_violations(res: Resolved, overrides: Mapping[str, str], environ: Mapping[str, str]) -> List[str]:
    """Ambient variables that would silently change (or be silently ignored by) this run."""
    out = []
    for name, val in sorted(environ.items()):
        if name in overrides or name in LOCATION_KEYS or name in DERIVED_KEYS:
            continue
        allowed = name in HERMETIC_ALLOWED or name.startswith(HERMETIC_ALLOWED_PREFIXES)
        if name.startswith("OPSD_") and not allowed and name not in res.env:
            out.append(
                f"ambient {name}={val!r} is not part of recipe {res.env['OPSD_ARM_ID']}; pass it as "
                f"{name}=... on the command line or unset it"
            )
        elif name in res.env and res.env[name] != val and not allowed:
            out.append(
                f"ambient {name}={val!r} differs from the recipe value {res.env[name]!r} (from "
                f"{res.provenance[name][-1][0]}); pass {name}=... on the command line to override, "
                "or unset it"
            )
    return out


# ------------------------------------------------------------------------------------------ provenance
def hint_content_sha256(path: str) -> str:
    """sha256 of the canonical JSON ``{task_id: task_hint}`` (step hints and meta excluded), i.e. what the
    trainer actually reads; identical to the bundle manifest's ``content_sha256``."""
    raw = json.loads(Path(path).read_text())
    th = {}
    for k, v in raw.items():
        if k == "__schema__":
            continue
        t = v if isinstance(v, str) else (v.get("task") if isinstance(v, dict) else None)
        if t:
            th[k] = t
    return hashlib.sha256(json.dumps(th, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cmd(args: Sequence[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(list(args), cwd=cwd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def skyrl_checkout() -> Optional[Path]:
    """The git checkout of the SkyRL the interpreter imports (``opsd._repo.skyrl_root()``), or None when
    ``skyrl`` is not installed or is not installed from the top of a git checkout (a wheel, a tarball)."""
    try:
        root = _repo.skyrl_root()
    except FileNotFoundError:
        return None
    top = _cmd(["git", "rev-parse", "--show-toplevel"], cwd=root) if root.is_dir() else None
    return root if top and Path(top).resolve() == root else None


# Run-wide settings exported by scripts/common/bootstrap.sh (only if unset) that can change results or
# their reproducibility; build_provenance records the values the run actually saw.
RUNTIME_ENV_NAMES = (
    "PYTHONHASHSEED",
    "VLLM_ALLREDUCE_USE_SYMM_MEM",
    "VLLM_HOST_IP",
    "HYDRA_FULL_ERROR",
    "RAY_TMPDIR",
    "LOGGER",
)
RUNTIME_ENV_PREFIXES = ("NCCL_",)
_SECRET_NAME_RX = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.I)


def runtime_env_record(environ: Optional[Mapping[str, str]] = None) -> dict:
    """The bootstrap's run-wide environment as seen by this process (unset -> None). ``ray_mode`` is
    ``local`` (the driver starts its own Ray) or ``external`` (RAY_ADDRESS names a cluster; not recorded)."""
    environ = os.environ if environ is None else environ
    rec: Dict[str, object] = {k: environ.get(k) for k in RUNTIME_ENV_NAMES}
    for k in sorted(environ):
        if k.startswith(RUNTIME_ENV_PREFIXES) and not _SECRET_NAME_RX.search(k):
            rec[k] = environ[k]
    rec["ray_mode"] = "local" if environ.get("RAY_ADDRESS", "") in ("", "local") else "external"
    try:
        import resource

        rec["open_files_soft_limit"] = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    except (ImportError, OSError, ValueError):  # pragma: no cover - non-POSIX
        rec["open_files_soft_limit"] = None
    return rec


def build_provenance(resolved_env_path: Path, verify_hints: bool = True, query_gpu: bool = True) -> dict:
    env = read_resolved_env(resolved_env_path)
    root = repo_root()
    sha = _cmd(["git", "rev-parse", "HEAD"], cwd=root)
    # --no-optional-locks: a status query must not rewrite the index (provenance is read-only, and the SkyRL
    # checkout is never modified).
    status = ["git", "--no-optional-locks", "status", "--porcelain", "--untracked-files=no"]
    dirty = _cmd(status, cwd=root)
    # SkyRL is an external checkout installed editable into the venv (not part of this repository): record the
    # one this interpreter imports, against the commit the repository is pinned to.
    skyrl = skyrl_checkout()
    skyrl_sha = _cmd(["git", "rev-parse", "HEAD"], cwd=skyrl) if skyrl else None
    skyrl_pinned = _repo.SKYRL_COMMIT
    skyrl_dirty = _cmd(status, cwd=skyrl) if skyrl else None
    versions = {}
    try:
        from importlib import metadata

        for pkg in ("torch", "vllm", "ray", "transformers", "flash-attn", "megatron-core"):
            try:
                versions[pkg] = metadata.version(pkg)
            except metadata.PackageNotFoundError:
                pass
    except Exception:  # noqa: BLE001 - provenance must never fail a launch
        pass
    hints = {}
    for var in sorted(k for k in env if k.endswith("HINTS_PATH") and env[k]):
        p = env[var]
        rec = {"path": p, "exists": os.path.isfile(p), "file_sha256": _sha256_file(Path(p))}
        if verify_hints and rec["exists"]:
            try:
                rec["content_sha256"] = hint_content_sha256(p)
            except (ValueError, OSError, AttributeError) as e:
                rec["content_sha256_error"] = str(e)
        if var == HINT_VAR and env.get(HINT_SHA_VAR):
            rec["declared_content_sha256"] = env[HINT_SHA_VAR]
            if "content_sha256" in rec:
                rec["content_sha256_matches"] = rec["content_sha256"] == env[HINT_SHA_VAR]
        hints[var] = rec
    gpu = None
    if query_gpu:
        q = _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
        gpu = q.splitlines() if q else None
    return {
        "arm_id": env.get("OPSD_ARM_ID"),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {"sha": sha, "dirty": bool(dirty) if dirty is not None else None},
        # The SkyRL checkout the venv imports: its commit, the commit this repository pins, and local edits.
        "skyrl": {
            "path": str(skyrl) if skyrl else None,
            "sha": skyrl_sha,
            "pinned_sha": skyrl_pinned,
            "matches_pin": (skyrl_sha == skyrl_pinned) if skyrl_sha and skyrl_pinned else None,
            "dirty": bool(skyrl_dirty) if skyrl_dirty is not None else None,
        },
        "uv_lock_sha256": _sha256_file(root / "uv.lock"),
        "resolved_env": {"path": str(resolved_env_path), "sha256": _sha256_file(Path(resolved_env_path))},
        "hints": hints,
        "python": sys.version.split()[0],
        "packages": versions,
        "gpus": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "runtime_env": runtime_env_record(),
    }


# ------------------------------------------------------------------------------------------------- CLI
def _filter_arms(benches: Iterable[Bench], table=None, status=None) -> List[Arm]:
    out = []
    for b in benches:
        for a in b.arms.values():
            if table and a.arm_id.table != table:
                continue
            if status and a.status != status:
                continue
            out.append(a)
    return out


def _cmd_list(a) -> int:
    names = [a.bench] if a.bench else list_benches(a.recipes_dir)
    if not names:
        print(f"no benches under {a.recipes_dir or default_recipes_dir()}", file=sys.stderr)
        return 1
    for arm in _filter_arms([load_bench(n, a.recipes_dir) for n in names], a.table, a.status):
        if a.long:
            print("\t".join([str(arm.arm_id), arm.status, arm.entry.objective or "-", arm.entry.paper or "-"]))
        else:
            print(arm.arm_id)
    return 0


def _resolve_from_args(a, overrides):
    res = resolve(a.arm_id, a.overlay, overrides, a.recipes_dir, allow_planned=a.allow_planned)
    for w in res.warnings:
        print(f"WARNING: {w}", file=sys.stderr)
    return res


def _cmd_show(a, overrides) -> int:
    res = _resolve_from_args(a, overrides)
    arm = res.arm
    if a.json:
        print(
            json.dumps(
                {
                    "arm_id": str(arm.arm_id),
                    "status": arm.status,
                    "paper": arm.entry.paper,
                    "objective": arm.entry.objective,
                    "table_entry": arm.entry.source,
                    "layers": res.layers,
                    "env": res.env,
                    "provenance": {k: [list(x) for x in v] for k, v in res.provenance.items()},
                },
                indent=2,
            )
        )
        return 0
    print(f"# arm:       {arm.arm_id}")
    print(f"# status:    {arm.status}")
    print(f"# paper:     {arm.entry.paper or '-'}")
    print(f"# objective: {arm.entry.objective or '-'}   (table entry {arm.entry.source})")
    if arm.entry.meta:
        print(f"# meta:      {arm.entry.meta}")
    print("# layers:    " + " < ".join(res.layers))
    width = max((len(k) for k in res.env), default=0)
    for k in sorted(res.env):
        chain = res.provenance.get(k, [])
        src = chain[-1][0] if chain else "?"
        over = ""
        if len(chain) > 1:
            over = "  (overrides " + ", ".join(f"{n}={v!r}" for n, v in chain[:-1]) + ")"
        print(f"{k.ljust(width)} = {res.env[k]!r:<40} [{src}]{over}")
    return 0


def _cmd_resolve(a, overrides) -> int:
    res = _resolve_from_args(a, overrides)
    if a.hermetic:
        bad = hermetic_violations(res, overrides, os.environ)
        if bad:
            for b in bad:
                print(f"HERMETIC: {b}", file=sys.stderr)
            return 3
    text = (
        format_env(res.env)
        if a.format == "env"
        else (
            format_env(res.env, quote=False)
            if a.format == "plain"
            else json.dumps(res.env, indent=2, sort_keys=True) + "\n"
        )
    )
    sys.stdout.write(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    if a.provenance_out:
        Path(a.provenance_out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.provenance_out).write_text(
            json.dumps(
                {
                    "arm_id": str(res.arm.arm_id),
                    "layers": res.layers,
                    "warnings": res.warnings,
                    "provenance": {k: [list(x) for x in v] for k, v in res.provenance.items()},
                },
                indent=2,
            )
            + "\n"
        )
    return 0


def _cmd_validate(a) -> int:
    names = [a.bench] if a.bench else list_benches(a.recipes_dir)
    errors = 0
    for n in names:
        try:
            b = load_bench(n, a.recipes_dir)
        except RecipeError as e:
            print(f"ERROR {n}: {e}")
            errors += 1
            continue
        print(f"{n}: {len(b.entries)} table entries, {len(b.arms)} arms, {len(b.pis)} pis")
        for s in b.skipped:
            print(f"  skipped {s}")
        for arm in b.arms.values():
            try:
                res = resolve(str(arm.arm_id), (), {}, a.recipes_dir, allow_planned=True)
            except RecipeError as e:
                print(f"  ERROR {arm.arm_id}: {e}")
                errors += 1
                continue
            if a.check_files:
                for var in (
                    k for k in res.env if k.endswith("_PATH") and res.env[k].startswith(res.env["OPSD_DATA_ROOT"])
                ):
                    if not os.path.exists(res.env[var]):
                        print(f"  MISSING {arm.arm_id}: {var}={res.env[var]}")
                        errors += 1
                p = res.env.get(HINT_VAR)
                want = res.env.get(HINT_SHA_VAR)
                if p and want and os.path.isfile(p) and hint_content_sha256(p) != want:
                    print(f"  SHA MISMATCH {arm.arm_id}: {p} content_sha256 != {want}")
                    errors += 1
    print("OK" if not errors else f"{errors} error(s)")
    return 1 if errors else 0


def _cmd_provenance(a) -> int:
    rec = build_provenance(Path(a.resolved_env), verify_hints=not a.no_verify_hints, query_gpu=not a.no_gpu_query)
    text = json.dumps(rec, indent=2, sort_keys=True) + "\n"
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    else:
        sys.stdout.write(text)
    bad = [v for v, r in rec["hints"].items() if r.get("content_sha256_matches") is False]
    if bad and a.strict:
        print(f"ERROR: hint content_sha256 differs from the recipe for {bad}", file=sys.stderr)
        return 4
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m opsd.recipes",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--recipes-dir",
        type=Path,
        default=None,
        help="recipes tree (default: $OPSD_RECIPES_DIR or <repo>/opsd/recipes)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list", help="list arm ids")
    p.add_argument("--bench")
    p.add_argument("--table")
    p.add_argument("--status", choices=STATUSES)
    p.add_argument("--long", action="store_true", help="also print status, objective and paper cell")
    for name, hlp in (("show", "resolved env with per-key provenance"), ("resolve", "sorted KEY=VALUE on stdout")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("arm_id")
        p.add_argument("overrides", nargs="*", metavar="KEY=VAL")
        p.add_argument("--overlay", action="append", default=[], help="overlay name or .env path (repeatable)")
        p.add_argument("--allow-planned", action="store_true", help="do not refuse status: planned")
        if name == "show":
            p.add_argument("--json", action="store_true")
        else:
            p.add_argument(
                "--format",
                choices=("env", "plain", "json"),
                default="env",
                help="env: shell-quoted where needed (source-able); plain: raw values",
            )
            p.add_argument(
                "--hermetic",
                action="store_true",
                help="fail if the ambient environment carries OPSD_* or recipe keys that are "
                "neither in the recipe nor passed as KEY=VAL",
            )
            p.add_argument("--out", help="also write the output to this file")
            p.add_argument("--provenance-out", help="write per-key provenance JSON here")
    p = sub.add_parser("validate", help="load every table, resolve every arm")
    p.add_argument("--bench")
    p.add_argument("--check-files", action="store_true", help="hint/task files must exist under OPSD_DATA_ROOT")
    p = sub.add_parser("provenance", help="write provenance.json for a resolved env")
    p.add_argument("resolved_env")
    p.add_argument("--out")
    p.add_argument("--no-verify-hints", action="store_true", help="skip recomputing hint content_sha256")
    p.add_argument("--no-gpu-query", action="store_true")
    p.add_argument("--strict", action="store_true", help="exit 4 if a hint content_sha256 differs from the recipe")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_parser()
    a, rest = ap.parse_known_args(argv)
    try:
        if a.cmd in ("show", "resolve"):
            stray = [r for r in rest if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", r)]
            if stray:
                ap.error(f"unrecognized arguments: {' '.join(stray)}")
            overrides = parse_overrides(list(a.overrides) + rest)
            return _cmd_show(a, overrides) if a.cmd == "show" else _cmd_resolve(a, overrides)
        if rest:
            ap.error(f"unrecognized arguments: {' '.join(rest)}")
        return {"list": _cmd_list, "validate": _cmd_validate, "provenance": _cmd_provenance}[a.cmd](a)
    except RecipeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
