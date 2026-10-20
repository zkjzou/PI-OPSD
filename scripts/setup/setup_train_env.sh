#!/usr/bin/env bash
# =====================================================================================================
# setup_train_env.sh -- build (or check) the training environment <repo root>/.venv. Safe to re-run: every
# step is idempotent, and nothing that is already in place is removed except what uv.lock says must go.
#
#   bash scripts/setup/setup_train_env.sh                    # FSDP environment (every 4B student)
#   bash scripts/setup/setup_train_env.sh --extra megatron   # Megatron environment (the 30B student)
#   bash scripts/setup/setup_train_env.sh --tau2 --dev       # + tau2-bench's runtime deps, + the test tools
#   bash scripts/setup/setup_train_env.sh --exact            # exactly these extras: drop the ones .venv has
#   bash scripts/setup/setup_train_env.sh --check [FLAGS]    # report the state only (same FLAGS as the build)
#
# Extras already installed are KEPT (uv sync alone would remove every extra it is not given): a re-run, or
# setup_tau2.sh, keeps `dev` when .venv has pytest, `tau2` when it has addict / deepdiff, and the Megatron
# backend when it has megatron-core and neither --extra nor OPSD_UV_EXTRA names a backend. So a re-run never
# drops the test tools or switches a Megatron .venv back to FSDP. An explicit --extra fsdp|megatron switches
# the backend; --exact syncs exactly the extras of this call.
#
# Steps
#   1. SkyRL: upstream NovaSky-AI/SkyRL, unmodified, at the commit this repository is pinned to (SKYRL_COMMIT
#      in opsd/_repo.py), checked out at $SKYRL_DIR -- by default the directory that [tool.uv.sources] in
#      pyproject.toml installs it from, <repo root>/../SkyRL-ecdaa946 as shipped. When it is missing:
#      `git clone $SKYRL_GIT_URL $SKYRL_DIR` + `git checkout <pin>`. When it exists it must be a git checkout
#      AT the pin with no local changes, or the script refuses: it never fetches into, resets or edits a
#      checkout.
#   2. From the repository root: uv sync --frozen --extra <fsdp|megatron> [--extra tau2] [--extra dev] -- the
#      environment of uv.lock, with skyrl / skyrl_gym installed editable from $SKYRL_DIR and opsd editable
#      from this repository. The extras are those of this call plus the ones .venv already has (above); with
#      --exact, exactly those of this call, so a later --exact call without --tau2 removes the tau2 packages.
#   3. Verification from a neutral directory, without PYTHONPATH: skyrl, skyrl_gym and opsd resolve to
#      $SKYRL_DIR/skyrl, $SKYRL_DIR/skyrl-gym/skyrl_gym and <repo root>/opsd; torch and vllm (megatron.core
#      with --extra megatron, tau2's dependencies with --tau2, pytest with --dev) are importable; and
#      `uv sync --frozen --check` with the same extras has nothing to change.
#
# Options / environment
#   --extra fsdp|megatron   the training backend (default: $OPSD_UV_EXTRA, else the one .venv has -- megatron
#                           when megatron-core is installed --, else fsdp). The two conflict: one .venv holds one.
#   --tau2                  also the `tau2` extra (tau2-bench's import-time dependencies; setup_tau2.sh passes it)
#   --dev                   also the `dev` extra (pytest, pre-commit)
#   --exact                 do not keep the extras .venv already has: exactly --extra / --tau2 / --dev
#   --check                 print the state of every step; exit 0 when the environment is complete, else 1
#   SKYRL_DIR               the SkyRL checkout (default: the [tool.uv.sources] path of pyproject.toml). To use
#                           another location, point BOTH paths there first -- skyrl = { path = "<dir>", ... }
#                           and skyrl-gym = { path = "<dir>/skyrl-gym", ... } -- and run `uv lock`; the script
#                           refuses a SKYRL_DIR that pyproject.toml does not install from.
#   SKYRL_GIT_URL           where a missing checkout is cloned from (default https://github.com/NovaSky-AI/SkyRL)
#   UV                      the uv binary (default: uv on PATH)
#   UV_PYTHON               interpreter for a NEW .venv (default 3.12, as in CI); an existing .venv is kept
# Needs network access (GitHub, PyPI, the PyTorch / vLLM / flashinfer indexes) except with --check. Writes only
# <repo root>/.venv, a new checkout at $SKYRL_DIR (uv's editable-build files inside it are covered by SkyRL's
# own .gitignore) and uv's cache. The launchers run $OPSD_PYTHON, else <repo root>/.venv/bin/python.
# Exit status 0 = success.
# =====================================================================================================
set -eo pipefail

log() { printf '[setup-train-env] %s\n' "$*" >&2; }
warn() { printf '[setup-train-env] WARNING: %s\n' "$*" >&2; }
die() { printf '[setup-train-env] ERROR: %s\n' "$*" >&2; exit 1; }
usage() { awk 'NR > 2 && /^# =====/ { exit } NR > 2 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; }

CHECK=0
TAU2=0
DEV=0
EXACT=0
EXTRA=""
while [ "$#" -gt 0 ]; do
    case "$1" in
        --check) CHECK=1; shift ;;
        --tau2) TAU2=1; shift ;;
        --dev) DEV=1; shift ;;
        --exact) EXACT=1; shift ;;
        --extra) [ -n "${2:-}" ] || die "--extra needs a value (fsdp | megatron)"; EXTRA="$2"; shift 2 ;;
        --extra=*) EXTRA="${1#--extra=}"; shift ;;
        -h | --help) usage; exit 0 ;;
        *) usage >&2; die "unknown argument: $1" ;;
    esac
done
EXTRA_FROM=""  # where an explicit backend came from; empty = not named, so .venv's own backend is kept
if [ -n "$EXTRA" ]; then
    EXTRA_FROM="--extra"
elif [ -n "${OPSD_UV_EXTRA:-}" ]; then
    EXTRA="$OPSD_UV_EXTRA"
    EXTRA_FROM="OPSD_UV_EXTRA"
fi
case "${EXTRA:-fsdp}" in fsdp | megatron) ;; *) die "${EXTRA_FROM:---extra} $EXTRA: expected fsdp or megatron" ;; esac

# ---- the repository root (the launchers' predicate) --------------------------------------------------------
# Physical paths (pwd -P): ../SkyRL-ecdaa946 below must name the directory uv and git resolve it to, which is
# relative to the PHYSICAL repository directory when the clone is reached through a symlink.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT="$_here"
while [ "$ROOT" != "/" ] && ! { [ -f "$ROOT/pyproject.toml" ] && [ -f "$ROOT/opsd/__init__.py" ] &&
    [ -f "$ROOT/opsd/skyrl_ext/__init__.py" ]; }; do
    ROOT="$(dirname "$ROOT")"
done
[ "$ROOT" != "/" ] ||
    die "repository root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py) not found above $_here"
[ -f "$ROOT/uv.lock" ] || die "$ROOT/uv.lock is missing from this checkout"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
UV="${UV:-uv}"
SKYRL_GIT_URL="${SKYRL_GIT_URL:-https://github.com/NovaSky-AI/SkyRL}"
PIN="$(sed -n 's/^SKYRL_COMMIT = "\([0-9a-f]\{40\}\)".*/\1/p' "$ROOT/opsd/_repo.py" 2>/dev/null | head -n 1)"
[ -n "$PIN" ] || die "no SKYRL_COMMIT = \"<40-hex sha>\" line in $ROOT/opsd/_repo.py (the SkyRL pin)"

_problems=0
_bad() {  # MSG: a problem (--check: counted; otherwise fatal)
    if [ "$CHECK" = 1 ]; then
        printf 'FAIL %s\n' "$*"
        _problems=$((_problems + 1))
    else
        die "$*"
    fi
}
_ok() { [ "$CHECK" = 1 ] && printf 'ok   %s\n' "$*"; return 0; }
_info() { [ "$CHECK" = 1 ] && printf 'info %s\n' "$*"; return 0; }

_abspath() {  # PATH [BASE]: absolute (relative to BASE, default $PWD), the existing part with symlinks resolved
    local p="$1" d
    case "$p" in /*) ;; *) p="${2:-$PWD}/$p" ;; esac
    if [ -d "$p" ]; then
        (cd "$p" && pwd -P)
        return 0
    fi
    d="$(dirname "$p")"
    if [ -d "$d" ]; then printf '%s/%s\n' "$(cd "$d" && pwd -P)" "$(basename "$p")"; else printf '%s\n' "$p"; fi
}

_uv_source() {  # NAME: the path [tool.uv.sources] in pyproject.toml installs NAME from ("" if there is none)
    sed -n "s/^$1[[:space:]]*=[[:space:]]*{[[:space:]]*path[[:space:]]*=[[:space:]]*\"\\([^\"]*\\)\".*/\\1/p" \
        "$ROOT/pyproject.toml" | head -n 1
}

_installed_extras() {  # the extras the existing .venv has, one per line (megatron / tau2 / dev); none: no .venv
    [ -x "$PY" ] || return 0
    # Installed distributions only (nothing is imported), from a neutral directory without PYTHONPATH.
    (cd / && env -u PYTHONPATH "$PY" - 2>/dev/null <<'PY') || :
import importlib.metadata as md


def has(*names):
    for name in names:
        try:
            md.distribution(name)
            return True
        except md.PackageNotFoundError:
            pass
    return False


if has("megatron-core"):
    print("megatron")
if has("addict", "deepdiff"):
    print("tau2")
if has("pytest", "pre-commit"):
    print("dev")
PY
}

# ---- the extras: this call's, plus the ones .venv already has (unless --exact) -----------------------------------
KEPT=()
if [ "$EXACT" != 1 ]; then
    _have="$(_installed_extras)"
    if [ -z "$EXTRA_FROM" ] && grep -qx megatron <<<"$_have"; then
        EXTRA=megatron
        KEPT+=("megatron (megatron-core is installed)")
    fi
    if [ "$TAU2" != 1 ] && grep -qx tau2 <<<"$_have"; then
        TAU2=1
        KEPT+=("tau2 (addict / deepdiff are installed)")
    fi
    if [ "$DEV" != 1 ] && grep -qx dev <<<"$_have"; then
        DEV=1
        KEPT+=("dev (pytest / pre-commit are installed)")
    fi
    if [ "$EXTRA" = fsdp ] && grep -qx megatron <<<"$_have"; then
        _msg="the backend fsdp ($EXTRA_FROM) replaces the Megatron stack installed in $VENV (the two conflict)"
        if [ "$CHECK" = 1 ]; then _info "$_msg"; else warn "$_msg"; fi
    fi
fi
EXTRA="${EXTRA:-fsdp}"
if [ "${#KEPT[@]}" -gt 0 ]; then
    _msg="keeping the extras $VENV already has: $(printf '%s, ' "${KEPT[@]}")"
    _msg="${_msg%, } (--exact drops them)"
    if [ "$CHECK" = 1 ]; then _info "$_msg"; else log "$_msg"; fi
fi

# ---- 1. the SkyRL checkout ----------------------------------------------------------------------------------
CFG_SKYRL="$(_uv_source skyrl)"
CFG_GYM="$(_uv_source skyrl-gym)"
[ -n "$CFG_SKYRL" ] || die "pyproject.toml has no [tool.uv.sources] entry skyrl = { path = \"...\", editable = true }"
CFG_DIR="$(_abspath "$CFG_SKYRL" "$ROOT")"
if [ -n "${SKYRL_DIR:-}" ]; then SKYRL_DIR="$(_abspath "$SKYRL_DIR")"; else SKYRL_DIR="$CFG_DIR"; fi
if [ "$SKYRL_DIR" != "$CFG_DIR" ]; then
    _bad "SKYRL_DIR=$SKYRL_DIR, but pyproject.toml installs SkyRL from \"$CFG_SKYRL\" ($CFG_DIR). To use" \
        "$SKYRL_DIR, set skyrl = { path = \"$SKYRL_DIR\", editable = true } and skyrl-gym = { path =" \
        "\"$SKYRL_DIR/skyrl-gym\", editable = true } under [tool.uv.sources], run \`uv lock\`, then this script again"
fi
if [ "$CFG_GYM" != "${CFG_SKYRL%/}/skyrl-gym" ]; then
    _bad "pyproject.toml installs skyrl-gym from \"${CFG_GYM:-<nothing>}\", expected \"${CFG_SKYRL%/}/skyrl-gym\"" \
        "(both [tool.uv.sources] paths must name the same SkyRL checkout)"
fi

_skyrl_verify() {  # SkyRL at $SKYRL_DIR: a git checkout of its own, at the pin, clean -> 0; else report, 1
    local top have dirty
    if [ ! -f "$SKYRL_DIR/skyrl/__init__.py" ] || [ ! -f "$SKYRL_DIR/skyrl-gym/skyrl_gym/__init__.py" ]; then
        _bad "$SKYRL_DIR is not a SkyRL checkout (no skyrl/__init__.py + skyrl-gym/skyrl_gym/__init__.py)"
        return 1
    fi
    top="$(git -C "$SKYRL_DIR" rev-parse --show-toplevel 2>/dev/null)" || top=""
    if [ -z "$top" ] || [ "$(cd "$top" && pwd -P)" != "$SKYRL_DIR" ]; then
        _bad "$SKYRL_DIR is not a git checkout of its own: cannot verify that it is SkyRL at $PIN" \
            "(clone it: git clone $SKYRL_GIT_URL $SKYRL_DIR && git -C $SKYRL_DIR checkout $PIN)"
        return 1
    fi
    have="$(git -C "$SKYRL_DIR" rev-parse HEAD 2>/dev/null)" || have=""
    if [ "$have" != "$PIN" ]; then
        _bad "$SKYRL_DIR is at ${have:-an unknown commit}, not the pinned $PIN. This script never changes an" \
            "existing checkout: git -C $SKYRL_DIR checkout $PIN (after a fetch if needed), or use another SKYRL_DIR"
        return 1
    fi
    dirty="$(git -C "$SKYRL_DIR" status --porcelain 2>/dev/null)" || dirty="?"
    if [ -n "$dirty" ]; then
        printf '%s\n' "$dirty" | head -n 10 | sed 's/^/    /' >&2
        _bad "$SKYRL_DIR has local changes or untracked files (above); SkyRL must stay unmodified upstream code"
        return 1
    fi
    _ok "SkyRL at $SKYRL_DIR: $PIN (the pin), clean"
    return 0
}

if [ ! -e "$SKYRL_DIR" ] || [ -z "$(ls -A "$SKYRL_DIR" 2>/dev/null)" ]; then
    if [ "$CHECK" = 1 ]; then
        _bad "no SkyRL checkout at $SKYRL_DIR (without --check this script clones $SKYRL_GIT_URL at $PIN there)"
    else
        command -v git >/dev/null 2>&1 || die "git is needed to clone SkyRL into $SKYRL_DIR"
        log "1/3 git clone $SKYRL_GIT_URL $SKYRL_DIR && git -C $SKYRL_DIR checkout $PIN"
        git clone --quiet "$SKYRL_GIT_URL" "$SKYRL_DIR" ||
            die "git clone $SKYRL_GIT_URL $SKYRL_DIR failed (remove what it left behind before re-running)"
        git -C "$SKYRL_DIR" -c advice.detachedHead=false checkout --quiet "$PIN" ||
            die "commit $PIN is not in $SKYRL_GIT_URL (the clone is left at $SKYRL_DIR)"
    fi
fi
if [ -e "$SKYRL_DIR" ] && [ -n "$(ls -A "$SKYRL_DIR" 2>/dev/null)" ]; then
    _skyrl_verify || :  # --check: counted; otherwise _bad has already exited
    [ "$CHECK" = 1 ] || log "1/3 SkyRL at $SKYRL_DIR: $PIN, clean"
    # uv resolves the [tool.uv.sources] path against the physical repository directory: it must land here too.
    _uv_dir="$(cd "$ROOT" && cd -P "$CFG_SKYRL" 2>/dev/null && pwd -P)" || _uv_dir=""
    [ "$_uv_dir" = "$SKYRL_DIR" ] ||
        _bad "uv resolves skyrl = { path = \"$CFG_SKYRL\" } from $ROOT to ${_uv_dir:-a missing directory}, not" \
            "$SKYRL_DIR"
fi

# ---- 2. uv sync -----------------------------------------------------------------------------------------------
SYNC_ARGS=(--frozen --extra "$EXTRA")
[ "$TAU2" = 1 ] && SYNC_ARGS+=(--extra tau2)
[ "$DEV" = 1 ] && SYNC_ARGS+=(--extra dev)
if [ "$CHECK" != 1 ]; then
    command -v "$UV" >/dev/null 2>&1 ||
        die "uv not found (https://docs.astral.sh/uv/getting-started/installation/); or set UV=<path to uv>"
    if [ -n "${UV_PROJECT_ENVIRONMENT:-}" ] &&
        [ "$(_abspath "$UV_PROJECT_ENVIRONMENT")" != "$(_abspath "$VENV")" ]; then
        warn "UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT is ignored: this script builds $VENV (the launchers'" \
            "default)"
    fi
    [ -x "$PY" ] || export UV_PYTHON="${UV_PYTHON:-3.12}"
    log "2/3 uv sync ${SYNC_ARGS[*]} -> $VENV"
    (cd "$ROOT" && UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync "${SYNC_ARGS[@]}") || die "uv sync failed (see above)"
    [ -x "$PY" ] || die "uv sync did not produce $PY"
    # uv builds skyrl / skyrl-gym editable inside the checkout; anything it leaves outside SkyRL's own
    # .gitignore would make the checkout look modified.
    _after="$(git -C "$SKYRL_DIR" status --porcelain 2>/dev/null)" || _after=""
    [ -z "$_after" ] ||
        warn "$SKYRL_DIR is not clean after the build (git -C $SKYRL_DIR status); leave SkyRL unmodified"
elif command -v "$UV" >/dev/null 2>&1; then
    _ok "uv: $("$UV" --version 2>/dev/null)"
else
    _info "uv not on PATH (needed to build and for the uv.lock comparison below; set UV=<path to uv>)"
fi

# ---- 3. verification ----------------------------------------------------------------------------------------
if [ ! -x "$PY" ]; then
    _bad "no environment at $VENV (run: bash scripts/setup/setup_train_env.sh, with the flags of this call)"
else
    _ok "python: $PY ($("$PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null))"
    # find_spec only locates the packages (nothing heavy is imported); a neutral cwd and no PYTHONPATH, so
    # this checks what the install itself provides.
    _report="$(cd / && env -u PYTHONPATH "$PY" - "$ROOT" "$SKYRL_DIR" "$EXTRA" "$TAU2" "$DEV" <<'PY'
import importlib.util
import os
import sys

root, skyrl_dir, extra, tau2, dev = sys.argv[1:6]
root, skyrl_dir = os.path.realpath(root), os.path.realpath(skyrl_dir)
want = {
    "skyrl": os.path.join(skyrl_dir, "skyrl"),
    "skyrl_gym": os.path.join(skyrl_dir, "skyrl-gym", "skyrl_gym"),
    "opsd": os.path.join(root, "opsd"),
    "torch": None,
    "vllm": None,
}
if extra == "megatron":
    want["megatron.core"] = None
if tau2 == "1":
    want.update({"addict": None, "deepdiff": None, "toml": None})
if dev == "1":
    want["pytest"] = None
for name, where in want.items():
    try:
        spec = importlib.util.find_spec(name)
    except Exception as e:  # noqa: BLE001 - reported, not raised
        print(f"FAIL {name}: {type(e).__name__}: {e}")
        continue
    if spec is None or not spec.origin:
        print(f"FAIL {name}: not importable")
        continue
    path = os.path.realpath(os.path.dirname(spec.origin))
    if where is None or path == where or path.startswith(where + os.sep):
        print(f"ok   {name} -> {path}")
    else:
        print(f"FAIL {name} -> {path} (expected under {where})")
PY
)" || _report="FAIL the verification script did not run ($PY)"
    while IFS= read -r _line; do
        case "$_line" in
            ok\ *) _ok "${_line#ok   }" ;;
            FAIL\ *) _bad "${_line#FAIL }" ;;
            ?*) _info "$_line" ;;
        esac
    done <<<"$_report"
    if command -v "$UV" >/dev/null 2>&1; then
        if _out="$(cd "$ROOT" && UV_PROJECT_ENVIRONMENT="$VENV" "$UV" sync "${SYNC_ARGS[@]}" --check --offline 2>&1)"
        then
            _ok "$VENV matches uv.lock (uv sync ${SYNC_ARGS[*]} --check)"
        else
            printf '%s\n' "$_out" | sed 's/\x1b\[[0-9;]*m//g' | head -n 20 | sed 's/^/    /' >&2
            _bad "$VENV does not match uv.lock for: uv sync ${SYNC_ARGS[*]} (see above; give --check the flags" \
                "of the build)"
        fi
    fi
fi

if [ "$CHECK" = 1 ]; then
    if [ "$_problems" = 0 ]; then
        echo "setup_train_env --check: the environment at $VENV is complete"
        exit 0
    fi
    _fix="bash scripts/setup/setup_train_env.sh"
    [ "$EXTRA" = fsdp ] || _fix="$_fix --extra $EXTRA"
    [ "$TAU2" = 1 ] && _fix="$_fix --tau2"
    [ "$DEV" = 1 ] && _fix="$_fix --dev"
    [ "$EXACT" = 1 ] && _fix="$_fix --exact"
    echo "setup_train_env --check: $_problems problem(s); build with: $_fix"
    exit 1
fi
log "done: $PY (launch an arm: bash scripts/<alfworld|appworld|tau2>/run.sh <arm_id>; tests, with --dev:" \
    ".venv/bin/python -m pytest from the repository root)"
