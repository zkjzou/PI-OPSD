#!/usr/bin/env bash
# =====================================================================================================
# setup_tau2.sh -- one-time setup for the tau2-bench launcher (scripts/tau2/run.sh). Safe to re-run: the
# end state is the same. Steps 2-5 re-verify and only do the work that is missing; step 1 is
# setup_train_env.sh, which syncs the environment to uv.lock with the tau2 extra added and KEEPS the extras the
# environment already has (dev when pytest is installed, the Megatron backend when megatron-core is), so it
# never removes the test tools or switches a Megatron environment to FSDP. SKIP_UV_SYNC=1 skips step 1.
#
#   bash scripts/setup/setup_tau2.sh                          # the .venv you built (+ tau2); FSDP if none yet
#   TAU2_UV_EXTRA=megatron bash scripts/setup/setup_tau2.sh   # 30B student (Megatron), whatever .venv has now
#   TAU2_UV_EXTRA=fsdp bash scripts/setup/setup_tau2.sh       # 4B students (FSDP), whatever .venv has now
#   bash scripts/setup/setup_tau2.sh --dev                    # ... + the dev extra (pytest, pre-commit)
#   bash scripts/setup/setup_tau2.sh --exact                  # exactly <backend> + tau2 [+ dev]: drop the rest
#   SKIP_UV_SYNC=1 bash scripts/setup/setup_tau2.sh           # environment already built (with --extra tau2)
#
# Steps
#   1. The training environment: `bash scripts/setup/setup_train_env.sh [--extra $TAU2_UV_EXTRA] --tau2 [--dev]
#      [--exact]`: the pinned SkyRL checkout next to the repository, then `uv sync --frozen` with those extras
#      plus the ones .venv already has (unless --exact). TAU2_UV_EXTRA (fsdp | megatron) names the backend;
#      unset, it is $OPSD_UV_EXTRA, else the one .venv has (megatron when megatron-core is installed), else
#      fsdp. The `tau2` extra holds tau2-bench's import-time dependencies.         (skip: SKIP_UV_SYNC=1)
#   2. tau2-bench runtime dependencies, checked: addict, deepdiff, orderly-set, cachebox and toml must import.
#      tau2-bench itself is never pip-installed (its litellm pin conflicts with the lock); it is imported from
#      the pinned checkout. This MUST precede step 4, which imports tau2.
#   3. AReaL RL task data (HF dataset inclusionAI/AReaL-tau2-data at the revision pinned in
#      opsd/tau2_bench/constants.py) staged into $AREAL_TAU2_DATA_DIR (default
#      ~/.cache/skyrl/areal-tau2-data) for TAU2_AREAL_DOMAINS (default "retail airline"), so training never
#      depends on a live download.
#   4. The pinned tau2-bench checkout (github.com/sierra-research/tau2-bench at constants.TAU2_REVISION) in
#      $TAU2_SOURCE_DIR: `python -m opsd.tau2_bench.prepare_data prepare --allow-judge`.
#      Its regenerated split manifests are written to a scratch directory and compared byte-for-byte with the
#      shipped ones (opsd/tau2_bench/data/); a difference is an error.
#   5. Optional: the PI data bundle (hint files) into $OPSD_DATA_ROOT, with OPSD_DATA_REPO + OPSD_DATA_REVISION
#      (a Hugging Face dataset id and commit) or OPSD_DATA_FROM_LOCAL=<unpacked bundle>.
#
# Steps 2-5 run <repo root>/.venv/bin/python, or $OPSD_PYTHON (then only with SKIP_UV_SYNC=1: step 1 builds
# <repo root>/.venv). Needs network access for steps 1-5 (GitHub, Hugging Face, PyPI). Writes only the
# repository's .venv (+ the SkyRL checkout setup_train_env.sh clones), $TAU2_SOURCE_DIR, $AREAL_TAU2_DATA_DIR,
# $TAU2_CACHE_DIR (default ~/.cache/skyrl) and $OPSD_DATA_ROOT.
# =====================================================================================================
set -eo pipefail

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
# ROOT = the repository root: pyproject.toml + the opsd package (opsd/, with recipes/) + scripts/
ROOT="$_here"
while [ "$ROOT" != "/" ] && ! { [ -f "$ROOT/pyproject.toml" ] && [ -f "$ROOT/opsd/__init__.py" ] &&
    [ -f "$ROOT/opsd/skyrl_ext/__init__.py" ]; }; do
    ROOT="$(dirname "$ROOT")"
done
[ "$ROOT" != "/" ] || {
    echo "[setup-tau2] FATAL: repository root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py)" \
        "not found above $_here" >&2
    exit 1
}

log() { printf '[setup-tau2] %s\n' "$*" >&2; }
die() { printf '[setup-tau2] ERROR: %s\n' "$*" >&2; exit 1; }

TRAIN_ENV_ARGS=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --dev | --exact) TRAIN_ENV_ARGS+=("$1"); shift ;;
        -h | --help)
            awk 'NR > 2 && /^# =====/ { exit } NR > 2 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) die "unknown argument: $1 (accepted: --dev, --exact; the backend is TAU2_UV_EXTRA=fsdp|megatron)" ;;
    esac
done
cd "$ROOT"

# Pinned tau2-bench revision (constants.TAU2_REVISION; a test keeps the two equal).
TAU2_REVISION=1d244f5dca42944b67a379b44bfeb9f5748f189d
TAU2_CACHE_DIR=${TAU2_CACHE_DIR:-$HOME/.cache/skyrl}
TAU2_SOURCE_DIR=${TAU2_SOURCE_DIR:-$TAU2_CACHE_DIR/tau2-bench/$TAU2_REVISION}
export AREAL_TAU2_DATA_DIR=${AREAL_TAU2_DATA_DIR:-$TAU2_CACHE_DIR/areal-tau2-data}
TAU2_AREAL_DOMAINS=${TAU2_AREAL_DOMAINS:-retail airline}
TAU2_UV_EXTRA=${TAU2_UV_EXTRA:-}  # empty: setup_train_env.sh keeps the backend .venv has (else fsdp)
UV=${UV:-uv}
PY=${OPSD_PYTHON:-$ROOT/.venv/bin/python}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"  # `opsd` = the package at <repo root>/opsd

# ---- 1. environment ---------------------------------------------------------------------------------------
if [ "${SKIP_UV_SYNC:-0}" != 1 ]; then
    case "$TAU2_UV_EXTRA" in
        fsdp | megatron) TRAIN_ENV_ARGS=(--extra "$TAU2_UV_EXTRA" "${TRAIN_ENV_ARGS[@]}") ;;
        "") ;;
        *) die "TAU2_UV_EXTRA=$TAU2_UV_EXTRA (expected fsdp or megatron)" ;;
    esac
    [ "$PY" = "$ROOT/.venv/bin/python" ] ||
        die "OPSD_PYTHON=$OPSD_PYTHON is not the environment step 1 builds ($ROOT/.venv): build yours with the" \
            "tau2 extra yourself and re-run with SKIP_UV_SYNC=1, or unset OPSD_PYTHON"
    log "1/5 bash scripts/setup/setup_train_env.sh --tau2 ${TRAIN_ENV_ARGS[*]}"
    UV="$UV" bash "$ROOT/scripts/setup/setup_train_env.sh" --tau2 "${TRAIN_ENV_ARGS[@]}" ||
        die "building the training environment failed (see above)"
else
    log "1/5 SKIP_UV_SYNC=1: using the existing environment ($PY)"
fi
[ -x "$PY" ] || die "python not found at $PY (set OPSD_PYTHON, or run without SKIP_UV_SYNC)"

# ---- 2. tau2-bench runtime dependencies (the `tau2` extra; before anything imports tau2) -------------------
if "$PY" -c "import addict, deepdiff, orderly_set, cachebox, toml" >/dev/null 2>&1; then
    log "2/5 tau2 runtime dependencies importable"
else
    die "tau2-bench's runtime dependencies (addict, deepdiff, orderly-set, cachebox, toml) are missing from $PY:" \
        "build the environment with the tau2 extra (bash scripts/setup/setup_train_env.sh --tau2" \
        "[--extra fsdp|megatron], i.e. uv sync --frozen --extra <fsdp|megatron> --extra tau2)"
fi

# ---- 3. AReaL task data at the pinned revision ----------------------------------------------------------
for _d in $TAU2_AREAL_DOMAINS; do
    log "3/5 staging AReaL $_d task data -> $AREAL_TAU2_DATA_DIR"
    "$PY" -c 'import sys
from opsd.tau2_bench.areal import stage_local
print(stage_local(domain=sys.argv[1]))' "$_d" >/dev/null || die "staging the AReaL $_d data failed"
done

# ---- 4. pinned tau2-bench checkout + manifest check -------------------------------------------------------
_manifests="$TAU2_CACHE_DIR/tau2-manifests/$TAU2_REVISION"
log "4/5 tau2-bench @ $TAU2_REVISION -> $TAU2_SOURCE_DIR (manifests -> $_manifests)"
"$PY" -m opsd.tau2_bench.prepare_data prepare --allow-judge \
    --source-dir "$TAU2_SOURCE_DIR" --output-dir "$_manifests"
_bad=0
for _f in "$_manifests"/*.jsonl; do
    _ship="opsd/tau2_bench/data/$(basename "$_f")"
    if [ ! -f "$_ship" ]; then
        log "  (not shipped) $(basename "$_f")"
    elif cmp -s "$_f" "$_ship"; then
        log "  identical  $(basename "$_f")"
    else
        log "  DIFFERENT  $(basename "$_f") -- the checkout or the AReaL data does not reproduce the shipped manifest"
        _bad=1
    fi
done
[ "$_bad" = 0 ] || [ "${TAU2_ALLOW_MANIFEST_DRIFT:-0}" = 1 ] ||
    die "regenerated manifests differ from the shipped ones (see above); TAU2_ALLOW_MANIFEST_DRIFT=1 continues anyway"

# ---- 5. PI data bundle (optional) -------------------------------------------------------------------------
if [ -n "${OPSD_DATA_FROM_LOCAL:-}" ]; then
    log "5/5 data bundle from $OPSD_DATA_FROM_LOCAL -> ${OPSD_DATA_ROOT:-<default data root>}"
    "$PY" -m opsd.data.fetch --from-local "$OPSD_DATA_FROM_LOCAL" --bench tau2
elif [ -n "${OPSD_DATA_REPO:-}" ]; then
    [ -n "${OPSD_DATA_REVISION:-}" ] || die "OPSD_DATA_REPO needs OPSD_DATA_REVISION (a commit sha)"
    log "5/5 data bundle $OPSD_DATA_REPO@$OPSD_DATA_REVISION -> ${OPSD_DATA_ROOT:-<default data root>}"
    "$PY" -m opsd.data.fetch --repo "$OPSD_DATA_REPO" --revision "$OPSD_DATA_REVISION" --bench tau2
else
    log "5/5 data bundle: skipped (set OPSD_DATA_REPO + OPSD_DATA_REVISION, or OPSD_DATA_FROM_LOCAL)"
fi

log "done. Launch an arm with: bash scripts/tau2/run.sh <arm_id>   (list: python -m opsd.recipes list --bench tau2)"
[ "$TAU2_SOURCE_DIR" = "$HOME/.cache/skyrl/tau2-bench/$TAU2_REVISION" ] ||
    log "note: pass TAU2_SOURCE_DIR=$TAU2_SOURCE_DIR to run.sh (non-default checkout location)"
