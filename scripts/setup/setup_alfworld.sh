#!/usr/bin/env bash
# =====================================================================================================
# setup_alfworld.sh -- one-time ALFWorld setup for scripts/alfworld/run.sh. Idempotent: every step
# is skipped when its result already exists (FORCE=1 redoes the package install and the parquets).
#
#   bash scripts/setup/setup_alfworld.sh            # venv + game data + parquets
#   bash scripts/setup/setup_alfworld.sh --check    # report what is missing, change nothing
#
# 1. the ALFWorld environment-server venv at $ALFWORLD_VENV (default $HOME/alfworld-venv), SEPARATE from
#    the training environment (textworld / gymnasium==0.29.1 / the PDDL stack conflict with it), from
#    scripts/setup/requirements/alfworld_server.txt pinned by alfworld_server.constraints.txt (the versions
#    the paper's servers ran with; Python 3.10);
# 2. the ALFWorld game data (`alfworld-download`) into $ALFWORLD_DATA (default $HOME/data/alfworld): the
#    servers load json_2.1.1/ and logic/ from there;
# 3. the SkyRL parquets in $ALFWORLD_PARQUET_DIR (default $ALFWORLD_DATA/parquet): train.parquet (the whole
#    train split; each run filters it to its task ids), validation.parquet (valid_seen, 140 tasks) and
#    validation_unseen.parquet (valid_unseen, 134 tasks). They store absolute game paths under
#    $ALFWORLD_DATA, so rebuild them (FORCE=1) if the data directory moves.
#
# The TRAINING environment is separate (<repo root>/.venv, from uv.lock with the pinned SkyRL checkout):
#   bash scripts/setup/setup_train_env.sh                     # or --extra megatron for the 30B student
# and the PI data bundle is fetched into $OPSD_DATA_ROOT with `python -m opsd.data.fetch`.
#
# Env: ALFWORLD_VENV, ALFWORLD_DATA, ALFWORLD_PARQUET_DIR, ALFWORLD_PYTHON (interpreter for a new venv,
# default python3.10, else python3), FORCE=1. Uses `uv` for the venv when it is on PATH, else venv + pip.
# =====================================================================================================
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ROOT = the repository root: pyproject.toml + the opsd package (opsd/) + scripts/ (SkyRL is not needed here)
ROOT="$HERE"
while [ "$ROOT" != "/" ] && ! { [ -f "$ROOT/pyproject.toml" ] && [ -f "$ROOT/opsd/__init__.py" ] &&
    [ -f "$ROOT/opsd/skyrl_ext/__init__.py" ]; }; do
    ROOT="$(dirname "$ROOT")"
done
[ "$ROOT" != "/" ] || {
    echo "setup_alfworld.sh: repository root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py)" \
        "not found above $HERE" >&2
    exit 2
}

CHECK_ONLY=0
case "${1:-}" in
    --check) CHECK_ONLY=1 ;;
    -h | --help) awk 'NR > 2 && /^# =====/ { exit } NR > 2 { sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"; exit 0 ;;
    "") ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
esac

_abs() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }
ALFWORLD_VENV="$(_abs "${ALFWORLD_VENV:-$HOME/alfworld-venv}")"
ALFWORLD_DATA="$(_abs "${ALFWORLD_DATA:-$HOME/data/alfworld}")"
ALFWORLD_PARQUET_DIR="$(_abs "${ALFWORLD_PARQUET_DIR:-$ALFWORLD_DATA/parquet}")"
export ALFWORLD_DATA    # alfworld-download and the dataset builder read it
REQ="$ROOT/scripts/setup/requirements/alfworld_server.txt"
CON="$ROOT/scripts/setup/requirements/alfworld_server.constraints.txt"
VPY="$ALFWORLD_VENV/bin/python"
FORCE=${FORCE:-0}

log() { printf '[setup_alfworld] %s\n' "$*" >&2; }

venv_ok() {
    [ -x "$VPY" ] && "$VPY" - <<'PY' >/dev/null 2>&1
import importlib.util, sys
mods = ("alfworld", "textworld", "fastapi", "uvicorn", "yaml", "pandas", "pyarrow")
sys.exit(1 if [m for m in mods if importlib.util.find_spec(m) is None] else 0)
PY
}
data_ok() { [ -d "$ALFWORLD_DATA/json_2.1.1" ] && [ -d "$ALFWORLD_DATA/logic" ]; }
parquets_ok() {
    local f
    for f in train validation validation_unseen; do [ -s "$ALFWORLD_PARQUET_DIR/$f.parquet" ] || return 1; done
}

if [ "$CHECK_ONLY" = 1 ]; then
    rc=0
    if venv_ok; then log "OK   venv $ALFWORLD_VENV"; else log "MISSING venv (or packages) at $ALFWORLD_VENV"; rc=1; fi
    if data_ok; then log "OK   game data $ALFWORLD_DATA"; else log "MISSING game data in $ALFWORLD_DATA"; rc=1; fi
    if parquets_ok; then log "OK   parquets $ALFWORLD_PARQUET_DIR"; else log "MISSING parquets in $ALFWORLD_PARQUET_DIR"; rc=1; fi
    exit "$rc"
fi

# ---- 1. the server venv ----
if venv_ok && [ "$FORCE" != 1 ]; then
    log "venv ready: $ALFWORLD_VENV"
else
    if [ ! -x "$VPY" ]; then
        if command -v uv >/dev/null 2>&1; then
            log "creating $ALFWORLD_VENV with uv (Python ${ALFWORLD_PYTHON:-3.10})"
            uv venv --python "${ALFWORLD_PYTHON:-3.10}" "$ALFWORLD_VENV"
        else
            PYBIN="${ALFWORLD_PYTHON:-$(command -v python3.10 || command -v python3)}"
            log "creating $ALFWORLD_VENV with $PYBIN"
            "$PYBIN" -m venv "$ALFWORLD_VENV"
        fi
    fi
    log "installing the server packages into $ALFWORLD_VENV"
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$VPY" -r "$REQ" -c "$CON"
    else
        "$VPY" -m pip install -r "$REQ" -c "$CON"
    fi
    venv_ok || { log "ERROR: the venv still lacks the server packages"; exit 1; }
fi

# ---- 2. the game data ----
if data_ok; then
    log "game data ready: $ALFWORLD_DATA"
else
    mkdir -p "$ALFWORLD_DATA"
    log "downloading the ALFWorld game data into $ALFWORLD_DATA"
    "$ALFWORLD_VENV/bin/alfworld-download" --data-dir "$ALFWORLD_DATA"
    data_ok || { log "ERROR: $ALFWORLD_DATA/json_2.1.1 or logic/ is still missing after the download"; exit 1; }
fi

# ---- 3. the parquets ----
if parquets_ok && [ "$FORCE" != 1 ]; then
    log "parquets ready: $ALFWORLD_PARQUET_DIR"
else
    log "building the parquets in $ALFWORLD_PARQUET_DIR"
    "$VPY" "$ROOT/opsd/alfworld/alfworld_dataset.py" \
        --local_dir "$ALFWORLD_PARQUET_DIR" --alfworld_data "$ALFWORLD_DATA" \
        --splits train valid_seen valid_unseen
    parquets_ok || { log "ERROR: parquet build did not produce all three files"; exit 1; }
fi

log "done. For run.sh: ALFWORLD_VENV=$ALFWORLD_VENV ALFWORLD_DATA=$ALFWORLD_DATA ALFWORLD_PARQUET_DIR=$ALFWORLD_PARQUET_DIR"
