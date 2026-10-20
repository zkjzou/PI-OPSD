# shellcheck shell=bash
# ---------------------------------------------------------------------------------------------------
# Shared preamble for the tau2 drivers in this directory. SOURCE it, do not execute it:
#   . "$(dirname "${BASH_SOURCE[0]}")/_tau2_env.sh"
#
# Sets (every one overridable from the environment):
#   REPO_ROOT       repository root: the nearest ancestor holding pyproject.toml, opsd/__init__.py and
#                   opsd/skyrl_ext/__init__.py
#   OPSD_DATA_ROOT  unpacked data bundle (default ${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data)
#   T2              $OPSD_DATA_ROOT/tau2          (splits/, hints/, reference_trajectories/, pi_metrics/)
#   PI_GRID_ROOT    scored divergence cells       (default ./outputs/pi_cells/tau2, <pool>/<student>/rows.parquet;
#                   score them first with python -m opsd.alfworld.analyze_pi_positions)
#   OUT_ROOT        where the drivers write       (default ./outputs, resolved against the CALLER's cwd)
#   PY              python interpreter            (default $REPO_ROOT/.venv/bin/python, else python3)
#   LLM_BASE_URL    OpenAI-compatible endpoint for the user simulator, the NL judge and any remote
#                   frontier policy (default $OPENAI_BASE_URL, else https://api.openai.com/v1)
#   LLM_KEY_ENV     NAME of the env var holding that endpoint's key (default OPENAI_API_KEY)
#   TAU2_LLM_ARGS   (array) the baseline.py flags that route the simulator AND the judge to
#                   LLM_BASE_URL / LLM_KEY_ENV. Every driver passes it on each baseline.py call, so
#                   these two overrides take effect everywhere (with the defaults the flags equal
#                   baseline.py's own defaults, i.e. nothing changes).
# and exports PYTHONPATH=$REPO_ROOT (the folder holding the `opsd` package) so `python -m opsd...` resolves
# from any cwd.
# ---------------------------------------------------------------------------------------------------
_t2_d="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$_t2_d" != "/" ] && ! { [ -f "$_t2_d/pyproject.toml" ] && [ -f "$_t2_d/opsd/__init__.py" ] &&
  [ -f "$_t2_d/opsd/skyrl_ext/__init__.py" ]; }; do
  _t2_d="$(dirname "$_t2_d")"
done
if [ "$_t2_d" = "/" ]; then
  echo "FATAL: repository root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py) not found above ${BASH_SOURCE[0]}" >&2
  exit 1
fi
REPO_ROOT=${REPO_ROOT:-$_t2_d}
unset _t2_d

OPSD_DATA_ROOT=${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}
T2=${T2:-$OPSD_DATA_ROOT/tau2}
PI_GRID_ROOT=${PI_GRID_ROOT:-$PWD/outputs/pi_cells/tau2}
case "${OUT_ROOT:-}" in
  "") OUT_ROOT="$PWD/outputs" ;;
  /*) ;;
  *)  OUT_ROOT="$PWD/$OUT_ROOT" ;;
esac
if [ -z "${PY:-}" ]; then
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then PY="$REPO_ROOT/.venv/bin/python"; else PY=python3; fi
fi
LLM_BASE_URL=${LLM_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}
LLM_KEY_ENV=${LLM_KEY_ENV:-OPENAI_API_KEY}
TAU2_LLM_ARGS=(--user-base-url "$LLM_BASE_URL" --user-api-key-env "$LLM_KEY_ENV"
               --judge-base-url "$LLM_BASE_URL" --judge-api-key-env "$LLM_KEY_ENV")
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPSD_DATA_ROOT PI_GRID_ROOT

# Fail fast when the endpoint key is missing: the simulator would otherwise 401 every episode, and
# baseline.py records those as infrastructure failures with exit code 0. xtrace is suspended around the
# test so a `set -x` driver never prints the key's value into its log.
require_llm_key () {
  local _xt=0
  case $- in *x*) _xt=1; set +x ;; esac
  if [ -z "${!LLM_KEY_ENV:-}" ]; then
    echo "FATAL: set $LLM_KEY_ENV (key for $LLM_BASE_URL: user simulator / judge / frontier policy)" >&2
    exit 1
  fi
  [ "$_xt" = 1 ] && set -x
  return 0
}

# `--task <id>` for every id in a JSON list file.
task_args () {
  "$PY" -c "import json,sys;print(' '.join('--task '+str(t) for t in json.load(open(sys.argv[1]))))" "$1"
}

# The pinned 211-task retail-train pool (tri352 tiers T1-T3) as a JSON list. $T211_TASKS wins; else the
# task set of the published t211 scored cell, written once under $OUT_ROOT.
t211_tasks () {
  if [ -n "${T211_TASKS:-}" ]; then echo "$T211_TASKS"; return 0; fi
  local f="$OUT_ROOT/tau2_t211_tasks.json" src="$PI_GRID_ROOT/t211/4bi/rows.parquet"
  if [ ! -s "$f" ]; then
    [ -f "$src" ] || { echo "FATAL: set T211_TASKS (no $src to derive the 211-task pool from)" >&2; return 1; }
    mkdir -p "$OUT_ROOT"
    "$PY" - "$src" "$f" <<'PYT' || return 1
import json, sys
import pandas as pd
ids = sorted(pd.read_parquet(sys.argv[1], columns=["task_id"]).task_id.astype(str).unique(),
             key=lambda t: (len(t), t))
assert len(ids) == 211, f"expected 211 tasks in the t211 cell, got {len(ids)}"
json.dump(ids, open(sys.argv[2], "w"), indent=0)
PYT
  fi
  echo "$f"
}
