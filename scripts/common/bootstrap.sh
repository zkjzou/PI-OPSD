#!/usr/bin/env bash
# shellcheck shell=bash
# =====================================================================================================
# bootstrap.sh -- the shared environment of every OPSD launcher. SOURCE it, do not execute it:
#
#     source "$ROOT/scripts/common/bootstrap.sh"      # ROOT = the repository root
#
# Typical launcher flow (scripts/<bench>/run.sh):
#     source ".../bootstrap.sh"                      # functions + base environment
#     opsd_resolve_recipe "$ARM_ID" "$@"             # -> $RUN_DIR/resolved.env, sourced; re-applies the env
#     [ "${DRY_RUN:-0}" = 1 ] && { opsd_dry_run_print "$MODULE" "${ARGV[@]}"; exit 0; }
#     source ".../preflight.sh"; opsd_preflight_all || exit 1
#     opsd_install_cleanup_trap                      # our own PIDs only, on EXIT/INT/TERM
#     opsd_spawn alfworld-server-0 python server.py --port 12500 >"$RUN_DIR/logs/s0.log" 2>&1
#     opsd_write_provenance "$RUN_DIR/resolved.env"
#     opsd_run python -m <entry module> "${ARGV[@]}"; rc=$?; exit $rc
#
# Sourcing runs opsd_bootstrap_env (skip with OPSD_BOOTSTRAP_AUTO=0). It is idempotent; call it again after
# RUN_DIR changes (opsd_resolve_recipe does). It:
#   * raises the open-file soft limit to min(65536, hard limit) (warns when the hard limit is lower);
#   * exports, each ONLY IF UNSET: VLLM_ALLREDUCE_USE_SYMM_MEM=0, NCCL_CUMEM_ENABLE=0, HYDRA_FULL_ERROR=1,
#     PYTHONHASHSEED=0, RAY_RUNTIME_ENV_HOOK (Ray's uv hook: a no-op unless the driver runs under `uv run`)
#     and LOGGER (wandb when WANDB_API_KEY is set, else console). Launched under `uv run` (an ancestor process
#     `uv run ...`), it instead exports RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 and leaves Ray's uv hook unset: Ray
#     would otherwise start every worker with `uv run` inside an uploaded copy of the working directory, where
#     pyproject's relative ../SkyRL-ecdaa946 source does not resolve; the workers then run the driver's
#     interpreter (.venv/bin/python), as without `uv run`. Plain `bash scripts/<bench>/run.sh` is the
#     documented launch. A different value already set for
#     PYTHONHASHSEED / NCCL_CUMEM_ENABLE / VLLM_ALLREDUCE_USE_SYMM_MEM is kept with a warning; whatever the
#     run sees is recorded under "runtime_env" in provenance.json. It never sets WANDB_BASE_URL/WANDB_ENTITY;
#   * prepends OPSD_NVML_LIB_DIR to LD_LIBRARY_PATH when set (hosts whose libcuda/NVML live outside the
#     loader path -- the symptom is Ray registering 0 GPUs);
#   * SINGLE-NODE mode (RAY_ADDRESS unset, empty or "local"): RAY_ADDRESS=local and a short per-run
#     RAY_TMPDIR=/tmp/opsd_ray_<uid>_<hash of RUN_DIR>, so the driver's ray.init() always starts its OWN
#     cluster and never attaches to another run's cluster found under a shared temp dir (keep RAY_TMPDIR
#     short: Ray's AF_UNIX socket paths must stay under 108 bytes; Ray creates the directory itself).
#     Local networking defaults (only if unset): NCCL_NET=Socket, VLLM_HOST_IP=127.0.0.1;
#   * MULTI-NODE mode (RAY_ADDRESS=<head-ip>:<port>, see ray_cluster.sh): none of the single-node settings.
# It NEVER runs `ray stop` and never signals a process it did not start (PIDs live in $RUN_DIR/pids/).
#
# Functions:
#   opsd_log / opsd_warn MSG...               message on stderr ("[opsd] ...")
#   opsd_die MSG [STATUS]                     message + exit STATUS (default 1)
#   opsd_repo_root                            print the repo root (the directory holding pyproject.toml,
#                                             opsd/__init__.py and opsd/skyrl_ext/__init__.py)
#   opsd_python                               print the interpreter: $OPSD_PYTHON, <root>/.venv/bin/python, python3
#   opsd_bootstrap_env                        (re)apply the environment above
#   opsd_ray_mode                             print "local" or "external"
#   opsd_resolve_recipe ARM_ID [ARGS...]      resolve the recipe (hermetic; ARGS = KEY=VAL / --overlay NAME)
#                                             in the CALLER's working directory (relative RUN_ROOT,
#                                             OPSD_DATA_ROOT, overlay and KEY=VAL paths mean "relative to where
#                                             you launched"), source it, write $RUN_DIR/resolved.env (under
#                                             DRY_RUN=1 a temp file, removed when the shell exits), export
#                                             OPSD_RESOLVED_ENV, re-apply the environment. A run directory is
#                                             never silently reused with other settings: when $RUN_DIR already
#                                             has a DIFFERENT resolved.env and this run would resume its newest
#                                             checkpoint (RESUME_MODE=latest, the default; no RESUME_PATH) while
#                                             $RUN_DIR holds checkpoints or exported evaluations, it fails with
#                                             the changed keys (pick a new RUN_NAME, or resume on purpose with
#                                             RESUME_MODE=from_path RESUME_PATH=...; DRY_RUN=1 only warns). An
#                                             evaluation-only re-invocation (EVAL_ONLY, EVAL_CKPT_STEPS) writes
#                                             resolved.eval-<sha12>.env instead, keeping the training record;
#                                             any other changed training run moves the earlier resolved.env and
#                                             provenance.json aside as *.prev-<sha12>.*. (The resolver already
#                                             gives overlay runs their own default RUN_NAME: <arm>__<overlay>.)
#   opsd_dry_run_print MODULE [ARGV...]       print `python -m MODULE ARGV` (one arg per line) + resolved env;
#                                             may be called once per trainer invocation
#   opsd_track_pid PID [LABEL]                record a RUNNING PID you started and its kernel start time in
#                                             $RUN_DIR/pids/ (cleanup reaches its process tree, or its process
#                                             group if it leads one, while that PID is alive)
#   opsd_untrack_pid PID                      forget PID
#   opsd_spawn LABEL CMD [ARGS...]            run CMD in the background in a NEW process group (setsid; with
#                                             job control on, the job's own group) and record the group id, the
#                                             spawn time and a per-spawn tag in its environment. CMD may start
#                                             background children and exit (a server pool launcher): cleanup
#                                             still finds them. PID in $OPSD_LAST_PID; redirect output on the
#                                             call: `opsd_spawn s0 cmd >log 2>&1`
#   opsd_run CMD [ARGS...]                    run CMD as the tracked foreground driver; INT/TERM reach it via
#                                             the cleanup trap (installed if missing); returns CMD's status
#   opsd_cleanup_own [LABEL_GLOB]             TERM, then after OPSD_CLEANUP_GRACE_S (20) KILL (ownership is
#                                             re-checked first), every tracked process: a spawned group whose
#                                             leader is alive -> the whole group (setsid: plus the rest of its
#                                             session); whose leader has exited -> setsid: every member of its
#                                             session started after the spawn (tagged or not: env -i children,
#                                             own process groups); job control: the group members started after
#                                             the spawn that carry its tag; a tracked PID -> its group if it
#                                             leads one, else its tree. A PID whose start time no longer
#                                             matches (reused) is never signalled
#   opsd_install_cleanup_trap                 EXIT/INT/TERM -> opsd_cleanup_own, keeping the exit status
#   opsd_wait_http URL TIMEOUT_S [PID...]     wait for an HTTP 2xx from URL; fail early if a PID dies
#   opsd_wait_ports HOST BASE COUNT TIMEOUT_S wait until every port BASE..BASE+COUNT-1 accepts TCP
#   opsd_write_provenance RESOLVED_ENV [OUT]  $RUN_DIR/provenance.json (provenance.<tag>.json next to a
#                                             resolved.<tag>.env) with git sha + dirty, the SkyRL checkout's
#                                             commit vs. the pin, uv.lock sha, resolved-env sha, hint content
#                                             sha256, GPU model, runtime env:
#                                             PYTHONHASHSEED, NCCL_*, vLLM/Ray/logger settings; never fatal
#
# Knobs (user/infrastructure settings, accepted by the resolver's hermetic check): OPSD_BOOTSTRAP_AUTO,
# OPSD_NVML_LIB_DIR, OPSD_PYTHON, OPSD_BOOTSTRAP_CLEANUP_GRACE_S (alias of OPSD_CLEANUP_GRACE_S).
# OPSD_BOOTSTRAP_SPAWN_TAG is set by opsd_spawn in the spawned command's environment only.
# Sourcing creates nothing on disk; DRY_RUN=1 also keeps opsd_resolve_recipe from creating RUN_DIR.
# Safe under `set -euo pipefail` callers.
# =====================================================================================================

_OPSD_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

opsd_log() { printf '[opsd] %s\n' "$*" >&2; }
opsd_warn() { printf '[opsd] WARNING: %s\n' "$*" >&2; }
opsd_die() { printf '[opsd] ERROR: %s\n' "$1" >&2; exit "${2:-1}"; }

opsd_repo_root() {  # the directory holding pyproject.toml, the opsd package (opsd/) and scripts/
    local d="$_OPSD_COMMON_DIR"
    while [ "$d" != "/" ]; do
        if [ -f "$d/pyproject.toml" ] && [ -f "$d/opsd/__init__.py" ] && [ -f "$d/opsd/skyrl_ext/__init__.py" ]; then
            printf '%s\n' "$d"
            return 0
        fi
        d="$(dirname "$d")"
    done
    opsd_warn "repository root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py) not found" \
        "above $_OPSD_COMMON_DIR"
    return 1
}

opsd_python() {
    if [ -n "${OPSD_PYTHON:-}" ]; then
        printf '%s\n' "$OPSD_PYTHON"
        return 0
    fi
    local root
    root="$(opsd_repo_root 2>/dev/null)" || root=""
    if [ -n "$root" ] && [ -x "$root/.venv/bin/python" ]; then
        printf '%s\n' "$root/.venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then
        command -v python3
    else
        command -v python
    fi
    return 0
}

opsd_ray_mode() {
    case "${RAY_ADDRESS:-}" in
        "" | local) echo local ;;
        *) echo external ;;
    esac
}

_opsd_raise_nofile() {
    local want=65536 hard cur
    hard="$(ulimit -Hn 2>/dev/null || echo 0)"
    cur="$(ulimit -Sn 2>/dev/null || echo 0)"
    if [ "$hard" != "unlimited" ] && [ "$hard" -lt "$want" ] 2>/dev/null; then
        opsd_warn "open-file hard limit is $hard (< $want); raising the soft limit to $hard only"
        want="$hard"
    fi
    if [ "$cur" = "unlimited" ] || { [ "$cur" -ge "$want" ] 2>/dev/null; }; then
        return 0
    fi
    ulimit -Sn "$want" 2>/dev/null || opsd_warn "could not raise the open-file limit to $want (now $cur)"
    return 0
}

_opsd_default() {  # VAR VALUE: export VAR=VALUE unless already set; remember that the bootstrap set it
    if [ -z "${!1:-}" ]; then
        export "$1=$2"
        case " ${_OPSD_BOOTSTRAP_SET:-} " in *" $1 "*) ;; *) _OPSD_BOOTSTRAP_SET="${_OPSD_BOOTSTRAP_SET:-} $1" ;; esac
    fi
    return 0
}

_opsd_contract() {  # VAR VALUE: _opsd_default, but warn (once) when VAR already holds a different value
    if [ -n "${!1:-}" ] && [ "${!1}" != "$2" ]; then
        case " ${_OPSD_CONTRACT_WARNED:-} " in
            *" $1 "*) ;;
            *)
                opsd_warn "$1=${!1} differs from the reproduction setting $1=$2 (kept; recorded in provenance.json)"
                _OPSD_CONTRACT_WARNED="${_OPSD_CONTRACT_WARNED:-} $1"
                ;;
        esac
    fi
    _opsd_default "$1" "$2"
}

_OPSD_UV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

_opsd_under_uv_run() {  # 0 when an ancestor process is `uv run ...` (Ray's own test: argv[0] named uv, argv[1] run)
    local pid="${PPID:-}" next k v a0 a1 n=0
    [ -r /proc/self/status ] || return 1  # no /proc (macOS): not detected
    while [ -n "$pid" ] && [ "$pid" -gt 1 ] 2>/dev/null && [ "$n" -lt 64 ]; do
        a0="" a1="" next=""
        { IFS= read -r -d '' a0 && IFS= read -r -d '' a1; } 2>/dev/null <"/proc/$pid/cmdline" || :
        if [ "${a0##*/}" = uv ] && [ "$a1" = run ]; then return 0; fi
        while IFS=$' \t' read -r k v; do
            if [ "$k" = PPid: ]; then
                next="$v"
                break
            fi
        done 2>/dev/null <"/proc/$pid/status" || :
        pid="$next"
        n=$((n + 1))
    done
    return 1
}

opsd_bootstrap_env() {
    _opsd_raise_nofile
    _opsd_contract VLLM_ALLREDUCE_USE_SYMM_MEM 0
    _opsd_contract NCCL_CUMEM_ENABLE 0
    _opsd_default HYDRA_FULL_ERROR 1
    _opsd_contract PYTHONHASHSEED 0
    if _opsd_under_uv_run; then
        # Ray's uv integration (on by default, and the hook below) would re-launch every worker with `uv run`
        # in an uploaded copy of the working directory, where ../SkyRL-ecdaa946 does not resolve.
        _opsd_default RAY_ENABLE_UV_RUN_RUNTIME_ENV 0
        [ "${RAY_RUNTIME_ENV_HOOK:-}" != "$_OPSD_UV_HOOK" ] || unset RAY_RUNTIME_ENV_HOOK
        if [ -z "${_OPSD_UV_RUN_NOTED:-}" ]; then
            opsd_log "launched under \`uv run\`: Ray's uv worker re-launch is off (RAY_ENABLE_UV_RUN_RUNTIME_ENV=0);" \
                "the workers run the driver's interpreter"
            _OPSD_UV_RUN_NOTED=1
        fi
    else
        _opsd_default RAY_RUNTIME_ENV_HOOK "$_OPSD_UV_HOOK"
    fi
    if [ -n "${WANDB_API_KEY:-}" ]; then _opsd_default LOGGER wandb; else _opsd_default LOGGER console; fi
    if [ -n "${OPSD_NVML_LIB_DIR:-}" ]; then
        case ":${LD_LIBRARY_PATH:-}:" in
            *":$OPSD_NVML_LIB_DIR:"*) ;;
            *) export LD_LIBRARY_PATH="$OPSD_NVML_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
        esac
    fi
    if [ "$(opsd_ray_mode)" = local ]; then
        _opsd_default RAY_ADDRESS local
        if [ -z "${RAY_TMPDIR:-}" ] || [ "${_OPSD_RAY_TMPDIR_AUTO:-0}" = 1 ]; then
            local key slug
            key="${RUN_DIR:-${RUN_NAME:-pid$$}}"
            slug="$(printf '%s' "$key" | cksum | awk '{printf "%08x", $1}')"
            RAY_TMPDIR="/tmp/opsd_ray_$(id -u)_${slug}"
            export RAY_TMPDIR
            _OPSD_RAY_TMPDIR_AUTO=1
        fi
        if [ "${#RAY_TMPDIR}" -gt 40 ]; then
            opsd_warn "RAY_TMPDIR=$RAY_TMPDIR is long; Ray's socket paths may exceed the 108-byte AF_UNIX limit"
        fi
        _opsd_default NCCL_NET Socket
        _opsd_default VLLM_HOST_IP 127.0.0.1
    fi
    return 0
}

# ------------------------------------------------------------------------------------------- recipes
_opsd_pymod() {  # MODULE [ARGS...]: `python -m MODULE ARGS` with the repo root FIRST on sys.path (so `opsd`
    # is this checkout's package, whatever else the interpreter has installed), run in the caller's working
    # directory (so relative paths keep meaning "relative to where you launched").
    # The working directory itself is taken OFF sys.path: a yaml.py / json.py where the user launched from
    # must not shadow the real module inside the resolver.
    local root
    root="$(opsd_repo_root)" || return 1
    "$(opsd_python)" -c 'import os, runpy, sys
root, here = sys.argv.pop(1), os.getcwd()
sys.path[:] = [root] + [p for p in sys.path if p not in ("", ".", here)]
runpy.run_module(sys.argv.pop(1), run_name="__main__", alter_sys=True)' "$root" "$@"
}

_opsd_sha256() {  # FILE -> hex sha256 (empty on failure)
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" 2>/dev/null | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" 2>/dev/null | awk '{print $1}'
    else
        "$(opsd_python)" -c 'import hashlib, sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' "$1" 2>/dev/null
    fi
}

_opsd_env_diff_keys() {  # OLD NEW: the keys whose KEY=VALUE line differs between two resolved envs
    awk -F= 'NR == FNR { old[$1] = $0; next } { new[$1] = $0 }
        END { for (k in old) if (old[k] != new[k]) print k; for (k in new) if (!(k in old)) print k }' "$1" "$2" \
        | sort | tr '\n' ' '
}

_opsd_trains() {  # does this invocation train? (EVAL_ONLY / EVAL_CKPT_STEPS only evaluate saved weights)
    case "${EVAL_ONLY:-0}" in 1 | true | True | TRUE | yes) return 1 ;; esac
    [ -z "${EVAL_CKPT_STEPS:-}" ]
}

_opsd_resumes_latest() {  # does this invocation continue from the newest checkpoint under $RUN_DIR/ckpts?
    # RESUME_MODE unset means latest (every launcher's default); an explicit RESUME_PATH is a checkpoint the
    # user chose, not "whatever is newest in this directory".
    _opsd_trains && [ -z "${RESUME_PATH:-}" ] && [ "${RESUME_MODE:-latest}" = latest ]
}

_opsd_has_training_output() {  # RUN_DIR: a checkpoint, or exported evaluations FINAL_TEST would rank
    local d
    for d in "$1/ckpts" "$1/exports"; do
        [ -d "$d" ] && [ -n "$(ls -A "$d" 2>/dev/null)" ] && return 0
    done
    return 1
}

# The resolution just written to TMP vs. what $RUN_DIR already holds. Prints the path the resolved env is to
# be stored at, or fails (after saying why) when the run must be refused:
#   * no earlier resolved.env, or the same content: $RUN_DIR/resolved.env;
#   * different content + training that resumes the newest checkpoint (RESUME_MODE=latest, no RESUME_PATH)
#     + $RUN_DIR holds checkpoints or exported evaluations: REFUSED, because they were produced with other
#     settings (a smoke run under an explicit RUN_NAME, a changed KEY=VAL, an edited recipe). DRY_RUN=1 warns;
#   * different content + an evaluation-only invocation (EVAL_ONLY / EVAL_CKPT_STEPS): the side file
#     $RUN_DIR/resolved.eval-<sha12>.env, so the training run's resolved.env (what the check above compares
#     against) and its provenance.json are kept;
#   * different content + any other training run (explicit RESUME_PATH, RESUME_MODE none/from_path, or
#     nothing to resume yet): $RUN_DIR/resolved.env; the earlier resolved.env / provenance.json are moved
#     to resolved.prev-<sha12>.env / provenance.prev-<sha12>.json.
_opsd_resolved_dest() {
    local tmp="$1" prev="$RUN_DIR/resolved.env" old new keys step msg
    if [ ! -f "$prev" ] || cmp -s "$prev" "$tmp"; then
        printf '%s\n' "$prev"
        return 0
    fi
    old="$(_opsd_sha256 "$prev")"
    new="$(_opsd_sha256 "$tmp")"
    keys="$(_opsd_env_diff_keys "$prev" "$tmp")"
    keys="${keys% }"
    if ! _opsd_trains; then
        printf '%s\n' "$RUN_DIR/resolved.eval-${new:0:12}.env"
        return 0
    fi
    if _opsd_resumes_latest && _opsd_has_training_output "$RUN_DIR"; then
        step="$(tr -d '[:space:]' <"$RUN_DIR/ckpts/latest_ckpt_global_step.txt" 2>/dev/null)" || step=""
        case "$step" in '' | *[!0-9]*) step="<N>" ;; esac
        msg="$RUN_DIR already holds a run of ${OPSD_ARM_ID:-this arm} that was resolved with DIFFERENT settings"
        msg="$msg (resolved.env sha256 ${old:0:12} -> ${new:0:12}; changed: ${keys:-?}), and RESUME_MODE=latest"
        msg="$msg would continue from its checkpoints and reuse its evaluations. Give this run its own directory"
        msg="$msg (RUN_NAME=<new name> or RUN_ROOT=<dir>), or, to continue those checkpoints with the new"
        msg="$msg settings on purpose, pass RESUME_MODE=from_path RESUME_PATH=$RUN_DIR/ckpts/global_step_$step"
        if [ "${DRY_RUN:-0}" != 1 ]; then
            printf '[opsd] ERROR: %s\n' "$msg" >&2
            return 1
        fi
        opsd_warn "$msg (DRY_RUN: a real run would refuse)"
    fi
    printf '%s\n' "$prev"
    if [ "${DRY_RUN:-0}" != 1 ]; then
        opsd_warn "the resolved env of $RUN_DIR changes (${keys:-?}); the earlier one is kept as resolved.prev-${old:0:12}.env"
        mv -f "$prev" "$RUN_DIR/resolved.prev-${old:0:12}.env"
        if [ -f "$RUN_DIR/provenance.json" ]; then mv -f "$RUN_DIR/provenance.json" "$RUN_DIR/provenance.prev-${old:0:12}.json"; fi
    fi
    return 0
}

_OPSD_DRY_TRAP_CMD='_opsd_rm_resolved_tmp'

# shellcheck disable=SC2120  # $3 comes from `eval set -- <trap -p output>`, not from a caller
_opsd_dry_exit_trap() {  # DRY_RUN: remove the temp resolved env however the shell exits (installed once)
    [ "${_OPSD_DRY_TRAP:-0}" = 1 ] && return 0
    local cur
    cur="$(trap -p EXIT)"
    if [ -z "$cur" ]; then
        trap -- "$_OPSD_DRY_TRAP_CMD" EXIT
    else  # chain: keep the caller's EXIT trap and the exit status it sees (`&& :` keeps set -e from firing)
        eval "set -- $cur"
        trap -- "_opsd_rc_=\$?; $_OPSD_DRY_TRAP_CMD; (exit \"\$_opsd_rc_\") && :; $3" EXIT
    fi
    _OPSD_DRY_TRAP=1
}

opsd_resolve_recipe() {
    local arm="${1:?usage: opsd_resolve_recipe ARM_ID [KEY=VAL|--overlay NAME ...]}"
    shift
    local tmp v dest
    opsd_repo_root >/dev/null || return 1
    tmp="$(mktemp "${TMPDIR:-/tmp}/opsd_resolved.XXXXXX")" || return 1
    # The hermetic check must see the USER's environment, not the defaults this file exported.
    if ! (
        for v in ${_OPSD_BOOTSTRAP_SET:-}; do unset "$v"; done
        _opsd_pymod opsd.recipes resolve "$arm" --hermetic "$@"
    ) >"$tmp"; then
        rm -f "$tmp"
        opsd_warn "recipe resolution failed for $arm (see the message above)"
        return 1
    fi
    local had_a=0
    case "$-" in *a*) had_a=1 ;; esac
    set -a
    # shellcheck disable=SC1090
    . "$tmp"
    [ "$had_a" = 1 ] || set +a
    if ! dest="$(_opsd_resolved_dest "$tmp")"; then
        rm -f "$tmp"
        return 1
    fi
    if [ "${DRY_RUN:-0}" = 1 ]; then
        _opsd_rm_resolved_tmp  # a previous call's temp file
        export OPSD_RESOLVED_ENV="$tmp"
        _OPSD_RESOLVED_TMP="$tmp"
        _opsd_dry_exit_trap
    else
        mkdir -p "$RUN_DIR"
        mv -f "$tmp" "$dest"
        export OPSD_RESOLVED_ENV="$dest"
        if [ "$dest" != "$RUN_DIR/resolved.env" ]; then
            opsd_log "evaluation-only run: resolved env in $dest (the training run's resolved.env is kept)"
        fi
    fi
    opsd_bootstrap_env
    return 0
}

opsd_dry_run_print() {  # may be called more than once (e.g. one block per trainer invocation)
    local module="${1:?usage: opsd_dry_run_print MODULE [ARGV...]}"
    shift
    printf 'DRY_RUN: %s -m %s \\\n' "$(opsd_python)" "$module"
    local a
    for a in "$@"; do printf '    %q \\\n' "$a"; done
    printf '\n# resolved environment (%s)\n' "${OPSD_RESOLVED_ENV:-<none>}"
    if [ -n "${OPSD_RESOLVED_ENV:-}" ] && [ -f "$OPSD_RESOLVED_ENV" ]; then cat "$OPSD_RESOLVED_ENV"; fi
    printf '# bootstrap: RAY_ADDRESS=%s RAY_TMPDIR=%s LOGGER=%s\n' "${RAY_ADDRESS:-}" "${RAY_TMPDIR:-}" "${LOGGER:-}"
    return 0
}

_opsd_rm_resolved_tmp() {  # the DRY_RUN=1 temp copy of the resolved env (a real run keeps $RUN_DIR/resolved.env)
    if [ -n "${_OPSD_RESOLVED_TMP:-}" ]; then
        rm -f "$_OPSD_RESOLVED_TMP"
        [ "${OPSD_RESOLVED_ENV:-}" != "$_OPSD_RESOLVED_TMP" ] || OPSD_RESOLVED_ENV=""
        _OPSD_RESOLVED_TMP=""
    fi
    return 0
}

# --------------------------------------------------------------------------------------- own processes
_opsd_pid_dir() {
    if [ -z "${RUN_DIR:-}" ]; then
        opsd_warn "RUN_DIR is not set; cannot track processes"
        return 1
    fi
    printf '%s\n' "$RUN_DIR/pids"
}

_opsd_starttime() {  # kernel start time (clock ticks since boot) of PID; fails if PID is gone
    local line
    local -a f
    { IFS= read -r line <"/proc/$1/stat"; } 2>/dev/null || return 1
    read -r -a f <<<"${line##*) }"  # fields after "pid (comm) "; starttime is field 22 overall
    [ -n "${f[19]:-}" ] || return 1
    printf '%s\n' "${f[19]}"
}

_opsd_now_ticks() {  # now, in the unit of _opsd_starttime (rounded down)
    local hz up
    hz="$(getconf CLK_TCK 2>/dev/null)" || hz=""
    { read -r up _ </proc/uptime; } 2>/dev/null || return 1
    awk -v u="$up" -v hz="${hz:-100}" 'BEGIN { printf "%d\n", u * hz }'
}

_opsd_has_tag() {  # PID TAG: PID's environment carries OPSD_BOOTSTRAP_SPAWN_TAG=TAG (unreadable -> no)
    local envs
    envs="$(tr '\0' '\n' <"/proc/$1/environ" 2>/dev/null)" || return 1
    case $'\n'"$envs"$'\n' in
        *$'\n'"OPSD_BOOTSTRAP_SPAWN_TAG=$2"$'\n'*) return 0 ;;
    esac
    return 1
}

_opsd_group_members() {  # PGID -> live (non-zombie) PIDs of that process group, one per line
    ps -e -o pid=,pgid=,stat= 2>/dev/null | awk -v g="$1" '$2 == g && $3 !~ /^Z/ { print $1 }'
}

_opsd_session_members() {  # SID -> "PID PGID" of every live (non-zombie) process in that session
    ps -e -o pid=,pgid=,sid=,stat= 2>/dev/null | awk -v s="$1" '$3 == s && $4 !~ /^Z/ { print $1, $2 }'
}

_opsd_write_pidfile() {  # PID START PGID SPAWN_TICKS TAG MODE LABEL  ("-" = unknown / not applicable)
    # MODE: setsid (CMD leads a new session: everything in it is ours) | jobctl (a job's process group in the
    # launcher's own session: members must carry TAG) | - (a plain tracked PID)
    local dir
    dir="$(_opsd_pid_dir)" || return 1
    mkdir -p "$dir"
    printf '%s %s %s %s %s %s %s\n' "$1" "$2" "$3" "$4" "$5" "$6" "$7" >"$dir/${7//\//_}.$1.pid"
}

opsd_track_pid() {
    local pid="${1:?usage: opsd_track_pid PID [LABEL]}" label="${2:-proc}" st
    _opsd_pid_dir >/dev/null || return 1
    st="$(_opsd_starttime "$pid")" || st=""
    if [ -z "$st" ]; then
        opsd_warn "opsd_track_pid: process $pid ($label) is not running"
        return 1
    fi
    _opsd_write_pidfile "$pid" "$st" - - - - "$label"
}

opsd_untrack_pid() {
    local dir
    dir="$(_opsd_pid_dir 2>/dev/null)" || return 0
    rm -f "$dir"/*."$1".pid 2>/dev/null
    return 0
}

opsd_spawn() {
    local label="${1:?usage: opsd_spawn LABEL CMD [ARGS...]}" ticks tag pid st pgid=- mode=-
    shift
    [ "$#" -gt 0 ] || { opsd_warn "opsd_spawn: no command"; return 1; }
    # Everything CMD starts inherits the new process group (pgid = CMD's PID, still valid after CMD exits:
    # the kernel never reuses a PID while a group or session of that id has members) and
    # OPSD_BOOTSTRAP_SPAWN_TAG; the spawn time rules out anything older. Under setsid CMD also leads a new
    # SESSION, which only its descendants can be in -- including ones that start their own process group or
    # scrub their environment (env -i), so cleanup sweeps the session. With job control on (interactive
    # shells) a background job already leads its own group in the launcher's session (and setsid would fork,
    # hiding the PID), so setsid is only used with it off, and the tag identifies the group's members.
    ticks="$(_opsd_now_ticks)" || ticks=-
    tag="$$.$ticks.$RANDOM$RANDOM$RANDOM"
    if [[ $- == *m* ]]; then
        OPSD_BOOTSTRAP_SPAWN_TAG="$tag" "$@" &
        pid=$!
        pgid=$pid
        mode=jobctl
    elif command -v setsid >/dev/null 2>&1; then
        OPSD_BOOTSTRAP_SPAWN_TAG="$tag" setsid "$@" &
        pid=$!
        pgid=$pid
        mode=setsid
    else
        opsd_warn "opsd_spawn: setsid not found; children that $1 leaves running after it exits cannot be cleaned up"
        OPSD_BOOTSTRAP_SPAWN_TAG="$tag" "$@" &
        pid=$!
    fi
    # shellcheck disable=SC2034  # read by the caller
    OPSD_LAST_PID=$pid
    st="$(_opsd_starttime "$pid")" || st=-  # "-": CMD already exited (and was reaped)
    _opsd_write_pidfile "$pid" "$st" "$pgid" "$ticks" "$tag" "$mode" "$label" || true
    return 0
}

_opsd_tree() {  # PID and all its descendants
    ps -e -o pid=,ppid= 2>/dev/null | awk -v root="$1" '
        { kids[$2] = kids[$2] " " $1 }
        END {
            out = root; queue = root
            while (queue != "") {
                n = split(queue, cur, " "); queue = ""
                for (i = 1; i <= n; i++) {
                    m = split(kids[cur[i]], ch, " ")
                    for (j = 1; j <= m; j++) { out = out " " ch[j]; queue = queue " " ch[j] }
                }
            }
            print out
        }'
}

_opsd_session_strays() {  # SID PGID TICKS: live members of session SID outside group PGID started at/after TICKS
    local p g pst out=""
    while read -r p g; do
        [ "$g" = "$2" ] && continue
        pst="$(_opsd_starttime "$p")" || continue
        [ "$pst" -ge "$3" ] 2>/dev/null || continue
        out="$out $p"
    done < <(_opsd_session_members "$1")
    printf '%s\n' "${out# }"
}

# pid file -> one target per line, "group:<pgid>" or "pids:<p1 p2 ...>"; fails when nothing in it is ours any more
_opsd_targets() {
    local pid st pgid ticks tag mode label cur mypgid p pst members=""
    read -r pid st pgid ticks tag mode label <"$1" || return 1
    case "$pid" in '' | *[!0-9]*) return 1 ;; esac
    if [ "$pid" -le 1 ] || [ "$pid" = "$$" ] || [ "$pid" = "${PPID:-0}" ]; then return 1; fi
    case "$pgid" in '' | *[!0-9]*) pgid="" ;; esac
    mypgid="$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
    cur="$(_opsd_starttime "$pid")" || cur=""
    if [ -n "$pgid" ] && [ "$pgid" -gt 1 ] && [ "$pgid" != "$mypgid" ]; then  # opsd_spawn'ed group
        if [ -n "$cur" ] && [ "$cur" = "$st" ]; then
            echo "group:$pgid"  # its leader is alive, so the group (and session) id cannot have been reused
            if [ "$mode" = setsid ]; then
                members="$(_opsd_session_strays "$pgid" "$pgid" "$ticks")"
                [ -z "$members" ] || echo "pids:$members"
            fi
            return 0
        fi
        # Leader gone. If its PID is in use again, the group emptied before the reuse: nothing left of ours.
        [ -e "/proc/$pgid" ] && return 1
        if [ "$mode" = setsid ]; then
            # Only descendants can be in the session, and its id cannot be reused while any process holds
            # it: every member started after the spawn is ours, tagged or not (env -i, a new process group).
            members="$(_opsd_session_strays "$pgid" "" "$ticks")"
        else
            for p in $(_opsd_group_members "$pgid"); do
                pst="$(_opsd_starttime "$p")" || continue
                [ "$pst" -ge "$ticks" ] 2>/dev/null || continue
                _opsd_has_tag "$p" "$tag" || continue
                members="$members $p"
            done
            members="${members# }"
        fi
        [ -n "$members" ] || return 1
        echo "pids:$members"
        return 0
    fi
    if [ -z "$cur" ] || [ "$cur" != "$st" ]; then return 1; fi  # gone, or the PID was reused
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
    if [ "$pgid" = "$pid" ] && [ "$pgid" != "$mypgid" ]; then
        echo "group:$pgid"
    else
        echo "pids:$(_opsd_tree "$pid")"
    fi
}

_opsd_collect_targets() {  # LABEL_GLOB -> one target per line, ownership checked now
    local dir f t
    dir="$(_opsd_pid_dir 2>/dev/null)" || return 0
    [ -d "$dir" ] || return 0
    for f in "$dir"/$1.pid; do
        [ -e "$f" ] || continue
        t="$(_opsd_targets "$f")" || t=""
        [ -n "$t" ] && printf '%s\n' "$t"
    done
    return 0
}

# shellcheck disable=SC2086  # pids:<list> is split on purpose
_opsd_signal_targets() {  # SIGNAL TARGET...
    local sig="$1" t
    shift
    for t in "$@"; do
        case "$t" in
            group:*) kill -"$sig" -- "-${t#group:}" 2>/dev/null || true ;;
            pids:*) kill -"$sig" ${t#pids:} 2>/dev/null || true ;;
        esac
    done
}

# shellcheck disable=SC2086  # pids:<list> is split on purpose
_opsd_targets_alive() {  # TARGET... -> success while any of them still has a live process
    local t
    for t in "$@"; do
        case "$t" in
            group:*) [ -n "$(_opsd_group_members "${t#group:}")" ] && return 0 ;;
            pids:*) kill -0 ${t#pids:} 2>/dev/null && return 0 ;;
        esac
    done
    return 1
}

opsd_cleanup_own() {
    local glob="${1:-*}" dir f grace i
    local -a targets=()
    dir="$(_opsd_pid_dir 2>/dev/null)" || return 0
    [ -d "$dir" ] || return 0
    mapfile -t targets < <(_opsd_collect_targets "$glob")
    if [ "${#targets[@]}" -gt 0 ]; then
        _opsd_signal_targets TERM "${targets[@]}"
        grace="${OPSD_CLEANUP_GRACE_S:-${OPSD_BOOTSTRAP_CLEANUP_GRACE_S:-20}}"
        for ((i = 0; i < grace * 4; i++)); do
            _opsd_targets_alive "${targets[@]}" || break
            sleep 0.25
        done
        # Re-check ownership before SIGKILL: a PID or group id may have been freed and reused meanwhile.
        mapfile -t targets < <(_opsd_collect_targets "$glob")
        if [ "${#targets[@]}" -gt 0 ]; then _opsd_signal_targets KILL "${targets[@]}"; fi
    fi
    for f in "$dir"/$glob.pid; do
        [ -e "$f" ] && rm -f "$f"
    done
    return 0
}

_opsd_on_exit() {
    local rc=$?
    trap - EXIT
    opsd_cleanup_own
    _opsd_rm_resolved_tmp
    exit "$rc"
}

_opsd_on_signal() {
    trap - EXIT INT TERM
    opsd_log "received SIG$1: stopping the processes this run started"
    opsd_cleanup_own
    _opsd_rm_resolved_tmp
    exit "$2"
}

opsd_install_cleanup_trap() {
    local cur
    cur="$(trap -p EXIT)"
    if [ -n "$cur" ] && [ "${_OPSD_TRAP_INSTALLED:-0}" != 1 ] && [ "$cur" != "trap -- '$_OPSD_DRY_TRAP_CMD' EXIT" ]; then
        opsd_warn "replacing an existing EXIT trap; call opsd_cleanup_own from your own trap instead if you need it"
    fi
    trap _opsd_on_exit EXIT
    trap '_opsd_on_signal INT 130' INT
    trap '_opsd_on_signal TERM 143' TERM
    _OPSD_TRAP_INSTALLED=1
    return 0
}

opsd_run() {
    [ "$#" -gt 0 ] || { opsd_warn "opsd_run: no command"; return 1; }
    [ "${_OPSD_TRAP_INSTALLED:-0}" = 1 ] || opsd_install_cleanup_trap
    "$@" &
    local pid=$! rc=0
    opsd_track_pid "$pid" driver || true
    while :; do
        wait "$pid"
        rc=$?
        kill -0 "$pid" 2>/dev/null || break  # a trapped signal interrupted `wait`: keep waiting
    done
    opsd_untrack_pid "$pid"
    return "$rc"
}

# --------------------------------------------------------------------------------------------- readiness
opsd_wait_http() {
    local url="${1:?usage: opsd_wait_http URL TIMEOUT_S [PID...]}" timeout="${2:-600}" code p deadline
    shift 2 || shift $#
    deadline=$(($(date +%s) + timeout))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        for p in "$@"; do
            kill -0 "$p" 2>/dev/null || { opsd_warn "process $p died while waiting for $url"; return 1; }
        done
        if command -v curl >/dev/null 2>&1; then
            code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$url" 2>/dev/null || true)"
        else
            code="$("$(opsd_python)" -c 'import sys,urllib.request as u
try: print(u.urlopen(sys.argv[1], timeout=5).status)
except Exception as e: print(getattr(e, "code", 0))' "$url" 2>/dev/null || true)"
        fi
        case "$code" in 2??) return 0 ;; esac
        sleep 2
    done
    opsd_warn "timed out after ${timeout}s waiting for $url (last HTTP status: ${code:-none})"
    return 1
}

opsd_wait_ports() {
    local host="${1:?usage: opsd_wait_ports HOST BASE COUNT TIMEOUT_S}" base="${2:?}" count="${3:?}" timeout="${4:-600}"
    local deadline port pending
    deadline=$(($(date +%s) + timeout))
    while :; do
        pending=""
        for ((port = base; port < base + count; port++)); do
            if ! (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then pending="$pending $port"; fi
        done
        [ -z "$pending" ] && return 0
        if [ "$(date +%s)" -ge "$deadline" ]; then
            opsd_warn "timed out after ${timeout}s; still not accepting:$(echo "$pending" | cut -c1-200)"
            return 1
        fi
        sleep 2
    done
}

# -------------------------------------------------------------------------------------------- provenance
opsd_write_provenance() {
    local resolved="${1:?usage: opsd_write_provenance RESOLVED_ENV [OUT]}" out="${2:-}" base
    if [ -z "$out" ]; then  # resolved.env -> provenance.json; resolved.<tag>.env -> provenance.<tag>.json
        base="$(basename "$resolved")"
        case "$base" in
            resolved.?*.env) out="$(dirname "$resolved")/provenance.${base#resolved.}" && out="${out%.env}.json" ;;
            *) out="${RUN_DIR:-.}/provenance.json" ;;
        esac
    fi
    opsd_repo_root >/dev/null || return 0
    if ! _opsd_pymod opsd.recipes provenance "$resolved" --out "$out"; then
        opsd_warn "could not write $out (continuing)"
    fi
    return 0
}

if [ "${OPSD_BOOTSTRAP_AUTO:-1}" != 0 ]; then
    opsd_bootstrap_env
fi
