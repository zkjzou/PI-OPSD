#!/usr/bin/env bash
# shellcheck shell=bash
# =====================================================================================================
# lib.sh -- helpers of the tau2 launcher (scripts/tau2/run.sh). SOURCE it after
# scripts/common/bootstrap.sh; it defines functions only and changes nothing on its own.
#
#   tau2_require_keys KEY...            fail (listing them) if any KEY is unset or empty
#   tau2_check_ambient_knobs FILE KEY...  fail if a KEY is set in the environment but not in the resolved env FILE
#   tau2_is_self_hosted URL             0 if URL is a plain-http endpoint (self-hosted, needs no API key)
#   tau2_check_llm_keys                 the simulator / eval-simulator / NL-judge keys a remote endpoint needs
#                                       (checked in this shell; the names land in TAU2_REMOTE_KEY_ENVS)
#   tau2_check_keys_on_ray_driver_node NAME...  multi-node: fail unless every NAME is set for a Ray task on
#                                       this node, which is where the generator reads them
#   tau2_truncate_manifest SRC N DEST   first N rows of a manifest into DEST (smoke runs)
#   tau2_resume_checkpoint CKPT_DIR     print the checkpoint dir a resume would load (empty if none)
#   tau2_maybe_reset_dataloader DIR     opt-in epoch-boundary data.pt reset (RESUME_RESET_DATALOADER; no longer
#                                       needed, the trainer drains an exhausted restored iterator itself)
#   tau2_node_ip                        first address of `hostname -I` (multi-node HTTP endpoint host)
# =====================================================================================================

tau2_require_keys() {
    local k missing=()
    for k in "$@"; do
        [ -n "${!k:-}" ] || missing+=("$k")
    done
    if [ "${#missing[@]}" -gt 0 ]; then
        opsd_warn "the resolved recipe does not set: ${missing[*]} (every tau2 arm must; see opsd/recipes/tau2/base.env)"
        return 1
    fi
    return 0
}

tau2_check_ambient_knobs() {  # RESOLVED_ENV NAME...: fail if NAME is set but not part of the resolved recipe
    local resolved="${1:?}" name bad=0
    shift
    for name in "$@"; do
        [ -n "${!name+x}" ] || continue
        grep -q "^${name}=" "$resolved" 2>/dev/null && continue
        opsd_warn "ambient $name=${!name} is not part of the recipe; pass it as $name=... on the command line (it is then recorded in resolved.env) or unset it"
        bad=1
    done
    return "$bad"
}

tau2_is_self_hosted() {
    case "${1:-}" in
        http://*) return 0 ;;
        *) return 1 ;;
    esac
}

# Effective base URL of an endpoint whose config default is $OPENAI_BASE_URL, else the public OpenAI API
# (opsd/tau2_bench/constants.py::default_llm_base_url). A comma-separated list (one URL per
# self-hosted simulator replica) is judged by its first entry.
_tau2_effective_url() {
    local u="${1:-}"
    [ -n "$u" ] || u="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
    printf '%s\n' "${u%%,*}"
}

# WHAT URL ENV_NAME: a remote URL needs the key in ENV_NAME -- record the name in TAU2_REMOTE_KEY_ENVS and
# return 1 (with a message) when it is empty in this shell. A self-hosted URL needs nothing.
_tau2_need_key() {
    local what="$1" url="$2" env_name="$3"
    tau2_is_self_hosted "$url" && return 0
    case " ${TAU2_REMOTE_KEY_ENVS:-} " in
        *" $env_name "*) ;;
        *) TAU2_REMOTE_KEY_ENVS="${TAU2_REMOTE_KEY_ENVS:+$TAU2_REMOTE_KEY_ENVS }$env_name" ;;
    esac
    [ -n "${!env_name:-}" ] && return 0
    opsd_warn "$what at $url needs an API key: set $env_name"
    return 1
}

tau2_check_llm_keys() {
    local bad=0
    TAU2_REMOTE_KEY_ENVS=""
    # training simulator
    _tau2_need_key "the simulated user ($SIMULATOR_MODEL)" "$(_tau2_effective_url "${SIMULATOR_BASE_URL:-}")" \
        "${SIM_API_KEY_ENV:-OPENAI_API_KEY}" || bad=1
    # evaluation simulator (empty model => evaluation reuses the training simulator)
    if [ -n "${EVAL_SIMULATOR_MODEL:-}" ]; then
        _tau2_need_key "the evaluation simulator ($EVAL_SIMULATOR_MODEL)" \
            "$(_tau2_effective_url "${EVAL_SIMULATOR_BASE_URL:-}")" "${EVAL_SIMULATOR_API_KEY_ENV:-OPENAI_API_KEY}" || bad=1
    fi
    # NL-assertion judge (gpt-4.1; 40 of the 114 retail_base evaluation tasks call it inside the reward)
    _tau2_need_key "the NL-assertion judge (openai/gpt-4.1)" "$(_tau2_effective_url "${JUDGE_BASE_URL:-}")" \
        "${JUDGE_API_KEY_ENV:-OPENAI_API_KEY}" || bad=1
    return "$bad"
}

# On an existing (multi-node) Ray cluster the generator runs inside the trainer's entrypoint task, pinned to
# the driver's node. A Ray task's environment is the one its raylet captured at `ray start`, plus the
# runtime env the trainer forwards -- which never carries a name containing KEY/TOKEN/SECRET/... So a key
# exported in the launching shell only passes tau2_check_llm_keys, while the generator falls back to a dummy
# key and every simulator / judge call fails with HTTP 401. This probe asks a Ray task on THIS node, given
# the same forwarded runtime env, which of the names it cannot see. (The Python source carries a marker
# line so tests can stand in for it without a cluster.)
_TAU2_RAY_KEY_PROBE='# tau2_ray_key_probe
import logging
import os
import sys

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

names = sys.argv[1:]
try:  # the runtime env the trainer gives its tasks (credential-like names are never part of it)
    from opsd.skyrl_ext.ray_runtime import collect_forwarded_env_vars

    env_vars = {k: v for k, v in collect_forwarded_env_vars().items() if k in names}
except ImportError as exc:  # a broken install: say so, then probe without the forwarded env
    print(f"[tau2] WARNING: no opsd.skyrl_ext.ray_runtime ({exc}); probing without the forwarded env", file=sys.stderr)
    env_vars = {}
ray.init(
    address=os.environ["RAY_ADDRESS"],
    log_to_driver=False,
    logging_level=logging.ERROR,
    runtime_env={"env_vars": env_vars} if env_vars else None,
)


@ray.remote(num_cpus=0)
def probe(names):
    import os
    import socket

    return socket.gethostname(), [n for n in names if not os.environ.get(n)]


here = NodeAffinitySchedulingStrategy(node_id=ray.get_runtime_context().get_node_id(), soft=False)
host, missing = ray.get(probe.options(scheduling_strategy=here).remote(names), timeout=600)
ray.shutdown()
print("TAU2_KEY_PROBE", host, *missing)
'

tau2_check_keys_on_ray_driver_node() {  # NAME...
    [ "$#" -gt 0 ] || return 0
    local out line host
    if ! out="$("$(opsd_python)" -c "$_TAU2_RAY_KEY_PROBE" "$@" 2>&1)"; then
        opsd_warn "could not run the API-key probe on the Ray cluster at ${RAY_ADDRESS:-?}:"
        printf '%s\n' "$out" | tail -n 20 >&2
        return 1
    fi
    printf '%s\n' "$out" | grep '^\[tau2\] WARNING' >&2 || true  # e.g. the forwarder could not be imported
    line="$(printf '%s\n' "$out" | grep '^TAU2_KEY_PROBE ' | tail -n 1)"
    if [ -z "$line" ]; then
        opsd_warn "the API-key probe on the Ray cluster at ${RAY_ADDRESS:-?} printed no result:"
        printf '%s\n' "$out" | tail -n 20 >&2
        return 1
    fi
    # shellcheck disable=SC2086
    set -- $line # TAU2_KEY_PROBE <host> [<missing name>...]
    host="$2"
    shift 2
    if [ "$#" -gt 0 ]; then
        opsd_warn "Ray tasks on this node ($host) do not see: $*. The simulator / judge would run without a key" \
            "(HTTP 401 on every call): a Ray task inherits the environment of \`ray start\`, and key variables are never" \
            "forwarded to it. Export $* in the shell that starts the head, BEFORE \`scripts/common/ray_cluster.sh head\`" \
            "(restart the head you started for this run), then launch again."
        return 1
    fi
    return 0
}

tau2_truncate_manifest() {
    local src="${1:?}" n="${2:?}" dest="${3:?}"
    case "$n" in '' | *[!0-9]*) opsd_warn "task limit '$n' is not a non-negative integer"; return 1 ;; esac
    mkdir -p "$(dirname "$dest")"
    head -n "$n" "$src" >"$dest"
    [ -s "$dest" ] || { opsd_warn "truncating $src to $n rows produced an empty manifest"; return 1; }
    return 0
}

tau2_resume_checkpoint() {
    local ckpt_dir="${1:?}" step
    if [ "${RESUME_MODE:-latest}" = from_path ] && [ -n "${RESUME_PATH:-}" ]; then
        printf '%s\n' "$RESUME_PATH"
        return 0
    fi
    if [ "${RESUME_MODE:-latest}" = latest ] && [ -f "$ckpt_dir/latest_ckpt_global_step.txt" ]; then
        step="$(tr -d '[:space:]' <"$ckpt_dir/latest_ckpt_global_step.txt")"
        case "$step" in '' | *[!0-9]*) return 0 ;; esac
        printf '%s\n' "$ckpt_dir/global_step_$step"
    fi
    return 0
}

# A resume from the LAST batch of an epoch restores an exhausted dataloader iterator. The trainer
# (OPSDRayPPOTrainer.load_checkpoints) now drains it itself, so the next epoch is a full pass without this.
# With RESUME_RESET_DATALOADER=auto the saved data.pt is inspected and set aside only when its iterator is
# exhausted (the data order then restarts from the seeded generator: same task multiset, a different order
# than an uninterrupted run); =1 always sets it aside; unset/0 never touches the checkpoint. The file is
# renamed, never deleted.
tau2_maybe_reset_dataloader() {
    local ckpt="${1:-}" mode="${RESUME_RESET_DATALOADER:-0}" verdict ts
    [ "$mode" != 0 ] || return 0
    [ -n "$ckpt" ] && [ -f "$ckpt/data.pt" ] || return 0
    case "$mode" in
        1 | true | yes) verdict="1 forced" ;;
        auto)
            verdict="$("$(opsd_python)" - "$ckpt/data.pt" "${ROLLOUT_BATCHES:-0}" <<'PY' 2>/dev/null
import sys, torch
s = torch.load(sys.argv[1], weights_only=False, map_location="cpu")
n, total = int(s.get("_num_yielded") or 0), int(sys.argv[2])
print(f"{int(bool(s.get('_iterator_finished')) or (total > 0 and n >= total))} {n}")
PY
)" || verdict=""
            ;;
        *)
            opsd_warn "RESUME_RESET_DATALOADER=$mode (expected 0, 1 or auto); leaving $ckpt/data.pt"
            return 0
            ;;
    esac
    case "$verdict" in
        "1 "*)
            ts="$(date +%Y%m%d_%H%M%S)"
            opsd_log "resume: data.pt is an exhausted epoch iterator (${verdict#1 }; RESUME_RESET_DATALOADER=$mode): moved to data.pt.exhausted-$ts so the next epoch is a full pass"
            mv -f "$ckpt/data.pt" "$ckpt/data.pt.exhausted-$ts"
            ;;
        "0 "*) opsd_log "resume: data.pt is mid-epoch (yielded ${verdict#0 } of ${ROLLOUT_BATCHES:-?}): kept" ;;
        *) opsd_warn "resume: could not read $ckpt/data.pt; kept (RESUME_RESET_DATALOADER=1 forces the reset)" ;;
    esac
    return 0
}

tau2_node_ip() {
    hostname -I 2>/dev/null | awk '{print $1}'
}
