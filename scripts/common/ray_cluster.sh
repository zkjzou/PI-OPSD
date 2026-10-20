#!/usr/bin/env bash
# =====================================================================================================
# ray_cluster.sh -- bring up a multi-node Ray cluster for the 2-node (30B) runs. Single-node runs do NOT
# need this: the trainer's ray.init() starts a private cluster (see bootstrap.sh).
#
#   head node:     scripts/common/ray_cluster.sh head   [--port 6379] [--temp-dir DIR] [--node-ip IP]
#   every worker:  scripts/common/ray_cluster.sh worker <HEAD_IP>:<PORT> [--temp-dir DIR] [--node-ip IP]
#   head node:     scripts/common/ray_cluster.sh wait   <NUM_NODES> [TIMEOUT_S] [<HEAD_IP>:<PORT>]
#   anywhere:      scripts/common/ray_cluster.sh status [<HEAD_IP>:<PORT>]
#   then, on the head only:  RAY_ADDRESS=<HEAD_IP>:<PORT> scripts/<bench>/run.sh <arm_id> NUM_NODES=2 ...
#
# Requirements: the same repository path, SkyRL checkout (../SkyRL-ecdaa946: the venv's editable skyrl /
# skyrl-gym point at its absolute path) and Python environment on every node; RUN_DIR (checkpoints are
# written by every rank) and OPSD_DATA_ROOT on a filesystem shared by all nodes; the Ray port (RAY_PORT,
# default 6379) and Ray's worker ports reachable between nodes. OPSD_*/ALFWORLD_*/APPWORLD_*/TAU2_*/SKYRL_*
# variables are forwarded to the workers by the trainer's Ray runtime environment, so nothing has to be
# exported before `ray start`.
#
# A worker retries joining until the head answers (OPSD_RAY_JOIN_TIMEOUT_S, default 1800), so the nodes can
# be started in any order. Env: RAY_PORT, NODE_IP (default: first address of `hostname -I`), RAY_TMPDIR
# (default /tmp/opsd_ray_<uid>; keep it short), OPSD_RAY_NUM_GPUS (default: autodetect), OPSD_DRY_RUN=1
# (print the commands instead of running them).
#
# STOPPING: this script never stops anything. `ray stop` on a node stops EVERY Ray process of your user on
# that host, including other runs' clusters; on a shared host stop only this cluster's processes (e.g. the
# `ray start` session you launched), or run `ray stop` only when you know nothing else of yours uses Ray.
# =====================================================================================================
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() { sed -n '2,29p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
die() { echo "ray_cluster.sh: $*" >&2; exit 2; }

python_bin() {
    if [ -n "${OPSD_PYTHON:-}" ]; then echo "$OPSD_PYTHON"; return; fi
    local d="$HERE"
    while [ "$d" != "/" ]; do
        if [ -f "$d/pyproject.toml" ] && [ -f "$d/opsd/__init__.py" ] && [ -f "$d/opsd/skyrl_ext/__init__.py" ] &&
            [ -x "$d/.venv/bin/python" ]; then
            echo "$d/.venv/bin/python"
            return
        fi
        d="$(dirname "$d")"
    done
    command -v python3 || command -v python
}

ray_bin() {
    local py
    py="$(python_bin)"
    if [ -x "$(dirname "$py")/ray" ]; then echo "$(dirname "$py")/ray"; else command -v ray || echo ray; fi
}

run() {
    if [ "${OPSD_DRY_RUN:-0}" = 1 ]; then
        printf 'DRY:'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

node_ip() { echo "${NODE_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"; }

parse_common() {  # sets PORT TEMP_DIR NODE_IP_ARG from flags
    PORT="${RAY_PORT:-6379}"
    TEMP_DIR="${RAY_TMPDIR:-/tmp/opsd_ray_$(id -u)}"
    NODE_IP_ARG="$(node_ip)"
    EXTRA=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --port) PORT="$2"; shift 2 ;;
            --temp-dir) TEMP_DIR="$2"; shift 2 ;;
            --node-ip) NODE_IP_ARG="$2"; shift 2 ;;
            --num-gpus) EXTRA+=("--num-gpus=$2"); shift 2 ;;
            *) die "unknown option $1" ;;
        esac
    done
    if [ -n "${OPSD_RAY_NUM_GPUS:-}" ]; then EXTRA+=("--num-gpus=$OPSD_RAY_NUM_GPUS"); fi
    [ -n "$NODE_IP_ARG" ] || die "cannot determine this node's IP; set NODE_IP or pass --node-ip"
    if [ "${#TEMP_DIR}" -gt 40 ]; then
        echo "ray_cluster.sh: WARNING: --temp-dir $TEMP_DIR is long; Ray's socket paths must stay < 108 bytes" >&2
    fi
}

cmd_head() {
    parse_common "$@"
    run "$(ray_bin)" start --head --port="$PORT" --node-ip-address="$NODE_IP_ARG" --temp-dir="$TEMP_DIR" \
        "${EXTRA[@]+"${EXTRA[@]}"}"
    echo "head is up. On every worker:  $0 worker $NODE_IP_ARG:$PORT"
    echo "then on this node:            $0 wait <NUM_NODES>  and  export RAY_ADDRESS=$NODE_IP_ARG:$PORT"
}

cmd_worker() {
    local addr="${1:-}"
    [ -n "$addr" ] && [ "${addr#-}" = "$addr" ] || die "usage: $0 worker <HEAD_IP>:<PORT> [options]"
    shift
    parse_common "$@"
    local deadline=$(($(date +%s) + ${OPSD_RAY_JOIN_TIMEOUT_S:-1800})) attempt=0
    while :; do
        attempt=$((attempt + 1))
        if run "$(ray_bin)" start --address="$addr" --node-ip-address="$NODE_IP_ARG" --temp-dir="$TEMP_DIR" \
            "${EXTRA[@]+"${EXTRA[@]}"}"; then
            echo "joined $addr (attempt $attempt)"
            return 0
        fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "ray_cluster.sh: could not join $addr within ${OPSD_RAY_JOIN_TIMEOUT_S:-1800}s" >&2
            return 1
        fi
        echo "head $addr not reachable yet (attempt $attempt); retrying in 15s" >&2
        sleep 15
    done
}

cmd_status() {
    local addr="${1:-${RAY_ADDRESS:-}}"
    [ -n "$addr" ] && [ "$addr" != local ] || die "usage: $0 status <HEAD_IP>:<PORT> (or set RAY_ADDRESS)"
    run "$(ray_bin)" status --address="$addr"
}

cmd_wait() {
    local n="${1:-}" timeout="${2:-1800}" addr="${3:-${RAY_ADDRESS:-}}"
    [[ "$n" =~ ^[0-9]+$ ]] || die "usage: $0 wait <NUM_NODES> [TIMEOUT_S] [<HEAD_IP>:<PORT>]"
    if [ -z "$addr" ] || [ "$addr" = local ]; then addr="$(node_ip):${RAY_PORT:-6379}"; fi
    if [ "${OPSD_DRY_RUN:-0}" = 1 ]; then
        echo "DRY: wait for $n alive node(s) at $addr (timeout ${timeout}s)"
        return 0
    fi
    local deadline=$(($(date +%s) + timeout)) alive
    while :; do
        alive="$("$(python_bin)" -c 'import sys, ray
ray.init(address=sys.argv[1], logging_level="ERROR", log_to_driver=False)
print(sum(1 for n in ray.nodes() if n.get("Alive")))' "$addr" 2>/dev/null | tail -n1 || true)"
        echo "alive nodes: ${alive:-?}/$n at $addr" >&2
        if [ -n "$alive" ] && [ "$alive" -ge "$n" ] 2>/dev/null; then return 0; fi
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "ray_cluster.sh: only ${alive:-0}/$n node(s) joined within ${timeout}s" >&2
            return 1
        fi
        sleep 10
    done
}

case "${1:-}" in
    head) shift; cmd_head "$@" ;;
    worker) shift; cmd_worker "$@" ;;
    status) shift; cmd_status "$@" ;;
    wait) shift; cmd_wait "$@" ;;
    stop) echo "ray_cluster.sh does not stop clusters; see STOPPING in --help" >&2; exit 2 ;;
    -h | --help | help) usage ;;
    *) usage >&2; exit 2 ;;
esac
