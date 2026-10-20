#!/usr/bin/env bash
# shellcheck shell=bash
# =====================================================================================================
# preflight.sh -- refuse to start a run on a machine that is not ready for it. Read-only: it inspects
# GPUs, listening ports, disk and input files, and never starts, stops or signals anything.
#
#   source scripts/common/preflight.sh; opsd_preflight_all       # from a launcher, after the recipe
#   bash scripts/common/preflight.sh                             # same, from the shell
#   bash scripts/common/preflight.sh --gpus 4 --ports 12500:8:alfworld --disk ./runs:50 --files a.json
#
# Functions (each prints PASS/FAIL lines on stderr and returns non-zero on failure):
#   opsd_preflight_gpus [N]               every GPU in CUDA_VISIBLE_DEVICES (all GPUs when unset) has no
#                                         compute process and < OPSD_PREFLIGHT_GPU_MAX_USED_MB (2048) MiB
#                                         used, and at least N (default NUM_GPUS_PER_NODE) are visible.
#                                         Indices and GPU-<uuid> entries are both accepted.
#   opsd_preflight_ports BASE COUNT [LABEL]   no listening TCP socket on BASE..BASE+COUNT-1 (`ss -ltn`, or
#                                         /proc/net/tcp{,6}). A busy port is never reused: a stale server
#                                         would answer the readiness probe for us.
#   opsd_preflight_disk PATH MIN_GB       free space on the filesystem holding PATH (or its nearest parent)
#   opsd_preflight_files PATH...          each path exists
#   opsd_preflight_hint_sha               OPSD_HINTS_PATH content sha256 == OPSD_HINTS_CONTENT_SHA256 (if set)
#   opsd_preflight_all                    all of the above, driven by the resolved recipe environment:
#       GPUs (NUM_GPUS_PER_NODE); ports HTTP_ENDPOINT_PORT, ALFWORLD_BASE_PORT+NUM_ALFWORLD_SERVERS,
#       APPWORLD_BASE_PORT+NUM_APPWORLD_SERVERS (only the current OPSD_BENCH's range when it is set) and
#       OPSD_PREFLIGHT_PORTS="BASE:COUNT[:LABEL] ..."; disk under RUN_ROOT (OPSD_PREFLIGHT_MIN_FREE_GB, 50)
#       and /tmp (warning below 10 GB); input files OPSD_HINTS_PATH TASK_IDS_PATH DEV_TASK_IDS_PATH
#       EVAL_HINT_TASK_IDS_PATH EVAL_HINT_DIR HINTS_DIR (when set); the hint content sha256.
#
# Knobs: OPSD_PREFLIGHT=0 (skip everything, with a warning), OPSD_PREFLIGHT_SKIP_GPUS=1,
# OPSD_PREFLIGHT_GPU_MAX_USED_MB, OPSD_PREFLIGHT_MIN_FREE_GB, OPSD_PREFLIGHT_PORTS,
# OPSD_PREFLIGHT_PORT_SOURCE=ss|proc (default: ss when installed), OPSD_PREFLIGHT_PROC_NET (files to parse
# in proc mode; default /proc/net/tcp /proc/net/tcp6).
# Re-run opsd_preflight_gpus right before the trainer starts if anything else may grab GPUs meanwhile.
# =====================================================================================================

_opsd_pf_ok() { printf '[preflight] PASS %s\n' "$*" >&2; }
_opsd_pf_fail() { printf '[preflight] FAIL %s\n' "$*" >&2; }
_opsd_pf_warn() { printf '[preflight] WARN %s\n' "$*" >&2; }

opsd_preflight_gpus() {
    local need="${1:-${NUM_GPUS_PER_NODE:-0}}" max_mb="${OPSD_PREFLIGHT_GPU_MAX_USED_MB:-2048}"
    local gpus apps vis entry idx uuid used napps bad=0 n=0
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        _opsd_pf_fail "nvidia-smi not found (set OPSD_PREFLIGHT_SKIP_GPUS=1 to skip the GPU check)"
        return 1
    fi
    if ! gpus="$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits 2>&1)"; then
        _opsd_pf_fail "nvidia-smi failed: $gpus"
        return 1
    fi
    apps="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
    if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
        vis="$(printf '%s\n' "$gpus" | awk -F', *' 'NF >= 3 {printf "%s%s", sep, $1; sep = ","}')"
    else
        vis="$CUDA_VISIBLE_DEVICES"
        if [ -z "$vis" ]; then
            _opsd_pf_fail "CUDA_VISIBLE_DEVICES is set but empty"
            return 1
        fi
    fi
    local IFS=,
    for entry in $vis; do
        entry="${entry// /}"
        local row
        row="$(printf '%s\n' "$gpus" | awk -F', *' -v e="$entry" \
            '$1 == e || $2 == e || (e ~ /^GPU-/ && index($2, e) == 1) {print $1 "|" $2 "|" $3; exit}')"
        if [ -z "$row" ]; then
            _opsd_pf_fail "GPU '$entry' from CUDA_VISIBLE_DEVICES does not exist"
            bad=1
            continue
        fi
        idx="${row%%|*}"; uuid="${row#*|}"; used="${uuid#*|}"; uuid="${uuid%%|*}"
        napps="$(printf '%s\n' "$apps" | awk -F', *' -v u="$uuid" '$1 == u {c++} END {print c + 0}')"
        n=$((n + 1))
        if [ "$napps" -gt 0 ] || [ "${used%%.*}" -ge "$max_mb" ] 2>/dev/null; then
            _opsd_pf_fail "GPU $idx is busy: $napps compute process(es), ${used} MiB used (limit ${max_mb} MiB)"
            bad=1
        fi
    done
    unset IFS
    if [ "$need" -gt "$n" ] 2>/dev/null; then
        _opsd_pf_fail "$n GPU(s) visible but the recipe needs NUM_GPUS_PER_NODE=$need"
        bad=1
    fi
    [ "$bad" = 0 ] && _opsd_pf_ok "$n GPU(s) free (${vis})"
    return "$bad"
}

_opsd_listening_ports() {
    local src="${OPSD_PREFLIGHT_PORT_SOURCE:-}"
    if [ -z "$src" ]; then
        if command -v ss >/dev/null 2>&1; then src=ss; else src=proc; fi
    fi
    if [ "$src" = ss ]; then
        ss -Hltn 2>/dev/null | awk '{n = split($4, a, ":"); if (a[n] ~ /^[0-9]+$/) print a[n]}'
    else
        local f hex
        # shellcheck disable=SC2086
        for f in ${OPSD_PREFLIGHT_PROC_NET:-/proc/net/tcp /proc/net/tcp6}; do
            [ -r "$f" ] || continue
            awk 'NR > 1 && $4 == "0A" {n = split($2, a, ":"); print a[n]}' "$f"
        done | while read -r hex; do echo $((16#$hex)); done
    fi | sort -un
}

opsd_preflight_ports() {
    local base="${1:?usage: opsd_preflight_ports BASE COUNT [LABEL]}" count="${2:?}" label="${3:-ports}" busy
    busy="$(_opsd_listening_ports | awk -v lo="$base" -v hi="$((base + count - 1))" '$1 >= lo && $1 <= hi' | tr '\n' ' ')"
    if [ -n "$busy" ]; then
        _opsd_pf_fail "$label: port(s) already listening in $base-$((base + count - 1)): $(echo "$busy" | cut -c1-120)" \
            "-- pick another base port; never reuse a busy one"
        return 1
    fi
    _opsd_pf_ok "$label: ports $base-$((base + count - 1)) free"
    return 0
}

opsd_preflight_disk() {
    local path="${1:?usage: opsd_preflight_disk PATH MIN_GB}" min_gb="${2:?}" p avail_kb
    p="$path"
    while [ ! -e "$p" ] && [ "$p" != "/" ] && [ "$p" != "." ]; do p="$(dirname "$p")"; done
    avail_kb="$(df -Pk "$p" 2>/dev/null | awk 'NR == 2 {print $4}')"
    if [ -z "$avail_kb" ]; then
        _opsd_pf_fail "cannot read free space for $path"
        return 1
    fi
    if [ "$avail_kb" -lt $((min_gb * 1024 * 1024)) ]; then
        _opsd_pf_fail "only $((avail_kb / 1024 / 1024)) GB free under $p (need $min_gb GB)"
        return 1
    fi
    _opsd_pf_ok "$((avail_kb / 1024 / 1024)) GB free under $p (need $min_gb GB)"
    return 0
}

opsd_preflight_files() {
    local f bad=0
    for f in "$@"; do
        if [ -e "$f" ]; then
            _opsd_pf_ok "found $f"
        else
            _opsd_pf_fail "missing $f (fetch the data bundle into OPSD_DATA_ROOT: python -m opsd.data.fetch)"
            bad=1
        fi
    done
    return "$bad"
}

opsd_preflight_hint_sha() {
    [ -n "${OPSD_HINTS_PATH:-}" ] && [ -n "${OPSD_HINTS_CONTENT_SHA256:-}" ] && [ -f "$OPSD_HINTS_PATH" ] || return 0
    local here root py got
    here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    root="$here"
    while [ "$root" != "/" ] && ! { [ -f "$root/pyproject.toml" ] && [ -f "$root/opsd/__init__.py" ] &&
        [ -f "$root/opsd/skyrl_ext/__init__.py" ]; }; do
        root="$(dirname "$root")"
    done
    py="${OPSD_PYTHON:-}"
    [ -n "$py" ] || { [ -x "$root/.venv/bin/python" ] && py="$root/.venv/bin/python"; } || py="python3"
    # no cd: a relative OPSD_HINTS_PATH is relative to where the launcher was started (the repo root goes
    # first on sys.path instead, and the working directory comes off it so nothing there shadows a module)
    if ! got="$("$py" -c 'import os, sys
root, here = sys.argv.pop(1), os.getcwd()
sys.path[:] = [root] + [p for p in sys.path if p not in ("", ".", here)]
from opsd.recipes.resolve import hint_content_sha256 as h
print(h(sys.argv[1]))' "$root" "$OPSD_HINTS_PATH" 2>&1)"; then
        _opsd_pf_fail "could not hash $OPSD_HINTS_PATH: $got"
        return 1
    fi
    if [ "$got" != "$OPSD_HINTS_CONTENT_SHA256" ]; then
        _opsd_pf_fail "hint content sha256 mismatch for $OPSD_HINTS_PATH: $got != $OPSD_HINTS_CONTENT_SHA256" \
            "(declared by the recipe for the published file). Re-fetch the data bundle; to run a modified" \
            "file on purpose pass OPSD_HINTS_CONTENT_SHA256= (empty) or its sha256 as KEY=VAL"
        return 1
    fi
    _opsd_pf_ok "hint content sha256 matches the recipe ($OPSD_HINTS_PATH)"
    return 0
}

opsd_preflight_all() {
    if [ "${OPSD_PREFLIGHT:-1}" = 0 ]; then
        _opsd_pf_warn "OPSD_PREFLIGHT=0: all checks skipped"
        return 0
    fi
    local bad=0 bench="${OPSD_BENCH:-}" spec base count label f files=()
    if [ "${OPSD_PREFLIGHT_SKIP_GPUS:-0}" = 1 ]; then
        _opsd_pf_warn "GPU check skipped (OPSD_PREFLIGHT_SKIP_GPUS=1)"
    else
        opsd_preflight_gpus || bad=1
    fi
    if [ -n "${HTTP_ENDPOINT_PORT:-}" ]; then opsd_preflight_ports "$HTTP_ENDPOINT_PORT" 1 http_endpoint || bad=1; fi
    if [ -n "${ALFWORLD_BASE_PORT:-}" ] && [ -n "${NUM_ALFWORLD_SERVERS:-}" ] && { [ -z "$bench" ] || [ "$bench" = alfworld ]; }; then
        opsd_preflight_ports "$ALFWORLD_BASE_PORT" "$NUM_ALFWORLD_SERVERS" alfworld_servers || bad=1
    fi
    if [ -n "${APPWORLD_BASE_PORT:-}" ] && [ -n "${NUM_APPWORLD_SERVERS:-}" ] && { [ -z "$bench" ] || [ "$bench" = appworld ]; }; then
        opsd_preflight_ports "$APPWORLD_BASE_PORT" "$NUM_APPWORLD_SERVERS" appworld_servers || bad=1
    fi
    for spec in ${OPSD_PREFLIGHT_PORTS:-}; do
        IFS=: read -r base count label <<<"$spec"
        opsd_preflight_ports "$base" "${count:-1}" "${label:-extra}" || bad=1
    done
    opsd_preflight_disk "${RUN_ROOT:-${RUN_DIR:-.}}" "${OPSD_PREFLIGHT_MIN_FREE_GB:-50}" || bad=1
    opsd_preflight_disk /tmp 10 >/dev/null 2>&1 || _opsd_pf_warn "less than 10 GB free in /tmp (Ray spills objects there)"
    for f in OPSD_HINTS_PATH TASK_IDS_PATH DEV_TASK_IDS_PATH EVAL_HINT_TASK_IDS_PATH EVAL_HINT_DIR HINTS_DIR; do
        [ -n "${!f:-}" ] && files+=("${!f}")
    done
    if [ "${#files[@]}" -gt 0 ]; then opsd_preflight_files "${files[@]}" || bad=1; fi
    opsd_preflight_hint_sha || bad=1
    if [ "$bad" = 0 ]; then
        _opsd_pf_ok "all preflight checks passed"
    else
        _opsd_pf_fail "preflight failed (fix the items above, or OPSD_PREFLIGHT=0 to override at your own risk)"
    fi
    return "$bad"
}

# ---- executed directly: parse flags (no flags = opsd_preflight_all) --------------------------------
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    if [ "$#" -eq 0 ]; then
        opsd_preflight_all
        exit $?
    fi
    _rc=0
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --all) opsd_preflight_all || _rc=1; shift ;;
            --gpus) opsd_preflight_gpus "${2:-}" || _rc=1; shift 2 ;;
            --ports) IFS=: read -r _b _c _l <<<"$2"; opsd_preflight_ports "$_b" "${_c:-1}" "${_l:-ports}" || _rc=1; shift 2 ;;
            --disk) opsd_preflight_disk "${2%:*}" "${2##*:}" || _rc=1; shift 2 ;;
            --files) shift; while [ "$#" -gt 0 ] && [ "${1#--}" = "$1" ]; do opsd_preflight_files "$1" || _rc=1; shift; done ;;
            -h | --help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
            *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
        esac
    done
    exit "$_rc"
fi
