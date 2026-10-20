#!/usr/bin/env bash
# shellcheck shell=bash
# =====================================================================================================
# servers.sh -- AppWorld helpers shared by scripts/appworld/{run,passk,pi_utility}.sh. SOURCE it after
# scripts/common/bootstrap.sh; it only defines functions.
#
#   aw_resolve_paths                 APPWORLD_VENV / APPWORLD_ROOT / APPWORLD_PY / APPWORLD_BIN (+ checks)
#   aw_bundle_dir                    print the unpacked AppWorld data bundle dir (may not exist)
#   aw_is_patched                    0 if the b0a8eae_* train tasks are patched in $APPWORLD_ROOT
#   aw_build_parquets DIR VARIANT [INCLUDE_DEV]
#                                    build (once, atomically, under a lock) the SkyRL parquets with
#                                    appworld_dataset.py --split grpo_recipe (+ train_dev), using the
#                                    AppWorld venv's python from $APPWORLD_ROOT; rebuilt when the prompt
#                                    variant, the b0a8eae patch state or the AppWorld version changes
#   aw_parquet_rows FILE             number of rows of a parquet file
#   aw_head_parquet SRC N DEST       first N rows of SRC into DEST (smoke runs)
#   aw_server_urls BASE COUNT        ["http://127.0.0.1:BASE",...] (a hydra list literal)
#   aw_start_server_pool BASE COUNT LOGDIR [LABEL]
#                                    one SUPERVISED `appworld serve environment` per port: each runs in a
#                                    restart loop, so a server that the env recycles (or that crashes) comes
#                                    back on the same port. Every supervisor is started with opsd_spawn as
#                                    <LABEL>-<port> (default LABEL appworld-server), i.e. tracked in
#                                    $RUN_DIR/pids/ and stopped by the run's cleanup trap (or explicitly with
#                                    opsd_cleanup_own '<LABEL>-*').
#   aw_wait_server_pool BASE COUNT NEED TIMEOUT_S
#                                    wait until every port accepts TCP; fails if none does, warns when
#                                    fewer than NEED (= concurrent rollouts) are up at the deadline
#   aw_check_port_owners BASE COUNT [LABEL]
#                                    fail if any listening socket in the range belongs to a process this run
#                                    did not start (a stale or foreign server would otherwise silently serve
#                                    our episodes)
#   aw_student KEY                   MODEL_PATH / SERVED_MODEL_NAME / VLLM_TP / PRESERVE_THINKING of a student
#                                    (4bi | 4bt | 30bi) for the offline generation scripts
#   aw_sampling PROFILE              TEMP / TOPP / TOPK of a sampling profile (onpolicy | official | thinking)
#   aw_serve_vllm OUTDIR             serve MODEL_PATH on the visible GPUs (one replica per VLLM_TP GPUs, ports
#                                    VLLM_BASE_PORT+), wait until healthy; sets VLLM_URLS (comma list)
# =====================================================================================================

aw_resolve_paths() {
    APPWORLD_VENV="${APPWORLD_VENV:-$HOME/appworld-venv}"
    APPWORLD_ROOT="${APPWORLD_ROOT:-$HOME/appworld_root}"
    APPWORLD_PY="${APPWORLD_PY:-$APPWORLD_VENV/bin/python}"
    APPWORLD_BIN="${APPWORLD_BIN:-$APPWORLD_VENV/bin/appworld}"
    export APPWORLD_VENV APPWORLD_ROOT APPWORLD_PY APPWORLD_BIN
    return 0
}

aw_check_install() {  # fail with a pointer to the setup script unless the AppWorld venv + data are usable
    local bad=0
    if [ ! -x "$APPWORLD_PY" ] || ! "$APPWORLD_PY" -c "import appworld, appworld.apps; appworld.apps.api_docs" >/dev/null 2>&1; then
        opsd_warn "AppWorld is not installed in APPWORLD_VENV=$APPWORLD_VENV"
        bad=1
    fi
    if [ ! -d "$APPWORLD_ROOT/data/tasks" ]; then
        opsd_warn "no AppWorld data under APPWORLD_ROOT=$APPWORLD_ROOT (expected $APPWORLD_ROOT/data/tasks)"
        bad=1
    fi
    if [ "$bad" = 1 ]; then
        opsd_warn "run: APPWORLD_VENV=$APPWORLD_VENV APPWORLD_ROOT=$APPWORLD_ROOT bash scripts/setup/setup_appworld.sh"
        return 1
    fi
    return 0
}

aw_bundle_dir() {
    printf '%s\n' "${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}/appworld/plain/opsd_appworld_pi"
}

aw_is_patched() {
    local f
    for f in "$APPWORLD_ROOT"/data/tasks/b0a8eae_{1,2,3}/ground_truth/evaluation.py; do
        [ -f "$f" ] && grep -q PATCHED "$f" || return 1
    done
    return 0
}

aw_parquet_rows() {
    "$APPWORLD_PY" -c 'import sys, pandas as pd; print(len(pd.read_parquet(sys.argv[1])))' "$1"
}

aw_head_parquet() {  # SRC N DEST
    "$APPWORLD_PY" -c 'import sys, pandas as pd
d = pd.read_parquet(sys.argv[1]).head(int(sys.argv[2]))
d.to_parquet(sys.argv[3], index=False)
print(len(d))' "$1" "$2" "$3"
}

_aw_data_stamp() {  # VARIANT
    local ver patched=0
    ver="$("$APPWORLD_PY" -c 'import appworld; print(getattr(appworld, "__version__", "?"))' 2>/dev/null || echo '?')"
    aw_is_patched && patched=1
    printf 'prompt_variant=%s b0a8eae_patched=%s appworld=%s\n' "$1" "$patched" "$ver"
}

aw_build_parquets() {  # DIR VARIANT [INCLUDE_DEV]
    local dir="${1:?usage: aw_build_parquets DIR VARIANT [INCLUDE_DEV]}" variant="${2:?}" include_dev="${3:-0}"
    local root files f stamp want ok lock tmp i holder rc
    root="$(opsd_repo_root)" || return 1
    files=(train.parquet validation.parquet eval_normal.parquet eval_challenge.parquet)
    [ "$include_dev" = 1 ] && files+=(train_dev.parquet)
    want="$(_aw_data_stamp "$variant")"
    mkdir -p "$dir" || return 1
    lock="$dir/.build.lock"
    for ((i = 0; i < 3600; i++)); do
        if mkdir "$lock" 2>/dev/null; then
            echo $$ >"$lock/pid"
            break
        fi
        holder="$(cat "$lock/pid" 2>/dev/null || true)"
        if [ -n "$holder" ] && ! kill -0 "$holder" 2>/dev/null; then
            opsd_warn "removing the stale parquet build lock of dead process $holder ($lock)"
            rm -rf "$lock"
            continue
        fi
        [ $((i % 60)) = 0 ] && opsd_log "waiting for another run to finish building $dir (lock held by ${holder:-?})"
        sleep 1
    done
    [ -d "$lock" ] && [ "$(cat "$lock/pid" 2>/dev/null)" = "$$" ] || { opsd_warn "could not take $lock"; return 1; }
    ok=1
    for f in "${files[@]}"; do [ -s "$dir/$f" ] || ok=0; done
    stamp="$(cat "$dir/.build_stamp" 2>/dev/null || true)"
    if [ "$ok" = 1 ] && [ "$stamp" = "$want" ]; then
        rm -rf "$lock"
        opsd_log "AppWorld parquets up to date in $dir ($want)"
        return 0
    fi
    [ -n "$stamp" ] && [ "$stamp" != "$want" ] && opsd_log "rebuilding $dir: was '$stamp', now '$want'"
    tmp="$dir/.build.$$"
    rm -rf "$tmp" && mkdir -p "$tmp"
    opsd_log "building AppWorld parquets ($want) -> $dir"
    rc=0
    (cd "$APPWORLD_ROOT" && APPWORLD_ROOT="$APPWORLD_ROOT" "$APPWORLD_PY" \
        "$root/opsd/appworld/appworld_dataset.py" \
        --local_dir "$tmp" --split grpo_recipe --prompt_variant "$variant") || rc=$?
    if [ "$rc" = 0 ] && [ "$include_dev" = 1 ]; then
        (cd "$APPWORLD_ROOT" && APPWORLD_ROOT="$APPWORLD_ROOT" "$APPWORLD_PY" \
            "$root/opsd/appworld/appworld_dataset.py" \
            --local_dir "$tmp" --split train_dev --prompt_variant "$variant") || rc=$?
    fi
    if [ "$rc" = 0 ]; then
        for f in "${files[@]}"; do
            [ -s "$tmp/$f" ] || { opsd_warn "appworld_dataset.py did not produce $f"; rc=1; }
        done
    fi
    if [ "$rc" = 0 ]; then
        for f in "$tmp"/*.parquet; do mv -f "$f" "$dir/"; done
        printf '%s\n' "$want" >"$dir/.build_stamp"
    fi
    rm -rf "$tmp" "$lock"
    return "$rc"
}

aw_server_urls() {  # BASE COUNT
    local base="$1" count="$2" i out="["
    for ((i = 0; i < count; i++)); do
        [ "$i" -gt 0 ] && out+=","
        out+="\"http://127.0.0.1:$((base + i))\""
    done
    printf '%s]\n' "$out"
}

aw_start_server_pool() {  # BASE COUNT LOGDIR [LABEL]
    local base="$1" count="$2" logdir="$3" label="${4:-appworld-server}" i port
    mkdir -p "$logdir"
    for ((i = 0; i < count; i++)); do
        port=$((base + i))
        # The env recycles a server by SIGTERM-ing the `appworld serve environment --port <port>` process
        # (found via /proc); this loop then restarts it on the same port.
        opsd_spawn "$label-$port" bash -c '
            cd "$1" || exit 1
            while true; do
                "$2" serve environment --port "$3" --no-show-usage
                echo "=== appworld server on port $3 exited rc=$? -- restarting ==="
                sleep 2
            done' aw-supervisor "$APPWORLD_ROOT" "$APPWORLD_BIN" "$port" >"$logdir/appworld_serve_$port.log" 2>&1
    done
    return 0
}

aw_wait_server_pool() {  # BASE COUNT NEED TIMEOUT_S
    local base="$1" count="$2" need="$3" timeout="${4:-900}" t0 up i
    t0=$SECONDS
    while :; do
        up=0
        for ((i = 0; i < count; i++)); do
            (exec 3<>"/dev/tcp/127.0.0.1/$((base + i))") 2>/dev/null && up=$((up + 1))
        done
        [ "$up" -ge "$count" ] && break
        if [ $((SECONDS - t0)) -ge "$timeout" ]; then break; fi
        opsd_log "appworld pool: $up/$count bound after $((SECONDS - t0))s"
        sleep 10
    done
    opsd_log "appworld pool ready: $up/$count bound after $((SECONDS - t0))s"
    if [ "$up" -eq 0 ]; then
        opsd_warn "no AppWorld server came up (see the appworld_serve_*.log files)"
        return 1
    fi
    if [ "$up" -lt "$need" ]; then
        opsd_warn "only $up servers bound but $need rollouts run concurrently: rollouts will run in waves" \
            "(raise APPWORLD_READY_TIMEOUT or lower the load)"
    fi
    return 0
}

aw_check_port_owners() {  # BASE COUNT [LABEL]
    local dir
    dir="$RUN_DIR/pids"
    "$(opsd_python)" - "$1" "$2" "$dir" "${3:-appworld-server}" <<'PY'
import glob, os, sys

base, count, piddir, label = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
ports = set(range(base, base + count))
listen = {}
for table in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        rows = open(table).read().splitlines()[1:]
    except OSError:
        continue
    for row in rows:
        f = row.split()
        if len(f) < 10 or f[3] != "0A":  # 0A = LISTEN
            continue
        port = int(f[1].rsplit(":", 1)[1], 16)
        if port in ports:
            listen.setdefault(port, set()).add(f[9])
groups = set()
for pf in glob.glob(os.path.join(piddir, f"{label}-*.pid")):
    try:  # "PID START PGID ..." (bootstrap.sh); opsd_spawn starts each supervisor in its own group
        f = open(pf).read().split()
        groups.add(int(f[2]) if len(f) > 2 and f[2].isdigit() else int(f[0]))
    except (OSError, ValueError, IndexError):
        pass
ours = set()
for d in os.listdir("/proc"):
    if not d.isdigit():
        continue
    try:
        if os.getpgid(int(d)) not in groups:
            continue
        fds = os.listdir(f"/proc/{d}/fd")
    except OSError:
        continue
    for fd in fds:
        try:
            link = os.readlink(f"/proc/{d}/fd/{fd}")
        except OSError:
            continue
        if link.startswith("socket:["):
            ours.add(link[8:-1])
foreign = sorted(p for p, inodes in listen.items() if not inodes <= ours)
if foreign:
    print(f"[opsd] ERROR: {len(foreign)} port(s) in {base}-{base + count - 1} are served by a process this run "
          f"did not start (stale or foreign AppWorld server?): {foreign[:20]}. Stop it or pick another "
          f"APPWORLD_BASE_PORT; never reuse a foreign server.", file=sys.stderr)
    sys.exit(1)
print(f"[opsd] {len(listen)} AppWorld port(s) in {base}-{base + count - 1}, all served by this run's servers",
      file=sys.stderr)
PY
}

# ------------------------------------------------------------------------- offline generation helpers
aw_student() {  # KEY -> MODEL_PATH SERVED_MODEL_NAME VLLM_TP PRESERVE_THINKING (as the original corpora used)
    case "$1" in
        4bi) MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507; SERVED_MODEL_NAME=Qwen3-4B-Instruct-2507; VLLM_TP=1; PRESERVE_THINKING=false ;;
        # The Thinking model's reasoning is carried across turns (no --reasoning-parser on the server, so
        # the <think> text stays in the message content the driver preserves).
        4bt) MODEL_PATH=Qwen/Qwen3-4B-Thinking-2507; SERVED_MODEL_NAME=Qwen3-4B-Thinking-2507; VLLM_TP=1; PRESERVE_THINKING=true ;;
        # 57 GiB of weights: TP2 (4 replicas on 8 GPUs) leaves room for the KV cache; TP divides the 4 KV heads.
        30bi) MODEL_PATH=Qwen/Qwen3-30B-A3B-Instruct-2507; SERVED_MODEL_NAME=Qwen3-30B-A3B-Instruct-2507; VLLM_TP=2; PRESERVE_THINKING=false ;;
        *) opsd_warn "unknown student '$1' (want 4bi | 4bt | 30bi)"; return 1 ;;
    esac
    return 0
}

aw_sampling() {  # PROFILE -> TEMP TOPP TOPK
    case "$1" in
        onpolicy) TEMP=1.0; TOPP=1.0; TOPK=-1 ;;  # the distribution every AppWorld training run rolls out from
        official) TEMP=0.7; TOPP=0.8; TOPK=20 ;;  # Qwen3-*-Instruct-2507's own generation config
        thinking) TEMP=0.6; TOPP=0.95; TOPK=20 ;; # Qwen3-4B-Thinking-2507's own generation config
        *) opsd_warn "unknown sampling profile '$1' (want onpolicy | official | thinking)"; return 1 ;;
    esac
    return 0
}

aw_visible_gpus() {  # comma list of the GPUs to use: $GPUS, else $CUDA_VISIBLE_DEVICES, else all
    if [ -n "${GPUS:-}" ]; then
        printf '%s\n' "$GPUS"
    elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        printf '%s\n' "$CUDA_VISIBLE_DEVICES"
    else
        nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -
    fi
}

aw_serve_vllm() {  # OUTDIR
    local outdir="$1" gpus i g devs port pids=() urls=() vllm
    vllm="$(dirname "$(opsd_python)")/vllm"
    [ -x "$vllm" ] || vllm=vllm
    IFS=, read -r -a gpus <<<"$(aw_visible_gpus)"
    [ "${#gpus[@]}" -gt 0 ] || { opsd_warn "no GPUs visible"; return 1; }
    [ $((${#gpus[@]} % VLLM_TP)) = 0 ] || { opsd_warn "${#gpus[@]} GPUs is not a multiple of TP=$VLLM_TP"; return 1; }
    mkdir -p "$outdir"
    for ((i = 0; i < ${#gpus[@]}; i += VLLM_TP)); do
        devs="$(IFS=,; echo "${gpus[*]:$i:$VLLM_TP}")"
        port=$((VLLM_BASE_PORT + i / VLLM_TP))
        CUDA_VISIBLE_DEVICES="$devs" PYTHONUNBUFFERED=1 opsd_spawn "vllm-$port" "$vllm" serve "$MODEL_PATH" \
            --served-model-name "$SERVED_MODEL_NAME" --port "$port" --tensor-parallel-size "$VLLM_TP" \
            --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEM_UTIL" \
            --enable-prefix-caching --trust-remote-code ${VLLM_EXTRA_ARGS:-} >"$outdir/vllm_$port.log" 2>&1
        pids+=("$OPSD_LAST_PID")
        urls+=("http://127.0.0.1:$port")
    done
    opsd_log "serving $MODEL_PATH: ${#urls[@]} replica(s) x TP$VLLM_TP on GPUs $(IFS=,; echo "${gpus[*]}")"
    for ((i = 0; i < ${#urls[@]}; i++)); do
        opsd_wait_http "${urls[$i]}/health" "${VLLM_READY_TIMEOUT:-1800}" "${pids[$i]}" ||
            { opsd_warn "${urls[$i]} never became healthy (see $outdir/vllm_*.log)"; return 1; }
    done
    VLLM_URLS="$(IFS=,; echo "${urls[*]}")"
    return 0
}
