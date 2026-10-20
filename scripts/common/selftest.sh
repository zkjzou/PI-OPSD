#!/usr/bin/env bash
# =====================================================================================================
# selftest.sh -- CPU-only self-test of bootstrap.sh, preflight.sh and ray_cluster.sh.
#
#     bash scripts/common/selftest.sh
#
# Starts no GPU work, no Ray and no servers: nvidia-smi and ss are replaced by stubs on PATH, ray_cluster.sh
# runs with OPSD_DRY_RUN=1, and the only processes it starts and stops are its own `sleep`s. Exit status is
# the number of failed checks (0 = all passed).
# =====================================================================================================
set -o pipefail
export PYTHONDONTWRITEBYTECODE=1  # never leave bytecode in the source tree

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
T="$(mktemp -d "${TMPDIR:-/tmp}/opsd_selftest.XXXXXX")"
LEFTOVER=()
cleanup() {
    local p
    for p in "${LEFTOVER[@]+"${LEFTOVER[@]}"}"; do kill "$p" 2>/dev/null; done
    rm -rf "$T"
}
trap cleanup EXIT

PASS=0
FAIL=0
ok() { PASS=$((PASS + 1)); printf 'ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL %s\n' "$1"; }
check() {  # check DESCRIPTION CMD...
    local d="$1"
    shift
    if "$@" >"$T/last.out" 2>&1; then ok "$d"; else bad "$d"; sed 's/^/     | /' "$T/last.out" | tail -n 15; fi
}
check_fails() {  # the command must fail
    local d="$1"
    shift
    if "$@" >"$T/last.out" 2>&1; then bad "$d (unexpectedly succeeded)"; sed 's/^/     | /' "$T/last.out" | tail -n 15; else ok "$d"; fi
}

# A clean subshell environment: no inherited Ray/OPSD/W&B state.
clean_env() {
    env -i PATH="$T/bin:$PATH" HOME="${HOME:-/tmp}" TMPDIR="$T" OPSD_PYTHON="${OPSD_PYTHON:-}" PYTHONDONTWRITEBYTECODE=1 "$@"
}
mkdir -p "$T/bin"

# ----------------------------------------------------------------------------------------- 1. syntax
for f in bootstrap.sh preflight.sh ray_cluster.sh selftest.sh; do
    check "bash -n $f" bash -n "$HERE/$f"
done
if grep -v '^\s*#' "$HERE/bootstrap.sh" "$HERE/preflight.sh" "$HERE/ray_cluster.sh" \
    | grep -E 'ray stop|pkill|killall' >"$T/grep.out"; then
    bad "no 'ray stop' / pkill / killall in code lines"
    cat "$T/grep.out"
else
    ok "no 'ray stop' / pkill / killall in code lines"
fi

# ------------------------------------------------------------------------------- 2. bootstrap: env
cat >"$T/env_probe.sh" <<'EOF'
set -euo pipefail
source "$1/bootstrap.sh"
printf 'RAY_ADDRESS=%s\nRAY_TMPDIR=%s\nNCCL_NET=%s\nVLLM_HOST_IP=%s\n' "${RAY_ADDRESS:-}" "${RAY_TMPDIR:-}" "${NCCL_NET:-}" "${VLLM_HOST_IP:-}"
printf 'SYMM=%s\nCUMEM=%s\nHYDRA=%s\nHASH=%s\nLOGGER=%s\nHOOK=%s\nUVRUN=%s\n' "$VLLM_ALLREDUCE_USE_SYMM_MEM" \
    "$NCCL_CUMEM_ENABLE" "$HYDRA_FULL_ERROR" "$PYTHONHASHSEED" "$LOGGER" "${RAY_RUNTIME_ENV_HOOK:-}" \
    "${RAY_ENABLE_UV_RUN_RUNTIME_ENV:-}"
printf 'NOFILE=%s\nHARD=%s\nLD=%s\nMODE=%s\nROOT_OK=%s\n' "$(ulimit -Sn)" "$(ulimit -Hn)" "${LD_LIBRARY_PATH:-}" "$(opsd_ray_mode)" \
    "$(r="$(opsd_repo_root)" && [ -f "$r/pyproject.toml" ] && [ -f "$r/opsd/__init__.py" ] \
        && [ -f "$r/opsd/skyrl_ext/__init__.py" ] && [ -f "$r/scripts/common/bootstrap.sh" ] && echo yes)"
EOF
val() { grep "^$1=" "$2" | head -n1 | cut -d= -f2-; }

clean_env RUN_DIR="$T/runA" bash "$T/env_probe.sh" "$HERE" >"$T/a.env" 2>"$T/a.err"
check "bootstrap sources cleanly under set -euo pipefail" test -s "$T/a.env"
check "single-node: RAY_ADDRESS=local" test "$(val RAY_ADDRESS "$T/a.env")" = local
RT="$(val RAY_TMPDIR "$T/a.env")"
check "single-node: per-run RAY_TMPDIR under /tmp, short" bash -c "[[ '$RT' == /tmp/opsd_ray_$(id -u)_* ]] && [ \${#RT} -le 40 ]" _
check "sourcing creates nothing on disk (Ray creates RAY_TMPDIR itself)" test ! -e "$RT"
check "single-node: NCCL_NET=Socket, VLLM_HOST_IP=127.0.0.1" \
    test "$(val NCCL_NET "$T/a.env")/$(val VLLM_HOST_IP "$T/a.env")" = "Socket/127.0.0.1"
check "exports SYMM_MEM=0 CUMEM=0 HYDRA=1 HASHSEED=0" \
    test "$(val SYMM "$T/a.env")$(val CUMEM "$T/a.env")$(val HYDRA "$T/a.env")$(val HASH "$T/a.env")" = "0010"
check "LOGGER=console without WANDB_API_KEY" test "$(val LOGGER "$T/a.env")" = console
check "uv runtime-env hook set, Ray's uv re-launch untouched (no \`uv run\` ancestor)" \
    test "$(val HOOK "$T/a.env")|$(val UVRUN "$T/a.env")" = "ray._private.runtime_env.uv_runtime_env_hook.hook|"
# Under `uv run` (an ancestor whose argv is `uv run ...`, Ray's own test), Ray would start every worker with `uv run`
# in an uploaded copy of the working directory: the bootstrap turns that off and leaves the uv hook unset.
mkdir -p "$T/uvrun"
printf 'bash "%s" "%s"\n' "$T/env_probe.sh" "$HERE" >"$T/uvrun/run"  # a child, not exec: `uv run` stays its ancestor
(cd "$T/uvrun" && clean_env RUN_DIR="$T/runU" bash -c 'exec -a uv bash run') >"$T/u.env" 2>"$T/u.err"
check "under \`uv run\`: RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 and no uv hook (workers run the driver's python)" \
    test "$(val HOOK "$T/u.env")|$(val UVRUN "$T/u.env")" = "|0"
check "repo root (pyproject.toml + opsd/ package + scripts/) found" test "$(val ROOT_OK "$T/a.env")" = yes
NOFILE="$(val NOFILE "$T/a.env")"
HARD="$(val HARD "$T/a.env")"
check "open-file soft limit >= min(65536, hard)" \
    bash -c "[ '$NOFILE' = unlimited ] || [ '$NOFILE' -ge 65536 ] || [ '$NOFILE' = '$HARD' ]"

clean_env RUN_DIR="$T/runA" bash "$T/env_probe.sh" "$HERE" >"$T/a2.env" 2>/dev/null
clean_env RUN_DIR="$T/runB" bash "$T/env_probe.sh" "$HERE" >"$T/b.env" 2>/dev/null
check "same RUN_DIR -> same RAY_TMPDIR" test "$(val RAY_TMPDIR "$T/a2.env")" = "$RT"
check "different RUN_DIR -> different RAY_TMPDIR" test "$(val RAY_TMPDIR "$T/b.env")" != "$RT"

clean_env RUN_DIR="$T/runC" PYTHONHASHSEED=7 LOGGER=tensorboard WANDB_API_KEY=x NCCL_NET=IB \
    OPSD_NVML_LIB_DIR=/opt/nvml LD_LIBRARY_PATH=/usr/lib DRY_RUN=1 bash "$T/env_probe.sh" "$HERE" >"$T/c.env" 2>/dev/null
check "user values win (PYTHONHASHSEED, LOGGER, NCCL_NET)" \
    test "$(val HASH "$T/c.env")/$(val LOGGER "$T/c.env")/$(val NCCL_NET "$T/c.env")" = "7/tensorboard/IB"
clean_env RUN_DIR="$T/runC" PYTHONHASHSEED=7 NCCL_CUMEM_ENABLE=0 DRY_RUN=1 bash "$T/env_probe.sh" "$HERE" >/dev/null 2>"$T/c.err"
check "a non-contract PYTHONHASHSEED is kept with one warning; matching values are silent" \
    bash -c "[ \$(grep -c 'PYTHONHASHSEED=7 differs' '$T/c.err') = 1 ] && ! grep -q NCCL_CUMEM '$T/c.err'"
check "OPSD_NVML_LIB_DIR prepended to LD_LIBRARY_PATH" test "$(val LD "$T/c.env")" = "/opt/nvml:/usr/lib"
check "DRY_RUN=1 creates no RAY_TMPDIR" test ! -e "$(val RAY_TMPDIR "$T/c.env")"
clean_env RUN_DIR="$T/runD" WANDB_API_KEY=x DRY_RUN=1 bash "$T/env_probe.sh" "$HERE" >"$T/d.env" 2>/dev/null
check "LOGGER=wandb when WANDB_API_KEY is set" test "$(val LOGGER "$T/d.env")" = wandb

clean_env RUN_DIR="$T/runE" RAY_ADDRESS=192.0.2.10:6379 bash "$T/env_probe.sh" "$HERE" >"$T/e.env" 2>/dev/null
check "multi-node: RAY_ADDRESS kept, no RAY_TMPDIR / local networking" \
    test "$(val RAY_ADDRESS "$T/e.env")|$(val RAY_TMPDIR "$T/e.env")|$(val NCCL_NET "$T/e.env")|$(val MODE "$T/e.env")" \
    = "192.0.2.10:6379|||external"
clean_env RUN_DIR="$T/runF" RAY_TMPDIR=/tmp/mine DRY_RUN=1 bash "$T/env_probe.sh" "$HERE" >"$T/f.env" 2>/dev/null
check "user RAY_TMPDIR is kept" test "$(val RAY_TMPDIR "$T/f.env")" = /tmp/mine

# ---------------------------------------------------------------------- 3. bootstrap: own processes
cat >"$T/procs.sh" <<'EOF'
set -euo pipefail
source "$1/bootstrap.sh"
OPSD_CLEANUP_GRACE_S=5
case "$2" in
  cleanup)
    opsd_spawn sleeper sleep 300
    tracked=$OPSD_LAST_PID
    opsd_spawn forker bash -c 'sleep 301 & wait'
    forker=$OPSD_LAST_PID
    sleep 300 &
    untracked=$!
    sleep 0.5
    child=$(ps -o pid= --ppid "$forker" | tr -d ' ' | head -n1)
    n_files=$(ls "$RUN_DIR/pids" | wc -l)
    opsd_cleanup_own
    sleep 0.3
    alive() { kill -0 "$1" 2>/dev/null && echo alive || echo dead; }
    echo "files=$n_files tracked=$(alive "$tracked") forker=$(alive "$forker") child=$(alive "${child:-0}") untracked=$(alive "$untracked") left=$(ls "$RUN_DIR/pids" | wc -l)"
    kill "$untracked"
    ;;
  reuse)
    sleep 300 &
    victim=$!
    mkdir -p "$RUN_DIR/pids"
    echo "$victim 1 - - - - bogus" >"$RUN_DIR/pids/bogus.$victim.pid"   # wrong start time = a reused PID
    opsd_cleanup_own
    kill -0 "$victim" 2>/dev/null && echo "victim=alive" || echo "victim=dead"
    kill "$victim"
    ;;
  pool)  # the spawned command backgrounds its servers and exits at once (a server-pool launcher)
    opsd_spawn pool bash -c 'sleep 307 & sleep 308 & exit 0'
    leader=$OPSD_LAST_PID
    sleep 0.5
    kids=$(ps -e -o pid=,pgid=,args= | awk -v g="$leader" '$2 == g && $3 == "sleep" { print $1 }' | tr '\n' ' ')
    n_files=$(ls "$RUN_DIR/pids" | wc -l)
    opsd_cleanup_own
    sleep 0.3
    alive=0
    for k in $kids; do kill -0 "$k" 2>/dev/null && alive=$((alive + 1)); done
    echo "leader=$(kill -0 "$leader" 2>/dev/null && echo alive || echo gone) kids=$(echo $kids | wc -w) files=$n_files alive_after=$alive left=$(ls "$RUN_DIR/pids" | wc -l)"
    for k in $kids; do kill "$k" 2>/dev/null || true; done
    ;;
  session)  # setsid sessions: children that scrub their environment (env -i, no tag) or leave the group
    sl="$(command -v sleep)"
    opsd_spawn envi bash -c "env -i $sl 319 & exit 0"          # leader exits; untagged child
    l1=$OPSD_LAST_PID
    opsd_spawn pgrp bash -c "set -m; $sl 320 & exit 0"          # leader exits; child in its own group
    l2=$OPSD_LAST_PID
    opsd_spawn live bash -c "set -m; $sl 321 & wait"            # leader alive; child in its own group
    l3=$OPSD_LAST_PID
    sleep 0.6
    kids=$(ps -e -o pid=,sid=,args= | awk -v a="$l1" -v b="$l2" -v c="$l3" \
        '($2 == a || $2 == b || $2 == c) && $3 ~ /sleep$/ { print $1 }' | tr '\n' ' ')
    opsd_cleanup_own
    sleep 0.3
    alive=0
    for k in $kids; do kill -0 "$k" 2>/dev/null && alive=$((alive + 1)); done
    echo "kids=$(echo $kids | wc -w) alive_after=$alive leader3=$(kill -0 "$l3" 2>/dev/null && echo alive || echo dead) left=$(ls "$RUN_DIR/pids" | wc -l)"
    for k in $kids; do kill "$k" 2>/dev/null || true; done
    ;;
  foreign)  # pid files that point at process groups we did not spawn: never signalled
    mkdir -p "$RUN_DIR/pids"
    setsid sleep 300 &   # a live group whose leader's start time does not match the record
    g1=$!
    echo "$g1 1 $g1 0 sometag setsid g1" >"$RUN_DIR/pids/g1.$g1.pid"
    setsid bash -c 'sleep 309 & exit 0' &   # leader exits; its child lives on without our tag
    g2=$!
    wait "$g2" 2>/dev/null || true
    sleep 0.3
    c2=$(ps -e -o pid=,pgid= | awk -v g="$g2" '$2 == g { print $1 }' | head -n1)
    echo "$g2 - $g2 0 nottheirs jobctl g2" >"$RUN_DIR/pids/g2.$g2.pid"
    setsid bash -c 'sleep 310 & exit 0' &   # a session that predates the recorded spawn time: not ours
    g3=$!
    wait "$g3" 2>/dev/null || true
    sleep 0.3
    c3=$(ps -e -o pid=,sid= | awk -v s="$g3" '$2 == s { print $1 }' | head -n1)
    echo "$g3 - $g3 999999999999 x setsid g3" >"$RUN_DIR/pids/g3.$g3.pid"
    opsd_cleanup_own
    a1=$(kill -0 "$g1" 2>/dev/null && echo alive || echo dead)
    a2=$(kill -0 "${c2:-0}" 2>/dev/null && echo alive || echo dead)
    a3=$(kill -0 "${c3:-0}" 2>/dev/null && echo alive || echo dead)
    echo "g1=$a1 g2child=$a2 g3child=$a3 left=$(ls "$RUN_DIR/pids" | wc -l)"
    kill "$g1" "${c2:-0}" "${c3:-0}" 2>/dev/null || true
    ;;
  trap)
    opsd_install_cleanup_trap
    opsd_spawn sleeper sleep 300
    echo "$OPSD_LAST_PID" >"$RUN_DIR/sleeper.pid"
    exit 3
    ;;
  run)
    opsd_run bash -c 'exit 7' && rc=0 || rc=$?
    echo "rc=$rc left=$(ls "$RUN_DIR/pids" 2>/dev/null | wc -l)"
    ;;
esac
EOF
clean_env RUN_DIR="$T/runP" bash "$T/procs.sh" "$HERE" cleanup >"$T/p.out" 2>"$T/p.err"
check "cleanup kills tracked process + process group child, spares untracked, removes pid files" \
    grep -qx "files=2 tracked=dead forker=dead child=dead untracked=alive left=0" "$T/p.out"
[ -s "$T/p.out" ] || cat "$T/p.err"
clean_env RUN_DIR="$T/runR" bash "$T/procs.sh" "$HERE" reuse >"$T/r.out" 2>&1
check "a PID whose start time changed (reused) is never signalled" grep -qx "victim=alive" "$T/r.out"
clean_env RUN_DIR="$T/runQ" bash "$T/procs.sh" "$HERE" pool >"$T/q.out" 2>&1
check "spawned command that backgrounds children and exits: children cleaned up" \
    grep -qx "leader=gone kids=2 files=1 alive_after=0 left=0" "$T/q.out"
clean_env RUN_DIR="$T/runG" bash "$T/procs.sh" "$HERE" foreign >"$T/g.out" 2>&1
check "groups we did not spawn (reused leader / untagged members / older session) are never signalled" \
    grep -qx "g1=alive g2child=alive g3child=alive left=0" "$T/g.out"
clean_env RUN_DIR="$T/runS" bash "$T/procs.sh" "$HERE" session >"$T/s.out" 2>&1
check "setsid session: env -i children, own-group children and a live leader's other groups cleaned up" \
    grep -qx "kids=3 alive_after=0 leader3=dead left=0" "$T/s.out"
clean_env RUN_DIR="$T/runT" bash "$T/procs.sh" "$HERE" trap >/dev/null 2>&1
rc=$?
check "cleanup trap keeps the exit status (3)" test "$rc" = 3
SP="$(cat "$T/runT/sleeper.pid" 2>/dev/null)"
sleep 0.3
check "cleanup trap stopped the spawned process on exit" bash -c "! kill -0 '${SP:-0}' 2>/dev/null"
[ -n "$SP" ] && LEFTOVER+=("$SP")
clean_env RUN_DIR="$T/runU" bash "$T/procs.sh" "$HERE" run >"$T/u.out" 2>&1
check "opsd_run returns the command's status and untracks it" grep -qx "rc=7 left=0" "$T/u.out"

# ------------------------------------------------------------------------------- 4. preflight (stubs)
cat >"$T/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  *--query-gpu=*) cat "$STUB_GPUS" ;;
  *--query-compute-apps=*) cat "$STUB_APPS" ;;
  *) exit 1 ;;
esac
EOF
cat >"$T/bin/ss" <<'EOF'
#!/usr/bin/env bash
cat "$STUB_SS"
EOF
chmod +x "$T/bin/nvidia-smi" "$T/bin/ss"
printf '0, GPU-aaaa-0000, 3\n1, GPU-bbbb-1111, 3\n2, GPU-cccc-2222, 5000\n3, GPU-dddd-3333, 0\n' >"$T/gpus"
printf 'GPU-bbbb-1111, 4242, 30000\n' >"$T/apps"
printf 'LISTEN 0 4096 0.0.0.0:12503 0.0.0.0:*\nLISTEN 0 128 [::]:18000 [::]:*\nLISTEN 0 5 127.0.0.53%%lo:53 0.0.0.0:*\n' >"$T/ss"
: >"$T/empty"
pf() {  # pf CUDA_VISIBLE_DEVICES CMD...  (an empty first argument leaves CUDA_VISIBLE_DEVICES unset)
    local vis="$1"
    shift
    clean_env STUB_GPUS="$T/gpus" STUB_APPS="$T/apps" STUB_SS="$T/ss" ${vis:+CUDA_VISIBLE_DEVICES=$vis} \
        bash -c 'source "$0/preflight.sh"; "$@"' "$HERE" "$@"
}
check "GPUs 0,3 free" pf 0,3 opsd_preflight_gpus 2
check_fails "GPU 1 has a compute process" pf 0,1 opsd_preflight_gpus
check_fails "GPU 2 has 5000 MiB used" pf 2 opsd_preflight_gpus
check "GPU selected by UUID" pf GPU-dddd-3333 opsd_preflight_gpus 1
check_fails "need more GPUs than visible" pf 0,3 opsd_preflight_gpus 4
check_fails "unknown GPU index" pf 7 opsd_preflight_gpus
check_fails "all GPUs (CUDA_VISIBLE_DEVICES unset) include busy ones" pf "" opsd_preflight_gpus
check_fails "ports 12500-12507 busy (12503)" pf 0 opsd_preflight_ports 12500 8 alfworld
check "ports 12600-12607 free" pf 0 opsd_preflight_ports 12600 8 alfworld
check_fails "port 18000 busy ([::] listener)" pf 0 opsd_preflight_ports 18000 1 http
printf 'sl local_address rem_address st\n0: 00000000:30D4 00000000:0000 0A\n1: 00000000:30D5 0100007F:1F90 01\n' >"$T/tcp"
check_fails "/proc/net/tcp parser: 12500 (0x30D4) listening" \
    env OPSD_PREFLIGHT_PORT_SOURCE=proc OPSD_PREFLIGHT_PROC_NET="$T/tcp" bash -c 'source "$0/preflight.sh"; opsd_preflight_ports 12500 1' "$HERE"
check "/proc/net/tcp parser: established (not LISTEN) 12501 is free" \
    env OPSD_PREFLIGHT_PORT_SOURCE=proc OPSD_PREFLIGHT_PROC_NET="$T/tcp" bash -c 'source "$0/preflight.sh"; opsd_preflight_ports 12501 1' "$HERE"
check "disk: 0 GB needed" pf 0 opsd_preflight_disk "$T/not/yet/created" 0
check_fails "disk: 10^9 GB needed" pf 0 opsd_preflight_disk "$T" 1000000000
check "files: existing" pf 0 opsd_preflight_files "$T/gpus"
check_fails "files: missing" pf 0 opsd_preflight_files "$T/nope.json"
check "preflight_all with free GPUs/ports (smoke-overlay port bases)" \
    pf 0,3 env NUM_GPUS_PER_NODE=2 OPSD_BENCH=alfworld ALFWORLD_BASE_PORT=9500 NUM_ALFWORLD_SERVERS=8 \
    HTTP_ENDPOINT_PORT=8600 RUN_ROOT="$T" OPSD_PREFLIGHT_MIN_FREE_GB=0 OPSD_HINTS_PATH="$T/gpus" \
    bash -c 'source "$0/preflight.sh"; opsd_preflight_all' "$HERE"
check_fails "preflight_all: busy server port range" \
    pf 0,3 env NUM_GPUS_PER_NODE=2 OPSD_BENCH=alfworld ALFWORLD_BASE_PORT=12500 NUM_ALFWORLD_SERVERS=8 \
    RUN_ROOT="$T" OPSD_PREFLIGHT_MIN_FREE_GB=0 bash -c 'source "$0/preflight.sh"; opsd_preflight_all' "$HERE"
check "preflight_all: other bench's range is not checked" \
    pf 0,3 env NUM_GPUS_PER_NODE=2 OPSD_BENCH=appworld ALFWORLD_BASE_PORT=12500 NUM_ALFWORLD_SERVERS=8 \
    RUN_ROOT="$T" OPSD_PREFLIGHT_MIN_FREE_GB=0 bash -c 'source "$0/preflight.sh"; opsd_preflight_all' "$HERE"
check_fails "preflight_all: missing hint file" \
    pf 0 env OPSD_PREFLIGHT_SKIP_GPUS=1 RUN_ROOT="$T" OPSD_PREFLIGHT_MIN_FREE_GB=0 OPSD_HINTS_PATH="$T/nope.json" \
    bash -c 'source "$0/preflight.sh"; opsd_preflight_all' "$HERE"
check "preflight.sh runs as a script with flags" \
    env PATH="$T/bin:$PATH" STUB_GPUS="$T/gpus" STUB_APPS="$T/apps" STUB_SS="$T/ss" CUDA_VISIBLE_DEVICES=3 \
    bash "$HERE/preflight.sh" --gpus 1 --ports 12600:4:x --disk "$T:0" --files "$T/gpus"
check "OPSD_PREFLIGHT=0 skips (with a warning)" pf 7 env OPSD_PREFLIGHT=0 bash -c 'source "$0/preflight.sh"; opsd_preflight_all' "$HERE"

PY="${OPSD_PYTHON:-}"
if [ -z "$PY" ]; then
    ROOT="$(clean_env bash -c 'source "$0/bootstrap.sh" >/dev/null 2>&1; opsd_repo_root' "$HERE" 2>/dev/null)"
    PY="$ROOT/.venv/bin/python"
    [ -x "$PY" ] || PY="$(command -v python3)"
fi
if "$PY" -c "import yaml" >/dev/null 2>&1; then
    printf '{"t1": "go north", "t2": {"task": "open box", "meta": {"x": 1}}}' >"$T/hints.json"
    SHA="$(printf '{"t1": "go north", "t2": "open box"}' | "$PY" -c 'import hashlib,json,sys; print(hashlib.sha256(json.dumps(json.loads(sys.stdin.read()), sort_keys=True, ensure_ascii=False).encode()).hexdigest())')"
    check "hint content sha256 matches" env OPSD_PYTHON="$PY" OPSD_HINTS_PATH="$T/hints.json" OPSD_HINTS_CONTENT_SHA256="$SHA" \
        bash -c 'source "$0/preflight.sh"; opsd_preflight_hint_sha' "$HERE"
    check_fails "hint content sha256 mismatch is fatal" env OPSD_PYTHON="$PY" OPSD_HINTS_PATH="$T/hints.json" \
        OPSD_HINTS_CONTENT_SHA256="$(printf '0%.0s' {1..64})" bash -c 'source "$0/preflight.sh"; opsd_preflight_hint_sha' "$HERE"
    # opsd_resolve_recipe on the resolver's synthetic test recipes (bench "toy")
    FIXR="$(cd "$HERE" && cd "$(clean_env bash -c 'source "$0/bootstrap.sh" >/dev/null 2>&1; opsd_repo_root' "$HERE")" && pwd)/opsd/recipes/tests/fixtures/recipes"
    cat >"$T/resolve.sh" <<'EOS'
set -euo pipefail
source "$1/bootstrap.sh"
before="${RAY_TMPDIR:-}"
opsd_resolve_recipe toy/main/4bi/glm_a MAX_TURNS=2 --overlay smoke
printf 'MAX_TURNS=%s\nNUM_GPUS=%s\nRUN_DIR=%s\nRESOLVED=%s\nMOVED=%s\nHINT=%s\n' "$MAX_TURNS" "$NUM_GPUS_PER_NODE" "$RUN_DIR" \
    "$OPSD_RESOLVED_ENV" "$([ "$before" != "$RAY_TMPDIR" ] && echo yes)" "$OPSD_HINTS_PATH"
opsd_dry_run_print toy_entry.main trainer.epochs=1 "a b"
if [ "${DRY_RUN:-0}" != 1 ]; then opsd_write_provenance "$OPSD_RESOLVED_ENV"; fi
EOS
    clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" OPSD_DATA_ROOT="$T/data" RUN_ROOT="$T/runs" \
        bash "$T/resolve.sh" "$HERE" >"$T/res.out" 2>"$T/res.err"
    check "opsd_resolve_recipe: KEY=VAL and overlay applied" test "$(val MAX_TURNS "$T/res.out")/$(val NUM_GPUS "$T/res.out")" = "2/4"
    check "opsd_resolve_recipe: resolved.env written into RUN_DIR (<arm>__<overlay>)" \
        test -f "$T/runs/toy__main__4bi__glm_a__smoke/resolved.env"
    check "opsd_resolve_recipe: RAY_TMPDIR re-derived from RUN_DIR" test "$(val MOVED "$T/res.out")" = yes
    check "opsd_resolve_recipe: hint path under OPSD_DATA_ROOT" test "$(val HINT "$T/res.out")" = "$T/data/toy/hints/glm/hints_actions.json"
    check "opsd_dry_run_print shows module, quoted argv and the resolved env" \
        bash -c "grep -q -- '-m toy_entry.main' '$T/res.out' && grep -qF 'a\\ b' '$T/res.out' && grep -q '^OPSD_ARM_ID=toy/main/4bi/glm_a' '$T/res.out'"
    check "opsd_write_provenance wrote provenance.json" test -s "$T/runs/toy__main__4bi__glm_a__smoke/provenance.json"
    check "provenance.json records the runtime env (PYTHONHASHSEED=0, local Ray)" \
        grep -q '"PYTHONHASHSEED": "0"' "$T/runs/toy__main__4bi__glm_a__smoke/provenance.json"
    mkdir -p "$T/cwd"
    (cd "$T/cwd" && clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" OPSD_DATA_ROOT=./data RUN_ROOT=./myruns \
        bash "$T/resolve.sh" "$HERE" >"$T/rel.out" 2>"$T/rel.err")
    check "relative RUN_ROOT / OPSD_DATA_ROOT resolve against the caller's directory" \
        bash -c "[ '$(val RUN_DIR "$T/rel.out")' = '$T/cwd/myruns/toy__main__4bi__glm_a__smoke' ] && [ '$(val HINT "$T/rel.out")' = '$T/cwd/data/toy/hints/glm/hints_actions.json' ]"
    mkdir -p "$T/shadow"
    echo 'raise SystemExit("the yaml.py in the working directory was imported")' >"$T/shadow/yaml.py"
    (cd "$T/shadow" && clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" OPSD_DATA_ROOT="$T/data" RUN_ROOT="$T/shadowruns" \
        bash "$T/resolve.sh" "$HERE" >"$T/shadow.out" 2>&1)
    check "a yaml.py in the caller's directory does not shadow the resolver's import" \
        bash -c "grep -q '^MAX_TURNS=2' '$T/shadow.out' && ! grep -q 'was imported' '$T/shadow.out'"
    _repo="$(clean_env bash -c 'source "$0/bootstrap.sh" >/dev/null 2>&1; opsd_repo_root' "$HERE")"
    (cd "$_repo" && clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" OPSD_DATA_ROOT="$T/data" RUN_ROOT="$T/rootruns" \
        DRY_RUN=1 bash "$T/resolve.sh" "$HERE" >"$T/fromroot.out" 2>&1)
    check "resolving from the repo root (the working directory holds the opsd package) works" \
        bash -c "grep -q '^MAX_TURNS=2' '$T/fromroot.out' && [ ! -e '$T/rootruns' ]"
    check "preflight hint hash: a yaml.py in the caller's directory is not imported either" \
        bash -c "cd '$T/shadow' && env OPSD_PYTHON='$PY' OPSD_HINTS_PATH='$T/hints.json' OPSD_HINTS_CONTENT_SHA256='$SHA' \
            bash -c 'source \"\$0/preflight.sh\"; opsd_preflight_hint_sha' '$HERE'"
    clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" OPSD_DATA_ROOT="$T/data" RUN_ROOT="$T/dry" DRY_RUN=1 \
        bash "$T/resolve.sh" "$HERE" >"$T/dry.out" 2>&1
    check "DRY_RUN=1: resolves without creating RUN_DIR" bash -c "grep -q '^MAX_TURNS=2' '$T/dry.out' && [ ! -e '$T/dry' ]"
    check "DRY_RUN=1: no resolved-env temp file is left behind" bash -c "! ls '$T'/opsd_resolved.* >/dev/null 2>&1"
    cat >"$T/dry2.sh" <<'EOS'
set -euo pipefail
source "$1/bootstrap.sh"
case "$2" in
  twice) opsd_resolve_recipe toy/main/4bi/glm_a; opsd_dry_run_print m1 a; opsd_dry_run_print m2 b ;;
  early) opsd_resolve_recipe toy/main/4bi/glm_a; exit 3 ;;
  chained) trap 'echo "caller trap saw rc=$?"' EXIT; opsd_resolve_recipe toy/main/4bi/glm_a; exit 4 ;;
esac
EOS
    dry2() { clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" OPSD_DATA_ROOT="$T/data" RUN_ROOT="$T/dry2" DRY_RUN=1 bash "$T/dry2.sh" "$HERE" "$1"; }
    dry2 twice >"$T/twice.out" 2>&1
    check "DRY_RUN=1: every opsd_dry_run_print shows the resolved env (one block per trainer call)" \
        bash -c "[ \$(grep -c '^OPSD_ARM_ID=toy/main/4bi/glm_a' '$T/twice.out') = 2 ] && ! grep -q '(<none>)' '$T/twice.out'"
    check "DRY_RUN=1: ...and the temp file is gone afterwards" bash -c "! ls '$T'/opsd_resolved.* >/dev/null 2>&1"
    dry2 early >/dev/null 2>&1
    rc=$?
    check "DRY_RUN=1: exiting between resolve and print keeps the status (3) and leaves no temp file" \
        bash -c "[ $rc = 3 ] && ! ls '$T'/opsd_resolved.* >/dev/null 2>&1"
    dry2 chained >"$T/chained.out" 2>&1
    rc=$?
    check "DRY_RUN=1: a caller's EXIT trap still runs and sees the exit status; no temp file" \
        bash -c "[ $rc = 4 ] && grep -qx 'caller trap saw rc=4' '$T/chained.out' && ! ls '$T'/opsd_resolved.* >/dev/null 2>&1"

    # A run directory is never resumed with other settings (the smoke-then-full / changed KEY=VAL trap).
    cat >"$T/guard.sh" <<'EOS'
set -euo pipefail
source "$1/bootstrap.sh"
shift
opsd_resolve_recipe toy/main/4bi/glm_a "$@"
if [ "${DRY_RUN:-0}" != 1 ]; then opsd_write_provenance "$OPSD_RESOLVED_ENV"; fi
echo "RESOLVED=$OPSD_RESOLVED_ENV"
EOS
    guard() { clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" OPSD_DATA_ROOT="$T/data" RUN_ROOT="$1" bash "$T/guard.sh" "$HERE" "${@:2}"; }
    GD="$T/rg/toy__main__4bi__glm_a"
    guard "$T/rg" >"$T/g0.out" 2>&1
    mkdir -p "$GD/ckpts/global_step_2"
    echo 2 >"$GD/ckpts/latest_ckpt_global_step.txt"
    cp "$GD/resolved.env" "$T/g0.env"
    check "resume guard: the same resolution resumes" guard "$T/rg"
    guard "$T/rg" MAX_TURNS=3 >"$T/g1.out" 2>&1
    rc=$?
    check "resume guard: other settings + RESUME_MODE=latest + a checkpoint -> refused, naming the changed keys and the ways out" \
        bash -c "[ $rc != 0 ] && grep -q 'DIFFERENT settings' '$T/g1.out' && grep -q 'changed: MAX_TURNS)' '$T/g1.out' \
            && grep -q 'RUN_NAME=<new name>' '$T/g1.out' && grep -qF 'RESUME_PATH=$GD/ckpts/global_step_2' '$T/g1.out'"
    check "resume guard: the refused run leaves resolved.env as it was" cmp -s "$GD/resolved.env" "$T/g0.env"
    clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" OPSD_DATA_ROOT="$T/data" RUN_ROOT="$T/rg" DRY_RUN=1 \
        bash "$T/guard.sh" "$HERE" MAX_TURNS=3 >"$T/g2.out" 2>&1
    rc=$?
    check "resume guard: DRY_RUN=1 only warns" bash -c "[ $rc = 0 ] && grep -q 'a real run would refuse' '$T/g2.out'"
    guard "$T/rg" MAX_TURNS=3 EVAL_ONLY=1 >"$T/g3.out" 2>&1
    rc=$?
    check "resume guard: an EVAL_ONLY re-invocation runs, in resolved.eval-<sha12>.env + provenance.eval-<sha12>.json" \
        bash -c "[ $rc = 0 ] && grep -q '^RESOLVED=$GD/resolved.eval-[0-9a-f]\{12\}.env' '$T/g3.out' \
            && ls '$GD'/provenance.eval-*.json >/dev/null 2>&1"
    check "resume guard: ...and keeps the training run's resolved.env" cmp -s "$GD/resolved.env" "$T/g0.env"
    guard "$T/rg" MAX_TURNS=3 RESUME_MODE=from_path RESUME_PATH="$GD/ckpts/global_step_2" >"$T/g4.out" 2>&1
    rc=$?
    check "resume guard: an explicit RESUME_PATH runs; the earlier resolved.env / provenance.json are kept as *.prev-<sha12>.*" \
        bash -c "[ $rc = 0 ] && grep -q '^MAX_TURNS=3' '$GD/resolved.env' && cmp -s '$GD'/resolved.prev-*.env '$T/g0.env' \
            && ls '$GD'/provenance.prev-*.json >/dev/null 2>&1"
    guard "$T/rg2" >/dev/null 2>&1
    check "resume guard: other settings with nothing to resume yet (no ckpts/exports) are accepted" guard "$T/rg2" MAX_TURNS=3
    printf 'LOGGER=wandb\nNCCL_NET=IB\n' >"$T/ov.env"
    cat >"$T/resolve2.sh" <<'EOS'
set -euo pipefail
source "$1/bootstrap.sh"
opsd_resolve_recipe toy/main/4bi/glm_a --overlay "$2"
echo "LOGGER=$LOGGER NCCL_NET=$NCCL_NET"
EOS
    clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" OPSD_DATA_ROOT="$T/data" RUN_ROOT="$T/runs3" DRY_RUN=1 \
        bash "$T/resolve2.sh" "$HERE" "$T/ov.env" >"$T/ov.out" 2>&1
    check "recipe values beat the bootstrap's own defaults (no false hermetic error)" \
        grep -qx "LOGGER=wandb NCCL_NET=IB" "$T/ov.out"
    check_fails "opsd_resolve_recipe is hermetic (stray ambient OPSD_* is rejected)" \
        clean_env OPSD_PYTHON="$PY" OPSD_RECIPES_DIR="$FIXR" RUN_ROOT="$T/runs2" OPSD_STRAY=1 bash "$T/resolve.sh" "$HERE"
else
    echo "skip hint-sha / recipe checks (no python with pyyaml; set OPSD_PYTHON)"
fi

# ------------------------------------------------------------------------------ 5. ray_cluster (dry)
RC="$HERE/ray_cluster.sh"
check "ray_cluster.sh head (dry) prints ray start --head" \
    bash -c "OPSD_DRY_RUN=1 NODE_IP=192.0.2.10 RAY_PORT=6390 bash '$RC' head --temp-dir /tmp/opsd_ray_t | grep -q 'start --head --port=6390 --node-ip-address=192.0.2.10 --temp-dir=/tmp/opsd_ray_t'"
check "ray_cluster.sh worker (dry) joins the head address" \
    bash -c "OPSD_DRY_RUN=1 NODE_IP=198.51.100.7 bash '$RC' worker 192.0.2.10:6390 | grep -q 'start --address=192.0.2.10:6390 --node-ip-address=198.51.100.7'"
check "ray_cluster.sh status (dry)" bash -c "OPSD_DRY_RUN=1 bash '$RC' status 192.0.2.10:6390 | grep -q 'status --address=192.0.2.10:6390'"
check "ray_cluster.sh wait (dry)" bash -c "OPSD_DRY_RUN=1 bash '$RC' wait 2 5 192.0.2.10:6390 | grep -q '2 alive node'"
check_fails "ray_cluster.sh refuses to stop clusters" bash "$RC" stop
check_fails "ray_cluster.sh worker without an address" env OPSD_DRY_RUN=1 bash "$RC" worker
check "ray_cluster.sh --help" bash "$RC" --help

echo
echo "selftest: $PASS passed, $FAIL failed"
exit "$FAIL"
