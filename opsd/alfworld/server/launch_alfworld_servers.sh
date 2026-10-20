#!/bin/bash
# Launch a pool of ALFWorld (AlfredTWEnv) FastAPI servers on consecutive ports. Echoes a JSON list of
# server URLs on the LAST line of stdout so the caller can capture it:
#   ALFWORLD_SERVER_URLS=$(bash launch_alfworld_servers.sh | tail -1)
#
# ALFWorld pulls in textworld + gymnasium==0.29.1 + a native PDDL/Inform7 stack that conflicts with the
# SkyRL/vLLM environment, so it lives in its OWN venv. The SkyRL training env only talks HTTP.
#
# This script installs and downloads NOTHING: it fails if the venv or the game data is missing. Create
# both once with
#   bash scripts/setup/setup_alfworld.sh
# Versions used for the paper runs (the setup script pins them):
#   alfworld==0.4.2 textworld==1.7.0 gymnasium==0.29.1 stable-baselines3==2.6.0
#   + fastapi, uvicorn[standard], pyyaml, pandas, pyarrow
#
# Env knobs:
#   ALFWORLD_VENV            venv with alfworld installed (must exist)     (default: $HOME/alfworld-venv)
#   ALFWORLD_DATA            alfworld-download target, holds json_2.1.1/   (default: $HOME/data/alfworld)
#   NUM_ALFWORLD_SERVERS     number of servers in the pool                 (default: 16)
#   ALFWORLD_BASE_PORT       first port; servers use BASE..BASE+N-1        (default: 8100)
#   ALFWORLD_HOST            bind host                                     (default: 0.0.0.0)
#   ALFWORLD_ADVERTISE_HOST  host the SkyRL client connects to             (default: 127.0.0.1)
#   ALFWORLD_MAX_STEPS       server-side episode cap                       (default: 50)
#   ALFWORLD_LOG_DIR         where server logs go                          (default: $PWD)
#   ALFWORLD_PID_DIR         where server PIDs are recorded                (default: $ALFWORLD_LOG_DIR)
#
# Ports: next to SkyRL-managed vLLM engines (engine i takes a port in [8000+100*i, 8100+100*i)) the
# default 8100 collides from the 2nd engine on; pass a base clear of that range (the launchers use 9100).
#
# PIDs: one file per server, $ALFWORLD_PID_DIR/alfworld_serve_<port>.pid, plus the space-separated list
# $ALFWORLD_PID_DIR/alfworld_serve.pids. A caller stops ONLY the servers it started:
#   kill $(cat "$ALFWORLD_PID_DIR/alfworld_serve.pids")
# Give every run its own ALFWORLD_PID_DIR (e.g. $RUN_DIR/pids) so concurrent runs never touch each
# other's pools.
#
# Memory: each /reset leaves a ~35 MB deleted-but-still-mapped library file in the server's $TMPDIR until
# that server process exits, so a long-lived pool grows by ~35 MB per episode. Restart the pool between
# long jobs and keep TMPDIR on a disk with room.
set -u

ALFWORLD_VENV=${ALFWORLD_VENV:-$HOME/alfworld-venv}
ALFWORLD_PY=${ALFWORLD_PY:-$ALFWORLD_VENV/bin/python}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/data/alfworld}
NUM_ALFWORLD_SERVERS=${NUM_ALFWORLD_SERVERS:-16}
ALFWORLD_BASE_PORT=${ALFWORLD_BASE_PORT:-8100}
ALFWORLD_HOST=${ALFWORLD_HOST:-0.0.0.0}
ALFWORLD_ADVERTISE_HOST=${ALFWORLD_ADVERTISE_HOST:-127.0.0.1}
ALFWORLD_MAX_STEPS=${ALFWORLD_MAX_STEPS:-50}
ALFWORLD_LOG_DIR=${ALFWORLD_LOG_DIR:-$PWD}
ALFWORLD_PID_DIR=${ALFWORLD_PID_DIR:-$ALFWORLD_LOG_DIR}
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- preconditions: an existing venv and downloaded data (no installs here) ----
if [ ! -x "$ALFWORLD_PY" ]; then
    echo "ERROR: no ALFWorld venv python at $ALFWORLD_PY (ALFWORLD_VENV=$ALFWORLD_VENV)." >&2
    echo "       Create it with 'bash scripts/setup/setup_alfworld.sh', or export ALFWORLD_VENV=<existing venv>." >&2
    exit 1
fi
if ! "$ALFWORLD_PY" - >&2 <<'PY'
import importlib.util
import sys

missing = [m for m in ("alfworld", "textworld", "fastapi", "uvicorn", "yaml") if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing packages: " + " ".join(missing))
PY
then
    echo "ERROR: the venv at $ALFWORLD_VENV lacks the ALFWorld server packages (see above)." >&2
    echo "       Re-run 'bash scripts/setup/setup_alfworld.sh'." >&2
    exit 1
fi
if [ ! -d "$ALFWORLD_DATA/json_2.1.1" ]; then
    echo "ERROR: ALFWorld game data not found: $ALFWORLD_DATA/json_2.1.1 (ALFWORLD_DATA=$ALFWORLD_DATA)." >&2
    echo "       Download it with 'bash scripts/setup/setup_alfworld.sh', or export ALFWORLD_DATA=<existing dir>." >&2
    exit 1
fi

mkdir -p "$ALFWORLD_LOG_DIR" "$ALFWORLD_PID_DIR"

# ---- launch the server pool ----
# IPv6 handling (some clusters advertise an IPv6 node address, e.g. 2001:db8::10; most hosts are IPv4):
#  - bracket the host so "http://[<ipv6>]:<port>" is a valid URL -- unbracketed it is an invalid URL
#    (urllib3 LocationParseError -> InvalidURL in the env client's _post, which crashed runs at init).
#  - bind the server to "::" (dual-stack on Linux) instead of 0.0.0.0 (IPv4-only), so it is actually
#    reachable at the IPv6 advertise address. This is local to the ALFWorld servers: the pinned upstream
#    SkyRL vLLM actor always binds 0.0.0.0, and IPv6 inference hosts are not supported in this release.
# IPv4 addresses and hostnames have no ':' and pass through unchanged.
case "$ALFWORLD_ADVERTISE_HOST" in
    *:*) ALFWORLD_URL_HOST="[$ALFWORLD_ADVERTISE_HOST]"; ALFWORLD_HOST="::" ;;
    *)   ALFWORLD_URL_HOST="$ALFWORLD_ADVERTISE_HOST" ;;
esac
ALFWORLD_SERVE_PIDS=""
URLS="["
for i in $(seq 0 $(($NUM_ALFWORLD_SERVERS - 1))); do
    PORT=$(($ALFWORLD_BASE_PORT + $i))
    ALFWORLD_DATA="$ALFWORLD_DATA" "$ALFWORLD_PY" "$THIS_DIR/alfworld_server.py" \
        --port "$PORT" --host "$ALFWORLD_HOST" --max-steps "$ALFWORLD_MAX_STEPS" \
        > "$ALFWORLD_LOG_DIR/alfworld_serve_$PORT.log" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" > "$ALFWORLD_PID_DIR/alfworld_serve_$PORT.pid"
    ALFWORLD_SERVE_PIDS="$ALFWORLD_SERVE_PIDS $SERVER_PID"
    URL="http://$ALFWORLD_URL_HOST:$PORT"
    if [ "$i" -gt 0 ]; then URLS+=","; fi
    URLS+="\"$URL\""
done
URLS+="]"

# Expose the PIDs so the caller can stop exactly this pool.
echo "$ALFWORLD_SERVE_PIDS" > "$ALFWORLD_PID_DIR/alfworld_serve.pids"
echo "Launched $NUM_ALFWORLD_SERVERS ALFWorld servers (ports $ALFWORLD_BASE_PORT..$(($ALFWORLD_BASE_PORT + $NUM_ALFWORLD_SERVERS - 1)); PIDs in $ALFWORLD_PID_DIR)" >&2
# JSON list on the LAST stdout line for the caller to capture.
echo "$URLS"
