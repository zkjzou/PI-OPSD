#!/usr/bin/env bash
# =====================================================================================================
# One-time AppWorld setup for the OPSD AppWorld experiments. Idempotent: every step checks its result first
# and is skipped when already done, so re-running it is safe (and is how you resume after a failure).
#
#   bash scripts/setup/setup_appworld.sh
#   APPWORLD_VENV=/path/venv APPWORLD_ROOT=/path/root bash scripts/setup/setup_appworld.sh
#
# Steps
#   1. AppWorld environment-server venv at $APPWORLD_VENV (default ~/appworld-venv): Python >= 3.11 (3.12
#      tested), `pip install -r scripts/setup/requirements/appworld_server.txt -c ...constraints.txt`
#      (appworld 0.1.3.post1 pinned; it requires pydantic<2, so it can NOT share the training venv),
#      `appworld install`.
#   2. AppWorld task data into $APPWORLD_ROOT/data (default ~/appworld_root) with `appworld download data`.
#      Never re-downloaded once present (a re-download would also undo step 4).
#   3. Every encrypted OPSD AppWorld bundle present under $OPSD_DATA_ROOT/appworld/, fetched first with
#        python -m opsd.data.fetch --repo <hf_dataset_id> --revision <commit_sha> --bench appworld
#      (or `--from-local <unpacked dataset copy> --bench appworld`; `--include 'appworld/<name>.bundle'` for one):
#        appworld_opsd_pi.bundle                       train-90 PI hints, reference trajectories, the b0a8eae
#                                                      patch (what training needs)
#        appworld_opsd_metrics.bundle                  error-step judge labels (train-90, dev, test_normal)
#        appworld_opsd_corpora_{train90,dev,test_normal}.bundle   n=128 no-hint student rollouts
#        appworld_opsd_dev_hints.bundle                the PI hints keyed to AppWorld dev
#        appworld_opsd_test_normal_hints.bundle        the PI hints keyed to AppWorld test_normal
#        appworld_opsd_xdnull.bundle                   the ALFWorld cross-domain null hints
#      Each is sha256-checked (against $OPSD_DATA_ROOT/manifest.json, which opsd.data.fetch writes, else the
#      value pinned in scripts/setup/appworld_bundles.py), decrypted with AppWorld's own unpack_bundle into
#      $OPSD_DATA_ROOT/appworld/plain/<its top-level directory>/ (plain/opsd_appworld_pi/{hints,patches,...},
#      plain/opsd_appworld_xdnull/alfworld/..., ...), and every unpacked file is checked against the sha256 of
#      the manifest's `contents` list (and opsd_appworld_pi/CONTENTS.json). A verified bundle is stamped and
#      skipped on the next run; bundles that are not there are skipped.
#      EVALUATION-SPLIT MATERIAL: metrics, corpora_dev, corpora_test_normal, dev_hints, test_normal_hints and
#      xdnull carry data keyed to AppWorld dev / test_normal (xdnull: ALFWorld-keyed hints whose text is AppWorld
#      test_normal reference solutions); they are for reproducing the paper's controls / analysis.
#      The ALFWorld arm alfworld/null/4bi/null_xdomain reads its hint file from the xdnull bundle
#      (plain/opsd_appworld_xdnull/alfworld/glt_xdmix_tau2_appworld.json), so it needs this script too:
#        python -m opsd.data.fetch --repo <hf_dataset_id> --revision <commit_sha> --bench appworld \
#            --include appworld/appworld_opsd_xdnull.bundle
#        bash scripts/setup/setup_appworld.sh      # steps 1-2 build the AppWorld venv the decryption needs
#      AppWorld's license requires that anything derived from its task data is redistributed only in an
#      encrypted form: do not re-share the decrypted files in plain text.
#   4. Patch the three b0a8eae_* train tasks (their ground truth compares a sum in seconds with a budget in
#      minutes, which makes them unwinnable) with the patch shipped inside the batch-1 bundle. Every paper arm
#      trained on the patched 90-task train split. WITHOUT that bundle this step is skipped: the dataset
#      builder then drops the three tasks (87 train tasks, 4 full steps per epoch at batch 18 instead of 5) and
#      scripts/appworld/run.sh refuses to train unless APPWORLD_ALLOW_UNPATCHED=1 -- such runs are not
#      reproductions of the paper numbers.
#
# Environment: APPWORLD_VENV, APPWORLD_ROOT, OPSD_DATA_ROOT (default ${XDG_CACHE_HOME:-~/.cache}/opsd/data),
#   APPWORLD_BUNDLE (location of the PI bundle; default $OPSD_DATA_ROOT/appworld/appworld_opsd_pi.bundle),
#   APPWORLD_PYTHON (the interpreter used to create the venv; default python3.12, else python3),
#   PIP_INDEX_URL (pip's own).
# Flags: --force-venv (recreate a venv that exists but is broken), --skip-bundle (steps 1-2 only),
#   --check (report the state, change nothing).
#
# The training environment itself is separate: `bash scripts/setup/setup_train_env.sh` (`--extra megatron`
# for the 30B student): the pinned SkyRL checkout next to the repository, then `uv sync --frozen`.
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
    echo "[setup_appworld] FATAL: repository root (pyproject.toml + opsd/__init__.py + opsd/skyrl_ext/__init__.py)" \
        "not found above $HERE" >&2
    exit 1
}

FORCE_VENV=0
SKIP_BUNDLE=0
CHECK_ONLY=0
for a in "$@"; do
    case "$a" in
        --force-venv) FORCE_VENV=1 ;;
        --skip-bundle) SKIP_BUNDLE=1 ;;
        --check) CHECK_ONLY=1 ;;
        -h | --help) sed -n '2,/^# =====/p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "[setup_appworld] unknown argument: $a (see --help)" >&2; exit 2 ;;
    esac
done

APPWORLD_VENV="${APPWORLD_VENV:-$HOME/appworld-venv}"
APPWORLD_ROOT="${APPWORLD_ROOT:-$HOME/appworld_root}"
OPSD_DATA_ROOT="${OPSD_DATA_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/opsd/data}"
APPWORLD_BUNDLE="${APPWORLD_BUNDLE:-$OPSD_DATA_ROOT/appworld/appworld_opsd_pi.bundle}"
PLAIN_DIR="$OPSD_DATA_ROOT/appworld/plain"
UNPACKED="$PLAIN_DIR/opsd_appworld_pi"
APY="$APPWORLD_VENV/bin/python"
ABIN="$APPWORLD_VENV/bin/appworld"
REQ="$ROOT/scripts/setup/requirements/appworld_server.txt"
CON="$ROOT/scripts/setup/requirements/appworld_server.constraints.txt"
BUNDLES_PY="$HERE/appworld_bundles.py"
# Published with the dataset (AppWorld's own convention for its encrypted bundles); see its README. The same
# password and salt open every OPSD AppWorld bundle.
BUNDLE_PASSWORD="${APPWORLD_BUNDLE_PASSWORD:-8kXonWNJyf2HOvBHKtUmSGlI}"
BUNDLE_SALT="${APPWORLD_BUNDLE_SALT:-h20G2urKCMNsvxoO}"
PATCHED_TASKS=(b0a8eae_1 b0a8eae_2 b0a8eae_3)

log() { printf '[setup_appworld] %s\n' "$*" >&2; }
die() { printf '[setup_appworld] ERROR: %s\n' "$*" >&2; exit 1; }

venv_ok() {
    [ -x "$APY" ] && [ -x "$ABIN" ] &&
        "$APY" -c "import appworld, appworld.apps; appworld.apps.api_docs; assert appworld.__version__ == '0.1.3.post1', appworld.__version__" >/dev/null 2>&1
}
data_ok() { [ -d "$APPWORLD_ROOT/data/tasks" ] && [ -n "$(ls -A "$APPWORLD_ROOT/data/tasks" 2>/dev/null | head -1)" ]; }
patched() {
    local t
    for t in "${PATCHED_TASKS[@]}"; do
        grep -q PATCHED "$APPWORLD_ROOT/data/tasks/$t/ground_truth/evaluation.py" 2>/dev/null || return 1
    done
}
bundles() {  # MODE (check | setup): scripts/setup/appworld_bundles.py over $OPSD_DATA_ROOT/appworld/*.bundle
    local py="$1"
    shift
    "$py" "$BUNDLES_PY" "$@" --data-root "$OPSD_DATA_ROOT" --pi-bundle "$APPWORLD_BUNDLE"
}

if [ "$CHECK_ONLY" = 1 ]; then
    venv_ok && log "venv:    OK ($APPWORLD_VENV)" || log "venv:    MISSING/BROKEN ($APPWORLD_VENV)"
    data_ok && log "data:    OK ($APPWORLD_ROOT/data)" || log "data:    MISSING ($APPWORLD_ROOT/data)"
    PYV="$APY"; [ -x "$PYV" ] || PYV=python3
    bundles "$PYV" check || log "bundles: could not be checked"
    patched && log "b0a8eae: patched (90 train tasks)" || log "b0a8eae: NOT patched (87 train tasks)"
    exit 0
fi

# ---- 1. venv ----------------------------------------------------------------------------------------
if venv_ok; then
    log "1/4 AppWorld venv OK: $APPWORLD_VENV"
else
    if [ -e "$APPWORLD_VENV" ]; then
        [ "$FORCE_VENV" = 1 ] || die "$APPWORLD_VENV exists but is not a working appworld==0.1.3.post1 venv; fix it, or re-run with --force-venv to recreate it"
        log "1/4 recreating $APPWORLD_VENV (--force-venv)"
        rm -rf "$APPWORLD_VENV"
    fi
    PYBOOT="${APPWORLD_PYTHON:-}"
    if [ -z "$PYBOOT" ]; then
        if command -v python3.12 >/dev/null 2>&1; then PYBOOT=python3.12; else PYBOOT=python3; fi
    fi
    "$PYBOOT" -c 'import sys; assert sys.version_info >= (3, 11), sys.version' 2>/dev/null ||
        die "$PYBOOT is older than Python 3.11 (AppWorld 0.1.x needs >= 3.11); set APPWORLD_PYTHON"
    log "1/4 creating $APPWORLD_VENV with $($PYBOOT --version 2>&1)"
    "$PYBOOT" -m venv "$APPWORLD_VENV"
    "$APY" -m pip install --upgrade pip -q
    "$APY" -m pip install -r "$REQ" -c "$CON"
    "$ABIN" install
    venv_ok || die "AppWorld is still not importable from $APPWORLD_VENV after installation"
fi

# ---- 2. task data -------------------------------------------------------------------------------------
mkdir -p "$APPWORLD_ROOT"
if data_ok; then
    log "2/4 AppWorld data present: $APPWORLD_ROOT/data"
else
    log "2/4 downloading the AppWorld data into $APPWORLD_ROOT"
    (cd "$APPWORLD_ROOT" && "$ABIN" download data)
    data_ok || die "no task data under $APPWORLD_ROOT/data after 'appworld download data'"
fi

# ---- 3. the encrypted OPSD bundles ---------------------------------------------------------------------
if [ "$SKIP_BUNDLE" = 1 ]; then
    log "3/4 skipped (--skip-bundle)"
else
    log "3/4 OPSD AppWorld bundles under $OPSD_DATA_ROOT/appworld -> $PLAIN_DIR"
    bundles "$APY" setup --password "$BUNDLE_PASSWORD" --salt "$BUNDLE_SALT" ||
        die "an AppWorld bundle could not be verified or unpacked (see above); re-fetch it with python -m opsd.data.fetch"
fi

# ---- 4. b0a8eae patch ---------------------------------------------------------------------------------
PATCH="$UNPACKED/patches/patch_appworld_b0a8eae.py"
if patched; then
    log "4/4 b0a8eae_* already patched (90 train tasks)"
elif [ -f "$PATCH" ]; then
    log "4/4 patching b0a8eae_1/2/3 in $APPWORLD_ROOT"
    "$APY" "$PATCH" --appworld-root "$APPWORLD_ROOT"
    patched || die "the b0a8eae patch did not apply"
else
    log "4/4 WARNING: no patch (PI bundle not unpacked): the train split will have 87 tasks, not 90"
fi

log "done: APPWORLD_VENV=$APPWORLD_VENV APPWORLD_ROOT=$APPWORLD_ROOT OPSD_DATA_ROOT=$OPSD_DATA_ROOT"
log "next: bash scripts/appworld/run.sh <arm id>   (python -m opsd.recipes list --bench appworld)"
