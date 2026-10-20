#!/usr/bin/env bash
# ---------------------------------------------------------------------------------------------------
# SHARED MEGATRON BACKEND BLOCK — sourced by run scripts that must support a MoE policy.
#
# PORTED from the Megatron block of the AppWorld run script rather than written fresh. That block is the
# ONLY validated 30B-MoE configuration in this repo: the AppWorld 30B-A3B DAPO run finished 150 steps and
# produced the best AppWorld number of any method. Every failure cited below was MEASURED on a real run —
# do not "simplify" them away.
#
# Factored into one sourced file instead of copied into both ALFWorld run scripts: two copies of a block
# this subtle drift apart, and the ALFWorld GRPO/DAPO and OPSD paths must agree on the backend exactly.
#
# CONTRACT
#   in : STRATEGY (fsdp2|megatron), NUM_GPUS_PER_NODE, NUM_NODES, USE_SAMPLE_PACKING,
#        MEGATRON_{TP,PP,CP,EP,ETP}, MOE_ROUTING_REPLAY, MEGATRON_OPTIMIZER_OFFLOAD[_FRACTION]
#   out: $MEGATRON_OVERRIDES — a hydra-override string, EMPTY unless STRATEGY=megatron, so every
#        pre-existing fsdp2 arm is bit-identical. Pass it unquoted into the hydra call.
#   also: exports CUDNN_PATH/CPATH/LD_LIBRARY_PATH on the megatron path only, and exits non-zero on a
#        parallelism-arithmetic violation. Interconnect-specific variables (e.g. a fabric provider's
#        fork-safety switch) are host-specific: export them yourself if your network stack needs them.
#
# WHY MEGATRON FOR A MoE, in order of importance:
#   1. `moe_enable_routing_replay` EXISTS ONLY HERE. Tiny numeric differences flip which of the top-8
#      experts fire, so vLLM's rollout logprobs and the trainer's diverge far more than for a dense model.
#      replay_utils.py fixes exactly that and imports megatron.core; on the FSDP path
#      `enable_return_routed_experts` has NO consumer. This is the main correctness risk in MoE RL.
#   2. Real expert parallelism, instead of all-gathering 604 M expert params per layer per micro-batch.
#   Empirically: a 30B MoE run down the ALFWorld FSDP colocate path emitted pure `++;` token garbage on
#   144/144 rollouts (two independent runs, 2026-08-19).
# ---------------------------------------------------------------------------------------------------

STRATEGY=${STRATEGY:-fsdp2}
MEGATRON_OVERRIDES=""

if [ "$STRATEGY" = "megatron" ]; then
    # cuDNN FOR MEGATRON/TE — REQUIRED, and it is a sudo quirk not a missing package. The megatron image
    # sets CUDNN_PATH/CPATH/LD_LIBRARY_PATH as ENV, but the container command is `sudo -E ... bash <run>`
    # and **sudo strips LD_LIBRARY_PATH even with -E** (unconditional blacklist), so transformer_engine
    # cannot find cuDNN 9: `OSError: libcudnn_graph.so.9: cannot open shared object file`. All six megatron
    # examples re-export these three. Same family as a libcuda.so.1 missing from LD_LIBRARY_PATH (Ray then
    # registers 0 GPUs).
    export CUDNN_PATH="${CUDNN_PATH:-/opt/cudnn}"
    export CPATH="${CUDNN_PATH}/include:${CPATH:-}"
    export LD_LIBRARY_PATH="${CUDNN_PATH}/lib:${LD_LIBRARY_PATH:-}"

    # ---- expandable_segments: NOT SET HERE. The AppWorld block sets it; that premise does not hold. ----
    # AppWorld's comment argues it is safe under Megatron because "validate_megatron_cfg ASSERTS
    # weight_sync_backend == nccl, so there is no CUDA-IPC transfer to break." MEASURED FALSE on the
    # ALFWorld path: the first weight sync of a 30B ALFWorld OPSD smoke run went through
    # weight_sync/cuda_ipc_strategy.py and died exactly the way the standing rule predicts --
    #   megatron_worker.py:1019 broadcast_to_inference_engines -> cuda_ipc_strategy.py:169 send_chunks
    #   -> new_inference_worker_wrap.py:155 update_weights_chunk -> rebuild_cuda_tensor
    #   -> RuntimeError: pidfd_getfd: Operation not permitted
    # (measured 2026-08-20). Expandable segments are incompatible with CUDA IPC, so
    # the share falls back to pidfd_getfd, which this container blocks. The standing rule -- NEVER set
    # expandable_segments on a colocate run -- holds on BOTH backends here; only the reason differs.
    # If a FRAGMENTATION OOM appears (reserved-but-unallocated large while a contiguous request fails),
    # the fix is CP or micro-batch, NOT this flag. Passing it explicitly is still honoured, so an operator
    # who has switched the transfer strategy to nccl can opt in.
    if [ -n "${PYTORCH_CUDA_ALLOC_CONF:-}" ]; then
        echo "NOTE: PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF was set explicitly." >&2
        case "$PYTORCH_CUDA_ALLOC_CONF" in
          *expandable_segments:True*)
            echo "      WARNING: expandable_segments breaks the CUDA-IPC weight sync on this path" >&2
            echo "      (pidfd_getfd: Operation not permitted at the first sync). Only safe if the" >&2
            echo "      weight transfer is genuinely nccl." >&2 ;;
        esac
    fi

    _MG="trainer.policy.megatron_config"
    MEGATRON_OVERRIDES="$_MG.tensor_model_parallel_size=${MEGATRON_TP:-4}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_MG.pipeline_model_parallel_size=${MEGATRON_PP:-1}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_MG.context_parallel_size=${MEGATRON_CP:-1}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_MG.expert_model_parallel_size=${MEGATRON_EP:-8}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_MG.expert_tensor_parallel_size=${MEGATRON_ETP:-1}"

    # ROUTER REPLAY (r3) IS A THREE-WAY COUPLING and validate_megatron_cfg asserts all three agree —
    # `moe_enable_routing_replay` alone kills the job in validate_cfg before a single rollout:
    #   AssertionError: rollout router replay (r3) is only supported when enable_return_routed_experts is True
    # Set together so the trio can never drift. NB r3 is UNVALIDATED on the colocate path: with it on,
    # generation died mid-rollout with a vLLM server returning an EMPTY body (JSONDecodeError), and it also
    # forces distributed_executor_backend=mp ("r3 can hang with the ray backend"). Default OFF here for the
    # same variable-isolation reason the AppWorld smoke defaulted it off — but a full run without it
    # FORFEITS the routing-mismatch mitigation that is the main reason to pick Megatron. Watch
    # policy/policy_kl and the importance ratio against a dense run to see whether that bites.
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_MG.moe_enable_routing_replay=${MOE_ROUTING_REPLAY:-false}"
    if [ "${MOE_ROUTING_REPLAY:-false}" = "true" ]; then
        _IE="generator.inference_engine"
        MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_IE.enable_return_routed_experts=true"
        MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_IE.distributed_executor_backend=mp"
    fi

    # gradient_accumulation_fusion MUST be off: the fused APEX kernel is not built in this image
    # (`fused_weight_gradient_mlp_cuda module is not found` in MegatronPolicyWorkerBase.init_model).
    # config.py's default already carries false, but something downstream turns it back on — which is why
    # the gsm8k 35b/122b examples set it explicitly for policy AND ref. Set for both here too.
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_MG.transformer_config_kwargs.gradient_accumulation_fusion=false"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES trainer.ref.megatron_config.transformer_config_kwargs.gradient_accumulation_fusion=false"

    # Activation recompute: all three Qwen3-30B-A3B examples keep these; match same-architecture precedent.
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_MG.transformer_config_kwargs.recompute_granularity=full"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_MG.transformer_config_kwargs.recompute_method=uniform"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_MG.transformer_config_kwargs.recompute_num_layers=1"

    # THE REF MODEL NEEDS ITS OWN PARALLELISM, and omitting it is a silent ~61 GB/GPU OOM, not an error.
    # THIS MATTERS MORE FOR ALFWORLD THAN IT DID FOR APPWORLD: pure DAPO builds no ref worker, but VANILLA
    # OPSD sets use_kl_in_reward=true and `use_ref_model = use_kl_loss or use_kl_in_reward`, so a ref worker
    # IS built — and here it is the FROZEN TEACHER, not a KL leash. Unconfigured it takes TP=1/PP=1/EP=1 and
    # every rank tries to hold the whole 30.5 B (~61 GB bf16), which reads as a memory failure but is a
    # config failure. MIRROR the policy's values — the two must agree.
    # No recompute_* on the ref: it runs under no_grad, so there are no stored activations to trade.
    # empty_cuda_cache matters more here than for the policy because colocate_all offloads/backloads the
    # ref around every forward.
    _RG="trainer.ref.megatron_config"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_RG.tensor_model_parallel_size=${MEGATRON_TP:-4}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_RG.pipeline_model_parallel_size=${MEGATRON_PP:-1}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_RG.context_parallel_size=${MEGATRON_CP:-1}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_RG.expert_model_parallel_size=${MEGATRON_EP:-8}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_RG.expert_tensor_parallel_size=${MEGATRON_ETP:-1}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_RG.moe_enable_routing_replay=${MOE_ROUTING_REPLAY:-false}"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_RG.empty_cuda_cache=true"

    # CPU OPTIMIZER OFFLOAD — the answer to the OOM, ON by default at this size. optim_step is the memory
    # PEAK (params + grads + fp32 optimizer state all resident): a 30.5 B MoE does NOT fit on 8x80 GB with
    # the optimizer on-GPU (measured: rollout/forward/backward all completed, then `Cuda failure 2 out of
    # memory` in optim_step). Offload frees ~45 GB/GPU. Both large-MoE examples default it ON with
    # fraction 1.0, so this is the intended configuration, not a desperate measure. Unlike FSDP's
    # cpu_offload (broken on this stack) this is Megatron's supported path: use_precision_aware_optimizer
    # keeps the offloaded state numerically correct and overlap_cpu_optimizer_d2h_h2d hides most of the
    # transfer behind compute. Cost: CPU RAM (the node has ~1.5 TB) plus some step time.
    _OPT_OFFLOAD=${MEGATRON_OPTIMIZER_OFFLOAD:-true}
    _OK="$_MG.optimizer_config_kwargs"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_OK.optimizer_cpu_offload=$_OPT_OFFLOAD"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_OK.use_precision_aware_optimizer=$_OPT_OFFLOAD"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_OK.overlap_cpu_optimizer_d2h_h2d=$_OPT_OFFLOAD"
    MEGATRON_OVERRIDES="$MEGATRON_OVERRIDES $_OK.optimizer_offload_fraction=${MEGATRON_OPTIMIZER_OFFLOAD_FRACTION:-1.0}"

    # ---- PARALLELISM ARITHMETIC, CHECKED BEFORE RAY STARTS ----------------------------------------
    # Megatron enforces TWO divisibility rules, and violating the second costs a whole submission:
    #   RuntimeError: world_size (8) is not divisible by expert_tensor_model_pipeline_parallel size (16)
    # CP IS PART OF THE MODEL-PARALLEL PRODUCT: DP = world/(TP*PP*CP). Omitting CP here once reported DP=2
    # for a config that actually ran DP=1 — a guard that prints a wrong number is worse than no guard.
    _WS=$(( NUM_GPUS_PER_NODE * NUM_NODES ))
    _TPPP=$(( ${MEGATRON_TP:-4} * ${MEGATRON_PP:-1} * ${MEGATRON_CP:-1} ))
    _ETMP=$(( ${MEGATRON_ETP:-1} * ${MEGATRON_EP:-8} * ${MEGATRON_PP:-1} ))
    if [ "$_TPPP" -gt "$_WS" ] || [ $(( _WS % _TPPP )) -ne 0 ]; then
        echo "FATAL: TP*PP*CP=$_TPPP must DIVIDE world_size=$_WS (it sets DP=world_size/(TP*PP*CP))." >&2
        exit 1
    fi
    if [ $(( _WS % _ETMP )) -ne 0 ]; then
        echo "FATAL: ETP*EP*PP=$_ETMP must DIVIDE world_size=$_WS." >&2
        echo "       Megatron calls this expert_tensor_model_pipeline_parallel size." >&2
        echo "       On 8 GPUs use TP=4 PP=1 EP=8 ETP=1 (product 8, DP=2)." >&2
        exit 1
    fi
    # CP>1 REQUIRES SAMPLE PACKING (validate_megatron_cfg asserts it, deep in config validation).
    # LIVE CONFLICT ON THIS PATH: the vanilla-OPSD arms run packing OFF (the ALFWorld OPSD launcher's default,
    # replicating the 4B protocol), so they CANNOT take CP>1 without changing that. Fail loudly here
    # instead of deep in validation.
    if [ "${MEGATRON_CP:-1}" -gt 1 ] && [ "${USE_SAMPLE_PACKING:-true}" != "true" ]; then
        echo "FATAL: MEGATRON_CP>1 requires sample packing (validate_megatron_cfg asserts it)." >&2
        echo "       Either set USE_SAMPLE_PACKING=true or keep MEGATRON_CP=1." >&2
        exit 1
    fi
    echo "megatron parallelism: TP=${MEGATRON_TP:-4} PP=${MEGATRON_PP:-1} EP=${MEGATRON_EP:-8}" \
         "ETP=${MEGATRON_ETP:-1} CP=${MEGATRON_CP:-1} -> DP=$(( _WS / _TPPP )), ETP*EP*PP=$_ETMP divides world=$_WS" \
         "| routing_replay=${MOE_ROUTING_REPLAY:-false} optimizer_cpu_offload=$_OPT_OFFLOAD"

    # Megatron needs its OWN environment (`uv sync --extra megatron`): pyproject declares the megatron
    # extra as CONFLICTING with other extras, so the fsdp venv cannot import megatron.core.
    # WARNING not a gate: as a gate with its stderr discarded this killed a job 3 s in with only
    # "megatron.core is not importable", and the real cause was the stripped LD_LIBRARY_PATH above. The
    # durable lesson is "never discard a diagnostic's error text". It runs AFTER the cuDNN export so it
    # validates a fixed environment, and the trainer is the real authority.
    if ! _mg_err=$(python -c "import megatron.core" 2>&1); then
        echo "WARNING: 'import megatron.core' failed here; continuing (the trainer is the real authority)." >&2
        printf '         %s\n' "$_mg_err" >&2
        echo "         If the trainer then dies on megatron, run from an environment built with the megatron extra." >&2
    fi
fi

# ---- PRE-STAGE THE WEIGHTS (opt-in; applies on BOTH backends) --------------------------------------
# Lives here rather than in the caller so PRESTAGE_MODEL has a real consumer -- an env var a run script
# never reads is a silent no-op, which is the dominant failure mode in these scripts.
# WHY: MODEL_PATH otherwise goes straight to hydra and EVERY worker resolves the HF id itself. Fine at
# 4B/8 GB; a live pull of a 61 GB MoE is the same shape as the failure that killed a Qwen3.5-4B run (16
# workers pulling 10 GB live -> from_pretrained "no file named ...shard"). One download per node, then all
# local workers hit a warm cache. Skipped when MODEL_PATH is already a local directory.
if [ "${PRESTAGE_MODEL:-0}" = "1" ] && [ -n "${MODEL_PATH:-}" ] && [ ! -d "$MODEL_PATH" ]; then
    echo "pre-staging $MODEL_PATH into ${HF_HOME:-default HF cache} ..."
    hf download "$MODEL_PATH" 2>/dev/null \
      || python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL_PATH')" \
      || echo "WARNING: pre-stage of $MODEL_PATH failed; workers will pull it live" >&2
fi
