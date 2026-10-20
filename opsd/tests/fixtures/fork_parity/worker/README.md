# Regression goldens: policy worker (beta teacher-KL + SDAR)

What the reference implementation's FSDP policy worker computed, so
`opsd.skyrl_ext.policy_worker.OPSDFSDPPolicyWorkerBase` can be held to the same numbers.

**Source.** The reference implementation: SkyRL `ecdaa946` with the OPSD loss terms written directly into
`skyrl/backends/skyrl_train/workers/worker.py` and `worker_utils.py`. Each case runs its path end to end:
`FSDPPolicyWorkerBase.forward_backward` -> `BatchIterator.batch_to_experience` (which attached
`exp.distill_log_probs`) -> `PolicyWorkerBase._forward_backward_micro` (which read `OPSD_DISTILL_LOSS_COEF`,
`OPSD_SDAR_LAMBDA`, `OPSD_SDAR_BETA` and `OPSD_SDAR_GATE` from `os.environ`; each case sets them).

**Harness (CPU, no Ray, no torch.distributed).** The worker is built with `object.__new__` and given what
`init_model` would set. Only these pieces are stubbed:

- the model: log-probs of the sampled tokens and full-sequence entropy, both computed from one seeded parameter `W`
  (5 rows x 8 positions x 6 vocab) with `requires_grad`;
- `strategy.backward`: records the loss value and calls `loss.backward()`;
- `strategy.all_reduce`: identity;
- the scheduler, the device mesh and `torch.cuda.current_device` (patched to the CPU).

The `MeshRank` uses dp_size 2. The stubs are the ones in `opsd/skyrl_ext/tests/test_policy_worker.py`.
Change them only together with a re-capture.

**Files.**

- `inputs.json`: the seeded inputs (seed 20261009): `W0`, the batch tensors (5 rows; row 3 is a fully loss-masked
  padding row) and the two teacher variants, `teacher` and `teacher_nonfinite` (NaN/±inf entries, encoded as the
  strings `"nan"`, `"inf"` and `"-inf"`). Also the shared algorithm config (`regular` loss, `token_mean`,
  `kl_loss_coef` 0.05, `entropy_loss_coef` 0.02, temperature 0.8).
- `goldens.json`: 32 cases. Each holds the case's `env`, `algo` overrides, `micro`, `dp_size`, `teacher`, `loss_fn`
  and `loss_fn_config`, and the result:
  - every loss passed to `strategy.backward`;
  - every micro-batch's status dict, minus `loss_fn_outputs`;
  - the reduced `WorkerOutput.metrics` and `loss_fn_outputs`;
  - `W.grad` after the whole `forward_backward`.

  One case records the error the reference raised instead: a bogus gate with lambda != 0.

**Cases.**

- beta {0, 0.1, 1e-4} x `use_kl_loss` on/off x `use_entropy_loss` on/off;
- SDAR lambda 0.5 x gate {gap, entropy, soft_or} x entropy loss on/off;
- beta and SDAR combined, with `OPSD_SDAR_BETA=2.0`;
- the gate alias `' Soft-Or '`;
- `OPSD_SDAR_BETA=''`, which parses as 0 (the no-gate ablation);
- a missing teacher, with both terms requested and with neither;
- a non-finite teacher;
- the `k1` estimator;
- a bogus gate with lambda 0, which the reference accepted;
- micro batch 1 and micro batch 3, in addition to the default micro batch 2 over 5 rows (chunks 2, 2, 1), which
  test teacher-chunk alignment;
- dp_size 1;
- `loss_fn=cross_entropy`, the SFT branch, which carries no OPSD terms;
- the Tinker `loss_fn` / `loss_fn_config` override.

**How it was captured.** The capture script is not part of this repository. Re-running it reproduces both
files byte for byte. All JSON is plain floats (Python `repr`, so
float32 values round-trip exactly). The files contain no host paths, usernames, secrets or environment values.

**Checked by.** `opsd/skyrl_ext/tests/test_policy_worker_parity.py`. The tolerance is float-rounding only:
rel 1e-6, abs 1e-9. On the capture machine the worker matched every value bit for bit.
