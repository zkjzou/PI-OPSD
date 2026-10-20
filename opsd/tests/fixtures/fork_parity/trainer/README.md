# Regression goldens: trainer cluster

Reference outputs of the reference implementation (`<ROOT>` = OPSD's trainer changes made directly in SkyRL
`ecdaa946`). The capture script is not part of this repository. All values are deterministic (fixed seeds,
no timings, no paths). Both captures ran twice and gave byte-identical files.

| File | What the reference code produced |
|---|---|
| `grpo_signal.json` | `RayPPOTrainer.compute_advantages_and_returns` from `<ROOT>/skyrl/train/trainer.py`, i.e. upstream's estimator plus the reference `grpo/*` gradient-signal hunk. It ran on a stubbed trainer (`object.__new__` + a SimpleNamespace cfg) with the real GRPO/RLOO estimators. Each case stores the input batch (rewards, response mask, uids, `is_last_step`, `pad_size`; padding added with the real `pad_training_input_batch`) and the outputs: the full `all_metrics` dict, the advantages and `data.metadata["metrics"]`. The cases cover groups of 4 that are all-solved, all-failed, mixed and partial; padding; uneven and singleton groups; float-rounding ties; negative rewards; RLOO; and step-wise batches, where only last-step rows count and intermediate steps carry a reward. |
| `resume_w0.json`, `resume_w2.json` | The reference `RayPPOTrainer.train()` coroutine (with its epoch ceiling, `global_step` stop guards and `resume_diag/*` logs), driven without models. Every model, generator and Ray touchpoint is an instance-level stub. `save_checkpoints` / `load_checkpoints` are the real ones (data.pt from a real torchdata `StatefulDataLoader`: 11 items, batch 3, `drop_last`, seed 42, `num_workers` 0 or 2 with fork workers). `trainer.ckpt_interval=2`, `eval_interval=4`, `eval_before_train` and `update_ref_every_epoch` are on. Recorded per run: the trained batches (uids per `global_step`), the logged `trainer/epoch`, the eval and ref-sync steps, every checkpoint's iterator state (batches consumed, `_iterator_finished`, sha256 of the state) and the tracker log keys. `fresh` is a 3-epoch run from scratch. `scenarios` resume from its checkpoints: an epoch boundary, mid-epoch, the in-loop end-of-training save, the reference `global_step_{T+1}` safety net (continued into an extra epoch, or relaunched as is), an `_iterator_finished=True` state (what upstream's safety net writes), and `trainer.epochs=0` eval-only runs. `source_states` holds the dataloader states the resumes start from, as tagged JSON (`__tensor__` base64, `__tuple__`, `__dict__` for non-string keys), so the tests resume from byte-identical inputs. |

## Consumers

- `opsd/skyrl_ext/tests/test_trainer_grpo_signal.py`: `OPSDRayPPOTrainer` must reproduce `all_metrics`
  (bit-identical `grpo/*`), the advantages and the batch metrics.
- `opsd/skyrl_ext/tests/test_trainer_resume.py`: `OPSDRayPPOTrainer` runs upstream `train()` plus its
  `load_checkpoints` drain on the same harness. It must train the same batches in the same order. Intended
  differences are asserted explicitly, not ignored: the corrected epoch index (and with it the
  `update_ref_every_epoch` sync) after a boundary resume; no stray training step when `trainer.epochs=0`;
  the full extra epoch when continuing from a `global_step_{T+1}` safety net; and upstream's
  `_iterator_finished=True` safety net.
