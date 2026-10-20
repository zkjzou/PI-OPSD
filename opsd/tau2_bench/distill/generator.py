"""τ² OPSD generator — emits the with-hint ``ref_prompt_token_ids`` the OPSD teacher forward needs.

The student rolls out with NO hint (``Tau2Generator.generate`` is untouched). Afterwards, per rollout,
we prepend the encoded ``<Secret information>`` block carrying the frontier teacher's trajectory to the
student's ``prompt_token_ids`` — OPSD's ``_ref_ids`` (``block + prompt_ids``). Because it is a token
PREFIX and not a re-tokenization, the SAME ``response_ids`` stay aligned token-for-token under the
hint-conditioned context, which is what makes the reverse-KL well defined.
``SelfDistillationTrainer`` then reads ``ref_prompt_token_ids`` + ``hints`` and runs the frozen
teacher-with-hint forward → ``hinted_log_probs`` → reward = −distill_coef · k1(student‖teacher).

⚠️ HINT KEY IS THE PLAIN ``task_id``, not τ1's ``{domain}:{task_id}``. ``build_tau2_pi_hints.py:141-143``
emits plain ids deliberately, for this consumer. We still try the τ1-style key first so a hint file
built by the τ1 tooling would not silently miss.

⚠️ A MISSING HINT IS NOT AN ERROR BY DEFAULT, AND THAT IS THE DANGEROUS PART. An absent per-task hint
falls back to a generic static string, which still produces a small nonzero KL — so a run whose hint
file never loaded looks like a weak result rather than a crash. Two guards:
  * ``opsd/task_hints_loaded`` is logged at construction; if it is 0 the artifact did not load (most
    often ``OPSD_HINTS_PATH`` is unset/wrong, or was exported AFTER ``ray start`` on a cluster the
    driver attached to, so the raylet never saw it).
  * ``OPSD_REQUIRE_HINT=1`` turns both a missing hint and an overflow drop into a hard failure.

⚠️ THE OVERFLOW GUARD WILL FIRE AT THE DEFAULT CAP. ``OPSD_REF_SEQ_CAP`` defaults to
``max_input_length + max_generate_length`` (39,000 on the paper recipe), but τ²'s hard bounds are
``MAX_PROMPT_LENGTH`` 6,144 + ``tau2.max_response_tokens`` 33,000 = 39,144 BEFORE the hint. Measured max
hint sizes over the 424-task pool are 642 / 9,101 / 12,547 tokens for actions / raw_no_think / raw, so
every format can overflow in the worst case. Set ``OPSD_REF_SEQ_CAP=52000`` — see
``scripts/analysis/build_tau2_pi_pool.py``, which re-checks this at build time.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from loguru import logger

from opsd.pi.artifact import load_task_hints
from opsd.tau2_bench.generator import Tau2Generator
from opsd.trainer.main_opsd_alfworld import SECRET_TEMPLATE

# Only reached when a task has no entry in the artifact. Deliberately generic: it must not encode
# domain knowledge, or a "no hints loaded" run would score misleadingly well.
_DEFAULT_STATIC_HINT = (
    "Resolve the customer's request efficiently: gather the needed identifiers first, then call the "
    "correct tools with valid arguments; confirm before any irreversible action."
)


class Tau2OPSDGenerator(Tau2Generator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        hints_path = os.environ.get("OPSD_HINTS_PATH", "").strip() or None
        self._hints_by_task = load_task_hints(hints_path)
        self._static_hint = os.environ.get("OPSD_STATIC_HINT", "").strip() or _DEFAULT_STATIC_HINT
        self._require_hint = os.environ.get("OPSD_REQUIRE_HINT", "").strip() in ("1", "true", "True")
        # Run-cumulative drop accounting (see _build_ref_prompts). Lives on the generator actor, which
        # is created once per run, so it accumulates for the whole run rather than per step.
        self._dropped_total = 0
        self._rollouts_total = 0

        try:
            default_cap = int(self.generator_cfg.max_input_length) + int(
                self.generator_cfg.sampling_params.max_generate_length
            )
        except Exception:  # noqa: BLE001 — a hand-built cfg stub in tests
            default_cap = 39000
        self._seq_cap = int(os.environ.get("OPSD_REF_SEQ_CAP", "") or default_cap)

        logger.info(
            f"Tau2OPSD generator | hints_path={hints_path} | per-task hints: "
            f"{len(self._hints_by_task)} | static fallback {len(self._static_hint)} chars | "
            f"seq_cap={self._seq_cap} | require_hint={self._require_hint}"
        )
        if not self._hints_by_task:
            msg = (
                "Tau2OPSD: ZERO per-task hints loaded -- every rollout will use the generic static "
                "hint and this run will not measure PI. Usual cause: OPSD_HINTS_PATH unset/wrong, or exported "
                "AFTER `ray start`, so the raylet snapshot never saw it."
            )
            if self._require_hint:
                raise RuntimeError(msg + " (OPSD_REQUIRE_HINT=1)")
            logger.error(msg)

    def _hint_for(self, domain: Any, task_id: Any) -> str:
        """Plain ``task_id`` is τ²'s key; the τ1-style ``domain:task_id`` is tried first for compat."""
        for key in (f"{domain}:{task_id}", str(task_id)):
            if key in self._hints_by_task:
                return self._hints_by_task[key]
        return self._static_hint

    def _ref_ids(self, prompt_ids: List[int], hint: str) -> List[int]:
        block = self.tokenizer.encode(SECRET_TEMPLATE.format(hint_lines=hint), add_special_tokens=False)
        return block + list(prompt_ids)

    def _build_ref_prompts(
        self,
        prompts: List[List[int]],
        responses: List[List[int]],
        env_extras: List[Dict[str, Any]],
        enforce: bool = True,
    ) -> Tuple[List[List[int]], List[str], Dict[str, float]]:
        """``enforce=False`` for eval batches: they are disjoint from the training pool by design,
        so a missing hint there is expected, not a fault. See the note in ``generate``."""
        ref_prompt_token_ids: List[List[int]] = []
        hints_used: List[str] = []
        n_hinted = n_dropped = n_static = 0
        hint_tokens = 0
        for i, p_ids in enumerate(prompts):
            extras = env_extras[i] if i < len(env_extras) else {}
            task_id = extras.get("task_id")
            hint = self._hint_for(extras.get("domain"), task_id)
            if hint is self._static_hint:
                n_static += 1
                # OPSD_REQUIRE_HINT NOW COVERS THE MISSING-HINT CASE, WHICH IT NEVER DID.
                # This module's docstring has always claimed the flag "turns both a missing hint and
                # an overflow drop into a hard failure", but only the overflow branch below raised. A
                # task_id absent from the artifact silently got the generic static string and merely
                # incremented opsd/hint_static_fallback -- the most dangerous outcome in this
                # generator, because a static hint still yields a small nonzero KL, so the arm trains,
                # logs healthy metrics, and means nothing on those tasks. It bites the checklist
                # family directly: those artifacts cover 323/352 of retail_tri352, so 8.2% of
                # rollouts would have been silently unhinted with REQUIRE_HINT=1 already set.
                if self._require_hint and enforce:
                    raise RuntimeError(
                        f"Tau2OPSD: no per-task hint for task {task_id!r} (domain "
                        f"{extras.get('domain')!r}); {len(self._hints_by_task)} hints loaded. The "
                        f"generic static fallback would train this rollout as if hinted while logging "
                        f"a healthy nonzero KL (OPSD_REQUIRE_HINT=1). Either restrict the pool "
                        f"manifest to the tasks the artifact covers, or set OPSD_REQUIRE_HINT=0 and "
                        f"report opsd/hint_static_fallback as a treatment dilution."
                    )
            ref = self._ref_ids(p_ids, hint) if hint else list(p_ids)
            if hint and len(ref) + len(responses[i]) > self._seq_cap:
                if self._require_hint and enforce:
                    raise RuntimeError(
                        f"Tau2OPSD: ref sequence for task {task_id} is "
                        f"{len(ref) + len(responses[i])} tokens > OPSD_REF_SEQ_CAP={self._seq_cap}. "
                        f"Refusing to silently drop the hint (OPSD_REQUIRE_HINT=1). Raise the cap or "
                        f"shorten the hints."
                    )
                # Fall back to the unhinted prompt for THIS rollout: k1 collapses to 0, so the sample
                # contributes nothing rather than training on a mismatched teacher context.
                ref, hint = list(p_ids), ""
                n_dropped += 1
            if hint:
                n_hinted += 1
                hint_tokens += len(ref) - len(p_ids)
            ref_prompt_token_ids.append(ref)
            hints_used.append(hint or "")
        n = max(len(prompts), 1)
        # ABSOLUTE COUNTS ALONGSIDE THE FRACTIONS. A dropped hint means that rollout trained with NO
        # privileged information, so the drop volume is a first-class result, not diagnostics: a
        # materially non-zero rate makes the arm a DIFFERENT experiment rather than a noisier one.
        # A fraction alone is hard to act on mid-run -- 0.008 reads as rounding until you multiply by
        # the batch and find it is one rollout every step. `_dropped_total` accumulates over the run so
        # the W&B panel shows cumulative loss without needing to integrate a rate by eye.
        self._dropped_total += n_dropped
        self._rollouts_total += len(prompts)
        metrics = {
            "opsd/hinted": n_hinted / n,
            "opsd/hint_dropped_overflow": n_dropped / n,
            "opsd/hint_dropped_count": float(n_dropped),
            "opsd/hint_dropped_total": float(self._dropped_total),
            "opsd/rollouts": float(len(prompts)),
            "opsd/rollouts_total": float(self._rollouts_total),
            "opsd/hint_static_fallback": n_static / n,
            "opsd/hint_tokens_mean": hint_tokens / max(n_hinted, 1),
            "opsd/task_hints_loaded": float(len(self._hints_by_task)),
        }
        return ref_prompt_token_ids, hints_used, metrics

    async def generate(self, input_batch: Any) -> Any:
        output = await super().generate(input_batch)  # unchanged, UNHINTED student rollout
        env_extras = list(input_batch.get("env_extras") or [])
        # ⚠️ EVAL BATCHES HAVE NO HINTS AND MUST NOT BE ENFORCED. The eval set (retail_base, 114
        # tasks keyed '0','1',...) is DISJOINT from every training pool (retail_N), by design -- 0/114
        # overlap with anlz/summ/chk artifacts. So during eval EVERY task legitimately lacks a hint.
        # The ref prompts are not scored at eval time either, so the whole hint path is inert there.
        # This is not hypothetical: on 2026-09-14 the missing-hint raise added below fired on the very
        # first step-0 eval prompt and killed FIVE arms before step 1, each exiting 0 through a
        # multi-arm launcher's per-arm failure isolation -- so they looked like clean successes with
        # zero steps. Enforce on TRAIN batches only.
        meta = input_batch.get("batch_metadata")
        phase = getattr(meta, "training_phase", None) if meta is not None else None
        is_eval = phase == "eval"
        ref_prompt_token_ids, hints_used, metrics = self._build_ref_prompts(
            output["prompt_token_ids"], output["response_ids"], env_extras, enforce=not is_eval
        )
        if is_eval:
            return output  # leave ref_prompt_token_ids/hints unset; nothing consumes them at eval
        output["ref_prompt_token_ids"] = ref_prompt_token_ids
        output["hints"] = hints_used
        output["rollout_metrics"] = {**(output.get("rollout_metrics") or {}), **metrics}
        logger.info(
            f"Tau2OPSD: {metrics['opsd/hinted']:.1%} of {len(ref_prompt_token_ids)} rollouts hinted "
            f"(mean {metrics['opsd/hint_tokens_mean']:.0f} hint tokens, "
            f"{metrics['opsd/hint_dropped_count']:.0f} dropped for overflow "
            f"({metrics['opsd/hint_dropped_overflow']:.1%}; "
            f"{metrics['opsd/hint_dropped_total']:.0f} so far this run), "
            f"{metrics['opsd/hint_static_fallback']:.1%} on the static fallback)"
        )
        return output


__all__ = ["Tau2OPSDGenerator"]
