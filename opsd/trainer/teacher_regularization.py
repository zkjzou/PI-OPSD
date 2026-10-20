"""OPSD hinted-teacher regularization — the SDPO (arXiv:2601.20802) EMA and trust-region teachers.

OPSD's hinted teacher is not a separate model: it is a forward pass through a *chosen worker* on the
with-hint prompt, so `OPSD_TEACHER` has always been a hard binary between the two endpoints

    frozen -> ref worker,    theta_0   (safe, but the teacher never improves)
    live   -> policy worker, theta     (collapses to the hint-blind attractor in our runs)

SDPO reports the same collapse for its unregularized teacher and shows that *both* endpoints are
dominated by a teacher held between them (LCBv6, alpha=0.01, to step 90):

    live q_theta 36.1 (diverges)  <  frozen q_theta_ref 48.8  <  EMA 49.3  <  trust-region 50.6

Both of their stabilizers implement one trust-region constraint, sum_t KL(q || q_theta_ref) <= eps:

    trust-region   q  ∝  exp((1-a) log q_theta_ref  +  a log q_theta)   -- output space, no new weights
    ema            theta' <- (1-a) theta' + a theta                     -- parameter space

Both reduce to `frozen` at a=0 and `live` at a=1, so OPSD_TEACHER_ALPHA is a single continuous knob
spanning the modes OPSD already had.

Knobs
-----
OPSD_TEACHER        frozen | live | ema | trust-region      (default frozen -- unchanged)
OPSD_TEACHER_ALPHA  a in [0,1] for ema / trust-region       (default 0.01, SDPO's rich-feedback value)
OPSD_EMA_INTERVAL   apply the EMA every K steps             (default 1; see `effective_alpha`)
OPSD_EMA_DIR        work dir for the EMA state              (default <export_path>/opsd_ema; use NVMe)

Both new modes REQUIRE a ref model (USE_KL_LOSS=true keeps one alive) and raise if there is none --
they must never degrade silently into `live`, which is the known-catastrophic configuration.
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import torch
from loguru import logger

VALID_MODES = ("frozen", "live", "ema", "trust-region")

# Modes that need theta_0 (i.e. a live ref worker) to be meaningful at all.
NEEDS_REF = ("frozen", "ema", "trust-region")


def resolve_teacher_mode() -> str:
    """Parse OPSD_TEACHER. Raises on an unknown mode rather than falling back to a default."""
    mode = os.environ.get("OPSD_TEACHER", "frozen").strip().lower()
    if mode in ("trust_region", "trustregion", "tr"):
        mode = "trust-region"
    if mode not in VALID_MODES:
        raise ValueError(f"OPSD_TEACHER must be one of {VALID_MODES}, got {mode!r}")
    return mode


def resolve_teacher_alpha() -> float:
    """Parse OPSD_TEACHER_ALPHA (SDPO's `a`). 0 => frozen teacher, 1 => live teacher."""
    alpha = float(os.environ.get("OPSD_TEACHER_ALPHA", "0.01") or 0.0)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"OPSD_TEACHER_ALPHA must be in [0,1], got {alpha}")
    return alpha


def require_ref_model(mode: str, ref_model) -> None:
    """Fail loudly when a ref-dependent mode is requested without a ref model.

    The pre-existing `frozen` path silently degrades to the policy worker when `self.ref_model is
    None` (USE_KL_LOSS=false => no ref model), and that silent degradation is exactly the live-teacher
    collapse. New modes refuse to start instead.
    """
    if mode in ("ema", "trust-region") and ref_model is None:
        raise ValueError(
            f"OPSD_TEACHER={mode} requires a reference model, but none was constructed. "
            "Set USE_KL_LOSS=true (or use_kl_in_reward) so SkyRL builds the ref worker. "
            "Refusing to fall back to the live teacher -- that is the collapse configuration."
        )


def effective_alpha(alpha: float, interval: int) -> float:
    """Per-application rate that reproduces a per-step EMA at `alpha` when applied every `interval`.

    A per-step EMA pulls the teacher a fraction `alpha` of the way to the student each step, so after
    K steps of a (slowly moving) student it has closed 1-(1-a)^K of the gap. Applying that compounded
    rate every K steps therefore tracks the same trajectory at the sampled points, which is what makes
    OPSD_EMA_INTERVAL a pure cost knob rather than a change of schedule.
    """
    interval = max(1, int(interval))
    return 1.0 - (1.0 - float(alpha)) ** interval


def ema_half_life(alpha: float) -> float:
    """Steps for the teacher to close half the gap to a static student. Sanity check vs run length:
    SDPO's alpha=0.01 is ~69 steps against ~80-90-step runs, i.e. deliberately near-frozen."""
    if alpha <= 0.0:
        return float("inf")
    if alpha >= 1.0:
        return 0.0
    import math

    return math.log(0.5) / math.log(1.0 - alpha)


def blend_teacher_log_probs(frozen_lp: torch.Tensor, live_lp: torch.Tensor, alpha: float) -> torch.Tensor:
    """SDPO's explicit trust-region teacher (their eq. 10 / Prop. B.1), at the realized tokens.

        q*(y_t) ∝ exp((1-a) log q_theta_ref(y_t) + a log q_theta(y_t))

    SDPO evaluates this as `torch.lerp(ref_logits, student_logits, a)` over FULL logits and lets the
    softmax normalize. OPSD's workers only return the log-prob of the realized token, so we lerp the
    log-probs directly and omit the per-token log-normalizer

        log Z_t = log sum_v q_theta_ref(v)^(1-a) q_theta(v)^a   <= 0   (Hoelder; =0 iff the two agree)

    Consequences, stated precisely because this is the one place we are not bit-identical to SDPO:
      * Downstream, OPSD only consumes the k1 estimator `student_lp - teacher_lp`, and dropping log Z
        makes that estimator EXACTLY the convex blend of the two KLs OPSD can already compute:
            k1(student || q*) ~= (1-a) * k1(student || frozen) + a * k1(student || live)
        i.e. the same shape as the existing OPSD_MIX_LAMBDA advantage mix, one level lower.
      * The omission is a bounded over-estimate: log Z <= 0, so we report a KL no smaller than the
        true one, by |log Z| (the Renyi divergence between the two teachers, which is small exactly
        when they agree -- early training, and always for small a).
      * It is EXACT at both endpoints (a=0 and a=1 give log Z = 0), so `trust-region` strictly
        generalizes `frozen` and `live` and reproduces them bit-for-bit at the extremes.
    Making this exact would require full [T, V] logits out of both workers; that is a much larger
    change to the forward path for a correction that vanishes in the regime SDPO actually runs in.
    """
    return torch.lerp(frozen_lp, live_lp.to(device=frozen_lp.device, dtype=frozen_lp.dtype), float(alpha))


class EMATeacherSync:
    """Drives theta' <- (1-a) theta' + a theta for the ref worker, which OPSD uses as the teacher.

    SDPO gets this almost for free: verl colocates actor and ref in one process, so their EMA is an
    in-place blend over `zip(teacher.parameters(), actor.parameters())`. SkyRL keeps `policy` and
    `ref` in separate Ray actor groups with no cross-group process group, so the transfer has to go
    through the one policy->ref primitive the framework already has -- `save_hf_model` + `init_model`,
    which is what `RayPPOTrainer.update_ref_with_policy` uses for its per-epoch hard copy. We reuse
    exactly that pair, blending on the driver in between, so this needs no changes under skyrl/.

    Precision: the exported HF checkpoint is bf16, whose relative epsilon (~3.9e-3) is LARGER than a
    typical alpha=0.01 update (~1e-4 relative once the teacher is close to the student), so blending
    in the stored dtype would silently round the EMA to a no-op. SDPO does not hit this because verl
    holds fp32 master weights for the colocated ref. We therefore keep our own fp32 master copy and
    only ever write the native dtype back into the HF checkpoint that `init_model` consumes.

    Cost per application: one full policy export + one ref re-init. Amortize with OPSD_EMA_INTERVAL
    (see `effective_alpha`) and keep OPSD_EMA_DIR on NVMe.
    """

    def __init__(self, dispatch, tokenizer, work_dir: str, alpha: float, interval: int = 1):
        self.dispatch = dispatch
        self.tokenizer = tokenizer
        self.work_dir = work_dir
        self.alpha = float(alpha)
        self.interval = max(1, int(interval))
        self.alpha_eff = effective_alpha(self.alpha, self.interval)
        # HF checkpoint the ref worker is re-initialized from (native dtype, valid index/config).
        self.hf_dir = os.path.join(work_dir, "teacher_hf")
        # fp32 master state -- the authoritative theta'. Never read by the workers.
        self.master_dir = os.path.join(work_dir, "teacher_fp32")
        self._initialized = False
        self.applications = 0

    # -- safetensors helpers ---------------------------------------------------------------------
    @staticmethod
    def _shard_files(d: str) -> List[str]:
        import glob

        return sorted(glob.glob(os.path.join(d, "*.safetensors")))

    @staticmethod
    def _load_shard(path: str) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]]]:
        from safetensors import safe_open
        from safetensors.torch import load_file

        meta = None
        try:
            with safe_open(path, framework="pt") as f:
                meta = f.metadata()
        except Exception:  # noqa: BLE001 — header metadata is optional
            meta = None
        return load_file(path), meta

    @staticmethod
    def _save_shard(tensors: Dict[str, torch.Tensor], path: str, meta: Optional[Dict[str, str]]) -> None:
        from safetensors.torch import save_file

        save_file(tensors, path, metadata=meta or {"format": "pt"})

    # -- lifecycle -------------------------------------------------------------------------------
    def _initialize_from_ref(self) -> None:
        """Snapshot the ref worker's current weights (theta_0) as both the HF checkpoint and the
        fp32 master. Done lazily on first application so a run that never reaches an EMA step pays
        nothing, and so we take theta' from what the ref worker ACTUALLY holds rather than re-deriving
        it from a config path that may point at an HF cache or a stale export."""
        os.makedirs(self.work_dir, exist_ok=True)
        if os.path.isdir(self.hf_dir):
            shutil.rmtree(self.hf_dir, ignore_errors=True)
        self.dispatch.save_hf_model("ref", self.hf_dir, self.tokenizer)

        os.makedirs(self.master_dir, exist_ok=True)
        for shard in self._shard_files(self.hf_dir):
            tensors, _ = self._load_shard(shard)
            master = {k: v.to(torch.float32) for k, v in tensors.items()}
            self._save_shard(master, os.path.join(self.master_dir, os.path.basename(shard)), {"format": "pt"})
            del tensors, master
        self._initialized = True
        logger.info(f"OPSD EMA teacher initialized from theta_0 at {self.hf_dir} (fp32 master {self.master_dir})")

    def should_apply(self, global_step: int) -> bool:
        return self.alpha > 0.0 and int(global_step) > 0 and int(global_step) % self.interval == 0

    def apply(self, global_step: int) -> Dict[str, float]:
        """theta' <- (1-a_eff) theta' + a_eff theta, then re-init the ref worker from theta'."""
        t0 = time.time()
        if not self._initialized:
            self._initialize_from_ref()

        policy_dir = os.path.join(self.work_dir, f"policy_step_{int(global_step)}")
        if os.path.isdir(policy_dir):
            shutil.rmtree(policy_dir, ignore_errors=True)
        self.dispatch.save_hf_model("policy", policy_dir, self.tokenizer)

        # Blend shard by shard. Policy and teacher checkpoints come from the same save path on the
        # same architecture, so shard basenames and key sets line up; assert rather than assume.
        gap_sq, ref_sq = 0.0, 0.0
        for pol_shard in self._shard_files(policy_dir):
            base = os.path.basename(pol_shard)
            master_shard = os.path.join(self.master_dir, base)
            hf_shard = os.path.join(self.hf_dir, base)
            if not os.path.isfile(master_shard):
                raise RuntimeError(f"EMA master shard missing for {base}; teacher state is inconsistent.")

            pol_tensors, _ = self._load_shard(pol_shard)
            master_tensors, master_meta = self._load_shard(master_shard)
            hf_tensors, hf_meta = self._load_shard(hf_shard)
            if set(pol_tensors) != set(master_tensors):
                raise RuntimeError(f"EMA/policy key mismatch in {base}: teacher state is inconsistent.")

            for name, master in master_tensors.items():
                theta = pol_tensors[name].to(torch.float32)
                # diagnostic: how far the teacher lags the student, before this step's blend
                diff = theta - master
                gap_sq += float(torch.sum(diff * diff))
                ref_sq += float(torch.sum(master * master))
                # theta' <- (1-a) theta' + a theta, accumulated in fp32
                master.mul_(1.0 - self.alpha_eff).add_(theta, alpha=self.alpha_eff)
                hf_tensors[name] = master.to(hf_tensors[name].dtype)

            self._save_shard(master_tensors, master_shard, master_meta)
            self._save_shard(hf_tensors, hf_shard, hf_meta)
            del pol_tensors, master_tensors, hf_tensors

        shutil.rmtree(policy_dir, ignore_errors=True)

        # Re-point the ref worker (and therefore the OPSD teacher) at theta'.
        self.dispatch.init_model("ref", self.hf_dir)
        self.applications += 1

        rel_gap = (gap_sq / ref_sq) ** 0.5 if ref_sq > 0 else 0.0
        return {
            "distill/ema_applications": float(self.applications),
            "distill/ema_alpha_eff": self.alpha_eff,
            "distill/ema_teacher_student_rel_l2": rel_gap,
            "distill/ema_sync_seconds": time.time() - t0,
        }
