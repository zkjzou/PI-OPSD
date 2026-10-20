"""τ²-specific wiring of DAPO's Soft Overlong Punishment (``tau2_bench/main.py``).

The arithmetic of ``apply_soft_overlong_punishment`` and the trainer-side seams shared by every benchmark
are tested in ``opsd/algorithms/tests/test_dapo_overlong_buffer.py``. τ² is the ONLY one of the three
envs with its own response cap, and copying ALFWorld's/AppWorld's budget formula here is a SILENT no-op --
these tests exist to catch that, and to keep the OPSD trainer sharing the DAPO trainer's penalty rather
than a copy of it.

The source-text checks read the files statically (no τ² simulator, policy or launcher is started). The
launcher is ``scripts/tau2/run.sh`` and the per-objective values live in ``opsd/recipes/tau2/``.
"""

import importlib.util
import re
import types
from pathlib import Path


def _repo_root() -> Path:
    """The repository root (scripts/, opsd/recipes/), located the layout-independent way (``opsd._repo``)."""
    from opsd._repo import repo_root

    return repo_root(__file__)


def _module_file(dotted: str) -> Path:
    """Source file of a module, located through the import system rather than by joining path components,
    so the lookup follows the module wherever the package layout puts it."""
    spec = importlib.util.find_spec(dotted)
    assert spec is not None and spec.origin, f"module {dotted} not found"
    return Path(spec.origin)


REPO = _repo_root()
TAU2_MAIN = _module_file("opsd.tau2_bench.main")
TAU2_OPSD_TRAINER = _module_file("opsd.tau2_bench.distill.trainer")
TAU2_GENERATOR = _module_file("opsd.tau2_bench.generator")
# The τ² launcher that maps the OVERLONG_* knobs onto hydra, and the recipes that set them per objective.
TAU2_RUN_SCRIPT = REPO / "scripts/tau2/run.sh"
TAU2_RECIPES = REPO / "opsd/recipes/tau2"


def _code(path: Path) -> str:
    """Source without full-line comments, so a commented-out call cannot satisfy a check."""
    return "\n".join(ln for ln in path.read_text().splitlines() if not ln.strip().startswith("#"))


def test_tau2_budget_is_max_response_tokens_not_the_other_envs_formula():
    """The budget MUST be tau2.max_response_tokens.

    generator.py already hard-drops anything past max_response_tokens (reward 0 AND loss_mask zeroed),
    so an onset derived from max_input_length + max_generate_length sits ABOVE a cliff that already
    discarded those trajectories: frac_penalized would read 0.000 forever with a correct-looking config.
    """
    code = _code(TAU2_MAIN)
    assert "budget=int(cfg.tau2.max_response_tokens)" in code, "budget must come from tau2.max_response_tokens"
    assert (
        "sampling_params.max_generate_length" not in code
    ), "budget uses the ALFWorld/AppWorld formula -- on τ² that is a silent no-op (see docstring)"


def test_tau2_penalty_onset_is_measured_from_max_response_tokens():
    """Behavioural twin of the check above: 33k cap, 13k buffer -> onset at 20k, linear to the cap, where a
    0/1 reward loses penalty_factor * reward_range / 2 = 0.5 (penalty_factor is defined on DAPO's +/-1 span)."""
    from opsd.tau2_bench.main import apply_tau2_overlong_penalty

    cfg = types.SimpleNamespace(
        tau2=types.SimpleNamespace(max_response_tokens=33000),
        trainer=types.SimpleNamespace(
            algorithm=types.SimpleNamespace(
                overlong_buffer=types.SimpleNamespace(len=13000, penalty_factor=1.0, reward_range=1.0)
            )
        ),
    )
    metrics: dict = {}
    g = {"rewards": [1.0, 1.0, 0.0], "response_ids": [[0] * 20000, [0] * 26500, [0] * 33000]}
    apply_tau2_overlong_penalty(cfg, g, metrics)
    assert [round(float(x), 4) for x in g["rewards"]] == [1.0, 0.75, -0.5]
    assert metrics["overlong_buffer/onset_tokens"] == 20000


def test_stats_go_to_all_metrics_not_rollout_metrics():
    """REGRESSION: ``rollout_metrics`` is consumed and POPPED inside generate(), which runs earlier in the
    same step than postprocess_generator_output -- so writing the penalty stats there is a silent no-op."""
    code = _code(TAU2_MAIN)
    # The penalty lives in apply_tau2_overlong_penalty(cfg, generator_output, metrics) so the OPSD
    # env-anchored trainer can share it; the trainer must hand it self.all_metrics, and it must write there.
    assert (
        "apply_tau2_overlong_penalty(self.cfg, generator_output, self.all_metrics)" in code
    ), "Tau2DAPOTrainer must pass self.all_metrics as the penalty's metrics sink"
    assert "metrics.update(stats)" in code, "penalty stats must go to the metrics sink (self.all_metrics)"
    assert (
        'rollout_metrics"]' not in code and '"rollout_metrics")' not in code
    ), "penalty stats written to rollout_metrics, which is already popped by then"


def test_tau2_opsd_trainer_shares_the_dapo_penalty_not_a_copy():
    """The OPSD env-anchored arms re-apply DAPO's penalty (their env reward is the training reward again).
    They must do so through the SAME helper, never a copied block that can drift from Tau2DAPOTrainer's."""
    code = _code(TAU2_OPSD_TRAINER)
    assert "from opsd.tau2_bench.main import apply_tau2_overlong_penalty" in code
    assert "apply_tau2_overlong_penalty(self.cfg, generator_output, self.all_metrics)" in code
    assert "apply_soft_overlong_punishment" not in code, "copied penalty block; call the shared helper"


def test_tau2_generator_hard_cut_still_exists_so_the_budget_choice_stays_valid():
    """The reason for the τ²-specific budget is a cliff in generator.py. If that cliff ever moves, the
    budget choice above has to be revisited -- so pin it."""
    assert (
        "len(response_ids) > max_response_tokens" in TAU2_GENERATOR.read_text()
    ), "the over-long hard cut moved; re-derive Tau2DAPOTrainer's budget"


def test_tau2_trainer_is_actually_installed_on_the_experiment():
    """A trainer that is never returned by get_trainer is the failure mode this whole file guards."""
    code = TAU2_MAIN.read_text()
    assert "class Tau2DAPOTrainer(OPSDRayPPOTrainer)" in code
    assert "def get_trainer(" in code, "Tau2Exp must override get_trainer or the penalty never runs"
    assert code.index("class Tau2DAPOTrainer") < code.index("def get_trainer("), "definition order"


def test_tau2_reward_range_defaults_to_binary_scale():
    """τ² rewards are binary 0/1, so the default reward_range is 1.0 -- ALFWorld's 10.0 would apply a
    penalty ten times too small relative to a success."""
    assert '"reward_range", 1.0' in _code(TAU2_MAIN), "τ² reward_range default must be 1.0 (binary reward)"


def test_tau2_run_script_exposes_the_overlong_knobs():
    src = _code(TAU2_RUN_SCRIPT)
    for knob in ("OVERLONG_BUFFER_LEN", "OVERLONG_PENALTY_FACTOR", "OVERLONG_REWARD_RANGE"):
        assert knob in src, f"{knob} not exposed by the run script"
    for path in ("len", "penalty_factor", "reward_range"):
        assert re.search(
            rf"trainer\.algorithm\.overlong_buffer\.{path}=", src
        ), f"overlong_buffer.{path} never reaches hydra"


def _recipe_env(path: Path) -> dict:
    return dict(
        ln.split("=", 1) for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    )


def test_tau2_recipes_price_overlong_against_the_tau2_budget():
    """The DAPO-anchored objectives (grpo, opsd_grpo, rlsd, lamsched) run DAPO's penalty with LEN=13000, i.e.
    onset at 33,000 - 13,000 = 20,000 tokens against tau2.max_response_tokens; vanilla OPSD trains no env
    reward, so it carries no penalty (LEN=0). Reward range 1.0 = tau2's binary reward."""
    base = _recipe_env(TAU2_RECIPES / "base.env")
    assert base["MAX_RESPONSE_TOKENS"] == "33000"
    assert base["OVERLONG_PENALTY_FACTOR"] == "1.0" and base["OVERLONG_REWARD_RANGE"] == "1.0"
    for objective in ("grpo", "opsd_grpo", "rlsd", "lamsched"):
        env = _recipe_env(TAU2_RECIPES / "objectives" / f"{objective}.env")
        assert env["OVERLONG_BUFFER_LEN"] == "13000", objective
    assert _recipe_env(TAU2_RECIPES / "objectives" / "opsd.env")["OVERLONG_BUFFER_LEN"] == "0"
