"""`reward_scale` on the ALFWorld env: default identity + the sign invariant it exists to protect.

Source-level assertions rather than a live env, matching test_alfworld_prompt.py's style here: the env's
__init__ opens a connection to an env-server pool, which is not available in CI.
"""

from pathlib import Path

import opsd.envs.alfworld.config
import opsd.envs.alfworld.env

# Located through the imported modules, so the test follows the package wherever it is installed.
ENV = Path(opsd.envs.alfworld.env.__file__)
CFG = Path(opsd.envs.alfworld.config.__file__)


def test_default_is_one_so_existing_arms_are_byte_identical():
    assert "reward_scale: float = 1.0" in CFG.read_text()


def test_scale_is_not_coerced_by_or():
    """`float(_cfg_get(...) or 1.0)` would rewrite an explicit 0.0 into 1.0 -- a silently wrong run."""
    src = ENV.read_text()
    assert '_cfg_get(cfg, "reward_scale", 1.0)' in src
    assert 'float(_cfg_get(cfg, "reward_scale", 1.0) or 1.0)' not in src, "the `or` coercion is back"
    assert "reward_scale must be > 0" in src, "a non-positive scale must raise, not train"


def test_every_reward_return_is_scaled():
    """All THREE terms (win bonus, invalid penalty, format bonus) must go through _r().

    Scaling only the win bonus breaks the OPSD success label, which reads `sum(env_reward) > 0` and is
    exact only while a win (10.0) outweighs up to 30 invalid actions (-0.1 each).
    """
    src = ENV.read_text()
    assert "_r = lambda raw: raw * self.reward_scale" in src
    for expected in (
        "reward=_r(-0.1 + fmt_reward), done=True",  # out of turns on an invalid action
        "reward=_r(-0.1 + fmt_reward), done=False",  # ordinary invalid action
        "reward=_r(fmt_reward),",  # /step failed
        "reward = _r(10.0 * float(self._won) + fmt_reward)",  # the terminal win/loss
    ):
        assert expected in src, f"unscaled reward return: {expected}"
    # No bare reward return may survive.
    assert "reward=-0.1 + fmt_reward" not in src
    assert "reward = 10.0 * float(self._won) + fmt_reward" not in src


def test_sign_invariant_holds_under_uniform_scaling():
    """A win stays positive and a loss non-positive at any scale -- the property `sum(r) > 0` relies on."""
    for scale in (1.0, 0.1, 0.01, 7.3):
        for n_invalid in range(0, 31):  # max_turns = 30
            win = scale * (10.0 - 0.1 * n_invalid)
            loss = scale * (-0.1 * n_invalid)
            assert win > 0, f"win misread as loss at scale={scale}, n_invalid={n_invalid}"
            assert loss <= 0, f"loss misread as win at scale={scale}, n_invalid={n_invalid}"


def test_win_bonus_only_rescale_would_break_the_label():
    """The counter-case the uniform design rejects: win -> 1.0 with the penalty left at -0.1."""
    broken = [n for n in range(31) if 1.0 - 0.1 * n <= 0]
    assert broken, "expected wins to go non-positive once >= 10 invalid actions accumulate"
    assert min(broken) == 10
