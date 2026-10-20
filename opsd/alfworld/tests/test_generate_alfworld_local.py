"""generate_alfworld_local.make_generator builds the stock SkyRLGymGenerator on OPSD's env config + registry.

Upstream ``SkyRLGymConfig`` has no ``alfworld`` field (the generator hands ``getattr(cfg, env_class, {})`` to the
env, so a missing field is a silent empty config), and upstream skyrl_gym does not know the ``alfworld`` id, so
the script must use ``OPSDSkyRLGymConfig`` and call ``register_opsd_envs()`` before building the generator.
No server, model or GPU: the tokenizer and inference client are stubs, and the registration is recorded.
"""

import sys

import opsd.envs.registry as registry_mod
from opsd.alfworld import generate_alfworld_local as gen_local
from opsd.skyrl_ext.config import OPSDSkyRLGymConfig


class _StubTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True, return_dict=False, **kw):
        return [1] * len(messages) + [2] + ([3] if add_generation_prompt else [])

    def decode(self, ids, **kw):
        return " ".join(str(i) for i in ids)


def test_make_generator_uses_the_opsd_env_config_and_registers_the_envs(monkeypatch):
    from skyrl.train.generators.skyrl_gym_generator import SkyRLGymGenerator

    calls = []
    monkeypatch.setattr(registry_mod, "register_opsd_envs", lambda: calls.append("register"))
    monkeypatch.setattr(sys, "argv", ["generate_alfworld_local.py", "--max-turns", "7", "--passcode"])
    args = gen_local.parse_args()
    urls = ["http://127.0.0.1:18100", "http://127.0.0.1:18101"]

    generator, sampling_params = gen_local.make_generator(args, urls, _StubTokenizer(), client=None)

    assert calls == ["register"]
    assert isinstance(generator, SkyRLGymGenerator)
    cfg = generator.skyrl_gym_cfg
    assert type(cfg) is OPSDSkyRLGymConfig
    assert cfg.alfworld.server_urls == urls
    assert cfg.alfworld.max_turns == 7 and cfg.alfworld.passcode_mode is True
    assert cfg.max_env_workers == args.max_env_workers
    assert sampling_params["max_tokens"] == args.max_generate_length
