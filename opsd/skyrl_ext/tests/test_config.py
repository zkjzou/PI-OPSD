"""OPSDTrainConfig: SkyRL's config tree plus OPSD's fields, parsed by upstream's own ``from_cli_overrides``.

Every recipe arm is parsed by the recipe tests in ``opsd/recipes/<bench>/tests/``, which feed each arm's dry-run
argv to its entry module's ``CONFIG_CLS``; this file pins the tree's shape and the parser behaviour it relies on.
"""

import __future__

import dataclasses
import json
import os
import pickle
import subprocess
import sys
from dataclasses import dataclass, field

import pytest
from omegaconf import OmegaConf

import opsd.envs.alfworld.config as alfworld_config
import opsd.envs.appworld.config as appworld_config
import opsd.skyrl_ext.config as opsd_config
from opsd.envs.alfworld.config import AlfworldEnvConfig
from opsd.envs.appworld.config import AppWorldEnvConfig
from opsd.skyrl_ext.config import (
    OPSDAlgorithmConfig,
    OPSDEnvironmentConfig,
    OPSDSkyRLGymConfig,
    OPSDTrainConfig,
    OPSDTrainerConfig,
    OverlongBufferConfig,
)
from skyrl.train.config import (
    AlgorithmConfig,
    BaseConfig,
    EnvironmentConfig,
    SkyRLGymConfig,
    SkyRLTrainConfig,
    TrainerConfig,
)
from skyrl.train.config.config import _resolve_class_type  # the parser's own nested-type resolution

ADDED_PREFIXES = (
    "trainer.algorithm.overlong_buffer.",
    "environment.skyrl_gym.alfworld.",
    "environment.skyrl_gym.appworld.",
)
# (OPSD class, upstream base, fields OPSD appends to the base)
TREE = [
    (OPSDTrainConfig, SkyRLTrainConfig, []),
    (OPSDTrainerConfig, TrainerConfig, []),
    (OPSDAlgorithmConfig, AlgorithmConfig, ["overlong_buffer"]),
    (OPSDEnvironmentConfig, EnvironmentConfig, []),
    (OPSDSkyRLGymConfig, SkyRLGymConfig, ["alfworld", "appworld"]),
]


def _flat(obj, prefix=""):
    out = {}
    for f in dataclasses.fields(obj):
        v = getattr(obj, f.name)
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            out.update(_flat(v, prefix + f.name + "."))
        else:
            out[prefix + f.name] = v
    return out


def _names(cls):
    return [f.name for f in dataclasses.fields(cls)]


@pytest.fixture(scope="module")
def default_cfg():
    return OPSDTrainConfig()


# --------------------------------------------------------------------------------------------- tree shape
def test_upstream_config_is_pristine():
    """None of the fork's fields is on SkyRL's own classes: OPSD reaches them only through the subclasses."""
    assert "overlong_buffer" not in _names(AlgorithmConfig)
    assert not {"alfworld", "appworld"} & set(_names(SkyRLGymConfig))
    assert "update_ref_every_n_steps" not in _names(TrainerConfig)
    with pytest.raises(ValueError, match=r"Invalid fields \{'overlong_buffer'\} for AlgorithmConfig"):
        SkyRLTrainConfig.from_cli_overrides(["trainer.algorithm.overlong_buffer.len=1"])
    with pytest.raises(ValueError, match=r"Invalid fields \{'alfworld'\} for SkyRLGymConfig"):
        SkyRLTrainConfig.from_cli_overrides(["environment.skyrl_gym.alfworld.max_turns=1"])


@pytest.mark.parametrize("cls,base,added", TREE, ids=[c.__name__ for c, _, _ in TREE])
def test_subclass_keeps_upstream_fields_in_order(cls, base, added):
    """Every upstream field keeps its name and position; OPSD's fields are appended."""
    assert issubclass(cls, base)
    assert _names(cls) == _names(base) + added


def test_inherited_defaults_are_upstreams(default_cfg):
    """Apart from the added fields, the default OPSD config IS the default SkyRL config, key for key."""
    ours, theirs = _flat(default_cfg), _flat(SkyRLTrainConfig())
    added = {k for k in ours if k.startswith(ADDED_PREFIXES)}
    assert sorted(added) == sorted(set(ours) - set(theirs))
    assert len(added) == 3 + len(_names(AlfworldEnvConfig)) + len(_names(AppWorldEnvConfig))
    assert {k: v for k, v in ours.items() if k not in added} == theirs


def test_default_node_types(default_cfg):
    cfg = default_cfg
    assert type(cfg.trainer) is OPSDTrainerConfig
    assert type(cfg.trainer.algorithm) is OPSDAlgorithmConfig
    assert type(cfg.environment) is OPSDEnvironmentConfig
    assert type(cfg.environment.skyrl_gym) is OPSDSkyRLGymConfig
    assert cfg.trainer.algorithm.overlong_buffer == OverlongBufferConfig()
    assert cfg.environment.skyrl_gym.alfworld == AlfworldEnvConfig()
    assert cfg.environment.skyrl_gym.appworld == AppWorldEnvConfig()


def test_overlong_buffer_defaults_are_off_and_dapo_scaled():
    ob = OverlongBufferConfig()
    assert (ob.len, ob.penalty_factor, ob.reward_range) == (0, 1.0, 2.0)
    assert isinstance(ob, BaseConfig)


def test_update_ref_every_n_steps_is_dropped():
    """Dropped by decision (every arm pinned it to 0); the key is now rejected like any unknown key."""
    assert "update_ref_every_n_steps" not in _names(OPSDTrainerConfig)
    with pytest.raises(ValueError, match=r"Invalid fields \{'update_ref_every_n_steps'\} for OPSDTrainerConfig"):
        OPSDTrainConfig.from_cli_overrides(["trainer.update_ref_every_n_steps=0"])


def _walk_types(cls, seen):
    if cls in seen:
        return
    seen.add(cls)
    for f in dataclasses.fields(cls):
        nested = _resolve_class_type(f.type)
        if nested is not None and dataclasses.is_dataclass(nested):
            _walk_types(nested, seen)


def test_no_string_annotations_anywhere_in_the_tree(default_cfg):
    """PEP 563 guard. A string annotation makes ``build_nested_dataclass`` pass the nested dict through raw, so
    ``cfg.trainer.algorithm.overlong_buffer`` would be a dict and DAPO a silent no-op."""
    for mod in (opsd_config, alfworld_config, appworld_config):
        assert getattr(mod, "annotations", None) is not __future__.annotations, mod.__name__
    seen = set()
    _walk_types(OPSDTrainConfig, seen)
    assert {OPSDTrainerConfig, OPSDAlgorithmConfig, OverlongBufferConfig, AlfworldEnvConfig, AppWorldEnvConfig} <= seen
    for cls in seen:
        for f in dataclasses.fields(cls):
            assert not isinstance(f.type, str), f"{cls.__name__}.{f.name} has a string annotation {f.type!r}"
    # every dataclass-valued default is reachable by the parser through its field type
    for cls in (OPSDTrainConfig, OPSDTrainerConfig, OPSDAlgorithmConfig, OPSDEnvironmentConfig, OPSDSkyRLGymConfig):
        node = cls()
        for f in dataclasses.fields(cls):
            v = getattr(node, f.name)
            if dataclasses.is_dataclass(v):
                assert _resolve_class_type(f.type) is type(v), f"{cls.__name__}.{f.name}"


# ------------------------------------------------------------------------------------------------ parsing
def test_cli_overrides_build_the_opsd_subtrees():
    cfg = OPSDTrainConfig.from_cli_overrides(
        [
            "trainer.algorithm.overlong_buffer.len=7000",
            "trainer.algorithm.overlong_buffer.penalty_factor=0.5",
            "trainer.algorithm.overlong_buffer.reward_range=10.0",
            "environment.env_class=alfworld",
            "environment.skyrl_gym.alfworld.server_urls=['http://127.0.0.1:8001','http://127.0.0.1:8002']",
            "environment.skyrl_gym.alfworld.task_types=[1,2]",
            "environment.skyrl_gym.alfworld.enable_thinking=true",
            "environment.skyrl_gym.alfworld.reward_scale=0.1",
            "environment.skyrl_gym.appworld.server_urls=['http://127.0.0.1:9001']",
            "environment.skyrl_gym.appworld.scaffold=loop",
            "environment.skyrl_gym.appworld.http_timeout=30",
        ]
    )
    ob = cfg.trainer.algorithm.overlong_buffer
    assert type(ob) is OverlongBufferConfig
    assert (ob.len, ob.penalty_factor, ob.reward_range) == (7000, 0.5, 10.0)
    alf, app = cfg.environment.skyrl_gym.alfworld, cfg.environment.skyrl_gym.appworld
    assert type(alf) is AlfworldEnvConfig and type(app) is AppWorldEnvConfig
    assert alf.server_urls == ["http://127.0.0.1:8001", "http://127.0.0.1:8002"]
    assert (alf.task_types, alf.enable_thinking, alf.reward_scale) == ([1, 2], True, 0.1)
    assert (app.server_urls, app.scaffold, app.http_timeout) == (["http://127.0.0.1:9001"], "loop", 30)
    # the stock SkyRLGymGenerator's lookup (getattr(skyrl_gym_cfg, env_class, dict())) finds the typed config
    assert getattr(cfg.environment.skyrl_gym, cfg.environment.env_class, dict()) is alf
    assert getattr(cfg.environment.skyrl_gym, "appworld", dict()) is app


def test_dict_overrides_parse_too():
    """``from_cli_overrides`` also takes a dotted-key dict (``None`` -> null), the way some tests call it."""
    cfg = OPSDTrainConfig.from_cli_overrides(
        {"trainer.algorithm.overlong_buffer.len": 512, "environment.skyrl_gym.appworld.scaffold": "union"}
    )
    assert cfg.trainer.algorithm.overlong_buffer.len == 512
    assert cfg.environment.skyrl_gym.appworld.scaffold == "union"


@pytest.mark.parametrize(
    "override,level",
    [
        ("trainer.algorithm.overlong_buffer.bogus=1", "OverlongBufferConfig"),
        ("environment.skyrl_gym.alfworld.bogus=1", "AlfworldEnvConfig"),
        ("environment.skyrl_gym.appworld.bogus=1", "AppWorldEnvConfig"),
        ("environment.skyrl_gym.bogus=1", "OPSDSkyRLGymConfig"),
        ("trainer.algorithm.bogus=1", "OPSDAlgorithmConfig"),
        ("trainer.bogus=1", "OPSDTrainerConfig"),
        ("trainer.policy.bogus=1", "PolicyConfig"),
        ("bogus.x=1", "OPSDTrainConfig"),
    ],
)
def test_unknown_keys_are_still_rejected(override, level):
    with pytest.raises(ValueError, match=rf"Invalid fields \{{'bogus'\}} for {level}\."):
        OPSDTrainConfig.from_cli_overrides([override])


def test_plus_prefix_is_still_rejected():
    with pytest.raises(ValueError, match="'\\+' prefix"):
        OPSDTrainConfig.from_cli_overrides(["+trainer.algorithm.overlong_buffer.len=1"])


def test_upstream_post_init_still_runs():
    cfg = OPSDTrainConfig.from_cli_overrides(
        [
            "trainer.policy.model.path=org/model",
            "trainer.max_prompt_length=77",
            "generator.sampling_params.temperature=0.6",
        ]
    )
    assert cfg.trainer.ref.model.path == "org/model"  # TrainerConfig.__post_init__
    assert cfg.generator.max_input_length == 77  # SkyRLTrainConfig.__post_init__
    assert cfg.trainer.algorithm.temperature == 0.6
    with pytest.raises(ValueError, match="logprobs_chunk_size"):
        OPSDTrainConfig.from_cli_overrides(["trainer.logprobs_chunk_size=0"])


def test_loss_config_rebuild_keeps_the_subclass():
    """The policy worker rebuilds the algorithm config with ``type(loss_config).from_dict_config(...)`` when a
    loss_fn_config override is given (upstream worker.py); the rebuilt config must keep overlong_buffer."""
    alg = OPSDTrainConfig.from_cli_overrides(["trainer.algorithm.overlong_buffer.len=13000"]).trainer.algorithm
    merged = OmegaConf.merge(OmegaConf.create(dataclasses.asdict(alg)), OmegaConf.create({"eps_clip_low": 0.1}))
    rebuilt = type(alg).from_dict_config(merged)
    assert type(rebuilt) is OPSDAlgorithmConfig and rebuilt.eps_clip_low == 0.1
    assert rebuilt.overlong_buffer == OverlongBufferConfig(len=13000)


@dataclass
class _BenchSection(BaseConfig):
    budget: int = 3


@dataclass
class _BenchConfig(OPSDTrainConfig):
    bench: _BenchSection = field(default_factory=_BenchSection)


def test_a_bench_subclass_inherits_every_opsd_field():
    """The pattern a bench config (e.g. tau2's) uses on top of OPSDTrainConfig: its own section plus all of ours."""
    cfg = _BenchConfig.from_cli_overrides(["bench.budget=9", "trainer.algorithm.overlong_buffer.len=13000"])
    assert type(cfg.bench) is _BenchSection and cfg.bench.budget == 9
    assert type(cfg.trainer.algorithm.overlong_buffer) is OverlongBufferConfig
    assert cfg.trainer.algorithm.overlong_buffer.len == 13000
    assert _names(_BenchConfig) == _names(OPSDTrainConfig) + ["bench"]


# ------------------------------------------------------------------------------------- Ray-worker pickling
def test_classes_pickle_by_reference(default_cfg):
    """``cfg.trainer`` goes to every policy / ref / critic actor; stdlib pickle must find its classes by name."""
    for cls, _, _ in TREE:
        assert cls.__module__ == "opsd.skyrl_ext.config"
    assert OverlongBufferConfig.__module__ == "opsd.skyrl_ext.config"
    blob = pickle.dumps(default_cfg.trainer)
    assert b"opsd.skyrl_ext.config" in blob and b"__main__" not in blob
    back = pickle.loads(blob)
    assert type(back) is OPSDTrainerConfig and back == default_cfg.trainer


_FRESH_PROCESS = r"""
import json, pickle, sys
cfg = pickle.loads(bytes.fromhex(sys.argv[1]))  # imports opsd.skyrl_ext.config by reference, as a Ray worker does
heavy = ["torch", "ray", "vllm", "transformers"]
heavy += ["opsd.envs.registry", "opsd.envs.alfworld.env", "opsd.envs.appworld.env"]
heavy = sorted(m for m in heavy if m in sys.modules)
from skyrl_gym.envs.registration import registry
ob = cfg.algorithm.overlong_buffer
print(json.dumps({
    "type": f"{type(cfg).__module__}.{type(cfg).__qualname__}",
    "overlong": f"{type(ob).__module__}.{type(ob).__qualname__}",
    "len": ob.len,
    "heavy": heavy,
    "registered": sorted(i for i in ("alfworld", "appworld") if i in registry),
    "opsd_file": sys.modules["opsd"].__file__,
}))
"""


def test_import_in_a_fresh_worker_process_is_light(tmp_path):
    """What a GPU worker does with ``cfg.trainer``: unpickle it in a fresh interpreter (other cwd, isolated mode,
    so only the installed packages are on sys.path). That imports the config module and must not pull in the
    training stack, the env code, or register any env."""
    cfg = OPSDTrainConfig.from_cli_overrides(["trainer.algorithm.overlong_buffer.len=7000"])
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run(
        [sys.executable, "-I", "-B", "-c", _FRESH_PROCESS, pickle.dumps(cfg.trainer).hex()],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["type"] == "opsd.skyrl_ext.config.OPSDTrainerConfig"
    assert out["overlong"] == "opsd.skyrl_ext.config.OverlongBufferConfig" and out["len"] == 7000
    assert out["heavy"] == [], f"unpickling cfg.trainer imported {out['heavy']}"
    assert out["registered"] == [], "importing the config tree must not register envs (call register_opsd_envs())"
    assert out["opsd_file"] == opsd_config.__file__.replace("skyrl_ext/config.py", "__init__.py")
