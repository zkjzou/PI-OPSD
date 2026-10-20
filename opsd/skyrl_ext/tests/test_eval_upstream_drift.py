"""Drift guard for the upstream code opsd.skyrl_ext.evaluate copies, wraps or replaces.

A SkyRL pin bump that touches any of these fails here: re-sync opsd/skyrl_ext/evaluate.py (or
opsd/skyrl_ext/main_generate.py) with the new upstream source (keep the OPSD changes listed in its header), re-run
the fork-parity tests, then update the hash.
"""

import hashlib
import importlib
import inspect

import pytest

# "<module>:<qualname>" -> sha256 of inspect.getsource at SkyRL ecdaa946
PINNED = {
    # copied into opsd.skyrl_ext.evaluate.evaluate
    "skyrl.train.evaluate:evaluate": "642e035e497e4202487bd1603d296152a96213d7a176cb61972e9a4c34d90566",
    # wrapped by opsd.skyrl_ext.evaluate.evaluate_step_wise (pass-through + post-processors)
    "skyrl.train.evaluate:evaluate_step_wise": "8dd9dce7c180c3fba04d5d9f482d60b6b196c4f84e91b8f07af25ada70e39004",
    # replaced by OPSDEvalMixin.eval (an upstream change here would be skipped silently)
    "skyrl.train.trainer:RayPPOTrainer.eval": "0f71daf19a6c88ad16cc1a6dd9edb842b396b4dbcfea6f575dfeef00f9c5a1d4",
    # copied
    "skyrl.train.utils.trainer_utils:calculate_per_dataset_metrics": (
        "37b535fe07f862ff9def00ecd6086b6d9d7af8c8aaaa1be1414bbee3866a7987"
    ),
    "skyrl.train.utils.trainer_utils:dump_per_dataset_eval_results": (
        "d0e6b6eeace218c2fc6b35db54d446aa71020d940991453766481e24d77005ff"
    ),
    # wrapped; its reward > 0 / last-token rule is re-implemented for the fallback
    "skyrl.train.generators.utils:get_metrics_from_generator_output": (
        "60bcd315f001ce859bc919e6c9e4d701853578bc12e3fcf1d0c0ec3eb1423da0"
    ),
    # copied into opsd.skyrl_ext.main_generate (OPSDEvalOnlyEntrypoint.run, eval_entrypoint, main)
    "skyrl.train.entrypoints.main_generate:EvalOnlyEntrypoint.run": (
        "b911700f9d51756aab9c8336152cef2fe810d3ab7e4e504b1752e8cb4658bd3e"
    ),
    "skyrl.train.entrypoints.main_generate:eval_entrypoint._function": (
        "54bfc28c84d0302b68ea926749f9a3e0ca1416deba5e63e446aa0a6294af6b88"
    ),
    "skyrl.train.entrypoints.main_generate:main": "bc1b0464feace7ee2573f0aa9d5fe8b8e3d58cd1d67b9098861c123dd40c86cb",
}


def _resolve(path):
    module, qualname = path.split(":")
    obj = importlib.import_module(module)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


@pytest.mark.parametrize("path", sorted(PINNED))
def test_upstream_source_unchanged(path):
    digest = hashlib.sha256(inspect.getsource(_resolve(path)).encode()).hexdigest()
    assert digest == PINNED[path], f"{path} changed upstream (sha256 {digest}); re-sync the opsd.skyrl_ext copy"
