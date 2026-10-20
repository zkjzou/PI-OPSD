"""
The Ray API-key probe embedded in scripts/tau2/lib.sh (``_TAU2_RAY_KEY_PROBE``) must import the forwarder from
``opsd.skyrl_ext.ray_runtime``. It wraps that import in a try/except, so a stale path would fail SILENTLY: the
probe would then run without the forwarded env and misreport prefixed, marker-free key names as missing.

pytest opsd/skyrl_ext/tests/test_ray_runtime_lib_sh_probe.py
"""

import ast
import importlib
from pathlib import Path

from opsd.skyrl_ext import ray_runtime

LIB_SH = next(
    parent / "scripts" / "tau2" / "lib.sh"
    for parent in Path(__file__).resolve().parents
    if (parent / "scripts" / "tau2" / "lib.sh").is_file()
)


def _probe_source() -> str:
    text = LIB_SH.read_text()
    start = text.index("_TAU2_RAY_KEY_PROBE='") + len("_TAU2_RAY_KEY_PROBE='")
    return text[start : text.index("\n'\n", start)]


def test_probe_compiles():
    compile(_probe_source(), str(LIB_SH), "exec")


def test_probe_imports_the_forwarder_from_ray_runtime():
    imports = [
        node
        for node in ast.walk(ast.parse(_probe_source()))
        if isinstance(node, ast.ImportFrom) and any(alias.name == "collect_forwarded_env_vars" for alias in node.names)
    ]
    assert [node.module for node in imports] == ["opsd.skyrl_ext.ray_runtime"]
    module = importlib.import_module(imports[0].module)
    assert module.collect_forwarded_env_vars is ray_runtime.collect_forwarded_env_vars
