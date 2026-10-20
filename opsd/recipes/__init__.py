"""Recipe resolver for the OPSD experiments: arm id -> resolved KEY=VALUE environment.

Run ``python -m opsd.recipes --help``; the recipe format is documented in ``resolve.py``.
The resolving function is exported as ``resolve_arm`` so that ``opsd.recipes.resolve`` keeps
naming the module.
"""

from opsd.recipes.resolve import (  # noqa: F401
    ArmId,
    RecipeError,
    Resolved,
    format_env,
    list_benches,
    load_bench,
    parse_arm_id,
)
from opsd.recipes.resolve import resolve as resolve_arm  # noqa: F401

__all__ = [
    "ArmId",
    "RecipeError",
    "Resolved",
    "format_env",
    "list_benches",
    "load_bench",
    "parse_arm_id",
    "resolve_arm",
]
