"""ALFWorld-namespace re-export of ``opsd.pi.adapters.skyrl_eval_dump`` (the SkyRL eval-dump loader).

Lets ALFWorld-side code write ``from opsd.alfworld.eval_dump_adapter import
load_skyrl_eval_dump``; the implementation lives only in the ``pi`` adapter.
"""

from opsd.pi.adapters.skyrl_eval_dump import (  # noqa: F401
    adapt_record,
    game_files_from_skyrl_eval_dump,
    load_skyrl_eval_dump,
    record_success,
)

__all__ = [
    "adapt_record",
    "game_files_from_skyrl_eval_dump",
    "load_skyrl_eval_dump",
    "record_success",
]
