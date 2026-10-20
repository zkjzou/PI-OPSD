r"""Build SkyRL-format parquet for AppWorld RL training.

Sources tasks from AppWorld's native splits via ``appworld.load_task_ids`` and renders one ReAct
prompt per task with Jinja2 from the task's ``supervisor`` details + ``instruction``, split into chat
messages on the ``USER:``/``ASSISTANT:`` markers. Prompt variants (``--prompt_variant``):

  * ``official`` (default; every paper arm): AppWorld's public ReAct code-agent ``instructions.txt``,
    vendored byte-identical as ``official_react_instructions.txt`` (Apache-2.0).
  * ``loop``: the LOOP react template. It is NOT redistributed here (Apple sample-code license);
    pass your own copy with ``--instructions-path``.
  * ``legacy``: the built-in ``notebooks/minimal_agent.ipynb`` prompt below plus a fence directive.
    Kept only so old runs reproduce (it scores ~0; see the measurements above ``_PROMPT_FILES``).

The policy emits one fenced ```python block per turn, which ``AppWorldEnv`` executes (parsing
contract: ``opsd/envs/appworld/scaffold.py``), and calls ``apis.supervisor.complete_task()`` to
finish.

Prerequisite (one-time; ``scripts/setup/setup_appworld.sh`` does all of it): a SEPARATE AppWorld
venv (``appworld==0.1.3.post1`` pins pydantic<2, so it cannot share the SkyRL/vLLM venv), then
``appworld install`` and ``appworld download data`` inside ``$APPWORLD_ROOT``. Run this script with
that venv's python from ``$APPWORLD_ROOT`` (it imports ``appworld``).

All four splits ship ground truth locally (``data/tasks/<id>/ground_truth``), so ``world.evaluate()``
yields a reward on any of them. ``train`` (90) is the RL train set (every paper arm). ``dev`` (57) is the
validation split; train on it only through the ``train_dev`` composite (the launcher's
``TRAIN_INCLUDE_DEV=1``), which is valid only when validation moves off dev (``VAL_ON_TEST=1``).
``test_normal`` (168) and ``test_challenge`` (417) are held-out test splits -- never train on them.

Usage (cwd = ``$APPWORLD_ROOT``; ``$REPO`` = this repository's root):
    $APPWORLD_VENV/bin/python $REPO/opsd/appworld/appworld_dataset.py \
        --local_dir ~/data/appworld --split grpo_recipe
    # one held-out split:
    $APPWORLD_VENV/bin/python $REPO/opsd/appworld/appworld_dataset.py \
        --local_dir ~/data/appworld --split test_normal
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# AppWorld native splits -> output parquet filename. train/dev drive RL; test_* are held-out eval.
_SPLIT_TO_FILE = {
    "train": "train.parquet",
    "dev": "validation.parquet",
    "test_normal": "test_normal.parquet",
    "test_challenge": "test_challenge.parquet",
    # ---- Legacy composites (NOT built by grpo_recipe any more) ----
    # WARNING: train_dev TRAINS ON DEV. dev is the validation set both reference papers report on
    # (Skill-SD evaluates on the 57-task dev set; LOOP validates on dev), and every AppWorld baseline
    # we have measured is a dev number — so training on it makes those comparisons invalid. Kept so
    # older configs still build and for the launcher's TRAIN_INCLUDE_DEV=1 variant, which is only valid
    # with validation moved off dev (VAL_ON_TEST=1); otherwise prefer train.parquet (90) +
    # validation.parquet (dev 57). No paper arm trains on dev.
    # train_dev: train (90) + dev (57) = 147 tasks -> the RL TRAIN set.
    # eval_normal_challenge: test_normal (168, all) + test_challenge (1 task / scenario => 139)
    #   = 307 tasks -> the held-out EVAL set. test_normal is in-distribution held-out; the 139
    #   one-per-scenario test_challenge tasks cover every hard scenario at 1/3 the eval cost.
    "train_dev": "train_dev.parquet",
    "eval_normal_challenge": "eval_normal_challenge.parquet",
    # ---- ALFWorld-style in-domain / OOD split (preferred; built by --split grpo_recipe) ----
    # Mirrors alfworld_dataset.py: the in-domain VALIDATION split (dev -> validation.parquet) keeps the
    # BASE data_source ("appworld", like train) so it overlays with the training metrics instead of
    # doubling the eval namespace. The held-out test splits each get their OWN data_source + bucket
    # (test_normal -> appworld_test_normal, see build_split), and the OOD split lives in its own file so
    # it can be gated off (EVAL_OOD) when you don't want to pay for it. ALFWorld:
    # validation.parquet / validation_unseen.parquet + EVAL_UNSEEN.
    "eval_normal": "eval_normal.parquet",  # test_normal (168) -> eval/appworld_test_normal/*
    "eval_challenge": "eval_challenge.parquet",  # test_challenge 1/scenario (139) -> eval/appworld_test_challenge/*
}

# AppWorld's ReAct prompt, verbatim from notebooks/minimal_agent.ipynb (PROMPT_TEMPLATE). It is a
# Jinja2 template with a few-shot demo; {{ supervisor.* }} and {{ instruction }} are filled per task.
# Roles are marked by leading ``USER:``/``ASSISTANT:`` lines (parsed into chat messages below).
PROMPT_TEMPLATE = """
USER:
I am your supervisor and you are a super intelligent AI Assistant whose job is to achieve my day-to-day tasks completely autonomously.

To do this, you will need to interact with app/s (e.g., spotify, venmo, etc) using their associated APIs on my behalf. For this you will undertake a *multi-step conversation* using a python REPL environment. That is, you will write the python code and the environment will execute it and show you the result, based on which, you will write python code for the next step and so on, until you've achieved the goal. This environment will let you interact with app/s using their associated APIs on my behalf.

Here are three key APIs that you need to know to get more information

# To get a list of apps that are available to you.
print(apis.api_docs.show_app_descriptions())

# To get the list of apis under any app listed above, e.g. supervisor
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))

# To get the specification of a particular api, e.g. supervisor app's show_account_passwords
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

Each code execution will produce an output that you can use in subsequent calls. Using these APIs, you can now generate code, that the environment will execute, to solve the task.

For example, consider the task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

Task:

What is the password for my Spotify account?

ASSISTANT:
# Okay. Lets first find which apps are available to get the password by looking at the app descriptions.
print(apis.api_docs.show_app_descriptions())

USER:
[
  {
    "name": "api_docs",
    "description": "An app to search and explore API documentation."
  },
  {
    "name": "supervisor",
    "description": "An app to access supervisor's personal information, account credentials, addresses, payment cards, and manage the assigned task."
  },
  ...
  {
    "name": "spotify",
    "description": "A music streaming app to stream songs and manage song, album and playlist libraries."
  },
  {
    "name": "venmo",
    "description": "A social payment app to send, receive and request money to and from others."
  },
  ...
]


ASSISTANT:
# Looks like the supervisor app could help me with that. Lets see what apis are available under this app.
print(apis.api_docs.show_api_descriptions(app_name='supervisor'))


USER:
[
  ...
  "show_account_passwords : Show your supervisor's account passwords."
  ...
]


ASSISTANT:
# I can use `show_account_passwords` to get the passwords. Let me see its detailed specification to understand its arguments and output structure.
print(apis.api_docs.show_api_doc(app_name='supervisor', api_name='show_account_passwords'))

USER:
{
  'app_name': 'supervisor',
  'api_name': 'show_account_passwords',
  'path': '/account_passwords',
  'method': 'GET',
  'description': "Show your supervisor's app account passwords.",
  'parameters': [],
  'response_schemas': {
    'success': [{'account_name': 'string', 'password': 'string'}],
    'failure': {'message': 'string'}
  }
}

ASSISTANT:
# Okay, it requires no arguments. So I can just call it directly.
print(apis.supervisor.show_account_passwords())

USER:
[
  {
    "account_name": "spotify",
    "password": "dummy_spotify_pass"
  },
  {
    "account_name": "file_system",
    "password": "dummy_fs_pass"
  },
  ...
]


ASSISTANT:
# So the Spotify password is an entry in the `passwords` list with the account_name=spotify.
spotify_password = [account_password["account_name"] == "spotify" for account_password in passwords][0]["password"]
print(spotify_password)


USER:
dummy_spotify_pass

ASSISTANT:
# When the task is completed, I need to call apis.supervisor.complete_task(). If there is an answer, I need to pass it as an argument `answer`. I will pass the spotify_password as an answer.
apis.supervisor.complete_task(answer=spotify_password)


USER:
Marked the active task complete.


----------------------------------------------

USER:
**Key instructions and disclaimers**:

1. The email addresses, access tokens and variables (e.g. spotify_password) in the example above were only for demonstration. Obtain the correct information by calling relevant APIs yourself.
2. Only generate valid code blocks, i.e., do not put them in ```...``` or add any extra formatting. Any thoughts should be put as code comments.
3. You can use the variables from the previous code blocks in the subsequent code blocks.
4. Write small chunks of code and only one chunk of code in every step. Make sure everything is working correctly before making any irreversible change.
5. The provided Python environment has access to its standard library. But modules and functions that have a risk of affecting the underlying OS, file system or process are disabled. You will get an error if do call them.
6. Any reference to a file system in the task instructions means the file system *app*, operable via given APIs, and not the actual file system the code is running on. So do not write code making calls to os-level modules and functions.
7. To interact with apps, only use the provided APIs, and not the corresponding Python packages. E.g., do NOT use `spotipy` for Spotify. Remember, the environment only has the standard library.
8. The provided API documentation has both the input arguments and the output JSON schemas. All calls to APIs and parsing its outputs must be as per this documentation.
9. For APIs that return results in "pages", make sure to consider all pages.
10. To obtain current date or time, use Python functions like `datetime.now()` or obtain it from the phone app. Do not rely on your existing knowledge of what the current date or time is.
11. For all temporal requests, use proper time boundaries, e.g., if I ask for something that happened yesterday, make sure to consider the time between 00:00:00 and 23:59:59. All requests are concerning a single, default (no) time zone.
12. Any reference to my friends, family or any other person or relation refers to the people in my phone's contacts list.
13. All my personal information, and information about my app account credentials, physical addresses and owned payment cards are stored in the "supervisor" app. You can access them via the APIs provided by the supervisor app.
14. Once you have completed the task, call `apis.supervisor.complete_task()`. If the task asks for some information, return it as the answer argument, i.e. call `apis.supervisor.complete_task(answer=<answer>)`. For tasks that do not require an answer, just skip the answer argument or pass it as None.
15. The answers, when given, should be just entity or number, not full sentences, e.g., `answer=10` for "How many songs are in the Spotify queue?". When an answer is a number, it should be in numbers, not in words, e.g., "10" and not "ten".
16. You can also pass `status="fail"` in the complete_task API if you are sure you cannot solve it and want to exit.
17. You must make all decisions completely autonomously and not ask for any clarifications or confirmations from me or anyone else.

USER:
Using these APIs, now generate code to solve the actual task:

My name is: {{ supervisor.first_name }} {{ supervisor.last_name }}. My personal email is {{ supervisor.email }} and phone number is {{ supervisor.phone_number }}.

Task:

{{ instruction }}
"""

# Fence contract appended to the task turn. AppWorld's native prompt says "no fences, thoughts as
# comments", but Qwen instruct models won't comply — they interleave natural-language prose with code,
# and since the whole turn is exec'd as Python, prose lines throw SyntaxError, wasting every turn (the
# agent loops and never completes → 0 reward). Asking for a single ```python fence lets the model reason
# freely in prose while AppWorldEnv._extract_code runs ONLY the fenced code. Verified: 0 syntax errors,
# clean completion, vs constant failures under the raw contract.
_FENCE_DIRECTIVE = (
    "\n\nIMPORTANT — OUTPUT FORMAT: You may briefly reason in plain text, but emit your action as a "
    "SINGLE fenced Python block:\n```python\n<your code>\n```\nOnly the code inside the fence is "
    "executed (variables persist across turns). Call `apis.supervisor.complete_task(answer=...)` inside "
    "the fence when the task is complete."
)

_ROLE_MARKER_RE = re.compile(r"(USER|ASSISTANT|SYSTEM):\n")

# Skill-Augmented variant (--skills_file): a plain-text block of distilled skills injected into the
# first/instructions turn so the policy sees them before acting. Empty => vanilla (every paper arm).
_SKILLS_BLOCK = ""


# --- prompt variants -------------------------------------------------------------------------
# MEASURED on Qwen3-4B-Thinking-2507, AppWorld dev/57, one identical scaffold (temp 0.6, 0 errors):
#   official  TGC 21.05 / completion 57.39     <- default
#   loop      TGC 10.53 / completion 50.65
#   legacy    TGC  0.00 / completion 30.8      <- the built-in PROMPT_TEMPLATE below
# "legacy" is the minimal_agent.ipynb prompt whose instruction #2 FORBIDS ``` fences while our
# _FENCE_DIRECTIVE demands one; that self-contradiction is why it scores 0. Kept only for repro.
_PROMPT_FILES = {
    "official": "official_react_instructions.txt",
}
# The LOOP react template (apple/ml-loop) is distributed under Apple's sample-code license rather than
# Apache-2.0, so it is not vendored. The variant still renders unchanged from a user-supplied copy.
LOOP_NOT_REDISTRIBUTED = (
    "LOOP prompt is not redistributed (Apple sample-code license); pass --instructions-path "
    "<your copy of the LOOP react template>"
)
_PROMPT_VARIANT = "official"
# --instructions-path: overrides the vendored prompt file for the `official` / `loop` variants.
_INSTRUCTIONS_PATH: Optional[str] = None


def _load_variant_template() -> Optional[str]:
    """Return the raw text of the selected prompt file, or None for the legacy built-in template."""
    if _PROMPT_VARIANT == "legacy":
        return None
    if _INSTRUCTIONS_PATH:
        return Path(_INSTRUCTIONS_PATH).expanduser().read_text()
    if _PROMPT_VARIANT == "loop":
        raise FileNotFoundError(LOOP_NOT_REDISTRIBUTED)
    fname = _PROMPT_FILES.get(_PROMPT_VARIANT)
    if fname is None:
        return None
    return (Path(__file__).parent / fname).read_text()


def render_prompt_messages(task) -> list:
    """Render the ReAct prompt for a task -> list[{"role","content"}].

    Mirrors MinimalReactAgent.prompt_messages (Jinja render + split on role markers), then merges
    consecutive same-role turns (the template ends with several USER turns) so the chat template
    sees a clean user/assistant alternation starting with `user` and ending with the task `user`.
    """
    from jinja2 import Template

    variant_tpl = _load_variant_template()
    if variant_tpl is not None:
        # The official/LOOP templates take `main_user` + `app_descriptions`; LOOP additionally names
        # the task `input_str` and injects `date`. Pass every name so either renders unchanged.
        sup = task.supervisor
        main_user = {
            "first_name": sup["first_name"],
            "last_name": sup["last_name"],
            "email": sup["email"],
            "phone_number": sup["phone_number"],
        }
        app_descriptions = json.dumps(
            [{"name": k, "description": v} for (k, v) in task.app_descriptions.items()], indent=1
        )
        rendered = Template(variant_tpl.lstrip()).render(
            instruction=task.instruction,
            input_str=task.instruction,
            date=str(getattr(task, "datetime", "") or ""),
            main_user=main_user,
            supervisor=sup,
            app_descriptions=app_descriptions,
        )
    else:
        rendered = Template(PROMPT_TEMPLATE.lstrip()).render(supervisor=task.supervisor, instruction=task.instruction)
    messages: list = []
    last_start = 0
    for match in _ROLE_MARKER_RE.finditer(rendered):
        end = match.span()[0]
        if messages:
            messages[-1]["content"] = rendered[last_start:end].strip()
        elif end != 0:
            raise ValueError(f"Prompt start has no role marker: {rendered[:end]!r}")
        messages.append({"role": match.group(1).lower(), "content": None})
        last_start = match.span()[1]
    messages[-1]["content"] = rendered[last_start:].strip()

    merged: list = []
    for msg in messages:
        if merged and merged[-1]["role"] == msg["role"]:
            merged[-1]["content"] += "\n\n" + msg["content"]
        else:
            merged.append(dict(msg))
    # Skill-Augmented: inject the distilled-skills block into the first (instructions) turn,
    # so the policy reads it before the few-shot examples and the task.
    if _SKILLS_BLOCK:
        merged[0]["content"] += "\n\n" + _SKILLS_BLOCK
    # The fence directive exists ONLY to patch the legacy template, whose instruction #2 forbids
    # fences. The official/LOOP prompts already mandate them ("end code blocks with ``` followed by
    # a newline"), so appending it there would just add redundant/conflicting formatting text.
    if variant_tpl is None:
        merged[-1]["content"] += _FENCE_DIRECTIVE
    return merged


def _one_task_per_scenario(task_ids: list) -> list:
    """Keep exactly one task (lowest-numbered variant) per scenario, deterministically.

    AppWorld task_ids are ``<scenario>_<variant>`` (e.g. ``5238afc_1``/``_2``/``_3``). test_challenge
    has 139 scenarios x 3 variants = 417 tasks; picking the ``_1`` variant per scenario yields 139
    representative tasks that cover every scenario at 1/3 the eval cost. Deterministic (reproducible
    eval set) and order-preserving in first-seen scenario order.
    """
    best: dict = {}
    order: list = []
    for task_id in task_ids:
        scenario, _, variant = task_id.rpartition("_")
        if scenario not in best:
            best[scenario] = task_id
            order.append(scenario)
        elif int(variant) < int(best[scenario].rpartition("_")[2]):
            best[scenario] = task_id
    return [best[s] for s in order]


# Tasks whose ground truth is internally inconsistent, so CORRECT agent behaviour scores 0.0.
#
# All three are variants of one train-split scenario (a workout-playlist task) and none appears in
# dev/test_normal/test_challenge, so dropping them cannot move any evaluation number we report.
#
# The defect: the task's ground-truth evaluator compares a duration in SECONDS against a workout
# length in MINUTES, so the only accepted answer is a playlist far too short for the workout. A policy
# that converts units correctly concludes that no playlist suffices and is scored 0.0; these were the
# only 3 of 90 train tasks the reference-trajectory author model could not solve, and resampling never
# helped (the reward requires reproducing the bug). Simply loosening the check would make the task
# reward-hackable (any long playlist passes without reading the workout plan), so the evaluator is not
# edited here.
#
# REPAIR: a patch (``patch_appworld_b0a8eae.py``) fixes the units and the task data so the intended
# answer is satisfiable and still requires reading the plan. Because it is derived from AppWorld's
# protected ground truth, AppWorld's terms only allow redistributing it ENCRYPTED, so it ships only
# inside the encrypted AppWorld data bundle, not in this repo. ``scripts/setup/setup_appworld.sh``
# unpacks the bundle into ``$OPSD_DATA_ROOT/appworld/plain/`` and runs the bundled
#     $APPWORLD_VENV/bin/python patch_appworld_b0a8eae.py --appworld-root "$APPWORLD_ROOT"
# right AFTER ``appworld download data`` (a re-download overwrites the patch) and BEFORE this script.
# Rather than a manual flag, we DETECT the repair below: a patched task is kept, an unpatched one is
# dropped. Without the bundle you therefore get the 87-task train set (every paper arm trained on the
# patched 90), never 90 tasks of which 3 reward the wrong behaviour.
#
# Set APPWORLD_KEEP_UNREWARDABLE=1 to train on them even UNPATCHED (e.g. to ablate their effect).
UNREWARDABLE_TASK_IDS = frozenset({"b0a8eae_1", "b0a8eae_2", "b0a8eae_3"})


def _is_repaired(task_id: str) -> bool:
    """True if patch_appworld_b0a8eae.py has been applied to this task's evaluator."""
    try:
        from appworld.common.path_store import path_store

        p = Path(path_store.data) / "tasks" / task_id / "ground_truth" / "evaluation.py"
        return p.is_file() and "PATCHED" in p.read_text()
    except Exception:
        return False


def build_rows(split: str, data_source: str = "appworld", one_per_scenario: bool = False) -> pd.DataFrame:
    from appworld import load_task_ids
    from appworld.task import Task

    task_ids = list(load_task_ids(split))
    if one_per_scenario:
        task_ids = _one_task_per_scenario(task_ids)
    if os.environ.get("APPWORLD_KEEP_UNREWARDABLE", "").lower() not in ("1", "true", "yes"):
        suspect = [t for t in task_ids if t in UNREWARDABLE_TASK_IDS]
        repaired = [t for t in suspect if _is_repaired(t)]
        dropped = [t for t in suspect if t not in repaired]
        if repaired:
            logger.info(
                "split=%s: keeping %d REPAIRED task(s) %s (patch_appworld_b0a8eae.py applied).",
                split,
                len(repaired),
                repaired,
            )
        if dropped:
            task_ids = [t for t in task_ids if t not in dropped]
            logger.warning(
                "split=%s: dropped %d unrewardable task(s) %s -- broken ground truth (seconds vs "
                "minutes) and NOT repaired; apply patch_appworld_b0a8eae.py from the encrypted AppWorld "
                "data bundle (scripts/setup/setup_appworld.sh) to train on them. %d tasks remain.",
                split,
                len(dropped),
                dropped,
                len(task_ids),
            )
    logger.info(
        "Loaded %d AppWorld task_ids for split=%s (one_per_scenario=%s, data_source=%s)",
        len(task_ids),
        split,
        one_per_scenario,
        data_source,
    )
    rows = []
    for i, task_id in enumerate(task_ids):
        task = Task.load(task_id=task_id)
        scenario, _, variant = task_id.rpartition("_")
        rows.append(
            {
                # data_source drives SkyRL's per-dataset EVAL metric buckets (eval/<data_source>/*).
                # For the eval composite we pass distinct sources per split so test_normal and
                # test_challenge get separate eval buckets; train stays a single "appworld".
                "data_source": data_source,
                # Full AppWorld ReAct few-shot prompt as chat messages; ends with the actual task as
                # the final user turn, so the model generates the first action next.
                "prompt": render_prompt_messages(task),
                "env_class": "appworld",
                # ground_truth carries the task_id; AppWorldEnv opens that world and world.evaluate()
                # supplies the actual reward.
                "reward_spec": {"method": "rule", "ground_truth": task_id},
                "extra_info": {
                    "task_id": task_id,
                    "split": split,
                    "scenario": scenario,
                    "variant": variant,
                    "index": i,
                },
            }
        )
    return pd.DataFrame(rows)


def build_split(split: str) -> pd.DataFrame:
    """Build a DataFrame for a native split OR a GRPO-recipe composite.

    Composites (see ``_SPLIT_TO_FILE``):
      * ``train_dev``             -> train (all) + dev (all), data_source="appworld" (the RL train set).
      * ``eval_normal_challenge`` -> test_normal (all, data_source="appworld_test_normal")
                                     + test_challenge (1 task/scenario, data_source="appworld_test_challenge").
        The per-split data_source makes SkyRL emit separate ``eval/appworld_test_normal/*`` and
        ``eval/appworld_test_challenge/*`` metric buckets (plus the combined ``eval/all/*``).
    """
    if split == "train_dev":
        return pd.concat([build_rows("train"), build_rows("dev")], ignore_index=True)
    if split == "eval_normal_challenge":
        # Legacy merged file (kept so older runs//configs referencing it still build).
        normal = build_rows("test_normal", data_source="appworld_test_normal")
        challenge = build_rows("test_challenge", data_source="appworld_test_challenge", one_per_scenario=True)
        return pd.concat([normal, challenge], ignore_index=True)
    # test_normal is a HELD-OUT TEST set, not the in-domain validation set (dev is — see build order
    # in main()), so it keeps its own bucket. Only train+dev share the base "appworld" data_source,
    # mirroring ALFWorld's train+valid_seen.
    if split == "eval_normal":
        return build_rows("test_normal", data_source="appworld_test_normal")
    if split == "eval_challenge":
        return build_rows("test_challenge", data_source="appworld_test_challenge", one_per_scenario=True)
    return build_rows(split)


def main():
    parser = argparse.ArgumentParser(description="Build SkyRL parquet from AppWorld native splits.")
    parser.add_argument("--local_dir", default="~/data/appworld", help="Output dir for the parquet files.")
    parser.add_argument(
        "--split",
        choices=[
            "train",
            "dev",
            "test_normal",
            "test_challenge",
            "both",
            "train_dev",
            "eval_normal_challenge",
            "eval_normal",
            "eval_challenge",
            "grpo_recipe",
        ],
        default="both",
        help="train->train.parquet, dev->validation.parquet, test_*->test_*.parquet, both->train+dev. "
        "eval_normal->test_normal (168, data_source=appworld_test_normal, held-out test), "
        "eval_challenge->1/scenario test_challenge (139, data_source=appworld_test_challenge, OOD); "
        "grpo_recipe builds train, dev (validation.parquet), eval_normal and eval_challenge. "
        "train_dev->train+dev (147; trains on dev, only valid with validation moved off dev). "
        "eval_normal_challenge is the LEGACY merged 307-task file (kept for older configs).",
    )
    parser.add_argument("--max_rows", type=int, default=None, help="Cap rows per split (debugging).")
    parser.add_argument(
        "--skills_file",
        default=None,
        help="Plain-text skills block to inject into the instructions turn -> builds the "
        "Skill-Augmented dataset. Omit for vanilla (every paper arm).",
    )
    parser.add_argument(
        "--prompt_variant",
        choices=("official", "loop", "legacy"),
        default="official",
        help="ReAct prompt: 'official' = AppWorld instructions.txt (default; TGC 21.05 on 4B-Thinking, "
        "61.9 on GPT-4o), 'loop' = Apple ml-loop react_template (10.53; not redistributed, needs "
        "--instructions-path), 'legacy' = the old built-in minimal_agent prompt whose instruction #2 "
        "contradicts our fence directive (0.00).",
    )
    parser.add_argument(
        "--instructions-path",
        "--instructions_path",
        dest="instructions_path",
        default=None,
        help="Jinja2 prompt template to use instead of the vendored file (official/loop variants). "
        "REQUIRED for --prompt_variant loop.",
    )
    args = parser.parse_args()

    global _PROMPT_VARIANT, _INSTRUCTIONS_PATH
    _PROMPT_VARIANT = args.prompt_variant
    if args.instructions_path:
        if _PROMPT_VARIANT == "legacy":
            parser.error("--instructions-path does not apply to --prompt_variant legacy (built-in template)")
        _INSTRUCTIONS_PATH = args.instructions_path
    if _PROMPT_VARIANT != "legacy":
        # fail fast if the prompt file is missing rather than silently falling back
        try:
            _load_variant_template()
        except FileNotFoundError as e:
            parser.error(str(e))
    logger.info("Prompt variant: %s (template: %s)", _PROMPT_VARIANT, _INSTRUCTIONS_PATH or "vendored")

    if args.skills_file:
        global _SKILLS_BLOCK
        with open(os.path.expanduser(args.skills_file)) as f:
            _SKILLS_BLOCK = f.read().strip()
        logger.info("Injecting %d-char skills block from %s (Skill-Augmented)", len(_SKILLS_BLOCK), args.skills_file)

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    if args.split == "both":
        splits = ["train", "dev"]
    elif args.split == "grpo_recipe":
        # ALFWorld-style role assignment (and what BOTH reference papers do):
        #   train (90)                 -> train.parquet        RL train        [ALFWorld: train]
        #   dev (57)                   -> validation.parquet   in-domain VAL   [ALFWorld: valid_seen]
        #   test_normal (168)          -> eval_normal.parquet  held-out TEST
        #   test_challenge (139, 1/sc) -> eval_challenge.parquet  OOD TEST     [ALFWorld: valid_unseen]
        # dev is the VALIDATION set, never training data: Skill-SD "train on all 90 available tasks
        # and evaluate on the 57-task development set", LOOP validates on dev / tests on test_normal,
        # and every baseline we have measured (4B 0/34.2, 2B 0/20, 4B-Thinking 0/30.8, the 21.05 TGC
        # prompt A/B) is a dev number — training on dev would invalidate all of those comparisons.
        splits = ["train", "dev", "eval_normal", "eval_challenge"]
    else:
        splits = [args.split]
    for split in splits:
        df = build_split(split)
        if args.max_rows is not None:
            df = df.iloc[: args.max_rows]
        out = os.path.join(local_dir, _SPLIT_TO_FILE[split])
        df.to_parquet(out, index=False)
        logger.info("Wrote %d rows -> %s", len(df), out)


if __name__ == "__main__":
    main()
