"""Config dataclass for the ALFWorld env (text-only ``AlfredTWEnv``, HTTP-service pattern).

Attached to ``environment.skyrl_gym.alfworld`` by the OPSD config tree (``opsd/skyrl_ext/config.py``,
a subclass of SkyRL's ``SkyRLGymConfig``). Mirrors ``AppWorldEnvConfig``: the SkyRL env is a thin HTTP
client over a pool of ``alfworld_server.py`` FastAPI servers (one ``AlfredTWEnv`` each, separate venv),
so nothing here imports alfworld/textworld.

Keep this module import-light (dataclasses/typing only): the config tree is unpickled in every Ray
worker. No ``from __future__ import annotations`` either: SkyRL's ``build_nested_dataclass`` needs real
class annotations.

The ``env_type`` / ``task_types`` / ``goal_desc_human_anns_prob`` mirror SDAR's
``configs/config_tw.yaml`` (the AlfredTWEnv base config); the server bakes them into its config.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AlfworldEnvConfig:
    # Pool of `alfworld_server.py` base URLs (one exclusive AlfredTWEnv per server at a time).
    server_urls: List[str] = field(default_factory=list)
    # AlfredTWEnv only (text-only). 'AlfredThorEnv' (vision) is explicitly out of scope.
    env_type: str = "AlfredTWEnv"
    # ALFRED task-type ids: 1 Pick&Place, 2 Examine-in-Light, 3 Clean&Place, 4 Heat&Place,
    # 5 Cool&Place, 6 Pick-Two&Place (matches configs/config_tw.yaml `env.task_types`). The
    # server filters its game files to these; kept here for documentation / dataset parity.
    task_types: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    # Episode cap (SDAR: env.max_steps; GRPO scripts use 30-50). Design baseline = 40.
    max_turns: int = 40
    # Number of most-recent (obs, action) pairs injected into the with-history prompt template.
    # 0 -> always use the no-history template (matches SDAR `env.history_length <= 0`). OPID
    # Qwen3 alfworld script uses history_length=5.
    history_length: int = 5
    seed: int = 0
    # Prompt style: "sdar" (verbatim SDAR templates, default) or "react" (faithful ReAct:
    # Task / <Think></Think> / Action / Observation; intended with a windowed step-wise generator
    # so the rolling chat history is the memory and `history_length` is set to 0).
    prompt_style: str = "sdar"
    # THINKING model? (Qwen3-*-Thinking with chat_template_kwargs.enable_thinking=true.) When True, the
    # action parser requires the committed <action> to appear AFTER a closed </think>: if the model ran
    # out mid-reasoning (no </think>), the rollout has NO valid action (rather than mis-parsing a
    # rehearsed / instruction-quoted <action> or a reasoning line). Leave False for enable_thinking=false
    # models (the template pre-closes <think> and the model emits a bare command). Mirror the generator's
    # generator.chat_template_kwargs.enable_thinking here.
    enable_thinking: bool = False
    # STRICTEST parse (thinking models): require the committed action to be a non-empty <action> tag
    # AFTER the LAST closed </think>. If </think> never closed (unfinished / hit token cap) or there is
    # no <action> after it, the turn has NO valid action (== empty <action></action>). Unlike
    # `enable_thinking` alone, an <action> emitted inside a still-open <think> does NOT count, and there
    # is no bare-line fallback. Opt-in (default False) so existing runs are unaffected.
    strict_action_outside_think: bool = False
    # prompt_style="react" ONLY: ask for <think> </think> explicitly in the init prompt.
    # Needed for NON-reasoning models. A reasoning-native checkpoint gets a think block for free because its
    # chat template prefills one -- Qwen3-4B-Thinking-2507 renders "...<|im_start|>assistant\n<think>\n" -- so
    # every prior ALFWorld arm produced <think> without the prompt ever asking. An Instruct checkpoint renders
    # only "...<|im_start|>assistant\n" (and silently ignores chat_template_kwargs.enable_thinking), while
    # ALFWORLD_REACT_INIT merely says "first reason", naming no tags. The result is prose reasoning and NO
    # tags. Turning this on swaps in an init line that names the tags. Default False keeps every existing arm
    # byte-identical. NB this is purely a PROMPT switch: parser strictness lives in `enable_thinking` /
    # `strict_action_outside_think`, which stay independent. The released ALFWorld 4bt (thinking) arms train
    # with it False; the checkpoint re-score behind six of their reported numbers set it True (see the
    # caveat at `invalid_action_feedback_guide`).
    think_tags_in_prompt: bool = False
    # prompt_style="react" ONLY: prepend a worked <think>/<action> example to the init prompt. IMPLIES
    # think_tags_in_prompt (the demo is incoherent without the instruction naming the tags). Needed when an
    # instruction alone loses to the model's own format prior: Qwen3-4B-Instruct-2507 produced <think> in 0 of
    # 73 turns under think_tags_in_prompt=True while opening 52/73 turns with the special token <tool_call>.
    # The demo is prompt TEXT (as in the original ReAct paper), not synthetic assistant messages, so init()
    # still returns exactly one message and no fake assistant turn can reach the loss mask. Default False.
    oneshot_example: bool = False
    # PER-TURN FORMAT REWARD: added to the turn's reward when the generation contains a well-formed
    # <think>...</think> block with at least `format_reward_min_think_chars` of content inside it. 0.0 (default)
    # = off, so every existing arm's reward is byte-identical.
    #
    # WHY. Nothing in the task reward pays for reasoning (reward = 10*won - 0.1*invalid), so GRPO deletes it:
    # alfworld_grpo_inst2507_4b started at 1171 assistant tokens with prose reasoning on every turn and was at
    # 225 tokens / 100% bare <action> within 3 logged steps. A stronger prompt raises the starting point but
    # cannot change the gradient; this is the term that makes the reasoning channel worth keeping.
    #
    # SIZING. Task success is 10.0 and the invalid-action penalty is 0.1/turn, so at the recommended 0.05 a
    # fully-compliant ~18-turn episode earns ~0.9, i.e. <10% of one success -- enough to beat noise, far too
    # small to prefer formatting over solving. Do NOT raise it toward 1.0: the policy would rather emit tidy
    # think blocks forever than finish the task.
    format_reward_coef: float = 0.0
    # UNIFORM REWARD SCALE, multiplying EVERY per-turn reward term (win bonus, invalid penalty, format
    # bonus) by the same factor. 1.0 (default) = byte-identical to every existing arm. Set 0.1 to put a
    # successful trajectory at 1.0 instead of 10.0.
    #
    # WHY IT IS UNIFORM, AND WHY YOU MUST NOT INSTEAD JUST SET THE WIN BONUS TO 1.0. The OPSD trainer reads
    # the success label as `sum(env_reward) > 0` (main_opsd_alfworld.py:387), which is exact only because a
    # win pays 10.0 against at most 30 * -0.1 of invalid-action penalty, so a win is >= 7.0 and a loss <= 0.
    # Drop the win bonus to 1.0 while leaving the penalty at -0.1 and any win with >= 10 invalid actions
    # goes NEGATIVE and is silently relabelled a loss -- at step 0 this run family averages ~5.8 invalid
    # actions/trajectory, so that tail is populated, not hypothetical. Scaling both together preserves the
    # sign invariant exactly.
    #
    # WHAT ELSE MOVES WITH IT. The scale is only inert where the env reward is multiplied away:
    #   * vanilla OPSD (OPSD_ENV_COEF=0)   -- reward = -distill_coef*k1; env reward zeroed. UNAFFECTED.
    #   * grpo_norm_by_std=true            -- A_GRPO is a z-score; scale cancels. UNAFFECTED.
    # It DOES change these, and they must be re-tuned together:
    #   * grpo_norm_by_std=false (the ALFWorld default) -- A_GRPO = score - group_mean, so |A_GRPO| falls
    #     ~10x. That is the point for the OPSD_MIX_LAMBDA arms (it makes lambda mean what it looks like),
    #     but it also cuts the effective LR on the GRPO term ~10x.
    #   * OPSD_ENV_COEF=1 with use_kl_in_reward=true -- reward = env - distill_coef*k1; rescaling env
    #     rebalances the two terms ~10x in the teacher's favour.
    #   * DAPO overlong buffer -- `reward_range` is in reward units (overlong_buffer.py:64 passes 10.0).
    #   * OFFLINE PI scorers threshold on the literal 10.0: pi/adapters/skyrl_eval_dump.py, pi/adapters/appworld.py.
    #     Artifacts scored under a different scale are not comparable to the existing corpora.
    reward_scale: float = 1.0
    # Minimum characters INSIDE <think>...</think> for the bonus to count. Guards the cheapest hack
    # (`<think></think>`, which is well-formed and free). It does not make the reasoning good -- it only sets
    # a floor on its length; a format reward buys FORMAT, and only task reward can buy useful reasoning.
    format_reward_min_think_chars: int = 30
    # NAME OF THE REASONING TAG the prompt asks for and the format reward pays for. "think" (default)
    # keeps every existing arm byte-identical.
    #
    # WHY THIS IS CONFIGURABLE. Qwen3-4B-Instruct-2507 will not emit <think> from any prompt -- three
    # variants measured EXACTLY 0% (tag instruction; + one-shot demo; + per-turn reminder and a
    # de-contradicted action guide). What it does instead is emit <tool_call> (special token 151657) as the
    # FIRST token of the turn and put the reasoning prose inside it. That is not a real tool call: 0 of 139
    # observed instances were followed by JSON. And it is triggered by us -- the baseline prompt, which names
    # only <action>, produced 0/96, while all three <think>-naming prompts produced 72-95%.
    # Decisively, as the prompt got more insistent about open-content-close discipline the CLOSING rate rose
    # 0% -> 21% -> 59%: the model adopts the structure we ask for and substitutes the tag NAME, because its
    # chat template uses <tool_call> and never <think> (both are in its vocabulary: 151657/151667).
    # So setting this to "tool_call" asks for the tag the model already reaches for. The reasoning is then the
    # model's OWN behaviour being reinforced rather than injected by a chat-template prefill, and the format
    # reward stops being inert -- at ~59% compliance there is real within-group variance for GRPO to act on,
    # where at 0% the bonus is identical across the group and contributes exactly no gradient.
    reasoning_tag: str = "think"
    # HOW the reasoning is delimited: "tag" (default) -> <reasoning_tag>...</reasoning_tag>; "prefix" ->
    # a plain label line, `reasoning_prefix` then the reasoning, terminated by the newline before <action>.
    #
    # WHY PREFIX EXISTS. Every XML-tag variant tried on Qwen3-4B-Instruct-2507 fought its tool-calling prior:
    # asking for <think> produced 0% <think> and 72-95% <tool_call> instead, and asking for <tool_call>
    # produced a well-formed block only 69% of the time -- 30% of turns closed with <tool_call> rather than
    # </tool_call>, with 31/31 of those placing it immediately before <action>, i.e. using it as a SEPARATOR.
    # A plain-text label has no closing token to get wrong and no special-token id to collide with
    # (<tool_call>=151657, <think>=151667), so it cannot trigger the tool-call reflex at all. This is also the
    # original ReAct formulation ("Thought: ... Action: ..."), which is what these prompt templates descend from.
    reasoning_style: str = "tag"
    reasoning_prefix: str = "THOUGHTS:"
    # Whether to list the admissible actions in each prompt. SDAR shows them; faithful ReAct does
    # NOT (the model must recall valid syntax). Honored by both prompt styles.
    show_admissible: bool = True
    # When an action is invalid (unparsable or not an admissible command), tell the model so in the
    # re-prompt instead of silently repeating the same observation. If `show_admissible` is False,
    # the feedback also lists the admissible commands so the model has a recovery path (common in
    # zero-shot agent setups). Default off (SDAR-faithful: silent re-prompt + -0.1 penalty).
    invalid_action_feedback: bool = False
    # Whether the invalid-action feedback repeats the FULL action guide (~700 chars: the 13-command
    # grammar plus the "use only objects that have appeared" rule) after the invalid-action notice.
    # The guide is already in the init turn, which the model still has in context, and early in training
    # an episode has 6-18 invalid actions, so repeating it adds several thousand characters per rollout.
    #
    # Released ALFWorld arms: True for the six trajectory-PI main-table arms of the thinking student
    # (alfworld/main/4bt/{glm,qwen}_{a,ao,aot}), which trained with the guide repeated; False for every
    # other arm. The recipes (opsd/recipes/alfworld/) set it per arm.
    # CAVEAT for those six arms: their reported numbers come from a checkpoint re-score that differs from
    # their training in TWO prompt settings: this one False, and `think_tags_in_prompt` True (the init
    # prompt names the <think> tags, which their training prompt did not). Every other 4bt arm was scored
    # during training, without the tags in the prompt. The re-score overrides are the `rescore:` entries in
    # opsd/recipes/alfworld/arms.yaml.
    invalid_action_feedback_guide: bool = False
    # Whether the invalid-action feedback (above) may append the admissible-command list when
    # `show_admissible` is False. Default True = current behavior (leak the list on error so the
    # model can recover). Set False to EXCLUDE admissible actions entirely — the model gets only the
    # grammar guide on error, never the live admissible set (harder / fully recall-based).
    invalid_action_show_admissible: bool = True
    # Prob of using human-annotated goal language instead of templated goals (config_tw.yaml).
    goal_desc_human_anns_prob: float = 0.0
    # Crash-safe server lease TTL (seconds); 0 -> derived from max_turns in the env.
    lease_ttl: float = 0.0
    # Per-request HTTP client timeout (seconds).
    http_timeout: int = 120
    # ALFWORLD-PASSCODE, a cheatable-verifier variant (see passcode.py). prompt_style="react" only.
    # Completing the task reveals a per-task private key, and the episode is scored ONLY on the key the
    # agent then submits with `submit <key>`. So a PI hint that leaks the key pays off without the task.
    # That is the point: it separates a hint's SHORTCUT value from its PROCEDURAL value, which PI
    # utility cannot. `success` becomes "correct key submitted". `task_completed`, `honest_success` and
    # `cheat_success` keep the two paths apart. Default False = every existing arm byte-identical.
    passcode_mode: bool = False
    # HMAC salt for the per-task key (and its decoy). The env and the offline hint builder derive keys
    # from (salt, task_id) independently, so they MUST agree -- change it only together with the hints.
    passcode_salt: str = "alfworld-passcode-v1"
    # Where a supplied hint goes. "user" (default) = appended to the first user turn, the PI-utility
    # convention. "system" = the hint text verbatim as a leading system turn. The driver passes the
    # OPSD teacher's rendered <Secret information> block, so in-context rollouts see exactly the
    # prompt the frozen teacher scores with.
    hint_placement: str = "user"
