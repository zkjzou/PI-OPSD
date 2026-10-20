"""Analyzer prompt templates + output parsing for PI construction.

Three analyzer tasks are used offline:
  - on-policy skill summary (Skill-SD style: success / mistake / golden_workflow)
  - contrastive hint (one positive + one negative trajectory for the same task)
  - checklist (a stronger agent's success vs the advised agent's failure -> a pre-action checklist)

All templates are **leakage-aware**: they ask for an abstracted, transferable *skill/strategy*, not the
literal winning action sequence. The mismatched-PI ablation in ``score_trajectories.py`` is the actual
leakage guard; the scrubber here is a cheap first line of defense.

ENVIRONMENT IS A PARAMETER, NOT A LITERAL. Every system message used to hardcode "the ALFRED/ALFWorld
embodied environment", so running any of these against AppWorld or tau2 told the analyzer it was reading
ALFWorld. The prompts are now ``*_TMPL`` strings with an ``{env}`` slot, and the bare ``SKILL_SYSTEM`` /
``CONTRASTIVE_SYSTEM`` / ``SKILL_UPDATE_SYSTEM`` names are kept as ALFWorld-formatted constants so
existing imports render **byte-identically** (asserted in tests/test_prompt_parity.py). Note the two blurb
lengths are load-bearing for that guarantee: SKILL/SKILL_UPDATE carried the parenthetical gloss, while
CONTRASTIVE carried only the bare name.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------------------------------
# Environment blurbs. `long` includes a parenthetical gloss of the environment; `short` is the bare name.
# --------------------------------------------------------------------------------------------------
ENV_SHORT = {
    "alfworld": "the ALFRED/ALFWorld embodied environment",
    "appworld": "the AppWorld environment",
    "tau2": "the tau2-bench dual-control environment",
}
ENV_LONG = {
    "alfworld": (
        "the ALFRED/ALFWorld embodied environment (a text world where an agent navigates a household "
        "and manipulates objects)"
    ),
    "appworld": (
        "the AppWorld environment (a suite of simulated consumer apps -- email, phone, shopping, music, "
        "file system -- which the agent drives by writing Python that calls documented "
        "`apis.<app>.<endpoint>` functions and finishes by calling `answer(...)`)"
    ),
    "tau2": (
        "the tau2-bench dual-control environment (an agent and a simulated user collaborate over a "
        "customer-service domain, and both can act on the shared database)"
    ),
}


def env_desc(bench: Optional[str] = None, *, long: bool = True) -> str:
    """Environment blurb for an analyzer system message. Unknown/None benchmark -> ALFWorld (the default
    every caller assumed before this was a parameter)."""
    key = (bench or "alfworld").strip().lower()
    table = ENV_LONG if long else ENV_SHORT
    return table.get(key, table["alfworld"])


# --------------------------------------------------------------------------------------------------
# On-policy skill summary (PI source: on-policy trajectory analysis)
# --------------------------------------------------------------------------------------------------
SKILL_SYSTEM_TMPL = (
    "You are an expert analyst of agent trajectories in {env}. You read ONE "
    "completed trajectory and its outcome and distill a TRANSFERABLE skill that would help on "
    "similar tasks. Never reveal or restate the exact winning action sequence verbatim; teach the "
    "general rule, not the literal answer. Respond with STRICT JSON only."
)


def skill_system(bench: Optional[str] = None) -> str:
    return SKILL_SYSTEM_TMPL.format(env=env_desc(bench, long=True))


SKILL_SYSTEM = skill_system("alfworld")

SKILL_USER = """Task: {task}
Outcome: {outcome}

Trajectory (observations and the agent's actions; <think> stripped):
{trajectory}

Return STRICT JSON with exactly these keys:
{{
  "success_analysis": "what general strategy worked or would work (1-2 sentences)",
  "mistake_analysis": "the key mistake(s) and their cause, or \\"\\" if none (1-2 sentences)",
  "golden_workflow": ["ordered list of abstract steps (verb + target type), no literal coordinates/answers"]
}}"""

# --------------------------------------------------------------------------------------------------
# AppWorld variant of the skill analyzer.
#
# SKILL_SYSTEM above is ALFWorld-specific ("navigates a household and manipulates objects"). Handing it
# an AppWorld trajectory -- Python executed against REST APIs -- makes the analyst reason about a
# household simulator while reading `apis.spotify.login(...)`. It does not error; it produces
# confidently wrong skills. So each env supplies its own wording.
#
# The AppWorld facts an analyst needs, none of them inferable from the ALFWorld framing: passwords come
# from the supervisor app and each app is logged into separately; list endpoints paginate; APIs are
# discovered via api_docs rather than known a priori; the episode only counts if
# `apis.supervisor.complete_task()` is called; and grading is hidden unit tests over the FINAL DATABASE
# STATE, so acting on too MANY entities fails exactly as hard as too few. That last point is the most
# common real failure we measured (qwen3.7 commented on 29 venmo payments when 12 qualified), which is
# why mistake_analysis explicitly steers toward scoping errors over surface details.
SKILL_SYSTEM_APPWORLD = (
    "You are an expert analyst of agent trajectories in AppWorld, where an agent completes everyday "
    "digital tasks by writing Python that calls REST-style app APIs (`apis.<app>.<method>(...)`) across "
    "apps such as spotify, venmo, gmail, phone, simple_note, amazon and splitwise. Key mechanics: "
    "account passwords come from the supervisor app and each app must be logged into separately for an "
    "access token; list endpoints are paginated; unfamiliar APIs are discovered with "
    "`apis.api_docs.show_api_descriptions` / `show_api_doc`; and the episode only counts if the agent "
    "calls `apis.supervisor.complete_task()`. Grading is hidden unit tests over the FINAL DATABASE "
    "STATE, so acting on too MANY entities fails just as hard as acting on too few -- correctly scoping "
    "'which items qualify' is usually the crux. You read ONE completed trajectory and its outcome and "
    "distill a TRANSFERABLE skill for similar tasks. Never restate literal values (emails, passwords, "
    "record ids, exact answers); teach the general rule, not the specific solution. STRICT JSON only."
)

SKILL_USER_APPWORLD = """Task: {task}
Outcome: {outcome}

Trajectory (the agent's reasoning, its Python actions, and the returned output):
{trajectory}

Return STRICT JSON with exactly these keys:
{{
  "success_analysis": "the general strategy that worked or would work (1-2 sentences)",
  "mistake_analysis": "the key mistake(s) and cause, or \\"\\" if none. Prefer naming a SCOPING error (acted on too many/few records, wrong date window, wrong relation) or a missing complete_task over surface details (1-2 sentences)",
  "golden_workflow": ["ordered abstract steps (verb + target type), e.g. 'resolve credentials for the target app', 'paginate the list endpoint until exhausted', 'filter to the qualifying subset', 'apply the mutation', 'call complete_task' -- no literal ids/emails/answers"]
}}"""

SKILL_SYSTEM_TAU2 = (
    "You are an expert analyst of agent trajectories in tau2-bench (retail/airline), where an agent "
    "acts as a customer-service representative over a MULTI-TURN CONVERSATION with a user, calling "
    "tools such as `find_user_id_by_email`, `get_user_details`, `get_order_details`, "
    "`exchange_delivered_order_items`, `modify_pending_order_items` and `transfer_to_human_agents`. "
    "Key mechanics: the agent MUST authenticate the user (email, or name + zip) before touching any "
    "record; a written DOMAIN POLICY constrains what is permissible and the agent must refuse or "
    "escalate when the request falls outside it; the agent must CONFIRM the details with the user "
    "before any irreversible mutation; and it is DUAL-CONTROL -- the user holds information and "
    "actions the agent does not, so the agent has to ELICIT what it needs rather than assume it. "
    "Grading combines the FINAL DATABASE STATE against expected write-actions with checks that the "
    "required information was actually communicated, so a correct mutation performed without "
    "authenticating, confirming, or telling the user still fails. You read ONE completed trajectory "
    "and its outcome and distill a TRANSFERABLE skill for similar tasks. Never restate literal values "
    "(emails, user ids, order ids, item ids, prices, addresses); teach the general rule, not the "
    "specific solution. STRICT JSON only."
)

SKILL_USER_TAU2 = """Task (the user\'s opening request): {task}
Outcome: {outcome}

Trajectory (the conversation: agent reasoning, tool calls, user and tool responses):
{trajectory}

Return STRICT JSON with exactly these keys:
{{
  "success_analysis": "the general strategy that worked or would work (1-2 sentences)",
  "mistake_analysis": "the key mistake(s) and cause, or \\"\\" if none. Prefer naming a POLICY or PROTOCOL error (mutated before authenticating, skipped user confirmation, acted outside the domain policy instead of refusing/escalating, assumed information instead of eliciting it, wrong order/item scope) over surface details (1-2 sentences)",
  "golden_workflow": ["ordered abstract steps (verb + target type), e.g. \\"authenticate the user\\", \\"retrieve the account and locate the referenced order\\", \\"check the request against the domain policy\\", \\"elicit the missing parameters from the user\\", \\"confirm the exact change with the user\\", \\"apply the mutation\\", \\"report the outcome back\\" -- no literal ids/emails/prices"]
}}"""

# env -> (system, user). Selected via OPSD_PI_PROMPT_ENV so a run picks its wording without threading
# env_class through every PI constructor.
_SKILL_PROMPTS_BY_ENV = {
    "alfworld": (SKILL_SYSTEM, SKILL_USER),
    "appworld": (SKILL_SYSTEM_APPWORLD, SKILL_USER_APPWORLD),
    "tau2": (SKILL_SYSTEM_TAU2, SKILL_USER_TAU2),
}


def skill_prompts(env_name=None, strict=False):
    """Return (system, user_template) for the skill analyzer; defaults to ALFWorld.

    `env_name` falls back to $OPSD_PI_PROMPT_ENV. An unknown name falls back to ALFWorld rather than
    raising -- a hard failure mid-training is worse than a slightly-off prompt -- but the caller logs
    which pair it selected so a mismatch is visible instead of silent.

    `strict=True` turns that fallback into an error. OFFLINE HINT GENERATION SHOULD ALWAYS PASS IT.
    The lenient default is right for a training run, but for generation it is a silent-wrong-answer
    machine: `--env tau2` before tau2 existed here returned the ALFWorld pair, whose system prompt
    opens "...in the ALFRED/ALFWorld embodied environment", and the run would finish at 100% coverage
    having analysed every tau2 conversation under the wrong environment framing. Same failure class
    as the AppWorld scaffold-demo task map.
    """
    import os

    name = (env_name or os.environ.get("OPSD_PI_PROMPT_ENV", "alfworld") or "alfworld").strip().lower()
    if name not in _SKILL_PROMPTS_BY_ENV:
        if strict:
            raise KeyError(
                f"skill_prompts: unknown env {name!r}; known={sorted(_SKILL_PROMPTS_BY_ENV)}. "
                "Refusing to fall back to ALFWorld wording for an offline generation run."
            )
        return _SKILL_PROMPTS_BY_ENV["alfworld"]
    return _SKILL_PROMPTS_BY_ENV[name]


# --------------------------------------------------------------------------------------------------
# Episode-skill (PI source: ONE episode -> a summary + a single policy-facing skill). Single self-
# contained user prompt (no separate system message); success -> workflow, failure -> avoidance rule.
# --------------------------------------------------------------------------------------------------
EPISODE_SKILL_USER = """Analyze the following agent episode and return ONLY valid JSON.
You need to complete two fields:
1. Write a concise episode_summary.
2. Write one episode_skill that extracts the successful trajectory into workflow: the core
decision rule and action ordering that made this trajectory work. / Write one episode_skill
that extracts the failed trajectory into avoidance rules: the core mistake and warning signs
that agent should avoid.
Important constraints:
- Write episode_skill as one short policy-facing skill, not as retrospective explanation of
the trajectory.
- Return only these top-level fields: episode_summary, episode_skill.
Return format:
{{
"episode_summary": "string",
"episode_skill": "string"
}}
Episode context:
- Task description: {task}
- episode_success: {outcome}
- Interaction trajectory: {trajectory}"""

# --------------------------------------------------------------------------------------------------
# Contrastive hint (PI source: TWO trajectories for the same task, each with its outcome — the pair may
# be success-vs-failure, but also success-vs-success or failure-vs-failure).
# --------------------------------------------------------------------------------------------------
CONTRASTIVE_SYSTEM_TMPL = (
    "You are an expert analyst of agent trajectories in {env}. You "
    "are shown TWO attempts at the SAME task, each labeled with its outcome and reward. Identify the "
    "pivotal behavioral difference between them and write a short, forward-looking hint the agent could "
    "use BEFORE acting next time. If one attempt did better, teach what to imitate from it and what to "
    "avoid from the other; if both share the same outcome, teach the decisive behavior they share (or, if "
    "both failed, the common pitfall to avoid). Teach the transferable rule; do not quote the exact "
    "action sequence verbatim. Respond with STRICT JSON only."
)


def contrastive_system(bench: Optional[str] = None) -> str:
    # NOTE: the SHORT blurb -- this template never carried the parenthetical gloss.
    return CONTRASTIVE_SYSTEM_TMPL.format(env=env_desc(bench, long=False))


CONTRASTIVE_SYSTEM = contrastive_system("alfworld")

CONTRASTIVE_USER = """Task: {task}

Trajectory A (outcome={a_outcome}, reward={a_reward}):
{a_trajectory}

Trajectory B (outcome={b_outcome}, reward={b_reward}):
{b_trajectory}

Return STRICT JSON with exactly these keys:
{{
  "key_difference": "the pivotal behavioral difference between A and B (1 sentence)",
  "preferred_behavior": "the behavior to repeat next time — from the better attempt, or the decisive move they share (1 sentence)",
  "hint": "a forward-looking hint, <= 60 words, abstract (no literal answer)"
}}"""

# --------------------------------------------------------------------------------------------------
# Contrastive REFINE (online): same output keys as CONTRASTIVE_USER, but improve a PRIOR hint in place.
# --------------------------------------------------------------------------------------------------
CONTRASTIVE_REFINE_USER = """Task: {task}

You have already accumulated the following hint for this task (from earlier attempts). Improve it using
the new pair below: keep what still holds, fix what the new evidence contradicts, and stay concise.

Current hint (may be empty):
{prior_hint}

Trajectory A (outcome={a_outcome}, reward={a_reward}):
{a_trajectory}

Trajectory B (outcome={b_outcome}, reward={b_reward}):
{b_trajectory}

Return STRICT JSON with exactly these keys:
{{
  "key_difference": "the pivotal behavioral difference between A and B (1 sentence)",
  "preferred_behavior": "the behavior to repeat next time — from the better attempt, or the decisive move they share (1 sentence)",
  "hint": "the improved forward-looking hint, <= 60 words, abstract (no literal answer)"
}}"""

# --------------------------------------------------------------------------------------------------
# Incremental skill update (online, PI source: prompt-level running skill folded one trajectory at a time)
# --------------------------------------------------------------------------------------------------
SKILL_UPDATE_SYSTEM_TMPL = (
    "You maintain a concise, running skill hint for a task in {env}. You are shown the "
    "CURRENT hint (possibly empty) and exactly ONE new agent trajectory with its outcome. Update the hint "
    "to fold in any new transferable lesson: keep what still holds, add what was newly learned, and drop "
    "what the new evidence contradicts. Teach the general, forward-looking strategy — never restate the "
    "exact winning action sequence verbatim. Respond with STRICT JSON only."
)


def skill_update_system(bench: Optional[str] = None) -> str:
    return SKILL_UPDATE_SYSTEM_TMPL.format(env=env_desc(bench, long=True))


SKILL_UPDATE_SYSTEM = skill_update_system("alfworld")

SKILL_UPDATE_USER = """Task: {task}
Trajectory outcome: {outcome}

Current hint (may be empty):
{prior_hint}

New trajectory (observations and the agent's actions; <think> stripped):
{trajectory}

Return STRICT JSON with exactly these keys:
{{
  "updated_hint": "the improved running hint (<= 80 words, abstract, no literal coordinates/answers); may equal the current hint if nothing new was learned",
  "changed": true or false
}}"""


# --------------------------------------------------------------------------------------------------
# Checklist (PI source: a STRONGER agent's success vs the ADVISED agent's failure on the same task).
#
# Differs from CONTRASTIVE in three ways that matter:
#   1. The pair is CROSS-SOURCE and asymmetric -- A is a frontier reference, B is the student being
#      advised -- so the prompt names the roles instead of calling them "attempt A/B" symmetrically.
#      Getting this wrong yields hints that describe the difference rather than instruct the student.
#   2. The output is a PRE-ACTION checklist of imperatives, not a retrospective one-sentence hint. The
#      injected PI is read before the agent acts, so a retrospective phrasing is unusable.
#   3. Sections are allowed to come back EMPTY. Forcing an item per section is what turns a checklist
#      into padding ("[ ] Be careful"), and padding is indistinguishable from a weak analyzer.
# --------------------------------------------------------------------------------------------------
CHECKLIST_SYSTEM_TMPL = (
    "You are an expert analyst of agent trajectories in {env}. You are shown TWO attempts at the SAME "
    "task: attempt A is by a much STRONGER REFERENCE AGENT and it SUCCEEDED; attempt B is by the "
    "WEAKER AGENT YOU ARE ADVISING and it FAILED. Write the PRE-ACTION CHECKLIST that would have "
    "carried B to A's outcome. The checklist is read by the weaker agent BEFORE it acts, so every item "
    "must be an imperative it can check off in advance -- never a retrospective observation about what "
    "happened. Teach the transferable procedure: never name the specific entity, value, file, or answer "
    "that this particular task happens to require, and never quote an action or line of code verbatim. "
    "An item that could only ever apply to this one task is a FAILED item. Respond with STRICT JSON only."
)


def checklist_system(bench: Optional[str] = None) -> str:
    return CHECKLIST_SYSTEM_TMPL.format(env=env_desc(bench, long=True))


CHECKLIST_USER = """Task: {task}

Attempt A -- STRONGER REFERENCE AGENT, SUCCEEDED:
{a_trajectory}

Attempt B -- THE AGENT YOU ARE ADVISING, FAILED:
{b_trajectory}

Write at most {max_items} items per section. Return an EMPTY LIST for any section the contrast gives you
nothing for -- do not pad, and do not restate the same instruction in more than one section.

Return STRICT JSON with exactly these keys:
{{
  "before_acting": ["imperatives to satisfy before taking the first action"],
  "while_acting": ["imperatives to satisfy at every step"],
  "before_finishing": ["imperatives to satisfy before submitting the final answer"],
  "pitfalls": ["a failure mode visible in attempt B, phrased as a warning sign to watch for"]
}}"""

# WIN-vs-WIN variant, for tasks where the student never fails so there is no failure to contrast.
# On ALFWorld's train1600 pool the base student wins ALL 32 rollouts on 14 tasks; without this they
# would get no hint at all and train hint-less inside an arm that is supposed to cover the pool.
# The pair stays TWO-SIDED on purpose -- a single trajectory gives the analyzer nothing to contrast and
# yields generic advice. Both succeeded, so the question shifts from "what went wrong" to "what does the
# stronger agent do that makes success reliable rather than lucky".
CHECKLIST_WINWIN_SYSTEM_TMPL = (
    "You are an expert analyst of agent trajectories in {env}. You are shown TWO attempts at the SAME "
    "task that BOTH SUCCEEDED: attempt A is by a much STRONGER REFERENCE AGENT, and attempt B is by the "
    "WEAKER AGENT YOU ARE ADVISING. Because both succeeded, do NOT invent failures. Write the PRE-ACTION "
    "CHECKLIST that would make the weaker agent's success RELIABLE rather than incidental -- the "
    "decisive behaviours the two share, plus anything A does that makes it more direct or more robust "
    "(fewer wasted steps, verifying before committing, avoiding a fragile assumption B happened to get "
    "away with). Every item must be an imperative the agent can check off BEFORE acting, never a "
    "retrospective observation. Teach the transferable procedure: never name the specific entity, value, "
    "or answer this particular task requires, and never quote an action verbatim. Respond with STRICT "
    "JSON only."
)


def checklist_winwin_system(bench: Optional[str] = None) -> str:
    return CHECKLIST_WINWIN_SYSTEM_TMPL.format(env=env_desc(bench, long=True))


CHECKLIST_WINWIN_USER = """Task: {task}

Attempt A -- STRONGER REFERENCE AGENT, SUCCEEDED:
{a_trajectory}

Attempt B -- THE AGENT YOU ARE ADVISING, ALSO SUCCEEDED:
{b_trajectory}

Write at most {max_items} items per section. Return an EMPTY LIST for any section the pair gives you
nothing for -- do not pad, and do not restate the same instruction in more than one section. In
particular, leave "pitfalls" empty unless attempt B shows a genuinely fragile step.

Return STRICT JSON with exactly these keys:
{{
  "before_acting": ["imperatives to satisfy before taking the first action"],
  "while_acting": ["imperatives to satisfy at every step"],
  "before_finishing": ["imperatives to satisfy before submitting the final answer"],
  "pitfalls": ["a fragile or wasteful step visible in attempt B, phrased as a warning sign"]
}}"""

CHECKLIST_SECTIONS = (
    ("before_acting", "Before you act"),
    ("while_acting", "At each step"),
    ("before_finishing", "Before you finish"),
    ("pitfalls", "Watch out for"),
)


def checklist_to_hint(parsed: Dict[str, Any], *, max_items: int = 5) -> str:
    """Render the checklist JSON into the injected hint.

    Dedup is GLOBAL across sections, not per-section: the analyzer restating "verify the result" under
    both `while_acting` and `before_finishing` is the most common padding mode, and a per-section dedup
    would keep both copies. Empty sections are dropped rather than rendered as an empty heading.
    """
    blocks: List[str] = []
    seen = set()
    for key, title in CHECKLIST_SECTIONS:
        raw = parsed.get(key) or []
        if isinstance(raw, str):  # analyzer collapsed a one-item list into a bare string
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            continue
        items: List[str] = []
        for entry in raw:
            text = " ".join(str(entry).split()).strip()
            text = text.lstrip("-*[]() ").strip()  # strip any bullet/checkbox the analyzer added itself
            if not text:
                continue
            fingerprint = text.lower().rstrip(".")
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            items.append(text)
            if len(items) >= max_items:
                break
        if items:
            blocks.append(f"## {title}\n" + "\n".join(f"[ ] {i}" for i in items))
    return "\n\n".join(blocks).strip()


def checklist_item_count(hint: str) -> int:
    """Number of rendered checklist items — used by the builders' verification blocks."""
    return hint.count("[ ] ")


def parse_analyzer_json(text: str) -> Optional[Dict[str, Any]]:
    """Parse strict JSON, tolerating ```json fences and surrounding prose. None on failure."""
    if not text:
        return None
    candidate = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace:
            candidate = brace.group(0)
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        return None


def scrub_hint(text: str, *, max_chars: int = 6000) -> str:
    """Cheap leakage defense: strip code fences, collapse whitespace, length-cap."""
    if not text:
        return ""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"[ \t]+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + " ..."
    return text


def skill_to_hint(parsed: Dict[str, Any]) -> str:
    """Render a parsed skill-summary JSON into a single hint string."""
    parts = []
    workflow = parsed.get("golden_workflow") or []
    if isinstance(workflow, list) and workflow:
        parts.append("Recommended approach:\n" + "\n".join(f"- {str(s)}" for s in workflow))
    success = (parsed.get("success_analysis") or "").strip()
    if success:
        parts.append(f"Key skill: {success}")
    mistake = (parsed.get("mistake_analysis") or "").strip()
    if mistake:
        parts.append(f"Avoid: {mistake}")
    return "\n".join(parts).strip()


def episode_to_hint(parsed: Dict[str, Any], *, include_summary: bool = False) -> str:
    """Render the episode-skill JSON into the injected hint. The policy-facing ``episode_skill`` is the
    hint; ``episode_summary`` is context (kept in meta, prepended only if ``include_summary``)."""
    skill = (parsed.get("episode_skill") or "").strip()
    if include_summary:
        summary = (parsed.get("episode_summary") or "").strip()
        if summary and skill:
            return f"Episode summary: {summary}\nSkill: {skill}"
    return skill


def contrastive_to_hint(parsed: Dict[str, Any]) -> str:
    """Render a parsed contrastive JSON into a single hint string."""
    hint = (parsed.get("hint") or "").strip()
    if hint:
        return hint
    # Fallback: synthesize from the structured fields if the model omitted the "hint" key. Accept both
    # the current (key_difference/preferred_behavior) and the legacy (divergence_point/what_positive_did_right) keys.
    preferred = (parsed.get("preferred_behavior") or parsed.get("what_positive_did_right") or "").strip()
    difference = (parsed.get("key_difference") or parsed.get("divergence_point") or "").strip()
    bits = [b for b in (preferred, f"Key difference: {difference}" if difference else "") if b]
    return " ".join(bits).strip()


def skill_update_to_hint(parsed: Dict[str, Any], prior: str = "") -> str:
    """Render a parsed incremental-skill-update JSON: the new running hint (fall back to prior)."""
    updated = (parsed.get("updated_hint") or "").strip()
    return updated or (prior or "").strip()
