"""Manage and load the exact upstream τ²-bench source tree.

Like the τ1 integration, the upstream package is intentionally **never installed**: we verify an
immutable checkout at a pinned commit, put ``<checkout>/src`` on ``sys.path``, and use only the
original environments, databases, tools, task definitions and reward implementation.  Keeping the
checkout byte-clean is what makes a run auditable — it is also why we never patch upstream files
(``verify_checkout`` enforces a clean tree).

``bootstrap_tau2`` is **order-sensitive**; see the docstring there.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from .constants import TAU2_REPOSITORY, TAU2_REVISION


class SourceCheckoutError(RuntimeError):
    """Raised when the upstream checkout cannot be trusted or loaded."""


class EgressNotAllowedError(RuntimeError):
    """Raised when τ² attempts an LLM call to a host we did not allowlist."""


def _git(args: Sequence[str], cwd: Path | None = None) -> str:
    process = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)
    if process.returncode:
        detail = process.stderr.strip() or process.stdout.strip()
        raise SourceCheckoutError(f"git {' '.join(args)} failed: {detail}")
    return process.stdout.strip()


# A tree exported WITHOUT git metadata (e.g. ``git archive`` of the pinned commit, for a node with no
# git remote) carries this marker instead; its sole content is the pinned revision it was built from.
REVISION_MARKER = ".tau2_revision"


def verify_checkout(source_dir: str | Path) -> Path:
    """Verify that ``source_dir`` is the pinned τ² source.

    Two provenance shapes are accepted, both anchored to ``TAU2_REVISION``:

    * a **git checkout** — HEAD must equal the pin and tracked files must be clean; or
    * an **exported tree** (no ``.git``) — a ``.tau2_revision`` marker must equal the pin.

    Either way ``src/tau2/registry.py`` must exist, so we never import an arbitrary tree.
    """

    source_path = Path(source_dir).expanduser().resolve()

    if (source_path / ".git").exists():
        head = _git(["rev-parse", "HEAD"], cwd=source_path)
        if head != TAU2_REVISION:
            raise SourceCheckoutError(f"tau2-bench checkout is at {head}, expected pinned revision {TAU2_REVISION}")
        tracked_changes = _git(["status", "--porcelain", "--untracked-files=no"], cwd=source_path)
        if tracked_changes:
            raise SourceCheckoutError(f"tau2-bench checkout has modified tracked files: {source_path}")
    else:
        marker = source_path / REVISION_MARKER
        if not marker.is_file():
            raise SourceCheckoutError(
                f"Not a git checkout and no {REVISION_MARKER} marker: {source_path} "
                "(expected a git clone from prepare_data, or an exported tree with the marker)"
            )
        staged_rev = marker.read_text().strip()
        if staged_rev != TAU2_REVISION:
            raise SourceCheckoutError(
                f"Exported tau2-bench tree is revision {staged_rev}, expected pinned {TAU2_REVISION}"
            )

    if not (source_path / "src" / "tau2" / "registry.py").is_file():
        raise SourceCheckoutError(f"Pinned checkout is missing src/tau2/registry.py: {source_path}")
    return source_path


def ensure_checkout(source_dir: str | Path) -> Path:
    """Clone and detach the official repository at the pinned revision, or verify it."""

    source_path = Path(source_dir).expanduser().resolve()
    just_cloned = False
    if not source_path.exists():
        source_path.parent.mkdir(parents=True, exist_ok=True)
        _git(["clone", "--filter=blob:none", "--no-checkout", TAU2_REPOSITORY, str(source_path)])
        just_cloned = True
    elif not (source_path / ".git").exists():
        raise SourceCheckoutError(f"Refusing to replace non-git path: {source_path}")

    current_head = _git(["rev-parse", "HEAD"], cwd=source_path)
    if current_head != TAU2_REVISION:
        # A fresh ``--no-checkout`` clone has an empty working tree that ``git status`` reports as
        # fully deleted; the dirty-guard only protects a user's *modified* checkout.
        if not just_cloned and _git(["status", "--porcelain"], cwd=source_path):
            raise SourceCheckoutError(f"Refusing to change a dirty tau2-bench checkout: {source_path}")
        try:
            _git(["cat-file", "-e", f"{TAU2_REVISION}^{{commit}}"], cwd=source_path)
        except SourceCheckoutError:
            _git(["fetch", "--depth=1", "origin", TAU2_REVISION], cwd=source_path)
        _git(["checkout", "--detach", TAU2_REVISION], cwd=source_path)
    return verify_checkout(source_path)


# --------------------------------------------------------------------------------------
# litellm hardening.  Importing any ``tau2.utils.llm_utils`` consumer mutates litellm
# process-globals; both mutations are actively harmful to us and are fixed here.
# --------------------------------------------------------------------------------------


def _widen_litellm_connection_pool(max_connections: int, request_timeout: float) -> None:
    """Replace τ²'s global 10-connection httpx client.

    ``tau2/utils/llm_utils.py`` sets, at import time::

        litellm.client_session = httpx.Client(limits=httpx.Limits(max_connections=10, ...))

    litellm's OpenAI transport reuses ``litellm.client_session`` for *every* sync call in the
    process (policy, user simulator and NL judge alike), so leaving it in place silently
    serialises any concurrency above 10 and eventually raises ``httpx.PoolTimeout``.
    """

    import httpx
    import litellm

    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_connections)
    litellm.client_session = httpx.Client(limits=limits, timeout=httpx.Timeout(request_timeout))
    # The async client is unused on our sync-orchestrator path; drop the narrow one it installed.
    litellm.aclient_session = None


def _disable_litellm_per_call_event_loop() -> bool:
    """Stop litellm spinning up a fresh event loop after EVERY sync completion. Returns True if patched.

    ``track_llm_api_timing`` wraps every sync provider call and, in its ``finally``, calls
    ``ServiceLogging.service_success_hook``. From a worker thread (where τ²'s orchestrator runs its policy,
    user-simulator and judge calls) there is no event loop, so the hook falls through to
    ``asyncio.run(async_service_success_hook(...))`` -- a brand-new loop created and torn down per call --
    to run a coroutine that only iterates ``litellm.service_callback``. That list is EMPTY here (we
    configure no service-level logging callbacks), so the whole thing is a no-op that still pays
    for a loop per call, from 64 threads at once.

    That is where a λ=0.99 checklist-hint DAPO×OPSD arm died at step 42 (2026-09-24):
    ``Fatal Python error: Segmentation fault`` with the frame in libuv's
    ``uv__signal_event``, stack ``asyncio/runners.py run <- litellm/_service_logger.py:132
    service_success_hook <- logging_utils.py sync_wrapper <- openai.py completion <- source.py guarded``.
    A local stress (64 threads x 400 calls, uvloop policy, SIGCHLD storm) did NOT reproduce it, so the exact
    libuv race is not pinned -- but this removes the code path it happened in from every τ² model call.

    Behaviour-preserving by construction: the fast path fires ONLY when ``litellm.service_callback`` is
    empty, i.e. exactly when the original hook did nothing. If anyone configures a service callback, the
    original hook runs unchanged.
    """

    try:
        from litellm._service_logger import ServiceLogging
    except Exception:  # noqa: BLE001 -- an import-shape change in litellm must never block a run
        return False
    if getattr(ServiceLogging.service_success_hook, "_tau2_no_loop", False):
        return True
    import litellm

    original = ServiceLogging.service_success_hook

    def service_success_hook(self, *args: Any, **kwargs: Any) -> Any:
        if not litellm.service_callback:
            if getattr(self, "mock_testing", False):  # keep litellm's own counters honest
                self.mock_testing_sync_success_hook += 1
            return None
        return original(self, *args, **kwargs)

    service_success_hook._tau2_no_loop = True  # type: ignore[attr-defined]
    ServiceLogging.service_success_hook = service_success_hook  # type: ignore[method-assign]
    return True


def _leg_of_completion_kwargs(kwargs: dict[str, Any]) -> str | None:
    """Which participant a ``litellm.completion`` call belongs to: agent / user / judge.

    Reads the ``metadata={"call_name": ...}`` tag that ``PolicySpec``/``UserSpec``/the judge args
    already attach for cost metering, and reuses metering's mapping so the two never diverge.
    Returns ``None`` when the call is untagged, in which case callers must not guess.
    """

    metadata = kwargs.get("metadata")
    if not isinstance(metadata, dict):
        return None
    call_name = metadata.get("call_name")
    if not call_name:
        return None
    try:
        from .metering import _LEG_BY_CALL_NAME

        return _LEG_BY_CALL_NAME.get(call_name)
    except Exception:  # noqa: BLE001
        return None


def _rebind_direct_importers(original: Any, replacement: Any) -> list[str]:
    """Repoint τ² modules that did ``from litellm import completion`` at our wrapper.

    **This is why the egress guard, the failure-leg attribution and the per-leg call stats were all
    silently inert.**  ``tau2/utils/llm_utils.py`` line 15 is ``from litellm import completion`` — a
    by-value binding — and ``bootstrap_tau2`` imports that module (step 1) *before* installing the
    guard (step 6).  Reassigning ``litellm.completion`` therefore never reached the reference τ²
    actually calls, so:

      * the guard never fail-closed on a non-allowlisted host — it protected nothing;
      * ``exc._tau2_leg`` was never stamped, so ``_classify_exception`` fell back to exception-type
        heuristics and filed **every** litellm error as ``policy_api_error``;
      * ``call_stats`` recorded zero calls.

    Observed live on a 2-node run: five *user-simulator* rate-limit failures (hosted
    ``deepseek-v4-flash`` simulator) reported as ``tau2/infra_policy=5`` with
    ``tau2/infra_simulator=0``.  The unit tests passed throughout because they call the *module
    attribute* the guard does patch — they proved the wrapper works, not that τ² routes through it.

    Scoped to ``tau2*`` modules on purpose: rebinding litellm's own internals could break its retry
    plumbing.  Idempotent (only swaps attributes that are still ``original``), and the same lesson
    ``install_thinking_normalization`` already applies to ``llm_agent.generate``.
    """

    rebound: list[str] = []
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("tau2"):
            continue
        for attr in ("completion", "acompletion"):
            try:
                if getattr(module, attr, None) is original:
                    setattr(module, attr, replacement)
                    rebound.append(f"{name}.{attr}")
            except Exception:  # noqa: BLE001 - never let patching abort bootstrap
                continue
    return rebound


_CALL_STATS: dict[str, dict[str, float]] = {}


def call_stats(reset: bool = False) -> dict[str, dict[str, float]]:
    """Per-leg LLM call volume, latency and observed retries since the last reset.

    Fills a real instrumentation hole: we counted per-leg *failures*
    (``tau2/infra_{policy,simulator,judge}``) but nothing about the **degraded-but-succeeding** case.
    A call that fails twice and succeeds on the third attempt is invisible — it logs nothing, raises
    nothing, and never reaches the failure metrics — yet it adds full request latency inside an
    episode's critical path, which is exactly the shape of an unexplained slow ``generate``.

    Three fields, and it matters which are exact:

    * ``calls``   — EXACT. This wrapper sees every top-level ``litellm.completion``. It is also the
      denominator for any tokens-per-minute claim; lacking it, I estimated the user simulator's load
      twice and was wrong both times.
    * ``seconds`` — EXACT wall time in the call. The usable proxy for client-side retries: litellm
      runs its ``num_retries`` loop INSIDE ``completion``, so a retried call is indistinguishable
      from a slow one to us, but it *is* slower. Watch mean latency, not just failures.
    * ``retries`` — BEST EFFORT, and specifically **gateway-side**: read from the
      ``x-litellm-attempted-retries`` response header when a LiteLLM gateway sets it. Our own
      ``num_retries=3`` (set in PolicySpec/UserSpec) is consumed inside litellm and is NOT reported
      anywhere, so this undercounts. Do not read it as "total retries".
    """

    out = {leg: dict(vals) for leg, vals in _CALL_STATS.items()}
    if reset:
        _CALL_STATS.clear()
    return out


def _record_call(leg: str | None, seconds: float, response: Any) -> None:
    """Accumulate one completed call. Never raises: instrumentation must not break an episode."""

    if leg is None:
        leg = "other"
    bucket = _CALL_STATS.setdefault(leg, {"calls": 0.0, "seconds": 0.0, "retries": 0.0})
    bucket["calls"] += 1
    bucket["seconds"] += seconds
    try:
        headers = (getattr(response, "_hidden_params", None) or {}).get("additional_headers") or {}
        for key in ("x-litellm-attempted-retries", "llm_provider-x-litellm-attempted-retries"):
            if key in headers:
                bucket["retries"] += float(headers[key] or 0)
                break
    except Exception:  # noqa: BLE001
        pass


def install_egress_guard(allowed_base_urls: Iterable[str]) -> None:
    """Fail closed if τ² tries to reach a host we did not allowlist.

    τ² defaults several auxiliary LLMs (NL-assertion judge, reviewers, classifiers) to
    ``gpt-4.1``/``claude-*``.  On a node without outbound access those calls hang or fail *inside*
    the reward function; with egress they would silently bill whatever account the ambient
    credentials belong to, un-metered.  Wrapping ``litellm.completion`` turns any such call into a
    loud, attributable exception.  The allowlist is the set of hosts of the *resolved* policy,
    simulator and judge base URLs.
    """

    import litellm

    allowed = {urlparse(u).netloc for u in allowed_base_urls if u}
    if getattr(litellm.completion, "_tau2_guarded", False):
        return
    inner = litellm.completion

    def guarded(*args: Any, **kwargs: Any) -> Any:
        host = urlparse(kwargs.get("api_base") or "").netloc
        if host not in allowed:
            raise EgressNotAllowedError(
                f"tau2 attempted a non-allowlisted LLM call: model={kwargs.get('model')!r} "
                f"api_base={kwargs.get('api_base')!r} call_name={kwargs.get('call_name')!r}. "
                f"Allowed hosts: {sorted(allowed)}"
            )
        import time as _time

        _t0 = _time.perf_counter()
        try:
            _response = inner(*args, **kwargs)
            _record_call(_leg_of_completion_kwargs(kwargs), _time.perf_counter() - _t0, _response)
            return _response
        except BaseException as exc:
            # Stamp WHICH LEG failed onto the exception before it propagates.
            # All three legs — policy, user simulator, NL judge — are litellm calls that raise the
            # same types (RateLimitError/APIConnectionError/Timeout/APIError), so a classifier
            # downstream can only guess, and used to attribute every one of them to the policy.
            # This wrapper is the one place that still knows the caller, via the metadata we attach
            # in PolicySpec/UserSpec/judge llm_args. Retries happen INSIDE `inner`, so anything
            # reaching here has already exhausted them.
            leg = _leg_of_completion_kwargs(kwargs)
            if leg is not None and getattr(exc, "_tau2_leg", None) is None:
                try:
                    exc._tau2_leg = leg  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001 - some exception types forbid attributes
                    pass
            raise

    guarded._tau2_guarded = True  # type: ignore[attr-defined]
    litellm.completion = guarded
    rebound = _rebind_direct_importers(inner, guarded)
    if rebound:
        print(f"[tau2] egress guard rebound direct importers: {', '.join(rebound)}", flush=True)


def install_thinking_normalization() -> None:
    """Split ``<think>…</think>`` out of the agent's ``content`` ourselves (idempotent).

    We must NOT rely on the engine's ``reasoning_parser`` for this.  On SkyRL's in-training engine the
    parser did not take effect (observed live: ``tau2/think_leak_frac = 0.925``), so ``content`` kept
    the raw ``<think>`` block — and because τ²'s ``flip_roles`` copies ``content`` verbatim into the
    *simulated user's* view, the user was reading the agent's private reasoning.  That silently
    corrupts the environment dynamics: the user gets privileged information it must never see.

    So we normalise every generated assistant message at the τ² boundary:
      * ``content``  -> only the post-``</think>`` remainder (what the user is allowed to read)
      * reasoning    -> stashed on ``raw_data`` as ``reasoning_content``, exactly where our encoder
                        and :func:`install_thinking_preservation` look for it.

    This makes the pipeline correct whether or not the server-side parser is active (if it *is*
    active, ``content`` has no ``</think>`` and this is a no-op), and keeps the preserved-thinking
    training path intact.
    """

    from tau2.utils import llm_utils

    if getattr(llm_utils.generate, "_tau2_normalize_thinking", False):
        return
    inner = llm_utils.generate

    def normalizing(*args: Any, **kwargs: Any) -> Any:
        message = inner(*args, **kwargs)
        content = getattr(message, "content", None) or ""
        if "</think>" not in content:
            return message  # parser already split it (or the turn had no reasoning)
        reasoning = content.split("</think>")[0].split("<think>")[-1].strip("\n")
        clean = content.split("</think>")[-1].lstrip("\n")
        try:
            raw = getattr(message, "raw_data", None)
            if not isinstance(raw, dict):
                raw = {}
            choices = raw.setdefault("choices", [{}])
            if not choices:
                choices.append({})
            msg_slot = choices[0].setdefault("message", {})
            # Only set it if absent, so a real parser-provided value always wins.
            msg_slot.setdefault("reasoning_content", reasoning)
            message.raw_data = raw
            message.content = clean
        except Exception:  # noqa: BLE001 - never break an episode over bookkeeping
            return message
        return message

    normalizing._tau2_normalize_thinking = True  # type: ignore[attr-defined]
    llm_utils.generate = normalizing
    # LLMAgent does `from tau2.utils.llm_utils import generate`, so rebind there too if imported.
    try:
        from tau2.agent import llm_agent

        if getattr(llm_agent, "generate", None) is not None:
            llm_agent.generate = normalizing
    except Exception:  # noqa: BLE001
        pass


_PRESERVE_STATS: dict[str, int] = {"assistant_msgs": 0, "with_thinking": 0}


def preservation_stats(reset: bool = False) -> dict[str, int]:
    """How often the agent's own prior reasoning actually reached its context.

    The counterpart to ``tau2/think_leak_frac``: that guard proves the *user simulator* never sees
    the chain-of-thought, this one proves the *agent* does.  Both must hold at once — preserved
    thinking is only coherent if the trained sequence and the sampled context contain the same
    tokens, and a silent failure here would score log-probs against a context the model never saw.

    Counts only assistant messages the **policy generated** (those carrying ``raw_data``), and how
    many of them reached the request with ``<think>`` inlined.  **Healthy is ~1.0; 0.0 means broken**
    (wrapper never installed, or it bailed on a shape mismatch).

    The ``raw_data`` restriction is the whole trick.  ``to_litellm_messages`` has a single call site
    (``llm_utils.generate``), shared by the agent *and* the user simulator, and in the simulator's
    request ``flip_roles`` relabels the user's own prior turns as role ``assistant``.  Counting those
    (and τ²'s hard-coded greeting, likewise never generated) diluted the metric to ~(N-1)/2N — it
    read 0.72 on a live run whose agent-side preservation was plausibly fine, which is useless as a
    guard.  Restricting to generated turns makes the healthy value 1.0 and any shortfall real.
    """
    stats = dict(_PRESERVE_STATS)
    if reset:
        _PRESERVE_STATS.update(assistant_msgs=0, with_thinking=0)
    return stats


def install_thinking_preservation() -> None:
    """Feed the agent **its own prior reasoning** back into its context (idempotent).

    Why this is required for training.  τ²'s ``generate`` builds the agent's request with
    ``to_litellm_messages``, which emits only ``{role, content, tool_calls}``.  Because the engine
    runs ``reasoning_parser=qwen3``, ``content`` is thinking-free — so **by default the agent never
    sees the reasoning it produced on earlier turns** (measured: the prompt grew 269 tok after a
    960-tok completion).

    That is fine for *eval*, but it breaks *GRPO*: our encoder renders the trained sequence with
    ``<think>`` preserved, so the log-probs would be scored against a context the model never
    actually conditioned on — a silent train/inference mismatch.  Here we re-inline the reasoning
    (recovered from ``AssistantMessage.raw_data``) into the content sent to the policy, so rollout
    and training see the *same* token stream.

    It cannot leak to the simulated user: the user's view is built by ``flip_roles``, which creates
    fresh ``UserMessage`` objects from ``content`` alone and drops ``raw_data`` — so those messages
    carry no reasoning and this wrapper is a no-op on them.

    Renders as ``<think>…</think>\\n\\n<content>``, which the preserve chat template
    (``qwen3thinking_preserve.jinja2``) parses back out — the same string it produces from a
    ``reasoning_content`` field, so generation and encoding agree token-for-token.
    """

    from tau2.utils import llm_utils

    if getattr(llm_utils.to_litellm_messages, "_tau2_preserve_thinking", False):
        return
    inner = llm_utils.to_litellm_messages

    def _reasoning_of(message: Any) -> str | None:
        raw = getattr(message, "raw_data", None)
        if not isinstance(raw, dict):
            return None
        try:
            return raw["choices"][0]["message"].get("reasoning_content") or None
        except (KeyError, IndexError, TypeError):
            return None

    def preserving(messages: list[Any]) -> list[dict[str, Any]]:
        out = inner(messages)
        if len(out) != len(messages):  # shape changed upstream; stay safe and don't touch it
            return out
        for src, dst in zip(messages, out):
            if dst.get("role") != "assistant":
                continue
            # Count ONLY messages the policy actually generated, i.e. those carrying ``raw_data``.
            # This wrapper is shared: ``to_litellm_messages`` has exactly one call site
            # (``llm_utils.generate``), which the **user simulator** uses too — and in its request
            # ``flip_roles`` turns the *user's own* prior turns into role ``assistant``.  Those can
            # never hold agent reasoning, so counting them diluted the metric toward ~(N-1)/2N
            # instead of ~1.0 and made a healthy run look broken (observed 0.72 live).  ``flip_roles``
            # builds fresh objects without ``raw_data``, so this test also excludes τ²'s hard-coded
            # greeting — leaving exactly the turns where thinking *should* be preserved.
            if not isinstance(getattr(src, "raw_data", None), dict):
                continue
            _PRESERVE_STATS["assistant_msgs"] += 1
            reasoning = _reasoning_of(src)
            if not reasoning:
                continue
            body = dst.get("content") or ""
            if "<think>" in body:  # already inlined
                _PRESERVE_STATS["with_thinking"] += 1
                continue
            dst["content"] = f"<think>\n{reasoning.strip()}\n</think>\n\n{body}"
            _PRESERVE_STATS["with_thinking"] += 1
        return out

    preserving._tau2_preserve_thinking = True  # type: ignore[attr-defined]
    llm_utils.to_litellm_messages = preserving
    # LLMAgent/UserSimulator import ``generate`` (which calls to_litellm_messages) from this module,
    # so patching the module attribute is enough — no need to touch the agent classes.


def redirect_nl_judge(model: str, llm_args: dict[str, Any]) -> None:
    """Point τ²'s NL-assertion judge at the configured endpoint (``judge_base_url``) and key.

    Measured on the ``base`` splits at the pin: airline 0 judge calls, telecom 0, **retail 40**
    (112 tasks name NL_ASSERTION in ``reward_basis`` but only 40 carry non-empty ``nl_assertions``).
    ``EvaluationType.ALL`` is run-or-crash — it raises if the basis names a component that was not
    evaluated — so the judge cannot simply be disabled; it must be redirected.

    ``evaluate_nl_assertions`` reads these as *module globals at call time*, so rebinding them is
    sufficient and requires no upstream edit.  We also force a JSON response format because the
    judge does an unguarded ``json.loads`` on the reply.
    """

    import tau2.evaluator.evaluator_nl_assertions as nl

    args = dict(llm_args)
    args.setdefault("response_format", {"type": "json_object"})
    # Tag the judge leg for cost metering (frontier spend incurred *inside* the reward function).
    args.setdefault("metadata", {"call_name": "nl_assertions_eval"})
    nl.DEFAULT_LLM_NL_ASSERTIONS = model
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = args


# Import names of τ²'s runtime deps that SkyRL's lockfile does not provide (pyproject.toml's `tau2` extra).
_TAU2_EXTRA_DEPS = frozenset({"addict", "deepdiff", "orderly_set", "cachebox", "toml"})


def bootstrap_tau2(
    source_dir: str | Path,
    *,
    max_connections: int = 64,
    request_timeout: float = 300.0,
    log_level: str = "ERROR",
    allowed_base_urls: Iterable[str] = (),
    judge_model: str | None = None,
    judge_llm_args: dict[str, Any] | None = None,
    preserve_thinking: bool = False,
) -> Path:
    """Import τ² from a verified checkout, with every known global hazard neutralised.

    The order below is load-bearing:

    1. ``TAU2_DATA_DIR`` **before** the first ``tau2`` import — ``tau2/utils/utils.py`` resolves
       ``DATA_DIR`` at import time.
    2. ``sys.path`` gets ``<checkout>/src`` (upstream uses a ``src/`` layout).
    3. Import ``tau2.utils.llm_utils``, which is what performs the litellm mutations…
    4. …then immediately undo the harmful ones (connection pool).
    5. Quiet loguru — τ² logs per-episode INFO and an ``logger.error`` per LLM exception, which is
       unusable at N=32+.
    6. Judge redirect + egress guard, before any completion can fire.
    7. Import ``tau2.registry`` (the expensive part: domain modules + pydantic models).
    8. Assert the imported package really came from the verified checkout.
    """

    source_path = verify_checkout(source_dir)
    os.environ.setdefault("TAU2_DATA_DIR", str(source_path / "data"))

    src_root = str(source_path / "src")
    if src_root not in sys.path:
        sys.path.insert(0, src_root)

    try:
        importlib.import_module("tau2.utils.llm_utils")
    except ModuleNotFoundError as exc:
        # τ² is not pip-installed, so its own deps are not resolved; these five are the ones SkyRL's
        # lockfile lacks (every ``tau2.utils`` import needs addict + deepdiff + toml).
        if (exc.name or "").split(".")[0] in _TAU2_EXTRA_DEPS:
            raise ModuleNotFoundError(
                f"{exc}. τ²'s extra runtime deps are missing: install the `tau2` extra, from the repository "
                "root `uv sync --frozen --extra fsdp --extra tau2` (--extra megatron instead of fsdp for the 30B) "
                "(addict==2.4.0 deepdiff==9.1.0 orderly-set==5.5.0 cachebox==6.2.0 toml==0.10.2).",
                name=exc.name,
            ) from exc
        raise
    _widen_litellm_connection_pool(max_connections=max_connections, request_timeout=request_timeout)
    # 4b. No fresh event loop per sync completion (a segfault site; see the helper).
    _disable_litellm_per_call_event_loop()

    try:
        from loguru import logger

        logger.remove()
        logger.add(sys.stderr, level=log_level)
    except Exception:  # noqa: BLE001 - logging config must never block a run
        pass

    if judge_model:
        redirect_nl_judge(judge_model, judge_llm_args or {})
    if allowed_base_urls:
        install_egress_guard(allowed_base_urls)
    # ALWAYS on: keeps the agent's <think> out of the content τ² shows the simulated user, without
    # trusting the engine's reasoning_parser (which silently did not apply on SkyRL's in-training
    # engine -> think_leak_frac 0.925). No-op when the parser IS active.
    install_thinking_normalization()
    # Training only. Eval deliberately leaves this off so the agent's context matches the published
    # τ²/leaderboard protocol (prior reasoning stripped) — see install_thinking_preservation.
    if preserve_thinking:
        install_thinking_preservation()

    package = importlib.import_module("tau2")
    importlib.import_module("tau2.registry")

    package_file = Path(package.__file__).resolve()
    if source_path not in package_file.parents:
        raise SourceCheckoutError(
            f"Imported tau2 from {package_file}, not verified checkout {source_path}. "
            "Use a fresh Python process for a different checkout."
        )
    return source_path


# --------------------------------------------------------------------------------------
# Task / environment accessors (thin wrappers over τ²'s registry)
# --------------------------------------------------------------------------------------


def _ensure_registered(domain: str) -> None:
    """Hook to register any domain added on top of τ²'s built-ins, before the registry is queried.

    No extra domains are registered in this release, so this is a no-op; it is kept as the single
    seam because it must run in EVERY process that touches the registry, not just the driver: Ray
    workers import this module fresh, and an unregistered domain surfaces as
    ``KeyError: Domain ... not found`` from inside the evaluator, i.e. during reward computation.
    """

    del domain


def load_tasks(domain: str, split: str | None = None) -> list[Any]:
    """Return τ² ``Task`` objects for ``domain``/``split`` (``None`` = every task).

    AReaL RL splits (see ``constants.AREAL_SPLIT_KEYS``) are not in τ²'s registry — they are loaded
    from HF and converted to native ``Task`` objects by ``areal.py``.
    """

    from .constants import AREAL_SPLIT_KEYS

    if split in AREAL_SPLIT_KEYS:
        from .areal import load_areal_tasks

        return load_areal_tasks(domain, split)

    _ensure_registered(domain)

    from tau2.registry import registry

    return registry.get_tasks_loader(domain)(split)


def load_task_splits(domain: str) -> dict[str, list[str]] | None:
    """Return ``{split_name: [task_id, ...]}`` for ``domain``, or ``None`` if unregistered."""

    from tau2.registry import registry

    loader = registry.get_task_splits_loader(domain)
    return loader() if loader is not None else None


def create_environment(domain: str, *, solo_mode: bool = False) -> Any:
    """Construct a fresh τ² ``Environment`` for ``domain``."""

    _ensure_registered(domain)

    from tau2.registry import registry

    return registry.get_env_constructor(domain)(solo_mode=solo_mode)


def get_user_tools(environment: Any, include: list[str] | None = None) -> list[Any] | None:
    """Return user-side tools, or ``None`` for single-control domains.

    Dual control is **telecom-only**: ``retail``/``airline`` raise
    ``ValueError("User tools not available")`` rather than returning an empty list.
    """

    try:
        return environment.get_user_tools(include=include) if include else environment.get_user_tools()
    except ValueError:
        return None
