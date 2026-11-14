"""A config-signature change rebuilds a session's cached agent. Review cancellation is per agent
instance, so a live turn on the rebuilt agent cannot reach a background review the old instance
still runs; the rebuild itself must retire it, without holding the cache lock or the turn."""

from __future__ import annotations

import threading
import time
import types
from collections import OrderedDict
from types import SimpleNamespace

import pytest

from agent import background_review, review_idle_queue
from gateway.run_agent_cache import GatewayAgentCacheMixin
from gateway.session_execution import GatewaySessionAgentMixin

SESSION_KEY = "agent:main:voice:dm:voice"


class _Parent:
    """The minimum a review parent needs: the review lock and run slot are created on demand."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.max_iterations = 0


class _Harness(GatewaySessionAgentMixin):
    def __init__(self, runner, ctx, built):
        self._runner = runner
        self._ctx = ctx
        self._built = built

    def _skip_context_files(self, platform_key) -> bool:
        return False

    def _cached_sid_is_dead(self, cache_lock, cache) -> tuple:
        return None, False

    def _current_message_count(self):
        return None

    def _build_fresh_agent(self, *args, **kwargs):
        agent = self._built.pop(0)
        if isinstance(agent, Exception):
            raise agent
        return agent


def _runner(sig: str):
    runner = SimpleNamespace(
        _agent_cache=OrderedDict(),
        _agent_cache_lock=threading.Lock(),
        _agent_config_signature=lambda *a, **k: sig,
        _extract_cache_busting_config=lambda cfg: {},
        _enforce_agent_cache_cap=lambda: None,
        _init_cached_agent_for_turn=lambda agent, depth: None,
        _apply_fallback_chain_to_agent=lambda agent, chain: None,
        _refresh_fallback_model=lambda: None,
        session_store=None,
    )
    # Record every release thread the lookup starts: retirement only ever runs through one, so an
    # empty list proves nothing will retire later.
    runner.spawned = []
    real_spawn = types.MethodType(GatewayAgentCacheMixin._spawn_release_thread, runner)

    def _spawn(target, args, name, **kwargs):
        runner.spawned.append(name)
        return real_spawn(target, args, name, **kwargs)

    runner._spawn_release_thread = _spawn
    runner._run_release_in_profile_scope = types.MethodType(
        GatewayAgentCacheMixin._run_release_in_profile_scope, runner)
    return runner


def _ctx():
    return SimpleNamespace(
        session_key=SESSION_KEY, session_id="s1", source=SimpleNamespace(platform=None),
        enabled_toolsets=[], user_config={}, interim_assistant_messages_enabled=False,
        _interrupt_depth=0,
    )


def _resolve(harness):
    route = {"model": "m", "runtime": {}}
    return harness._resolve_turn_agent(route, "voice", "", 10, None, None)


def _admit_review(parent):
    """Put an admitted, still-running review on ``parent`` and return its run and fork."""
    run = background_review.prepare_background_review_run(parent)
    assert run is not None
    fork = SimpleNamespace(name=f"review-of-{parent.name}")
    assert run.begin_request(fork)
    return run, fork


@pytest.fixture
def queue(monkeypatch):
    q = review_idle_queue.ReviewIdleQueue()
    monkeypatch.setattr(q, "_ensure_thread", lambda: None)
    monkeypatch.setattr(review_idle_queue, "QUEUE", q)
    return q


@pytest.fixture
def interrupts(monkeypatch):
    """Record each hard interrupt sent to a review fork, and acknowledge it like a fork would."""
    seen = []
    runs = {}

    def _interrupt(review_agent, *, message, tool_reason):
        seen.append((review_agent, tool_reason))
        run = runs.get(id(review_agent))
        if run is not None:
            run.request_done.set()

    monkeypatch.setattr(background_review, "_interrupt_background_review", _interrupt)
    return seen, runs


def _wait_for(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_signature_change_retires_the_replaced_agents_review(queue, interrupts):
    seen, runs = interrupts
    old, new = _Parent("old"), _Parent("new")
    run, fork = _admit_review(old)
    runs[id(fork)] = run
    runner = _runner("sig-new")
    runner._agent_cache[SESSION_KEY] = (old, "sig-old", None, "s1")

    agent, reused = _resolve(_Harness(runner, _ctx(), [new]))

    assert agent is new and reused is False
    assert runner._agent_cache[SESSION_KEY][0] is new
    assert _wait_for(lambda: seen), "the replaced agent's review got no interrupt"
    assert seen[0][0] is fork
    # The old instance is fenced: no new review starts, and a late requeue is refused.
    assert background_review.background_review_retired(old)
    assert background_review.prepare_background_review_run(old) is None
    queue.enqueue(old, SESSION_KEY, {})
    assert queue.pending_count() == 0
    # The new agent is untouched and can review.
    assert not background_review.background_review_retired(new)


def test_a_hung_cancel_does_not_hold_the_turn_or_the_cache_lock(monkeypatch, queue):
    lock_free_during_cancel = []
    runner = _runner("sig-new")

    def _interrupt(review_agent, *, message, tool_reason):
        got = runner._agent_cache_lock.acquire(timeout=0.5)
        if got:
            runner._agent_cache_lock.release()
        lock_free_during_cancel.append(got)
        # Never acknowledge: the fork is wedged.

    monkeypatch.setattr(background_review, "_interrupt_background_review", _interrupt)
    monkeypatch.setattr(background_review, "_BACKGROUND_REVIEW_CANCEL_TIMEOUT_SECONDS", 5.0)
    old, new = _Parent("old"), _Parent("new")
    run, _fork = _admit_review(old)
    runner._agent_cache[SESSION_KEY] = (old, "sig-old", None, "s1")
    try:
        started = time.monotonic()
        agent, _ = _resolve(_Harness(runner, _ctx(), [new]))
        elapsed = time.monotonic() - started
        assert agent is new
        assert elapsed < 1.0, f"the lookup waited {elapsed:.2f}s on the cancel"
        assert _wait_for(lambda: lock_free_during_cancel)
        assert lock_free_during_cancel == [True]
    finally:
        run.request_done.set()


def test_same_signature_reuses_and_keeps_the_review(queue, interrupts):
    seen, runs = interrupts
    cached = _Parent("cached")
    run, fork = _admit_review(cached)
    runs[id(fork)] = run
    runner = _runner("sig")
    runner._agent_cache[SESSION_KEY] = (cached, "sig", None, "s1")

    agent, reused = _resolve(_Harness(runner, _ctx(), []))

    assert agent is cached and reused is True
    assert runner.spawned == []
    assert seen == []
    assert not background_review.background_review_retired(cached)


def test_a_failed_rebuild_leaves_the_cached_agent_and_its_review(queue, interrupts):
    seen, runs = interrupts
    old = _Parent("old")
    run, fork = _admit_review(old)
    runs[id(fork)] = run
    runner = _runner("sig-new")
    runner._agent_cache[SESSION_KEY] = (old, "sig-old", None, "s1")

    with pytest.raises(RuntimeError):
        _resolve(_Harness(runner, _ctx(), [RuntimeError("build failed")]))

    assert runner.spawned == []
    assert runner._agent_cache[SESSION_KEY][0] is old
    assert seen == []
    assert not background_review.background_review_retired(old)
