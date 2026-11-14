"""Idle deferral for background reviews on the managed local runtime.

On the managed llama-server the post-turn review fork monopolizes the GPU the next prompt
needs and the next live turn cancels it (decode cost paid, learning lost). Reviews bound for
the managed endpoint are therefore queued and dispatched when the machine is quiet
(``auxiliary.background_review.defer``: ``auto`` = exactly that case, ``never`` = old behavior;
explicit /refine never defers). One slot per session, newest snapshot wins (a review replays
the whole conversation, so coalescing is dedup, not loss); aged-out items (defer_max_age_s,
default 30 min) dispatch regardless of idleness; in-memory best-effort like the immediate
fork. Idle truth is the supervisor's /slots held for a settle window (defer_idle_s, default
15 s). Endpoints the operator marks single-stream (max_in_flight_requests: 1) defer the same way.
"""

from __future__ import annotations

import contextvars
import json
import logging
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_IDLE_SETTLE_S = 15.0  # quiet window: two back-to-back prompts must not look idle, a coffee break must
_POLL_INTERVAL_S = 5.0  # poll cadence while non-empty; the thread parks when empty
_MAX_AGE_DEFAULT_S = 30.0 * 60.0  # dispatch regardless of idleness past this age


def defer_mode(task_cfg: Optional[dict[str, Any]]) -> str:
    """'auto' (default) or 'never' from auxiliary.background_review.defer."""
    raw = str((task_cfg or {}).get("defer", "auto")).strip().lower()
    return raw if raw in ("auto", "never") else "auto"


def defer_max_age_s(task_cfg: Optional[dict[str, Any]]) -> float:
    try:
        value = float((task_cfg or {}).get("defer_max_age_s", _MAX_AGE_DEFAULT_S))
    except (TypeError, ValueError):
        return _MAX_AGE_DEFAULT_S
    return value if value > 0 else _MAX_AGE_DEFAULT_S


def defer_idle_s(task_cfg: Optional[dict[str, Any]]) -> float:
    """Seconds the process must be free of live turns before a queued review dispatches
    (``auxiliary.background_review.defer_idle_s``; default the settle window). A voice conversation
    pauses for longer than the default between turns, so its profile can ask for a real break."""
    try:
        value = float((task_cfg or {}).get("defer_idle_s", _IDLE_SETTLE_S))
    except (TypeError, ValueError):
        return _IDLE_SETTLE_S
    return value if value > 0 else _IDLE_SETTLE_S


def review_targets_managed_local(agent: Any, task_cfg: Optional[dict[str, Any]]) -> bool:
    """Would this review fork decode on the llama-server WE manage? Exact netloc match against the
    supervisor state file; any failure reads False (immediate spawn is the safe default). The cheap
    TTL-cached netloc probe runs FIRST so cloud-only installs skip runtime resolution on the turn's tail."""
    try:
        from agent.auxiliary_managed_local import _is_managed_local_endpoint, _managed_local_netloc

        if not _managed_local_netloc():
            return False
        from agent.background_review import _resolve_review_runtime

        runtime = _resolve_review_runtime(agent, task_cfg)
        return _is_managed_local_endpoint(runtime.get("base_url"))
    except Exception:
        return False


def _provider_entries(config: Any) -> list[tuple[str, dict[str, Any]]]:
    """``(provider id, entry)`` for every ``providers.<id>`` block and every legacy
    ``custom_providers`` list item (id ``custom:<name>``); malformed sections are skipped."""
    entries: list[tuple[str, dict[str, Any]]] = []
    if not isinstance(config, dict):
        return entries
    providers = config.get("providers")
    if isinstance(providers, dict):
        entries += [(str(pid), block) for pid, block in providers.items() if isinstance(block, dict)]
    custom = config.get("custom_providers")
    if isinstance(custom, list):
        entries += [(f"custom:{item.get('name')}", item) for item in custom
                    if isinstance(item, dict) and item.get("name")]
    return entries


_DEFAULT_PORTS = {"http": 80, "https": 443}


def _host_port(url: Any) -> tuple[str, Optional[int]]:
    """Lowercased hostname and port (the scheme's default when absent) of ``url``; ``("", None)``
    for a missing or malformed URL (e.g. an unclosed IPv6 bracket, a non-numeric port)."""
    from urllib.parse import urlsplit

    if not (isinstance(url, str) and url.strip()):
        return "", None
    raw = url.strip()
    try:
        parts = urlsplit(raw if "//" in raw else "//" + raw)
        host, port = (parts.hostname or "").lower(), parts.port
    except ValueError:
        return "", None
    return host, port if port is not None else _DEFAULT_PORTS.get(parts.scheme)


def _entry_url(block: dict[str, Any]) -> str:
    """The entry's endpoint URL: ``base_url``, else ``url``, else ``api``, as Hermes writes them."""
    for key in ("base_url", "url", "api"):
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _mark(block: dict[str, Any]) -> Optional[int]:
    """The entry's positive ``max_in_flight_requests``, else None."""
    try:
        value = int(block.get("max_in_flight_requests"))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _provider_max_in_flight(provider_id: Optional[str], base_url: str) -> Optional[int]:
    """``max_in_flight_requests`` for the review runtime's endpoint. The entry named by
    ``provider_id`` decides when there is one. Otherwise the entries on the same host and port
    decide: the first marked one, or None when none of them is marked, since the endpoint's own entry
    says it has no mark. Only when no entry names the host and port does the first marked entry on
    the same host apply. ``None`` when nothing is marked or the config is unreadable: unlimited is
    the fail-open default, so only an explicit ``1`` ever newly defers anyone."""
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:  # health: allow BLE001 -- an unreadable config reads as unmarked: spawn now, the old default
        return None
    entries = _provider_entries(config)
    if provider_id:
        named = [block for pid, block in entries if pid == provider_id]
        if named:
            return _mark(named[0])
    host, port = _host_port(base_url)
    if not host:
        return None
    exact: list[Optional[int]] = []
    same_host: list[int] = []
    for _pid, block in entries:
        entry_host, entry_port = _host_port(_entry_url(block))
        if entry_host != host:
            continue
        mark = _mark(block)
        if entry_port == port:
            exact.append(mark)
        elif mark is not None:
            same_host.append(mark)
    if exact:
        return next((mark for mark in exact if mark is not None), None)
    return same_host[0] if same_host else None


def review_targets_single_stream(agent: Any, task_cfg: Optional[dict[str, Any]]) -> bool:
    """True when the review fork would decode on an endpoint the operator marked single-stream
    (``max_in_flight_requests: 1`` on its ``providers`` or ``custom_providers`` entry).

    Companion to :func:`review_targets_managed_local` for serves the supervisor does not manage,
    on this machine or another one on the network: an immediately spawned review contends with the
    next live turn for the only slot, loses via hard interrupt, and costs a full re-prefill per kill.
    A review replays the whole conversation, so deferring it to the idle queue loses nothing. The mark
    is the operator's statement that the endpoint does not queue requests itself, so it applies on
    any host. Any failure reads False (immediate spawn is the pre-existing default)."""
    try:
        from agent.background_review import _resolve_review_runtime

        runtime = _resolve_review_runtime(agent, task_cfg)
        base_url = runtime.get("base_url") or ""
        if not _host_port(base_url)[0]:
            return False
        return _provider_max_in_flight(runtime.get("provider"), base_url) == 1
    except Exception:  # health: allow BLE001 -- runs on the turn's tail; any failure spawns now, the old default
        return False


@dataclass(slots=True)
class _PendingReview:
    agent: Any
    session_key: str
    kwargs: dict[str, Any]
    enqueued_at: float
    context: contextvars.Context


class ReviewIdleQueue:
    """Session-coalescing queue + idle-gated dispatcher thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: dict[str, _PendingReview] = {}
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._live_turns = 0
        self._quiet_since: Optional[float] = None
        # Test seams — replaced by unit tests, never in production.
        self._now: Callable[[], float] = time.monotonic
        self._server_idle: Callable[[], bool] = _managed_server_idle

    def note_turn_started(self) -> None:
        with self._lock:
            self._live_turns += 1
            self._quiet_since = None

    def note_turn_finished(self) -> None:
        with self._lock:
            self._live_turns = max(0, self._live_turns - 1)
            if self._live_turns == 0:
                self._quiet_since = self._now()
        self._wake.set()

    def enqueue(self, agent: Any, session_key: str, kwargs: dict[str, Any]) -> None:
        """Add (or replace — newest snapshot wins) a session's pending review, keeping the ORIGINAL
        enqueue time on coalesce so a busy session cannot push its age-out forever."""
        from agent.background_review import holding_review_fence

        # Parent -> queue lock order; retirement fences insertion before purging. The
        # dispatcher never takes a parent lock while holding the queue lock.
        with holding_review_fence(agent) as retired:
            if retired:
                return
            with self._lock:
                existing = self._pending.get(session_key)
                enqueued_at = existing.enqueued_at if existing is not None else self._now()
                self._pending[session_key] = _PendingReview(
                    agent, session_key, kwargs, enqueued_at, contextvars.copy_context())
        self._ensure_thread()
        self._wake.set()
        logger.info("Background review deferred (session=%s, queued=%d)", session_key[-12:], len(self._pending))

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def discard_parent(self, agent: Any) -> None:
        """Remove only this instance's items, never a successor sharing its session key."""
        with self._lock:
            keys = [key for key, item in self._pending.items() if item.agent is agent]
            for key in keys:
                del self._pending[key]

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, daemon=True, name="bg-review-idle-queue")
                self._thread.start()

    def quiet_period(self) -> tuple[float, Optional[float]]:
        """Process-wide quiet clock: (seconds since the last live turn ended, the stamp that quiet period began
        at). ``(0.0, None)`` while any turn runs or before the first turn ends."""
        with self._lock:
            if self._live_turns > 0 or self._quiet_since is None:
                return 0.0, None
            return self._now() - self._quiet_since, self._quiet_since

    def _quiet_for(self) -> float:
        """Seconds this process has been turn-free (0 while a turn runs)."""
        return self.quiet_period()[0]

    def _pop_dispatchable(self) -> Optional[_PendingReview]:
        """Oldest aged-out item, else the oldest item once quiet+idle hold. The quiet window is the
        oldest item's ``defer_idle_s``."""
        with self._lock:
            if not self._pending:
                return None
            now = self._now()
            aged = [p for p in self._pending.values()
                    if now - p.enqueued_at >= defer_max_age_s(p.kwargs.get("task_cfg"))]
            candidate = min(aged, key=lambda p: p.enqueued_at) if aged else None
            oldest = min(self._pending.values(), key=lambda p: p.enqueued_at)
            window = defer_idle_s(oldest.kwargs.get("task_cfg"))
        if candidate is None and (self._quiet_for() < window or not self._server_idle()):
            return None
        with self._lock:
            if candidate is None:
                if not self._pending:
                    return None
                candidate = min(self._pending.values(), key=lambda p: p.enqueued_at)
            return self._pending.pop(candidate.session_key, None)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                if not self._pending:
                    self._wake.clear()
                    continue
            item = None
            try:
                item = self._pop_dispatchable()
                if item is not None:
                    # The shared dispatcher has no caller profile. Enter the item's context
                    # before reading config AND spawning the context-propagating review worker.
                    item.context.run(self._dispatch, item)
            except Exception:
                logger.warning("Deferred review dispatch failed", exc_info=True)
            if item is None:
                time.sleep(_POLL_INTERVAL_S)

    def _dispatch(self, item: _PendingReview) -> None:
        from agent.background_review import background_review_retired

        # A popped item is no longer purgeable. Preparation/admission also check the
        # fence in case retirement races this check or the enabled/config re-read.
        if background_review_retired(item.agent):
            return
        if not self._still_enabled(item):
            logger.info(
                "Deferred background review dropped: reviews were disabled while it was queued (session=%s)",
                item.session_key[-12:])
            return
        logger.info(
            "Dispatching deferred background review (session=%s, waited=%.0fs, queued=%d)",
            item.session_key[-12:], self._now() - item.enqueued_at, self.pending_count())
        item.agent._spawn_background_review_now(**item.kwargs)

    @staticmethod
    def _still_enabled(item: _PendingReview) -> bool:
        """Re-check the enabled gate at DISPATCH time (disabling reviews while queued must stick). Fail-open."""
        try:
            from agent.background_review import load_background_review_settings

            return load_background_review_settings()[0]
        except Exception:
            return True


def _managed_server_idle() -> bool:
    """No processing slot on any loaded model of the managed router; unreachable/no state file reads idle."""
    try:
        from hermes_cli.local_runtime.supervisor import state_path
        from urllib.parse import quote

        state = json.loads(state_path().read_text(encoding="utf-8-sig"))
        base = str(state.get("base_url", "")).rsplit("/v1", 1)[0]
        headers = {"Authorization": f"Bearer {state.get('api_key', '')}"}
        if not base:
            return True

        def _get(path: str) -> Any:
            with urllib.request.urlopen(urllib.request.Request(f"{base}{path}", headers=headers), timeout=3) as r:
                return json.loads(r.read())

        loaded = [m["id"] for m in _get("/models").get("data", [])
                  if (m.get("status") or {}).get("value") in ("loaded", "ready")]
        return not any(
            # autoload=false: a model unloaded since the /models read must not load again for a status read.
            s.get("is_processing") for mid in loaded
            for s in _get(f"/slots?model={quote(mid)}&autoload=false") if isinstance(s, dict)
        )
    except Exception:
        return True


# Module singleton — one queue per process, like the load-progress watcher.
QUEUE = ReviewIdleQueue()
