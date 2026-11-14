"""The quiet window a deferred background review waits for is a setting.

``auxiliary.background_review.defer_idle_s`` sets how long the process must be free of live turns
before a queued review dispatches. A voice conversation pauses for longer than the 15 s default
between turns, so a review dispatched in such a pause would still meet the next turn. The window
is read from the oldest queued item's settings, as the age-out is; ``defer_max_age_s`` still
dispatches an item past its age whatever the window.
"""

from agent.review_idle_queue import _IDLE_SETTLE_S, ReviewIdleQueue, defer_idle_s


class _FakeAgent:
    session_id = "sess-x"

    def _spawn_background_review_now(self, **kwargs):
        pass


def _queue():
    q = ReviewIdleQueue()
    clock = {"t": 0.0}
    q._now = lambda: clock["t"]
    q._server_idle = lambda: True
    q._ensure_thread = lambda: None
    return q, clock


def _quiet_from_now(q):
    q.note_turn_started()
    q.note_turn_finished()


def test_defer_idle_parsing():
    assert _IDLE_SETTLE_S == 15.0
    assert defer_idle_s(None) == 15.0
    assert defer_idle_s({}) == 15.0
    assert defer_idle_s({"defer_idle_s": 600}) == 600.0
    assert defer_idle_s({"defer_idle_s": "600"}) == 600.0
    assert defer_idle_s({"defer_idle_s": "soon"}) == 15.0
    assert defer_idle_s({"defer_idle_s": 0}) == 15.0
    assert defer_idle_s({"defer_idle_s": -5}) == 15.0


def test_default_window_is_fifteen_seconds():
    q, clock = _queue()
    q.enqueue(_FakeAgent(), "s1", {"task_cfg": {}})
    _quiet_from_now(q)
    clock["t"] += 14.0
    assert q._pop_dispatchable() is None
    clock["t"] += 2.0
    assert q._pop_dispatchable() is not None


def test_configured_window_holds_until_it_elapses():
    q, clock = _queue()
    q.enqueue(_FakeAgent(), "s1", {"task_cfg": {"defer_idle_s": 600, "defer_max_age_s": 28800}})
    _quiet_from_now(q)
    clock["t"] += 599.0
    assert q._pop_dispatchable() is None
    clock["t"] += 2.0
    item = q._pop_dispatchable()
    assert item is not None and item.session_key == "s1"


def test_a_new_turn_restarts_the_configured_window():
    q, clock = _queue()
    q.enqueue(_FakeAgent(), "s1", {"task_cfg": {"defer_idle_s": 600}})
    _quiet_from_now(q)
    clock["t"] += 500.0
    _quiet_from_now(q)
    clock["t"] += 500.0
    assert q._pop_dispatchable() is None
    clock["t"] += 101.0
    assert q._pop_dispatchable() is not None


def test_aged_out_item_dispatches_inside_the_window():
    q, clock = _queue()
    q.enqueue(_FakeAgent(), "s1", {"task_cfg": {"defer_idle_s": 600, "defer_max_age_s": 100}})
    q.note_turn_started()  # never goes quiet
    clock["t"] += 99.0
    assert q._pop_dispatchable() is None
    clock["t"] += 2.0
    assert q._pop_dispatchable() is not None


def test_window_comes_from_the_oldest_item():
    q, clock = _queue()
    q.enqueue(_FakeAgent(), "old", {"task_cfg": {"defer_idle_s": 600}})
    clock["t"] += 1.0
    q.enqueue(_FakeAgent(), "new", {"task_cfg": {}})
    _quiet_from_now(q)
    clock["t"] += 20.0
    assert q._pop_dispatchable() is None
    clock["t"] += 600.0
    assert q._pop_dispatchable().session_key == "old"
