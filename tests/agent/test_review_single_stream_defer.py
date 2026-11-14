"""Deferral of background reviews bound for an endpoint the operator marked single-stream.

Companion contracts to tests/agent/test_review_idle_queue.py (managed-local deferral). A review
whose fork would decode on an endpoint marked ``max_in_flight_requests: 1`` (in a ``providers``
entry or a legacy ``custom_providers`` entry) is deferred to the idle queue; otherwise it contends
with the next live turn for the only slot and loses via hard interrupt.

- marked 1, loopback or LAN host        -> defer (True)
- unmarked, or marked with another value -> immediate (False; pre-existing default)
- the mark belongs to another host      -> immediate (False)
- config unreadable                     -> immediate (False; fail open)
- defer: never                          -> immediate (False; explicit opt-out respected)

Adapted from upstream PR 119676 (rktchip), which limited the mark to loopback hosts and to
``providers`` entries.
"""

from __future__ import annotations

from unittest.mock import patch

from agent.review_idle_queue import review_targets_single_stream
from run_agent import _review_should_defer


class _FakeAgent:
    def __init__(self, provider="qwen27b-exl3", model="Qwen3.8-27B-EXL3-3.5bpw",
                 base_url="http://127.0.0.1:8290/v1"):
        self.provider = provider
        self.model = model
        self._base_url = base_url
        self._credential_pool = None
        self.request_overrides = {}
        self.max_tokens = None
        self.acp_command = None
        self.acp_args = []

    def _current_main_runtime(self):
        return {"api_key": None, "base_url": self._base_url, "api_mode": None}


def _cfg(providers=None, custom_providers=None):
    cfg = {}
    if providers is not None:
        cfg["providers"] = providers
    if custom_providers is not None:
        cfg["custom_providers"] = custom_providers
    return cfg


def _defers(agent, config, task_cfg=None) -> bool:
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        targeted = review_targets_single_stream(agent, task_cfg or {})
        assert _review_should_defer(agent, task_cfg or {}) is targeted
        return targeted


def test_loopback_single_stream_defers():
    config = _cfg({"qwen27b-exl3": {"base_url": "http://127.0.0.1:8290/v1", "max_in_flight_requests": 1}})
    assert _defers(_FakeAgent(), config) is True


def test_loopback_without_key_spawns_immediately():
    config = _cfg({"qwen27b-exl3": {"base_url": "http://127.0.0.1:8290/v1"}})
    assert _defers(_FakeAgent(), config) is False


def test_other_value_spawns_immediately():
    config = _cfg({"qwen27b-exl3": {"base_url": "http://127.0.0.1:8290/v1", "max_in_flight_requests": 4}})
    assert _defers(_FakeAgent(), config) is False


def test_matched_by_hostname_when_id_differs():
    config = _cfg({"tabby-local": {"base_url": "http://127.0.0.1:8290/v1", "max_in_flight_requests": 1}})
    assert _defers(_FakeAgent(provider="other-id"), config) is True


def test_lan_providers_entry_defers():
    agent = _FakeAgent(provider="rig", base_url="http://192.0.2.10:8080/v1")
    config = _cfg({"rig": {"base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}})
    assert _defers(agent, config) is True


def test_lan_custom_providers_entry_defers():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    config = _cfg(custom_providers=[
        {"name": "llamacpp", "base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}])
    assert _defers(agent, config) is True


def test_custom_providers_entry_matched_by_name():
    agent = _FakeAgent(provider="custom:llamacpp", base_url="http://rig.lan:8080/v1")
    config = _cfg(custom_providers=[
        {"name": "llamacpp", "base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}])
    assert _defers(agent, config) is True


def test_unmarked_custom_providers_entry_spawns_immediately():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    config = _cfg(custom_providers=[{"name": "llamacpp", "base_url": "http://192.0.2.10:8080/v1"}])
    assert _defers(agent, config) is False


def test_mark_on_another_host_spawns_immediately():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.11:8080/v1")
    config = _cfg(
        {"rig": {"base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}},
        [{"name": "llamacpp", "base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}],
    )
    assert _defers(agent, config) is False


def test_malformed_sections_spawn_immediately():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    config = {"providers": ["not", "a", "dict"], "custom_providers": {"llamacpp": {}}}
    assert _defers(agent, config) is False


def test_malformed_entry_url_is_skipped():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    config = _cfg({"broken": {"base_url": "http://[::1", "max_in_flight_requests": 4},
                   "rig": {"base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}})
    assert _defers(agent, config) is True


def test_unreadable_config_fails_open():
    with patch("hermes_cli.config.load_config_readonly", side_effect=OSError("nope")):
        assert review_targets_single_stream(_FakeAgent(), {}) is False


def test_defer_never_respected():
    config = _cfg({"qwen27b-exl3": {"base_url": "http://127.0.0.1:8290/v1", "max_in_flight_requests": 1}})
    with patch("hermes_cli.config.load_config_readonly", return_value=config):
        assert _review_should_defer(_FakeAgent(), {"defer": "never"}) is False


def test_unmarked_entry_on_the_same_host_does_not_hide_a_marked_one():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    config = _cfg({"stt": {"base_url": "http://192.0.2.10:9000/v1"}},
                  [{"name": "llamacpp", "base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}])
    assert _defers(agent, config) is True


def test_marked_entry_found_whatever_the_order():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    config = _cfg({"rig": {"base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}},
                  [{"name": "stt", "base_url": "http://192.0.2.10:9000/v1"}])
    assert _defers(agent, config) is True


def test_same_host_and_port_wins_over_same_host():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    other_port_marked = _cfg({"a": {"base_url": "http://192.0.2.10:9000/v1", "max_in_flight_requests": 1},
                              "b": {"base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 4}})
    assert _defers(agent, other_port_marked) is False
    this_port_marked = _cfg({"a": {"base_url": "http://192.0.2.10:9000/v1", "max_in_flight_requests": 4},
                             "b": {"base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}})
    assert _defers(agent, this_port_marked) is True


def test_default_port_matches_an_explicit_one():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10/v1")
    config = _cfg({"a": {"base_url": "http://192.0.2.10:9000/v1", "max_in_flight_requests": 4},
                   "b": {"base_url": "http://192.0.2.10:80/v1", "max_in_flight_requests": 1}})
    assert _defers(agent, config) is True


def test_url_under_api_or_url_is_read():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    for field in ("api", "url"):
        config = _cfg({"llamacpp": {field: "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}})
        assert _defers(agent, config) is True, field


def test_unmarked_entry_for_the_exact_endpoint_decides():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    config = _cfg({"llm": {"base_url": "http://192.0.2.10:8080/v1"},
                   "other": {"base_url": "http://192.0.2.10:9000/v1", "max_in_flight_requests": 1}})
    assert _defers(agent, config) is False


def test_first_marked_entry_on_the_exact_endpoint_decides():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    config = _cfg({"plain": {"base_url": "http://192.0.2.10:8080/v1"}},
                  [{"name": "llamacpp", "base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}])
    assert _defers(agent, config) is True


def test_same_host_fallback_when_no_entry_names_the_endpoint():
    agent = _FakeAgent(provider="custom", base_url="http://192.0.2.10:8080/v1")
    config = _cfg({"other": {"base_url": "http://192.0.2.10:9000/v1", "max_in_flight_requests": 1}})
    assert _defers(agent, config) is True
