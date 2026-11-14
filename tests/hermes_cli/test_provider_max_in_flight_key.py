"""``max_in_flight_requests`` is a provider entry key: the background-review idle queue reads it
to defer reviews for an endpoint with one slot. The normalizer must not report it as unknown, and
the legacy ``custom_providers`` to ``providers`` conversion (the config migration, the dashboard's
save and activate) must keep it, or the mark is lost without a word."""

import logging

import pytest

from hermes_cli.config_providers import (
    _PROVIDER_NORMALIZE_WARNED,
    _custom_provider_entry_to_provider_config,
    _normalize_custom_provider_entry,
)

ENTRY = {"name": "llamacpp", "base_url": "http://192.0.2.10:8080/v1", "max_in_flight_requests": 1}


@pytest.fixture(autouse=True)
def _reset_warn_cache():
    _PROVIDER_NORMALIZE_WARNED.clear()
    yield
    _PROVIDER_NORMALIZE_WARNED.clear()


def test_key_is_not_reported_as_unknown(caplog):
    with caplog.at_level(logging.WARNING):
        _normalize_custom_provider_entry(dict(ENTRY), provider_key="llamacpp")
    assert not [r for r in caplog.records if "unknown config keys" in r.getMessage()]


def test_normalized_entry_keeps_the_key():
    normalized = _normalize_custom_provider_entry(dict(ENTRY), provider_key="llamacpp")
    assert normalized["max_in_flight_requests"] == 1


def test_conversion_to_providers_keeps_the_key():
    converted = _custom_provider_entry_to_provider_config(dict(ENTRY), provider_key="llamacpp")
    assert converted["api"] == "http://192.0.2.10:8080/v1"
    assert converted["max_in_flight_requests"] == 1


@pytest.mark.parametrize("bad", [0, -1, True, "1", 1.5, None])
def test_invalid_values_are_dropped(bad):
    entry = dict(ENTRY, max_in_flight_requests=bad)
    assert "max_in_flight_requests" not in _normalize_custom_provider_entry(entry, provider_key="llamacpp")
    assert "max_in_flight_requests" not in _custom_provider_entry_to_provider_config(entry, provider_key="llamacpp")
