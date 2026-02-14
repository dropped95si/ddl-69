from ddl69.utils.signals import (
    infer_horizon_days_from_rules,
    infer_row_horizon_days,
    parse_horizon_days,
)


def test_parse_horizon_days_units() -> None:
    assert parse_horizon_days({"value": 2, "unit": "weeks"}) == 14.0
    assert parse_horizon_days({"value": 6, "unit": "months"}) == 180.0
    assert parse_horizon_days({"value": 1, "unit": "year"}) == 365.0
    assert parse_horizon_days("18mo") == 540.0


def test_infer_horizon_days_from_rules_uses_longest_available() -> None:
    rules = [
        {"rule": "A", "h60": {"samples": 10}, "h90": {"samples": 7}},
        {"rule": "B", "h120": {"samples": 5}},
    ]
    assert infer_horizon_days_from_rules(rules) == 120.0


def test_infer_row_horizon_prefers_explicit_fields() -> None:
    row = {
        "horizon_days": 45,
        "learned_top_rules": [{"h120": {"samples": 5}}],
    }
    assert infer_row_horizon_days(row, default_horizon_days=5) == 45


def test_infer_row_horizon_uses_rules_when_missing_explicit() -> None:
    row = {
        "learned_top_rules": [{"h60": {"samples": 10}, "h90": {"samples": 3}}],
    }
    assert infer_row_horizon_days(row, default_horizon_days=5) == 90


def test_infer_row_horizon_uses_bucket_fallback() -> None:
    assert infer_row_horizon_days({"timeframe": "long"}, default_horizon_days=5) == 400
    assert infer_row_horizon_days({"timeframe": "swing"}, default_horizon_days=5) == 180
    assert infer_row_horizon_days({"timeframe": "day"}, default_horizon_days=5) == 10

