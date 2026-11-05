import pytest

from agentic_platform.security import FixedWindowRateLimiter, verify_api_key


def test_api_key_is_optional_but_enforced_when_configured():
    assert verify_api_key(None, None)
    assert verify_api_key("secret", "secret")
    assert not verify_api_key("secret", None)
    assert not verify_api_key("secret", "wrong")


def test_api_key_comparison_handles_non_ascii_values():
    assert verify_api_key("کلید-محرمانه", "کلید-محرمانه")
    assert not verify_api_key("کلید-محرمانه", "wrong")


def test_fixed_window_rate_limit_and_reset():
    limiter = FixedWindowRateLimiter(requests_per_minute=2)
    assert limiter.allow("client-a", now=120.0) == (True, 0)
    assert limiter.allow("client-a", now=121.0) == (True, 0)
    allowed, retry_after = limiter.allow("client-a", now=122.0)
    assert not allowed
    assert retry_after == 58
    assert limiter.allow("client-b", now=122.0) == (True, 0)
    assert limiter.allow("client-a", now=180.0) == (True, 0)
    limiter.reset()
    assert limiter.allow("client-a", now=122.0) == (True, 0)


def test_rate_limiter_rejects_invalid_limit():
    with pytest.raises(ValueError):
        FixedWindowRateLimiter(0)
