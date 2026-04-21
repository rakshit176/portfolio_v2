# tests/unit/test_rate_limiter.py
import pytest
import asyncio
from utils.rate_limiter import RateLimiter, MaxRetriesExceeded

@pytest.mark.asyncio
async def test_rate_limiter_allows_requests_within_limit():
    """Should allow requests when under RPM limit."""
    limiter = RateLimiter(max_rpm=10)

    call_count = 0
    async def mock_call():
        nonlocal call_count
        call_count += 1
        return "success"

    # 5 calls should be fine
    for _ in range(5):
        result = await limiter.call_with_retry(mock_call, provider="groq")
        assert result == "success"

    assert call_count == 5

@pytest.mark.asyncio
async def test_rate_limiter_tracks_requests_per_provider():
    """Should track requests separately for each provider."""
    limiter = RateLimiter(max_rpm=2)

    async def mock_call():
        return "success"

    # 2 calls to groq - should work
    await limiter.call_with_retry(mock_call, provider="groq")
    await limiter.call_with_retry(mock_call, provider="groq")

    # 2 calls to gemini - should work (separate bucket)
    await limiter.call_with_retry(mock_call, provider="gemini")
    await limiter.call_with_retry(mock_call, provider="gemini")

@pytest.mark.asyncio
async def test_rate_limiter_backs_off_on_limit():
    """Should back off when limit is exceeded."""
    limiter = RateLimiter(max_rpm=1)

    async def mock_call():
        return "success"

    # First call succeeds
    await limiter.call_with_retry(mock_call, provider="groq")

    # Second call should trigger backoff
    import time
    start = time.time()
    await limiter.call_with_retry(mock_call, provider="groq")
    elapsed = time.time() - start

    # Should have waited at least 1 second (exponential backoff)
    assert elapsed >= 1.0

@pytest.mark.asyncio
async def test_rate_limiter_raises_after_max_retries():
    """Should raise MaxRetriesExceeded after all retries exhausted."""
    limiter = RateLimiter(max_rpm=0)  # Zero allowed

    class RateLimitError(Exception):
        pass

    async def failing_call():
        raise RateLimitError("Rate limited")

    with pytest.raises(MaxRetriesExceeded):
        await limiter.call_with_retry(
            failing_call,
            provider="groq",
            exceptions=(RateLimitError,)
        )
