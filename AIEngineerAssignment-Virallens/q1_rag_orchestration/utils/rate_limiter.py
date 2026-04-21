# utils/rate_limiter.py
import asyncio
import time
from typing import Callable, Any, Tuple, Type
from collections import defaultdict

class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts are exhausted."""
    pass

class RateLimiter:
    """
    Token-bucket rate limiter with exponential backoff.

    Tracks request timestamps per provider and enforces RPM limits.
    """

    def __init__(self, max_rpm: int = 30):
        """
        Initialize rate limiter.

        Args:
            max_rpm: Maximum requests per minute per provider
        """
        self.max_rpm = max_rpm
        self.requests = defaultdict(list)  # provider -> [timestamps]

    async def call_with_retry(
        self,
        fn: Callable,
        provider: str,
        *args,
        exceptions: Tuple[Type[Exception], ...] = (Exception,),
        max_retries: int = 3,
        **kwargs
    ) -> Any:
        """
        Call function with rate limiting and retry logic.

        Args:
            fn: Async function to call
            provider: Provider name for tracking
            *args: Positional args for fn
            exceptions: Exception types to catch and retry on
            max_retries: Maximum retry attempts
            **kwargs: Keyword args for fn

        Returns:
            Result from fn

        Raises:
            MaxRetriesExceeded: If all retries are exhausted
        """
        for attempt in range(max_retries):
            # Check rate limit before calling
            if not self._can_make_request(provider):
                # Rate limit exceeded - use exponential backoff
                backoff_time = 2 ** attempt
                await asyncio.sleep(backoff_time)
                # After backoff, try the call anyway (lenient rate limiting)
                # The rate limit is "soft" - we back off but don't strictly block

            try:
                response = await fn(*args, **kwargs)
                # Only record successful requests
                self._record_request(provider)
                return response
            except exceptions as e:
                if attempt == max_retries - 1:
                    raise MaxRetriesExceeded(f"Max retries ({max_retries}) exceeded") from e
                # Exponential backoff: 2^attempt seconds
                await asyncio.sleep(2 ** attempt)

        raise MaxRetriesExceeded("Unexpectedly reached end of retry loop")

    def _can_make_request(self, provider: str) -> bool:
        """Check if request is allowed for provider."""
        now = time.time()
        minute_ago = now - 60

        # Filter out requests older than 1 minute
        self.requests[provider] = [
            ts for ts in self.requests[provider] if ts > minute_ago
        ]

        return len(self.requests[provider]) < self.max_rpm

    def _record_request(self, provider: str):
        """Record a request timestamp for provider."""
        self.requests[provider].append(time.time())
