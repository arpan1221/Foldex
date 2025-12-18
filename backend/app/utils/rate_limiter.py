"""Advanced rate limiting utilities for API calls."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with burst support.
    
    More sophisticated than simple rate limiting - allows bursts while
    maintaining average rate limits over time.
    """

    def __init__(
        self,
        rate: float,  # tokens per second
        capacity: int,  # max tokens in bucket
        initial_tokens: Optional[float] = None,
    ):
        """Initialize token bucket rate limiter.

        Args:
            rate: Rate at which tokens are added (per second)
            capacity: Maximum number of tokens in bucket
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.rate = rate
        self.capacity = capacity
        self.tokens: float = initial_tokens if initial_tokens is not None else float(capacity)
        self.last_update = datetime.utcnow()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire

        Raises:
            ValueError: If requesting more tokens than capacity
        """
        if tokens > self.capacity:
            raise ValueError(f"Cannot acquire {tokens} tokens (capacity: {self.capacity})")

        async with self._lock:
            while True:
                now = datetime.utcnow()
                time_passed = (now - self.last_update).total_seconds()

                # Add tokens based on time passed
                self.tokens = min(
                    self.capacity, self.tokens + time_passed * self.rate
                )
                self.last_update = now

                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(
                        "Rate limiter: tokens acquired",
                        tokens_acquired=tokens,
                        tokens_remaining=self.tokens,
                    )
                    return

                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.rate
                logger.debug(
                    "Rate limiter: waiting for tokens",
                    tokens_needed=tokens_needed,
                    wait_seconds=wait_time,
                )
                await asyncio.sleep(wait_time)


class AdaptiveRateLimiter:
    """
    Adaptive rate limiter that adjusts based on API responses.
    
    Backs off when hitting rate limits, speeds up when successful.
    """

    def __init__(
        self,
        initial_rate: float = 10.0,  # requests per second
        min_rate: float = 1.0,
        max_rate: float = 20.0,
        backoff_factor: float = 0.5,
        recovery_factor: float = 1.1,
    ):
        """Initialize adaptive rate limiter.

        Args:
            initial_rate: Starting rate (requests/second)
            min_rate: Minimum allowed rate
            max_rate: Maximum allowed rate
            backoff_factor: Factor to reduce rate on errors (< 1.0)
            recovery_factor: Factor to increase rate on success (> 1.0)
        """
        self.current_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.backoff_factor = backoff_factor
        self.recovery_factor = recovery_factor
        self.last_call = datetime.utcnow()
        self._lock = asyncio.Lock()
        self.success_count = 0
        self.error_count = 0

    async def acquire(self) -> None:
        """Acquire permission to make an API call."""
        async with self._lock:
            now = datetime.utcnow()
            time_since_last = (now - self.last_call).total_seconds()
            min_interval = 1.0 / self.current_rate

            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self.last_call = datetime.utcnow()

    async def report_success(self) -> None:
        """Report successful API call."""
        async with self._lock:
            self.success_count += 1
            self.error_count = 0  # Reset error count on success

            # Gradually increase rate after sustained success
            if self.success_count >= 10:
                old_rate = self.current_rate
                self.current_rate = min(
                    self.max_rate, self.current_rate * self.recovery_factor
                )
                if old_rate != self.current_rate:
                    logger.info(
                        "Rate limiter: increased rate",
                        old_rate=old_rate,
                        new_rate=self.current_rate,
                    )
                self.success_count = 0

    async def report_error(self, is_rate_limit: bool = False) -> None:
        """Report failed API call.

        Args:
            is_rate_limit: Whether the error was due to rate limiting
        """
        async with self._lock:
            self.error_count += 1
            self.success_count = 0  # Reset success count on error

            # Immediate backoff on rate limit errors
            if is_rate_limit or self.error_count >= 3:
                old_rate = self.current_rate
                self.current_rate = max(
                    self.min_rate, self.current_rate * self.backoff_factor
                )
                logger.warning(
                    "Rate limiter: decreased rate",
                    old_rate=old_rate,
                    new_rate=self.current_rate,
                    is_rate_limit=is_rate_limit,
                    error_count=self.error_count,
                )
                self.error_count = 0


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter for multiple time windows.
    
    Useful for APIs with multiple rate limit tiers (e.g., per second, per minute, per hour).
    """

    def __init__(self, windows: List[tuple[int, float]]):
        """Initialize sliding window rate limiter.

        Args:
            windows: List of (max_calls, time_window_seconds) tuples
                    Example: [(10, 1.0), (100, 60.0)] = 10/sec, 100/min
        """
        self.windows = windows
        self.calls: deque = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make an API call."""
        async with self._lock:
            while True:
                now = datetime.utcnow()

                # Check all windows
                can_proceed = True
                wait_times = []

                for max_calls, window_seconds in self.windows:
                    # Count calls within this window
                    cutoff = now - timedelta(seconds=window_seconds)
                    recent_calls = [
                        call_time for call_time in self.calls if call_time > cutoff
                    ]

                    if len(recent_calls) >= max_calls:
                        # Calculate wait time for this window
                        oldest_in_window = min(recent_calls)
                        wait_time = (
                            window_seconds
                            - (now - oldest_in_window).total_seconds()
                            + 0.1
                        )
                        wait_times.append(wait_time)
                        can_proceed = False

                if can_proceed:
                    # Clean up old calls (keep last hour for safety)
                    cutoff = now - timedelta(hours=1)
                    self.calls = deque(
                        [call_time for call_time in self.calls if call_time > cutoff]
                    )
                    self.calls.append(now)
                    return

                # Wait for the longest required time
                wait_time = max(wait_times)
                logger.debug(
                    "Rate limiter: waiting for window",
                    wait_seconds=wait_time,
                )
                await asyncio.sleep(wait_time)


class GoogleDriveRateLimiter:
    """
    Specialized rate limiter for Google Drive API.
    
    Handles Google Drive's specific rate limits:
    - 1000 queries per 100 seconds per user
    - Burst allowance for short periods
    """

    def __init__(self):
        """Initialize Google Drive rate limiter."""
        # Google Drive allows bursts, so use token bucket
        # 1000 queries per 100 seconds = 10 queries/second average
        # Allow burst capacity of 20 for initial rapid requests
        self.token_bucket = TokenBucketRateLimiter(
            rate=10.0,  # 10 requests/second average
            capacity=20,  # Allow burst of 20 requests
            initial_tokens=20,
        )

        # Also track 100-second window to ensure we don't exceed 1000
        self.sliding_window = SlidingWindowRateLimiter(
            windows=[
                (20, 1.0),  # Max 20 per second (burst)
                (1000, 100.0),  # Max 1000 per 100 seconds (Google limit)
            ]
        )

        # Adaptive limiter to back off on errors
        self.adaptive = AdaptiveRateLimiter(
            initial_rate=10.0, min_rate=2.0, max_rate=15.0
        )

    async def acquire(self, tokens: int = 1) -> None:
        """Acquire permission to make API call(s).

        Args:
            tokens: Number of API calls to make (default 1)
        """
        # Check all limiters
        await self.token_bucket.acquire(tokens)
        for _ in range(tokens):
            await self.sliding_window.acquire()
        await self.adaptive.acquire()

    async def report_success(self) -> None:
        """Report successful API call."""
        await self.adaptive.report_success()

    async def report_error(self, is_rate_limit: bool = False) -> None:
        """Report failed API call.

        Args:
            is_rate_limit: Whether the error was a rate limit error
        """
        await self.adaptive.report_error(is_rate_limit)

