"""Raw HTTP fetching for vlr.gg: rate limiting, retries, session reuse."""

from __future__ import annotations

import time

import requests

BASE_URL = "https://www.vlr.gg"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class Fetcher:
    """Thin wrapper around requests with rate limiting and retry-on-failure.

    vlr.gg's robots.txt only disallows /search/auto and /rr/, so plain GETs
    against event/team/stats pages are fine, but we still self-throttle to
    avoid hammering the site.
    """

    def __init__(
        self,
        min_interval: float = 1.5,
        max_retries: int = 3,
        timeout: float = 15.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def get_html(self, path: str, params: dict | None = None) -> str:
        """Fetch a page (path relative to vlr.gg, or a full URL) and return its HTML."""
        url = path if path.startswith("http") else f"{BASE_URL}{path}"

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            self._last_request_at = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.text
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts") from last_error

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "Fetcher":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
