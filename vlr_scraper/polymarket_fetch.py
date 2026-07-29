"""Raw HTTP fetching for Polymarket's public read APIs: rate limiting, retries,
session reuse. Mirrors fetch.Fetcher's shape but talks to three different
hosts (no auth needed for any of these -- all public market-data reads):

  - gamma-api.polymarket.com  -- event/market metadata (discovery, token ids,
    schedule times, resolved outcomes)
  - clob.polymarket.com       -- per-token historical price series + the
    live order book
  - data-api.polymarket.com   -- historical trade tape
"""

from __future__ import annotations

import time

import requests

GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
DATA_BASE_URL = "https://data-api.polymarket.com"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


class PolymarketFetcher:
    """Thin JSON-GET wrapper with rate limiting and retry-on-failure, shared
    across the three Polymarket hosts (each call passes its own base_url)."""

    def __init__(
        self,
        min_interval: float = 0.25,
        max_retries: int = 4,
        timeout: float = 20.0,
        user_agent: str = DEFAULT_USER_AGENT,
    ):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_request_at = 0.0
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "application/json"})

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        wait = self.min_interval - elapsed
        if wait > 0:
            time.sleep(wait)

    def get_json(self, url: str, params: dict | None = None):
        """GET a full URL (already including host) and return parsed JSON.
        Treats a 404 as "no data" (returns None) rather than retrying --
        Polymarket's APIs use 404 for e.g. a token with no trade history yet,
        which isn't a transient failure.

        clob.polymarket.com sometimes answers with HTTP 200 and a JSON body
        like {"error": "pg: connection pool timeout"} instead of a non-2xx
        status -- observed on /prices-history under load. Treat that as a
        transient failure (retry) if the message says so; otherwise (e.g.
        "invalid filters: ...", a real 400-shaped problem) it'll never
        succeed on retry, so return None immediately instead of burning
        the retry budget."""
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            self._last_request_at = time.monotonic()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, dict) and "error" in data and "history" not in data:
                    msg = str(data["error"]).lower()
                    if any(kw in msg for kw in ("timeout", "connection", "unavailable")) and attempt < self.max_retries:
                        time.sleep(2**attempt)
                        continue
                    return None
                return data
            except requests.RequestException as exc:
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(2**attempt)
        if last_error is None:
            return None
        raise RuntimeError(f"Failed to fetch {url} after {self.max_retries} attempts") from last_error

    def get_gamma(self, path: str, params: dict | None = None):
        return self.get_json(f"{GAMMA_BASE_URL}{path}", params=params)

    def get_clob(self, path: str, params: dict | None = None):
        return self.get_json(f"{CLOB_BASE_URL}{path}", params=params)

    def get_data_api(self, path: str, params: dict | None = None):
        return self.get_json(f"{DATA_BASE_URL}{path}", params=params)

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "PolymarketFetcher":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
