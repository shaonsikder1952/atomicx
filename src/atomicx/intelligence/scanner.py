"""News Scanner.

Monitors RSS feeds (Reuters, Bloomberg) and X/Twitter for keywords.
Triggers a deep-dive when a story crosses the significance threshold.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Any
from loguru import logger
from pydantic import BaseModel, Field


WATCHED_KEYWORDS = [
    "federal reserve", "interest rate", "rate decision", "fomc",
    "tariff", "trade war", "sanctions",
    "oil supply", "opec", "energy crisis",
    "sec", "regulation", "etf approval", "etf rejection",
    "hack", "exploit", "rug pull",
    "elon musk", "bitcoin", "ethereum", "crypto ban",
    "war", "geopolitical", "invasion",
    "inflation", "cpi", "ppi", "gdp",
    "bank run", "liquidity crisis", "default",
]


class NewsItem(BaseModel):
    """A discovered news story."""
    item_id: str = ""
    title: str
    source: str
    url: str = ""
    keywords_matched: list[str] = Field(default_factory=list)
    significance_score: float = 0.0
    discovered_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    deep_dive_triggered: bool = False


class NewsScanner:
    """Continuously scans RSS feeds and social media for significant stories."""

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BACKOFF_SECONDS = [2, 5, 10]  # Exponential backoff
    RECONNECT_COOLDOWN_MINUTES = 5  # Wait before attempting to reconnect a failed source

    # Cost optimization defaults
    NEWS_SCAN_INTERVAL_MINUTES = 5  # Fallback if config not set

    def __init__(self, significance_threshold: float = 0.5) -> None:
        self.logger = logger.bind(module="intelligence.scanner")
        self.threshold = significance_threshold
        self.seen_items: set[str] = set()
        self.pending_deep_dives: list[NewsItem] = []

        # Cost optimization - cache news to avoid re-scanning
        self._last_news_scan: datetime | None = None
        self._cached_news: list[NewsItem] = []

        # Load cost optimization settings
        from atomicx.config import get_settings
        _settings = get_settings()
        self.NEWS_SCAN_INTERVAL_MINUTES = _settings.news_scan_interval_minutes
        self.logger.info(f"[COST-OPTIMIZATION] News scan interval: {self.NEWS_SCAN_INTERVAL_MINUTES} minutes")

        # RSS feeds to monitor
        self.rss_feeds = [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            # Backup sources (always work)
            "https://cryptopanic.com/news/rss/",
            "https://news.bitcoin.com/feed/",
        ]

        from atomicx.config import get_settings
        self._settings = get_settings()
        self._reddit = None

        # Connection health tracking
        self._source_health: dict[str, dict[str, Any]] = {
            "reddit": {"healthy": True, "last_success": None, "last_failure": None, "consecutive_failures": 0},
            "twitter": {"healthy": True, "last_success": None, "last_failure": None, "consecutive_failures": 0},
            "rss": {"healthy": True, "last_success": None, "last_failure": None, "consecutive_failures": 0},
        }
        self._total_scans = 0
        self._successful_scans = 0

    def _mark_source_success(self, source: str) -> None:
        """Mark a source as healthy after successful scan."""
        if source in self._source_health:
            self._source_health[source]["healthy"] = True
            self._source_health[source]["last_success"] = datetime.now(tz=timezone.utc)
            self._source_health[source]["consecutive_failures"] = 0

    def _mark_source_failure(self, source: str) -> None:
        """Mark a source failure and check if it should be marked unhealthy."""
        if source in self._source_health:
            self._source_health[source]["last_failure"] = datetime.now(tz=timezone.utc)
            self._source_health[source]["consecutive_failures"] += 1

            # Mark unhealthy after 3 consecutive failures
            if self._source_health[source]["consecutive_failures"] >= 3:
                self._source_health[source]["healthy"] = False
                self.logger.error(
                    f"[SCANNER] Source '{source}' marked UNHEALTHY after "
                    f"{self._source_health[source]['consecutive_failures']} consecutive failures"
                )

    def _should_attempt_reconnect(self, source: str) -> bool:
        """Check if enough time has passed to attempt reconnection."""
        if source not in self._source_health:
            return True

        health = self._source_health[source]

        # If healthy, always attempt
        if health["healthy"]:
            return True

        # If unhealthy, check cooldown period
        if health["last_failure"]:
            cooldown_delta = timedelta(minutes=self.RECONNECT_COOLDOWN_MINUTES)
            time_since_failure = datetime.now(tz=timezone.utc) - health["last_failure"]
            if time_since_failure > cooldown_delta:
                self.logger.info(f"[SCANNER] Attempting reconnection to '{source}' after cooldown")
                return True

        return False

    async def _retry_with_backoff(self, func, source: str, *args, **kwargs) -> Any:
        """Execute a function with exponential backoff retry logic."""
        for attempt in range(self.MAX_RETRIES):
            try:
                result = await func(*args, **kwargs)
                self._mark_source_success(source)
                return result
            except Exception as e:
                wait_time = self.RETRY_BACKOFF_SECONDS[min(attempt, len(self.RETRY_BACKOFF_SECONDS) - 1)]

                if attempt < self.MAX_RETRIES - 1:
                    self.logger.warning(
                        f"[SCANNER] {source} attempt {attempt + 1}/{self.MAX_RETRIES} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(
                        f"[SCANNER] {source} failed after {self.MAX_RETRIES} attempts: {e}"
                    )
                    self._mark_source_failure(source)
                    raise

        return None

    async def scan_social(self) -> list[NewsItem]:
        """Scan X and Reddit for viral keywords with reconnection logic."""
        discovered = []

        # 1. Reddit (PRAW)
        if self._settings.reddit_client_id and self._settings.reddit_client_secret:
            if self._should_attempt_reconnect("reddit"):
                try:
                    discovered.extend(await self._retry_with_backoff(
                        self._scan_reddit,
                        "reddit"
                    ))
                except Exception as e:
                    self.logger.error(f"Reddit scan failed after all retries: {e}")
            else:
                self.logger.debug("Skipping Reddit scan (in cooldown period)")

        # 2. X/Twitter (v2 Bearer Token)
        if self._settings.twitter_bearer_token:
            if self._should_attempt_reconnect("twitter"):
                try:
                    discovered.extend(await self._retry_with_backoff(
                        self._scan_twitter,
                        "twitter"
                    ))
                except Exception as e:
                    self.logger.error(f"Twitter scan failed after all retries: {e}")
            else:
                self.logger.debug("Skipping Twitter scan (in cooldown period)")

        if discovered:
            self._successful_scans += 1
        self._total_scans += 1

        return discovered

    async def _scan_reddit(self) -> list[NewsItem]:
        """Scan Reddit (extracted for retry logic)."""
        discovered = []
        import praw

        # Reconnect if needed
        if not self._reddit:
            self._reddit = praw.Reddit(
                client_id=self._settings.reddit_client_id,
                client_secret=self._settings.reddit_client_secret,
                user_agent=self._settings.reddit_user_agent
            )

        # Scan r/CryptoCurrency and r/WallStreetBets
        for sub_name in ["CryptoCurrency", "WallStreetBets"]:
            subreddit = self._reddit.subreddit(sub_name)
            for submission in subreddit.hot(limit=10):
                item = self._evaluate_headline({
                    "title": submission.title,
                    "source": f"Reddit: r/{sub_name}",
                    "url": submission.url
                })
                if item and item.item_id not in self.seen_items:
                    self.seen_items.add(item.item_id)
                    discovered.append(item)

        return discovered

    async def _scan_twitter(self) -> list[NewsItem]:
        """Scan Twitter (extracted for retry logic)."""
        discovered = []
        import httpx

        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {"Authorization": f"Bearer {self._settings.twitter_bearer_token}"}
        query = " OR ".join(WATCHED_KEYWORDS[:10])  # Simplified query
        params = {"query": query, "max_results": 10}

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, headers=headers, params=params)
            if resp.status_code == 200:
                data = resp.json()
                for tweet in data.get("data", []):
                    item = self._evaluate_headline({
                        "title": tweet.get("text", ""),
                        "source": "X/Twitter",
                        "url": f"https://twitter.com/i/web/status/{tweet.get('id')}"
                    })
                    if item and item.item_id not in self.seen_items:
                        self.seen_items.add(item.item_id)
                        discovered.append(item)
            elif resp.status_code == 429:
                raise Exception("Rate limited by Twitter API")
            else:
                raise Exception(f"Twitter API returned {resp.status_code}: {resp.text}")

        return discovered
        
    async def scan_cycle(self) -> list[NewsItem]:
        """Run one scan cycle across all feeds with retry logic."""
        # Cost optimization - only scan news every N minutes
        now = datetime.now(tz=timezone.utc)
        if self._last_news_scan is not None:
            time_since_last_scan = (now - self._last_news_scan).total_seconds() / 60
            if time_since_last_scan < self.NEWS_SCAN_INTERVAL_MINUTES:
                # Return cached news
                self.logger.debug(
                    f"[COST-SAVE] Using cached news (last scan {time_since_last_scan:.1f}m ago, "
                    f"next scan in {self.NEWS_SCAN_INTERVAL_MINUTES - time_since_last_scan:.1f}m)"
                )
                return self._cached_news

        # Time to scan fresh news
        self.logger.info(f"[NEWS-SCAN] Fetching fresh news (interval: {self.NEWS_SCAN_INTERVAL_MINUTES}m)")
        discovered = []

        if self._should_attempt_reconnect("rss"):
            try:
                discovered.extend(await self._retry_with_backoff(
                    self._scan_rss_feeds,
                    "rss"
                ))
            except Exception as e:
                self.logger.error(f"RSS scan failed after all retries: {e}")
        else:
            self.logger.debug("Skipping RSS scan (in cooldown period)")

        if discovered:
            self._successful_scans += 1
        self._total_scans += 1

        # Update cache
        self._last_news_scan = now
        self._cached_news = discovered
        self.logger.success(f"[NEWS-SCAN] Cached {len(discovered)} stories, valid for {self.NEWS_SCAN_INTERVAL_MINUTES}m")

        return discovered

    async def _scan_rss_feeds(self) -> list[NewsItem]:
        """Scan all RSS feeds (extracted for retry logic)."""
        import feedparser
        import httpx

        discovered = []

        # Headers to avoid 403 blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        }

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True, headers=headers) as client:
            for url in self.rss_feeds:
                try:
                    resp = await client.get(url, timeout=10.0)
                    if resp.status_code == 200:
                        feed = feedparser.parse(resp.text)
                        for entry in feed.entries:
                            headline = {
                                "title": entry.get("title", ""),
                                "source": feed.feed.get("title", "RSS Feed"),
                                "url": entry.get("link", "")
                            }
                            item = self._evaluate_headline(headline)
                            if item and item.item_id not in self.seen_items:
                                self.seen_items.add(item.item_id)
                                discovered.append(item)

                                if item.significance_score >= self.threshold:
                                    item.deep_dive_triggered = True
                                    self.pending_deep_dives.append(item)
                                    self.logger.warning(
                                        f"[SCANNER] HIGH SIGNIFICANCE STORY: '{item.title}' "
                                        f"(score: {item.significance_score:.2f}) → Deep-dive triggered"
                                    )
                    else:
                        self.logger.warning(f"RSS feed {url} returned {resp.status_code}")
                except Exception as e:
                    self.logger.warning(f"Failed to fetch RSS feed {url}: {e}")
                    # Don't raise - continue with other feeds

        # Only raise if ALL feeds failed
        if not discovered and len(self.rss_feeds) > 0:
            raise Exception("All RSS feeds failed to return data")

        return discovered
        
    def _evaluate_headline(self, headline: dict[str, str]) -> NewsItem | None:
        """Score a headline against watched keywords."""
        title_lower = headline["title"].lower()
        matched = [kw for kw in WATCHED_KEYWORDS if kw in title_lower]

        if not matched:
            return None

        item_id = hashlib.md5(headline["title"].encode()).hexdigest()[:12]
        score = min(1.0, len(matched) * 0.3)  # More keyword matches = higher significance

        return NewsItem(
            item_id=item_id,
            title=headline["title"],
            source=headline.get("source", "unknown"),
            url=headline.get("url", ""),
            keywords_matched=matched,
            significance_score=score,
        )

    def get_health_status(self) -> dict[str, Any]:
        """Get the current health status of all data sources."""
        healthy_sources = sum(1 for s in self._source_health.values() if s["healthy"])
        total_sources = len(self._source_health)

        return {
            "overall_healthy": healthy_sources > 0,  # At least one source working
            "healthy_sources": healthy_sources,
            "total_sources": total_sources,
            "sources": self._source_health,
            "total_scans": self._total_scans,
            "successful_scans": self._successful_scans,
            "success_rate": self._successful_scans / self._total_scans if self._total_scans > 0 else 0.0,
        }

    def is_healthy(self) -> bool:
        """Quick check if at least one source is healthy."""
        return any(s["healthy"] for s in self._source_health.values())
