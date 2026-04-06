"""Browser Pool with Playwright CDP Control.

Extracted from OpenClaw's browser automation patterns.
Solves WAF/403 blocks with real browser fingerprints and JavaScript execution.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger


@dataclass
class BrowserContext:
    """A managed browser context with CDP control."""
    context_id: str
    context: Any  # playwright.async_api.BrowserContext
    page: Any     # playwright.async_api.Page
    created_at: datetime
    last_used: datetime
    request_count: int = 0


class BrowserPool:
    """Managed browser pool with Playwright CDP control.

    Extracted from OpenClaw's browser automation system.

    Key Features:
    - Persistent browser contexts (reduce startup latency)
    - Stealth mode (bypass WAF detection)
    - JavaScript execution (dynamic content loading)
    - Screenshot capability (OCR for embedded charts)
    - Automatic cleanup (idle timeout)
    """

    def __init__(
        self,
        pool_size: int = 3,
        idle_timeout_seconds: int = 600,  # 10 minutes
        headless: bool = True,
    ):
        """Initialize browser pool.

        Args:
            pool_size: Number of persistent browser contexts
            idle_timeout_seconds: Close contexts after this idle time
            headless: Run browsers in headless mode
        """
        self._pool_size = pool_size
        self._idle_timeout = idle_timeout_seconds
        self._headless = headless

        self._playwright = None
        self._browser = None
        self._contexts: dict[str, BrowserContext] = {}
        self._context_lock = asyncio.Lock()

        self.logger = logger.bind(module="intelligence.browser_pool")

    async def _ensure_browser(self) -> None:
        """Ensure Playwright browser is launched."""
        if self._browser is not None:
            return

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "Playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        self.logger.info("[BROWSER-POOL] Launching Playwright Chromium...")

        self._playwright = await async_playwright().start()

        # Launch Chromium with stealth settings
        self._browser = await self._playwright.chromium.launch(
            headless=self._headless,
            args=[
                "--disable-blink-features=AutomationControlled",  # Hide automation
                "--disable-dev-shm-usage",                        # Docker compatibility
                "--no-sandbox",                                   # Docker compatibility
                "--disable-setuid-sandbox",
            ]
        )

        self.logger.success(
            f"[BROWSER-POOL] ✓ Browser launched (headless={self._headless})"
        )

    async def get_page(
        self, context_id: str = "default"
    ) -> tuple[Any, BrowserContext]:
        """Get or create a browser page from pool.

        Args:
            context_id: Context identifier (isolates sessions)

        Returns:
            (page, context) tuple
        """
        await self._ensure_browser()

        async with self._context_lock:
            # Check if context exists and is still valid
            if context_id in self._contexts:
                ctx = self._contexts[context_id]

                # Check idle timeout
                idle_seconds = (datetime.now(tz=timezone.utc) - ctx.last_used).total_seconds()
                if idle_seconds < self._idle_timeout:
                    ctx.last_used = datetime.now(tz=timezone.utc)
                    ctx.request_count += 1
                    return ctx.page, ctx
                else:
                    # Timeout exceeded, close and recreate
                    self.logger.debug(
                        f"[BROWSER-POOL] Context {context_id} idle for {idle_seconds:.0f}s, closing"
                    )
                    await ctx.context.close()
                    del self._contexts[context_id]

            # Create new context
            if len(self._contexts) >= self._pool_size:
                # Pool full, close oldest context
                oldest_id = min(
                    self._contexts.keys(),
                    key=lambda k: self._contexts[k].last_used
                )
                self.logger.debug(
                    f"[BROWSER-POOL] Pool full, closing oldest context: {oldest_id}"
                )
                await self._contexts[oldest_id].context.close()
                del self._contexts[oldest_id]

            # Create context with stealth settings
            context = await self._browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                locale="en-US",
                timezone_id="America/New_York",
                permissions=["geolocation"],
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
            )

            # Create page
            page = await context.new_page()

            # Stealth JavaScript injections
            await page.add_init_script("""
                // Hide webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });

                // Mock plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });

                // Mock languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });

                // Mock chrome property
                window.chrome = {
                    runtime: {}
                };

                // Mock permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)

            # Store context
            ctx = BrowserContext(
                context_id=context_id,
                context=context,
                page=page,
                created_at=datetime.now(tz=timezone.utc),
                last_used=datetime.now(tz=timezone.utc),
                request_count=1,
            )
            self._contexts[context_id] = ctx

            self.logger.success(
                f"[BROWSER-POOL] ✓ Created context: {context_id}"
            )

            return page, ctx

    async def fetch(
        self,
        url: str,
        wait_for: str = "domcontentloaded",
        timeout: int = 30000,
        screenshot: bool = False,
        context_id: str = "default",
    ) -> dict[str, Any]:
        """Fetch URL with browser and extract content.

        Args:
            url: URL to fetch
            wait_for: Wait condition ("load", "domcontentloaded", "networkidle")
            timeout: Timeout in milliseconds
            screenshot: Take screenshot for OCR
            context_id: Browser context identifier

        Returns:
            Dict with html, text, screenshot (if requested)
        """
        page, ctx = await self.get_page(context_id)

        try:
            self.logger.debug(f"[BROWSER-POOL] Fetching: {url}")

            # Navigate to URL
            response = await page.goto(url, wait_until=wait_for, timeout=timeout)

            if response is None:
                raise RuntimeError(f"Failed to load {url}")

            # Extract content
            html = await page.content()
            text = await page.inner_text("body")

            result = {
                "url": url,
                "status_code": response.status,
                "html": html,
                "text": text,
                "screenshot": None,
            }

            # Take screenshot if requested
            if screenshot:
                screenshot_bytes = await page.screenshot(
                    type="png",
                    full_page=True
                )
                result["screenshot"] = screenshot_bytes

            self.logger.success(
                f"[BROWSER-POOL] ✓ Fetched {url} ({response.status}) - {len(text)} chars"
            )

            return result

        except Exception as e:
            self.logger.error(f"[BROWSER-POOL] ✗ Failed to fetch {url}: {e}")
            raise

    async def execute_script(
        self,
        script: str,
        context_id: str = "default",
    ) -> Any:
        """Execute JavaScript in browser context.

        Args:
            script: JavaScript code to execute
            context_id: Browser context identifier

        Returns:
            Script return value
        """
        page, _ = await self.get_page(context_id)
        return await page.evaluate(script)

    async def cleanup_idle_contexts(self) -> None:
        """Close contexts that have exceeded idle timeout."""
        async with self._context_lock:
            now = datetime.now(tz=timezone.utc)
            to_remove = []

            for context_id, ctx in self._contexts.items():
                idle_seconds = (now - ctx.last_used).total_seconds()
                if idle_seconds >= self._idle_timeout:
                    self.logger.debug(
                        f"[BROWSER-POOL] Closing idle context: {context_id} "
                        f"(idle: {idle_seconds:.0f}s)"
                    )
                    await ctx.context.close()
                    to_remove.append(context_id)

            for context_id in to_remove:
                del self._contexts[context_id]

    async def shutdown(self) -> None:
        """Shutdown browser pool and cleanup all contexts."""
        self.logger.info("[BROWSER-POOL] Shutting down...")

        # Close all contexts
        async with self._context_lock:
            for ctx in self._contexts.values():
                try:
                    await ctx.context.close()
                except Exception as e:
                    self.logger.warning(f"[BROWSER-POOL] Error closing context: {e}")

            self._contexts.clear()

        # Close browser
        if self._browser:
            await self._browser.close()
            self._browser = None

        # Stop playwright
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None

        self.logger.success("[BROWSER-POOL] ✓ Shutdown complete")

    def get_stats(self) -> dict[str, Any]:
        """Get browser pool statistics.

        Returns:
            Dict with pool size, active contexts, request counts
        """
        return {
            "pool_size": self._pool_size,
            "active_contexts": len(self._contexts),
            "contexts": {
                ctx_id: {
                    "requests": ctx.request_count,
                    "age_seconds": (datetime.now(tz=timezone.utc) - ctx.created_at).total_seconds(),
                    "idle_seconds": (datetime.now(tz=timezone.utc) - ctx.last_used).total_seconds(),
                }
                for ctx_id, ctx in self._contexts.items()
            }
        }


# Global singleton
_global_browser_pool: Optional[BrowserPool] = None


def get_browser_pool() -> BrowserPool:
    """Get or create global browser pool instance."""
    global _global_browser_pool
    if _global_browser_pool is None:
        _global_browser_pool = BrowserPool(
            pool_size=3,                 # 3 persistent contexts
            idle_timeout_seconds=600,    # 10 min idle timeout
            headless=True,               # Headless mode
        )
    return _global_browser_pool


async def shutdown_browser_pool() -> None:
    """Shutdown global browser pool."""
    global _global_browser_pool
    if _global_browser_pool is not None:
        await _global_browser_pool.shutdown()
        _global_browser_pool = None
