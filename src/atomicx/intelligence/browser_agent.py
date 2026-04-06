"""Browser Agent — Real Web Research.

Performs real web scraping and content extraction for deep-dive research
on significant news stories. Uses httpx + BeautifulSoup for lightweight
extraction without requiring a headless browser.

Falls back to a structured "no data" response if scraping fails,
instead of returning fake hardcoded people.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any
from loguru import logger

from atomicx.intelligence.scanner import NewsItem

# BeautifulSoup is optional — graceful degradation
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class BrowserAgent:
    """Autonomous web research agent for deep-dive analysis.
    
    Performs real HTTP fetches and extracts structured information
    from news articles. Does NOT use mock/hardcoded data.
    """
    
    def __init__(self) -> None:
        self.logger = logger.bind(module="intelligence.browser")
        if not HAS_BS4:
            self.logger.warning(
                "beautifulsoup4 not installed. Browser Agent will operate in "
                "headline-only mode. Install with: pip install beautifulsoup4"
            )
    
    async def deep_dive(self, news_item: NewsItem) -> dict[str, Any]:
        """Perform real research on a high-significance news story.
        
        1. Fetch the original article URL
        2. Extract article text and metadata
        3. Identify mentioned people/entities
        4. Return structured findings (real data only — no fakes)
        """
        self.logger.info(f"[BROWSER] Starting deep-dive on: '{news_item.title}'")
        
        research: dict[str, Any] = {
            "story": news_item.title,
            "source": news_item.source,
            "url": news_item.url,
            "related_articles": [],
            "related_people": [],
            "causal_entities": [],  # FIX: Track institutional drivers
            "extracted_text": "",
            "sentiment_signal": "neutral",
            "urgency": "low",
            "data_quality": "none",  # Tracks how much real data we got
            "word_count": 0,  # FIX: Track content depth
        }

        # Step 1: Fetch the original article
        if news_item.url:
            article_data = await self._fetch_and_extract(news_item.url)
            if article_data:
                text = article_data.get("text", "")
                word_count = article_data.get("word_count", 0)

                research["extracted_text"] = text[:2000]
                research["related_people"] = article_data.get("people", [])
                research["causal_entities"] = article_data.get("causal_entities", [])
                research["word_count"] = word_count

                # FIX: Better data quality assessment
                if word_count > 500:
                    research["data_quality"] = "full"
                elif word_count > 100:
                    research["data_quality"] = "partial"
                else:
                    research["data_quality"] = "minimal"

                # Derive sentiment from keyword analysis of actual article text
                research["sentiment_signal"] = self._analyze_sentiment(text)

                # Urgency from keyword density
                research["urgency"] = self._assess_urgency(
                    text, news_item.keywords_matched
                )

        if research["data_quality"] in ("none", "minimal"):
            # We couldn't fetch the article — derive what we can from the headline
            research["sentiment_signal"] = self._analyze_sentiment(news_item.title)
            if research["data_quality"] == "none":
                research["data_quality"] = "headline_only"
            self.logger.warning(
                f"[BROWSER] Limited content extracted (quality={research['data_quality']}, "
                f"words={research['word_count']}). Analysis may be degraded."
            )

        people_count = len(research["related_people"])
        entities_count = len(research["causal_entities"])
        self.logger.info(
            f"[BROWSER] Deep-dive complete: data_quality={research['data_quality']}, "
            f"words={research['word_count']}, {people_count} people, "
            f"{entities_count} entities, sentiment={research['sentiment_signal']}"
        )
        
        return research
    
    async def _fetch_and_extract(self, url: str) -> dict[str, Any] | None:
        """Fetch a URL and extract structured content.

        FIX: Enhanced headers + delays + retries to bypass WAF/bot detection.
        """
        import httpx
        import random

        # FIX: Random delay before fetch (mimic human reading time)
        await asyncio.sleep(random.uniform(0.5, 2.0))

        # FIX: Extended user agent pool with latest versions
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
        ]

        # FIX: More realistic headers with Referer and additional tracking prevention
        selected_ua = random.choice(user_agents)
        headers = {
            "User-Agent": selected_ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "Accept-Language": "en-US,en;q=0.9,es;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "cross-site",  # Changed from "none" to look more natural
            "Sec-Fetch-User": "?1",
            "Sec-Ch-Ua": '"Chromium";v="123", "Not:A-Brand";v="8"',
            "Sec-Ch-Ua-Mobile": "?0",
            "Sec-Ch-Ua-Platform": '"Windows"',
            "Cache-Control": "max-age=0",
            "Referer": "https://www.google.com/",  # FIX: Add referer to look like organic traffic
        }

        # FIX: Retry logic with exponential backoff
        max_retries = 2
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=20.0,  # Increased timeout for slower responses
                    headers=headers,
                    # FIX: Disable HTTP/2 to avoid some WAF fingerprinting
                    http2=False,
                ) as client:
                    resp = await client.get(url)

                    # FIX: Better status code handling
                    if resp.status_code == 403:
                        self.logger.warning(f"[BROWSER] Blocked by WAF (403) for {url}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(random.uniform(2.0, 4.0))
                            continue
                        return None
                    elif resp.status_code == 429:
                        self.logger.warning(f"[BROWSER] Rate limited (429) for {url}")
                        await asyncio.sleep(random.uniform(5.0, 10.0))
                        continue
                    elif resp.status_code != 200:
                        self.logger.debug(f"[BROWSER] HTTP {resp.status_code} for {url}")
                        return None

                    content_type = resp.headers.get("content-type", "")
                    if "html" not in content_type and "text" not in content_type:
                        self.logger.debug(f"[BROWSER] Non-HTML content type: {content_type}")
                        return None

                    html = resp.text

                    # FIX: Detect Cloudflare challenge pages
                    if "cf-browser-verification" in html or "Just a moment" in html:
                        self.logger.warning(f"[BROWSER] Cloudflare challenge detected for {url}")
                        return None

                    if HAS_BS4:
                        return self._parse_html(html)
                    else:
                        return self._parse_html_minimal(html)

            except httpx.TimeoutException:
                self.logger.debug(f"[BROWSER] Timeout for {url} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(1.0, 3.0))
                    continue
            except Exception as e:
                self.logger.debug(f"[BROWSER] Fetch failed for {url}: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                    continue

        return None
    
    def _parse_html(self, html: str) -> dict[str, Any]:
        """Extract structured data from HTML using BeautifulSoup.

        FIX: Enhanced content extraction for crypto news sites that use
        non-standard HTML structures beyond <article> tags.
        """
        soup = BeautifulSoup(html, "html.parser")

        # Remove script/style elements
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
            tag.decompose()

        # FIX: Enhanced article extraction with multiple selectors
        # Try multiple content container strategies used by major crypto sites
        text = ""

        # Strategy 1: Standard <article> tag
        article = soup.find("article")
        if article:
            text = article.get_text(separator=" ", strip=True)

        # Strategy 2: Common crypto news site wrappers (Cointelegraph, CoinDesk, etc.)
        if not text or len(text) < 200:
            content_selectors = [
                {"class": lambda x: x and "article" in x.lower()},
                {"class": lambda x: x and "content" in x.lower()},
                {"class": lambda x: x and "post-body" in x.lower()},
                {"class": lambda x: x and "entry-content" in x.lower()},
                {"class": lambda x: x and "story-body" in x.lower()},
                {"id": lambda x: x and "content" in x.lower()},
            ]

            for selector in content_selectors:
                content_div = soup.find("div", selector)
                if content_div:
                    text = content_div.get_text(separator=" ", strip=True)
                    if len(text) > 200:  # Found substantial content
                        logger.debug(f"[PARSER] Found content via selector: {selector}")
                        break

        # Strategy 3: Meta tag fallback (og:description or meta description)
        if not text or len(text) < 100:
            logger.warning("[PARSER] Article extraction failed, falling back to meta tags")
            # Try Open Graph description
            og_desc = soup.find("meta", property="og:description")
            if og_desc and og_desc.get("content"):
                text = og_desc["content"]
                logger.debug("[PARSER] Using og:description fallback")
            else:
                # Try standard meta description
                meta_desc = soup.find("meta", attrs={"name": "description"})
                if meta_desc and meta_desc.get("content"):
                    text = meta_desc["content"]
                    logger.debug("[PARSER] Using meta description fallback")

        # Strategy 4: Last resort - all paragraphs
        if not text or len(text) < 50:
            paragraphs = soup.find_all("p")
            text = " ".join(p.get_text(strip=True) for p in paragraphs)
            logger.debug(f"[PARSER] Using paragraph fallback ({len(paragraphs)} paragraphs)")

        # Extract people mentioned (simple NER via capitalized name patterns)
        people = self._extract_people_from_text(text)

        # FIX: Extract causal drivers (institutional entities)
        causal_entities = self._extract_causal_entities(text)

        return {
            "text": text[:5000],  # Cap at 5000 chars
            "people": people,
            "causal_entities": causal_entities,
            "title": soup.title.string if soup.title else "",
            "word_count": len(text.split()),
        }
    
    def _parse_html_minimal(self, html: str) -> dict[str, Any]:
        """Minimal HTML parsing without BeautifulSoup."""
        # Strip HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        
        people = self._extract_people_from_text(text)
        
        return {
            "text": text[:5000],
            "people": people,
        }
    
    def _extract_people_from_text(self, text: str) -> list[dict[str, str]]:
        """Extract likely person names from text using pattern matching.

        NOT an NLP model — just finds "Firstname Lastname" patterns
        that appear near role indicators (CEO, Chairman, etc.)
        """
        people = []
        seen = set()

        # Common role keywords that precede or follow names
        role_patterns = [
            r"(?:CEO|CFO|CTO|Chairman|President|Secretary|Director|Minister|"
            r"Governor|Senator|Analyst|Founder|Chief)\s+([A-Z][a-z]+ [A-Z][a-z]+)",
            r"([A-Z][a-z]+ [A-Z][a-z]+),?\s+(?:the\s+)?(?:CEO|CFO|CTO|Chairman|"
            r"President|Secretary|Director|Minister|Governor|Senator|Analyst|Founder|Chief)",
        ]

        for pattern in role_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                if name not in seen and len(name) > 4:
                    seen.add(name)
                    people.append({
                        "name": name,
                        "role": "mentioned_in_article",
                        "context": text[max(0, match.start()-50):match.end()+50].strip(),
                    })

        return people[:10]  # Cap at 10 people

    def _extract_causal_entities(self, text: str) -> list[str]:
        """Extract institutional/regulatory entities that drive market moves.

        FIX: "Deep Entity Recognition" for causal drivers beyond just people.
        These are the organizations whose actions CAUSE price movements.
        """
        entities = []
        text_upper = text.upper()

        # Regulatory & Government
        regulatory = ["SEC", "CFTC", "FEDERAL RESERVE", "FED", "TREASURY", "DOJ",
                     "EUROPEAN CENTRAL BANK", "ECB", "BANK OF ENGLAND", "BOE"]

        # Exchanges & Institutions
        exchanges = ["BINANCE", "COINBASE", "KRAKEN", "FTX", "GEMINI", "BITFINEX",
                    "BLACKROCK", "GRAYSCALE", "FIDELITY", "JPMORGAN", "GOLDMAN SACHS"]

        # Protocols & Projects
        protocols = ["BITCOIN", "ETHEREUM", "TETHER", "USDC", "CIRCLE", "MICROSTRATEGY"]

        all_keywords = regulatory + exchanges + protocols

        for keyword in all_keywords:
            if keyword in text_upper:
                entities.append(keyword.title())

        return list(set(entities))[:15]  # Unique, cap at 15
    
    def _analyze_sentiment(self, text: str) -> str:
        """Derive sentiment from keyword analysis of text.
        
        Returns: 'bullish', 'bearish', 'dovish', 'hawkish', 'neutral'
        """
        text_lower = text.lower()
        
        bullish_kw = ["rally", "surge", "approval", "bullish", "growth", "recovery", "soar"]
        bearish_kw = ["crash", "dump", "ban", "hack", "collapse", "plunge", "selloff"]
        dovish_kw = ["rate cut", "easing", "stimulus", "dovish", "accommodation"]
        hawkish_kw = ["rate hike", "tightening", "hawkish", "restrictive", "inflation risk"]
        
        scores = {
            "bullish": sum(1 for kw in bullish_kw if kw in text_lower),
            "bearish": sum(1 for kw in bearish_kw if kw in text_lower),
            "dovish": sum(1 for kw in dovish_kw if kw in text_lower),
            "hawkish": sum(1 for kw in hawkish_kw if kw in text_lower),
        }
        
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "neutral"
    
    def _assess_urgency(self, text: str, matched_keywords: list[str]) -> str:
        """Assess urgency based on content and keyword density."""
        urgent_kw = ["breaking", "emergency", "crash", "hack", "exploit", "ban", "war"]
        text_lower = text.lower()
        
        urgent_count = sum(1 for kw in urgent_kw if kw in text_lower)
        
        if urgent_count >= 2 or len(matched_keywords) >= 3:
            return "high"
        elif urgent_count >= 1 or len(matched_keywords) >= 2:
            return "medium"
        return "low"
