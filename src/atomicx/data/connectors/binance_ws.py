"""Binance WebSocket connector for real-time market data streaming.

Handles:
- Real-time trade data (tick-by-tick)
- Kline/candlestick updates (1m, 5m, 15m, 1h, 4h, 1d)
- Order book depth updates
- Funding rate updates (for perpetuals)

Publishes all data to Kafka topics for downstream consumption.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any

import aiohttp
from loguru import logger

from atomicx.config import get_settings
from atomicx.common.cache import get_sensory_cache


class BinanceWebSocketConnector:
    """Connects to Binance WebSocket API and streams market data.

    Supports multiple concurrent streams per symbol.
    Automatically reconnects on disconnection with exponential backoff.
    """

    def __init__(
        self,
        symbols: list[str] | None = None,
        on_trade: Any = None,
        on_kline: Any = None,
        on_depth: Any = None,
        on_cvd: Any = None,  # INSTITUTIONAL FIX: CVD callback
        on_liquidation: Any = None,  # INSTITUTIONAL FIX: Liquidation callback
    ) -> None:
        settings = get_settings()
        self._base_url = settings.binance_ws_url
        self._symbols = symbols or settings.default_symbols
        self._on_trade = on_trade
        self._on_kline = on_kline
        self._on_depth = on_depth
        self._on_cvd = on_cvd
        self._on_liquidation = on_liquidation
        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None

        # INSTITUTIONAL FIX: Separate WebSocket for global liquidation stream
        # The !forceOrder@arr stream doesn't work with combined streams
        self._liquidation_session: aiohttp.ClientSession | None = None
        self._liquidation_ws: aiohttp.ClientWebSocketResponse | None = None
        self._liquidation_task: asyncio.Task | None = None

        self._cache = get_sensory_cache()
        self._running = False
        self._reconnect_delay = 1.0
        self._max_reconnect_delay = 60.0
        # FIX: Heartbeat watchdog to detect zombie connections
        self._last_message_time: datetime = datetime.now(tz=timezone.utc)
        self._watchdog_task: asyncio.Task | None = None
        self._stale_threshold_seconds = 60  # Force reconnect if no data for 60s

        # FIX: CVD (Cumulative Volume Delta) tracking - detects orderbook spoofing
        # Tracks actual executed volume (aggressive buys vs sells)
        self._cvd_cache: dict[str, float] = {}  # symbol -> cumulative delta
        self._cvd_reset_time: datetime = datetime.now(tz=timezone.utc)

    def _normalize_symbol(self, symbol: str) -> str:
        """Convert 'BTC/USDT' to 'btcusdt' for Binance streams."""
        return symbol.replace("/", "").lower()

    def _denormalize_symbol(self, binance_symbol: str) -> str:
        """Convert Binance format 'BTCUSDT' back to internal 'BTC/USDT' format.

        This ensures cache keys match what Variable Engine expects.
        """
        # Assume all pairs end with USDT for now (can be extended later)
        if binance_symbol.endswith("USDT") and len(binance_symbol) > 4:
            base = binance_symbol[:-4]  # "BTCUSDT" → "BTC"
            return f"{base}/USDT"
        return binance_symbol

    def _build_stream_url(self) -> str:
        """Build combined stream URL for all symbols and data types.

        INSTITUTIONAL FIX: Use proper combined stream endpoint with comma-separated streams.
        Format: wss://stream.binance.com:9443/stream?streams=stream1/stream2/stream3

        NOTE: Binance combined streams use SLASH separator, not comma!
        """
        streams = []
        for symbol in self._symbols:
            s = self._normalize_symbol(symbol)
            streams.append(f"{s}@trade")
            streams.append(f"{s}@kline_1m")
            # Depth format for combined stream: @depth<levels> OR @depth (no update speed in combined)
            streams.append(f"{s}@depth20")

        # FIX: Use SLASH-separated format for combined streams (Binance docs confirm this)
        stream_names = "/".join(streams)
        return f"{self._base_url}/stream?streams={stream_names}"

    def _build_liquidation_stream_url(self) -> str:
        """Build URL for global liquidation stream (Pain Map).

        INSTITUTIONAL FIX: Separate WebSocket connection required.
        The !forceOrder@arr global stream doesn't work with combined streams.

        Format: wss://fstream.binance.com/ws/!forceOrder@arr

        NOTE: Binance Futures WebSocket does NOT use port 9443 (unlike spot streams).
        """
        # Use futures stream endpoint (fstream) - no port needed
        return "wss://fstream.binance.com/ws/!forceOrder@arr"

    async def _watchdog(self) -> None:
        """Heartbeat watchdog to detect zombie connections.

        FIX: If no messages received for 60s, force reconnect even if connection "appears" open.
        Prevents the "15:12 UTC freeze" where zombie sockets hang without formal close.
        """
        logger.info("[WATCHDOG] Heartbeat monitor started (60s threshold)")

        while self._running:
            await asyncio.sleep(10)  # Check every 10 seconds

            if not self._running:
                break

            now = datetime.now(tz=timezone.utc)
            time_since_last_msg = (now - self._last_message_time).total_seconds()

            if time_since_last_msg > self._stale_threshold_seconds:
                logger.error(
                    f"[WATCHDOG] ZOMBIE CONNECTION DETECTED! No data for {time_since_last_msg:.0f}s. "
                    f"Last message: {self._last_message_time.strftime('%H:%M:%S UTC')}. Force reconnecting..."
                )

                # Force close zombie connection
                if self._ws and not self._ws.closed:
                    await self._ws.close()

                break  # Exit watchdog loop to trigger reconnect

    async def _liquidation_stream(self) -> None:
        """Separate WebSocket connection for global liquidation stream.

        INSTITUTIONAL FIX: The !forceOrder@arr stream doesn't work in combined streams,
        so we need a dedicated connection for Pain Map data.
        """
        if not self._on_liquidation:
            logger.info("[LIQUIDATION-STREAM] Callback not configured, skipping liquidation stream")
            return

        liquidation_url = self._build_liquidation_stream_url()
        reconnect_delay = 1.0

        logger.info(f"[LIQUIDATION-STREAM] Starting dedicated liquidation WebSocket: {liquidation_url}")

        while self._running:
            try:
                self._liquidation_session = aiohttp.ClientSession()
                logger.debug(f"[LIQUIDATION-STREAM] Attempting connection to {liquidation_url}")

                self._liquidation_ws = await self._liquidation_session.ws_connect(
                    liquidation_url,
                    heartbeat=20,
                    timeout=aiohttp.ClientTimeout(total=30)
                )
                reconnect_delay = 1.0  # Reset on successful connection
                logger.success("[LIQUIDATION-STREAM] Connected to global liquidation feed!")

                message_count = 0
                async for msg in self._liquidation_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        message_count += 1
                        if message_count == 1:
                            logger.info(f"[LIQUIDATION-STREAM] First message received: {msg.data[:200]}")

                        data = json.loads(msg.data)
                        event_type = data.get("e", "")
                        if event_type == "forceOrder":
                            await self._process_liquidation(data)
                        else:
                            if message_count <= 3:
                                logger.debug(f"[LIQUIDATION-STREAM] Non-forceOrder message: {event_type}")

                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        logger.error(f"[LIQUIDATION-STREAM] WebSocket error: {self._liquidation_ws.exception()}")
                        break
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                        logger.warning("[LIQUIDATION-STREAM] Connection closed")
                        break

            except Exception as e:
                if not self._running:
                    break
                logger.error(
                    f"[LIQUIDATION-STREAM] Error: {e}. Reconnecting in {reconnect_delay}s"
                )
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, self._max_reconnect_delay)

            finally:
                if self._liquidation_session and not self._liquidation_session.closed:
                    await self._liquidation_session.close()

    async def start(self) -> None:
        """Start the WebSocket connection with auto-reconnect.

        INSTITUTIONAL FIX: Runs two concurrent WebSocket connections:
        1. Combined stream for trades, klines, and depth (per symbol)
        2. Dedicated stream for global liquidations (!forceOrder@arr)
        """
        self._running = True
        logger.info(
            "Starting Binance WebSocket connector",
            symbols=self._symbols,
        )

        # Start liquidation stream in background (separate connection)
        if self._on_liquidation:
            self._liquidation_task = asyncio.create_task(self._liquidation_stream())
            logger.info("[LIQUIDATION-STREAM] Background task launched")

        while self._running:
            try:
                # Start watchdog
                if self._watchdog_task is None or self._watchdog_task.done():
                    self._watchdog_task = asyncio.create_task(self._watchdog())

                await self._connect_and_listen()
            except Exception as e:
                if not self._running:
                    break
                logger.error(
                    f"WebSocket error: {e}. Reconnecting in {self._reconnect_delay}s"
                )
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(
                    self._reconnect_delay * 2, self._max_reconnect_delay
                )

    async def _connect_and_listen(self) -> None:
        """Establish WebSocket connection and process messages."""
        url = self._build_stream_url()
        logger.info(f"Connecting to Binance WebSocket: {url[:80]}...")

        self._session = aiohttp.ClientSession()
        try:
            self._ws = await self._session.ws_connect(url, heartbeat=20)
            self._reconnect_delay = 1.0  # Reset on successful connection
            logger.success(f"Binance WebSocket connected: {len(self._symbols)} symbols active")

            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING):
                    logger.warning("WebSocket connection closed")
                    break
        finally:
            if self._session and not self._session.closed:
                await self._session.close()

    async def _handle_message(self, data: dict) -> None:
        """Route incoming WebSocket messages to appropriate handlers."""
        # FIX: Update last message timestamp (heartbeat watchdog)
        self._last_message_time = datetime.now(tz=timezone.utc)

        # Combined stream format: data is nested under 'data' key
        if "stream" in data:
            stream = data["stream"]
            payload = data.get("data", data)
        else:
            # Single stream format
            stream = data.get("e", "")
            payload = data

        event_type = payload.get("e", "")

        if event_type == "trade":
            await self._process_trade(payload)
        elif event_type == "kline":
            await self._process_kline(payload)
        elif event_type == "depthUpdate" or "lastUpdateId" in payload:
            await self._process_depth(payload)
        elif event_type == "forceOrder":
            # INSTITUTIONAL: Global Liquidation Stream (Pain Map)
            await self._process_liquidation(payload)

    async def _process_trade(self, data: dict) -> None:
        """Process individual trade events."""
        trade = {
            "timestamp": datetime.fromtimestamp(data["T"] / 1000, tz=timezone.utc),
            "symbol": data["s"],
            "price": Decimal(data["p"]),
            "quantity": Decimal(data["q"]),
            "is_buyer_maker": data["m"],
            "trade_id": data["t"],
        }
        if self._on_trade:
            await self._on_trade(trade)

        # FIX: Update sensory cache with denormalized symbol (BTC/USDT format)
        symbol_internal = self._denormalize_symbol(data["s"])
        self._cache.update(symbol_internal, "LAST_PRICE", float(data["p"]))

        # FIX: Calculate CVD (Cumulative Volume Delta) - Anti-Spoofing Layer
        # Tracks ACTUAL executed volume to detect if orderbook walls are real
        # is_buyer_maker=True → Seller hit bid (bearish) → negative delta
        # is_buyer_maker=False → Buyer hit ask (bullish) → positive delta
        quantity = float(data["q"])
        delta = -quantity if data["m"] else quantity  # Seller aggressive = negative

        # Reset CVD daily (prevents unbounded growth)
        now = datetime.now(tz=timezone.utc)
        if (now - self._cvd_reset_time).total_seconds() > 86400:  # 24 hours
            logger.info("[CVD] 24-hour reset triggered")
            self._cvd_cache.clear()
            self._cvd_reset_time = now

        # Update cumulative delta
        current_cvd = self._cvd_cache.get(symbol_internal, 0.0)
        new_cvd = current_cvd + delta
        self._cvd_cache[symbol_internal] = new_cvd

        # Push to sensory cache
        self._cache.update(symbol_internal, "CVD", new_cvd)

        # INSTITUTIONAL FIX: Persist CVD to database
        if self._on_cvd:
            await self._on_cvd(symbol_internal, new_cvd)

        # Log significant CVD changes (>1000 BTC/ETH equivalent)
        if abs(delta) > 10:  # Large single trade
            logger.debug(
                f"[CVD] {symbol_internal}: Large {('BUY' if delta > 0 else 'SELL')} "
                f"delta={delta:.2f}, cumulative CVD={new_cvd:.2f}"
            )

    async def _process_kline(self, data: dict) -> None:
        """Process kline/candlestick events."""
        k = data["k"]
        kline = {
            "timestamp": datetime.fromtimestamp(k["t"] / 1000, tz=timezone.utc),
            "symbol": data["s"],
            "timeframe": k["i"],
            "open": Decimal(k["o"]),
            "high": Decimal(k["h"]),
            "low": Decimal(k["l"]),
            "close": Decimal(k["c"]),
            "volume": Decimal(k["v"]),
            "quote_volume": Decimal(k["q"]),
            "trade_count": k["n"],
            "is_closed": k["x"],
        }
        if self._on_kline:
            await self._on_kline(kline)

    async def _process_depth(self, data: dict) -> None:
        """Process order book depth snapshots."""
        bids = data.get("bids", data.get("b", []))
        asks = data.get("asks", data.get("a", []))

        # FIX: Symbol fallback for depth snapshots (which don't include "s" field)
        # Depth snapshots don't include "s", so extract from subscribed symbols
        symbol = data.get("s", "")
        if not symbol and self._symbols:
            # Convert internal format (BTC/USDT) to Binance format (BTCUSDT)
            symbol = self._normalize_symbol(self._symbols[0]).upper()  # "BTC/USDT" → "BTCUSDT"

        depth = {
            "timestamp": datetime.now(tz=timezone.utc),
            "symbol": symbol,
            "bids": [[Decimal(p), Decimal(q)] for p, q in bids[:20]],
            "asks": [[Decimal(p), Decimal(q)] for p, q in asks[:20]],
        }

        # Compute derived fields
        if depth["bids"] and depth["asks"]:
            best_bid = depth["bids"][0][0]
            best_ask = depth["asks"][0][0]
            depth["spread"] = best_ask - best_bid
            depth["mid_price"] = (best_bid + best_ask) / 2
            depth["bid_total_volume"] = sum(q for _, q in depth["bids"])
            depth["ask_total_volume"] = sum(q for _, q in depth["asks"])

        if self._on_depth:
            await self._on_depth(depth)

        # Update Sensory Cache with Microstructure Metrics
        if "bid_total_volume" in depth and "ask_total_volume" in depth:
            v_bids = float(depth["bid_total_volume"])
            v_asks = float(depth["ask_total_volume"])
            total = v_bids + v_asks
            if total > 0:
                imbalance = (v_bids - v_asks) / total  # -1 (short heavy) to +1 (long heavy)
                # FIX: Denormalize symbol to match Variable Engine format
                symbol_binance = depth["symbol"]
                symbol_internal = self._denormalize_symbol(symbol_binance)
                self._cache.update(symbol_internal, "OB_IMBALANCE", imbalance)
                self._cache.update(symbol_internal, "BID_VOL", v_bids)
                self._cache.update(symbol_internal, "ASK_VOL", v_asks)

    async def _process_liquidation(self, data: dict) -> None:
        """Process liquidation events (Pain Map).

        INSTITUTIONAL: Tracks where retail traders are getting liquidated.
        This reveals hidden support/resistance levels.

        Event format:
        {
            "e": "forceOrder",
            "E": 1568014460893,  # event time
            "o": {
                "s": "BTCUSDT",   # symbol
                "S": "SELL",      # side
                "o": "LIMIT",     # order type
                "f": "IOC",       # time in force
                "q": "0.014",     # quantity
                "p": "9910",      # price
                "ap": "9910",     # average price
                "X": "FILLED",    # status
                "l": "0.014",     # last filled quantity
                "z": "0.014",     # filled quantity
                "T": 1568014460893  # trade time
            }
        }
        """
        order = data.get("o", {})
        if not order:
            return

        liquidation = {
            "timestamp": datetime.fromtimestamp(data["E"] / 1000, tz=timezone.utc),
            "symbol": order["s"],
            "side": order["S"],  # SELL = long liquidation, BUY = short liquidation
            "price": Decimal(order["p"]),
            "quantity": Decimal(order["q"]),
            "order_type": order.get("o", "UNKNOWN"),
        }

        # Log significant liquidations (> $10k at current BTC price ~$67k)
        qty = float(liquidation["quantity"])
        price = float(liquidation["price"])
        notional = qty * price

        if notional > 10000:  # $10k+ liquidation
            side_emoji = "🔻" if liquidation["side"] == "SELL" else "🔺"
            logger.warning(
                f"[LIQUIDATION] {side_emoji} {liquidation['symbol']}: "
                f"{liquidation['side']} ${notional:,.0f} @ ${price:,.2f}"
            )

        # Store in sensory cache for microstructure analysis
        symbol_internal = self._denormalize_symbol(liquidation["symbol"])
        self._cache.update(symbol_internal, "LAST_LIQUIDATION_PRICE", float(price))
        self._cache.update(symbol_internal, "LAST_LIQUIDATION_SIDE", liquidation["side"])

        # INSTITUTIONAL FIX: Persist liquidation to database
        if self._on_liquidation:
            # Convert to internal format for OHLCVSaver
            liquidation_normalized = {
                "timestamp": liquidation["timestamp"],
                "symbol": symbol_internal,
                "side": liquidation["side"],
                "price": liquidation["price"],
                "quantity": liquidation["quantity"],
                "order_type": liquidation["order_type"],
            }
            await self._on_liquidation(liquidation_normalized)

    async def subscribe(self, symbol: str) -> None:
        """Add a symbol to the live stream dynamically."""
        if symbol not in self._symbols:
            self._symbols.append(symbol)
            logger.info(f"Subscribing to new symbol: {symbol}")
            if self._ws:
                await self._ws.close() # Reconnect-and-listen will handle the rest

    async def stop(self) -> None:
        """Gracefully stop the WebSocket connection.

        INSTITUTIONAL FIX: Cleans up both main and liquidation streams.
        """
        self._running = False

        # Stop main combined stream
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()

        # Stop liquidation stream
        if self._liquidation_ws and not self._liquidation_ws.closed:
            await self._liquidation_ws.close()
        if self._liquidation_session and not self._liquidation_session.closed:
            await self._liquidation_session.close()

        # Wait for liquidation task to finish
        if self._liquidation_task and not self._liquidation_task.done():
            self._liquidation_task.cancel()
            try:
                await self._liquidation_task
            except asyncio.CancelledError:
                pass

        logger.info("Binance WebSocket connector stopped (main + liquidation streams)")
