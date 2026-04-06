"""Asset initialization manager - orchestrates autonomous asset setup.

When a new asset is added through the UI, this manager:
1. Validates the symbol
2. Detects asset type
3. Downloads historical data (all timeframes)
4. Updates initialization progress
5. Subscribes to real-time data
6. Adds to cognitive loop

All without manual intervention or system restarts.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from sqlalchemy import select, update

from atomicx.data.connectors.router import get_connector_router, AssetType
from atomicx.data.storage.database import get_session
from atomicx.data.storage.models import PortfolioAsset


class AssetInitializationManager:
    """Manages autonomous asset initialization and tracking."""

    # Timeframes to backfill
    TIMEFRAMES = ["1m", "5m", "15m", "1h", "4h", "1d"]
    # Candles per timeframe
    CANDLES_PER_TIMEFRAME = 500

    def __init__(self):
        self.logger = logger.bind(module="asset.manager")
        self.router = get_connector_router()
        self._initialization_tasks: dict[str, asyncio.Task] = {}

    async def initialize_asset(
        self,
        symbol: str,
        asset_type: str | None = None,
        callback: Any | None = None
    ) -> dict[str, Any]:
        """Initialize a new asset completely autonomously.

        Args:
            symbol: Asset symbol (e.g., "BTC/USDT", "AAPL", "CL=F")
            asset_type: Optional asset type override (auto-detected if None)
            callback: Optional callback function to call when complete

        Returns:
            Dictionary with initialization results
        """
        self.logger.info(f"⚡ Starting autonomous initialization for {symbol}")

        try:
            # Step 1: Validate and detect asset type
            await self._update_status(symbol, "initializing", 0)

            if not asset_type:
                detected_type, normalized_symbol = self.router.detect_asset_type(symbol)
                asset_type = detected_type.value
                symbol = normalized_symbol
                self.logger.info(f"Auto-detected {symbol} as {asset_type}")

            # Validate symbol exists
            is_valid, error = await self.router.validate_symbol(symbol)
            if not is_valid:
                await self._update_status(
                    symbol, "error", 0,
                    error_message=error or "Symbol validation failed"
                )
                return {"status": "error", "message": error}

            # Update database with data source
            data_source = self.router.get_data_source_name(
                AssetType(asset_type) if isinstance(asset_type, str) else asset_type
            )
            await self._update_data_source(symbol, data_source)

            # Step 2: Backfill historical data for all timeframes
            await self._update_status(symbol, "backfilling", 10)
            await self._backfill_historical_data(symbol)

            # Step 3: Mark as active
            await self._update_status(symbol, "active", 100)
            await self._update_last_update(symbol)

            self.logger.success(
                f"✅ Asset {symbol} fully initialized and ready for trading"
            )

            # Step 4: Trigger callback (add to cognitive loop)
            if callback:
                try:
                    callback(symbol)
                    self.logger.info(f"Triggered callback for {symbol} (added to cognitive loop)")
                except Exception as e:
                    self.logger.error(f"Callback failed for {symbol}: {e}")

            return {
                "status": "success",
                "symbol": symbol,
                "asset_type": asset_type,
                "data_source": data_source,
                "message": f"{symbol} ready for analysis"
            }

        except Exception as e:
            self.logger.error(f"Failed to initialize {symbol}: {e}")
            await self._update_status(
                symbol, "error", 0,
                error_message=str(e)
            )
            return {"status": "error", "message": str(e)}

    async def initialize_asset_background(
        self,
        symbol: str,
        asset_type: str | None = None,
        callback: Any | None = None
    ) -> None:
        """Initialize asset in background (non-blocking).

        Creates an async task that runs initialization.
        """
        task = asyncio.create_task(
            self.initialize_asset(symbol, asset_type, callback)
        )
        self._initialization_tasks[symbol] = task

        # Add done callback to clean up
        task.add_done_callback(lambda t: self._initialization_tasks.pop(symbol, None))

        self.logger.info(f"Started background initialization for {symbol}")

    async def _backfill_historical_data(self, symbol: str) -> None:
        """Download historical data for all timeframes.

        Downloads 500 candles for each timeframe:
        1m, 5m, 15m, 1h, 4h, 1d
        """
        connector = await self.router.get_connector(symbol)
        await connector.start()

        total_timeframes = len(self.TIMEFRAMES)
        completed = 0

        try:
            for timeframe in self.TIMEFRAMES:
                try:
                    self.logger.info(
                        f"Backfilling {symbol} {timeframe} "
                        f"({completed + 1}/{total_timeframes})..."
                    )

                    # Fetch historical candles
                    bars = await connector.get_historical_ohlcv(
                        timeframe=timeframe,
                        limit=self.CANDLES_PER_TIMEFRAME
                    )

                    # Save to database
                    await self._save_ohlcv_bars(symbol, timeframe, bars)

                    completed += 1
                    progress = 10 + int((completed / total_timeframes) * 80)  # 10-90%
                    await self._update_status(symbol, "backfilling", progress)

                    self.logger.success(
                        f"✓ Saved {len(bars)} {timeframe} candles for {symbol}"
                    )

                except Exception as e:
                    self.logger.warning(
                        f"Failed to backfill {timeframe} for {symbol}: {e}. Continuing..."
                    )

            # Compute initial variables (takes progress from 90% to 95%)
            await self._update_status(symbol, "backfilling", 90)
            await self._compute_initial_variables(symbol)
            await self._update_status(symbol, "backfilling", 95)

        finally:
            await connector.stop()

    async def _save_ohlcv_bars(
        self,
        symbol: str,
        timeframe: str,
        bars: list[Any]
    ) -> None:
        """Save OHLCV bars to database."""
        if not bars:
            return

        from atomicx.data.storage.models import OHLCV

        async with get_session() as session:
            for bar in bars:
                # Upsert (insert or update)
                ohlcv = OHLCV(
                    timestamp=bar.timestamp,
                    symbol=symbol,
                    timeframe=timeframe,
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                    quote_volume=bar.quote_volume,
                    trade_count=bar.trade_count,
                )

                # Use merge to handle duplicates
                await session.merge(ohlcv)

            await session.commit()

    async def _compute_initial_variables(self, symbol: str) -> None:
        """Compute initial variable values for the asset.

        This populates the variable engine with the first set of indicators.
        """
        try:
            from atomicx.variables.engine import VariableComputeEngine

            var_engine = VariableComputeEngine()

            # Add symbol to engine
            if hasattr(var_engine, '_symbols'):
                if symbol not in var_engine._symbols:
                    var_engine._symbols.append(symbol)

            # Compute snapshot (this will calculate all 46 indicators)
            await var_engine.compute_snapshot(symbol)

            self.logger.success(f"Computed initial variables for {symbol}")

        except Exception as e:
            self.logger.error(f"Failed to compute initial variables for {symbol}: {e}")
            # Don't fail initialization over this

    async def _update_status(
        self,
        symbol: str,
        status: str,
        progress: int,
        error_message: str | None = None
    ) -> None:
        """Update asset status in database."""
        async with get_session() as session:
            stmt = (
                update(PortfolioAsset)
                .where(PortfolioAsset.symbol == symbol)
                .values(
                    status=status,
                    backfill_progress=progress,
                    error_message=error_message,
                    updated_at=datetime.now(timezone.utc)
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def _update_data_source(self, symbol: str, data_source: str) -> None:
        """Update data source in database."""
        async with get_session() as session:
            stmt = (
                update(PortfolioAsset)
                .where(PortfolioAsset.symbol == symbol)
                .values(data_source=data_source)
            )
            await session.execute(stmt)
            await session.commit()

    async def _update_last_update(self, symbol: str) -> None:
        """Update last_update timestamp."""
        async with get_session() as session:
            stmt = (
                update(PortfolioAsset)
                .where(PortfolioAsset.symbol == symbol)
                .values(last_update=datetime.now(timezone.utc))
            )
            await session.execute(stmt)
            await session.commit()

    async def get_initialization_status(self, symbol: str) -> dict[str, Any]:
        """Get current initialization status for an asset."""
        async with get_session() as session:
            result = await session.execute(
                select(PortfolioAsset).where(PortfolioAsset.symbol == symbol)
            )
            asset = result.scalar_one_or_none()

            if not asset:
                return {"status": "not_found"}

            return {
                "symbol": asset.symbol,
                "status": asset.status,
                "progress": asset.backfill_progress,
                "data_source": asset.data_source,
                "error_message": asset.error_message,
                "last_update": asset.last_update.isoformat() if asset.last_update else None,
            }


# Global singleton
_asset_manager: AssetInitializationManager | None = None


def get_asset_manager() -> AssetInitializationManager:
    """Get the global asset manager instance."""
    global _asset_manager
    if _asset_manager is None:
        _asset_manager = AssetInitializationManager()
    return _asset_manager
