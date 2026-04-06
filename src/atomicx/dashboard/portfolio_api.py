"""Portfolio Management API endpoints.

Separate from trading signals — this tracks user's actual positions and P&L.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from loguru import logger

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel


router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])

# ═══ FIX: Global callback for dynamic asset addition ═══
# Set by run.py to connect portfolio to cognitive loop
_on_add_symbol_callback = None


def set_add_symbol_callback(callback):
    """Set the callback that's triggered when a new asset is added."""
    global _on_add_symbol_callback
    _on_add_symbol_callback = callback
    logger.info("[PORTFOLIO-API] Asset addition callback registered")


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic Models
# ═══════════════════════════════════════════════════════════════════════════


class AddAssetRequest(BaseModel):
    symbol: str
    asset_type: str = "crypto"  # crypto, stock, commodity, forex


class AddPositionRequest(BaseModel):
    symbol: str
    side: str  # long, short
    entry_price: float
    quantity: float
    notes: str | None = None


class ClosePositionRequest(BaseModel):
    exit_price: float
    notes: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# API Endpoints
# ═══════════════════════════════════════════════════════════════════════════


@router.post("/assets/add")
async def add_asset(request: AddAssetRequest):
    """Add a new asset to portfolio."""
    try:
        from sqlalchemy import select
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import PortfolioAsset

        async with get_session() as session:
            # Check if already exists
            result = await session.execute(
                select(PortfolioAsset).where(PortfolioAsset.symbol == request.symbol)
            )
            existing = result.scalar_one_or_none()

            if existing:
                if not existing.is_active:
                    # Reactivate if was deactivated
                    existing.is_active = True
                    await session.commit()
                    logger.info(f"[PORTFOLIO] Reactivated asset: {request.symbol}")
                    return {"status": "reactivated", "asset_id": existing.id}
                elif existing.status == "error":
                    # Autonomous recovery: wipe error and retry logic
                    existing.error_message = None
                    existing.status = "initializing"
                    existing.backfill_progress = 0
                    await session.commit()
                    logger.warning(f"[PORTFOLIO] Asset {request.symbol} was in error state, autonomously retrying initialization.")
                    
                    # Call the background initialization again
                    from atomicx.data.asset_manager import get_asset_manager
                    asset_manager = get_asset_manager()
                    await asset_manager.initialize_asset_background(
                        symbol=request.symbol,
                        asset_type=request.asset_type,
                        callback=_on_add_symbol_callback
                    )
                    return {
                        "status": "restarting",
                        "asset_id": existing.id,
                        "message": f"Retrying initialization for {request.symbol}. Check status at /api/portfolio/assets/{request.symbol}/status"
                    }
                else:
                    return {"status": "already_exists", "asset_id": existing.id}

            # Create new asset
            asset = PortfolioAsset(
                symbol=request.symbol,
                asset_type=request.asset_type
            )
            session.add(asset)
            await session.commit()
            await session.refresh(asset)

            logger.info(f"[PORTFOLIO] Added new asset: {request.symbol} ({request.asset_type})")

            # ═══ AUTONOMOUS ASSET INITIALIZATION ═══
            # Trigger background initialization: backfill data, compute variables, start tracking
            from atomicx.data.asset_manager import get_asset_manager

            asset_manager = get_asset_manager()
            await asset_manager.initialize_asset_background(
                symbol=request.symbol,
                asset_type=request.asset_type,
                callback=_on_add_symbol_callback  # Add to cognitive loop when ready
            )

            logger.success(
                f"[PORTFOLIO] Asset {request.symbol} initialization started in background"
            )

            return {
                "status": "created",
                "asset_id": asset.id,
                "message": f"{request.symbol} is initializing. Check status at /api/portfolio/assets/{request.symbol}/status"
            }

    except Exception as e:
        # Handle duplicate key error gracefully
        if "duplicate key" in str(e) or "UniqueViolationError" in str(e):
            logger.warning(f"[PORTFOLIO] Asset {request.symbol} already exists (race condition)")
            return {"status": "already_exists", "message": f"Asset {request.symbol} is already in your portfolio"}
        logger.error(f"[PORTFOLIO] Failed to add asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets")
async def get_assets():
    """Get all portfolio assets."""
    try:
        from sqlalchemy import select
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import PortfolioAsset, PortfolioStats

        async with get_session() as session:
            # Get all active assets
            result = await session.execute(
                select(PortfolioAsset).where(PortfolioAsset.is_active == True).order_by(PortfolioAsset.created_at)
            )
            assets = result.scalars().all()

            # Get stats for each asset
            assets_data = []
            for asset in assets:
                stats_result = await session.execute(
                    select(PortfolioStats).where(PortfolioStats.symbol == asset.symbol)
                )
                stats = stats_result.scalar_one_or_none()

                assets_data.append({
                    "id": asset.id,
                    "symbol": asset.symbol,
                    "asset_type": asset.asset_type,
                    "created_at": asset.created_at.isoformat(),
                    "stats": {
                        "total_invested": float(stats.total_invested) if stats else 0.0,
                        "total_pnl": float(stats.total_pnl) if stats else 0.0,
                        "total_pnl_percentage": float(stats.total_pnl_percentage) if stats else 0.0,
                        "open_positions": stats.open_positions if stats else 0,
                        "closed_positions": stats.closed_positions if stats else 0,
                        "win_rate": float(stats.win_rate) if stats else 0.0
                    }
                })

            return {"assets": assets_data}

    except Exception as e:
        logger.error(f"[PORTFOLIO] Failed to get assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/positions/add")
async def add_position(request: AddPositionRequest):
    """Open a new position."""
    try:
        from sqlalchemy import select
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import PortfolioAsset, Position

        async with get_session() as session:
            # Get asset
            result = await session.execute(
                select(PortfolioAsset).where(PortfolioAsset.symbol == request.symbol)
            )
            asset = result.scalar_one_or_none()

            if not asset:
                raise HTTPException(status_code=404, detail=f"Asset {request.symbol} not in portfolio. Add it first.")

            # Create position
            position = Position(
                portfolio_asset_id=asset.id,
                symbol=request.symbol,
                side=request.side,
                entry_price=Decimal(str(request.entry_price)),
                quantity=Decimal(str(request.quantity)),
                notes=request.notes,
                status="open"
            )
            session.add(position)
            await session.commit()
            await session.refresh(position)

            # Update stats
            await _update_portfolio_stats(session, request.symbol)

            logger.info(f"[PORTFOLIO] Opened position: {request.symbol} {request.side} @ {request.entry_price}")
            return {"status": "created", "position_id": position.id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PORTFOLIO] Failed to add position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/positions/{position_id}/close")
async def close_position(position_id: int, request: ClosePositionRequest):
    """Close an open position."""
    try:
        from sqlalchemy import select
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import Position

        async with get_session() as session:
            # Get position
            result = await session.execute(
                select(Position).where(Position.id == position_id)
            )
            position = result.scalar_one_or_none()

            if not position:
                raise HTTPException(status_code=404, detail=f"Position {position_id} not found")

            if position.status != "open":
                raise HTTPException(status_code=400, detail=f"Position is already {position.status}")

            # Calculate P&L
            exit_price = Decimal(str(request.exit_price))
            entry_value = position.entry_price * position.quantity
            exit_value = exit_price * position.quantity

            if position.side == "long":
                pnl = exit_value - entry_value
            else:  # short
                pnl = entry_value - exit_value

            pnl_percentage = (pnl / entry_value) * 100

            # Update position
            position.exit_time = datetime.now(timezone.utc)
            position.exit_price = exit_price
            position.pnl = pnl
            position.pnl_percentage = pnl_percentage
            position.status = "closed"
            if request.notes:
                position.notes = (position.notes or "") + "\n" + request.notes

            await session.commit()

            # Update stats
            await _update_portfolio_stats(session, position.symbol)

            logger.info(f"[PORTFOLIO] Closed position {position_id}: P&L={float(pnl):.2f} ({float(pnl_percentage):.2f}%)")
            return {
                "status": "closed",
                "pnl": float(pnl),
                "pnl_percentage": float(pnl_percentage)
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PORTFOLIO] Failed to close position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/positions")
async def get_positions(symbol: str | None = None, status: str | None = None):
    """Get all positions, optionally filtered by symbol or status."""
    try:
        from sqlalchemy import select
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import Position

        async with get_session() as session:
            query = select(Position)

            if symbol:
                query = query.where(Position.symbol == symbol)
            if status:
                query = query.where(Position.status == status)

            query = query.order_by(Position.entry_time.desc())

            result = await session.execute(query)
            positions = result.scalars().all()

            positions_data = []
            for pos in positions:
                positions_data.append({
                    "id": pos.id,
                    "symbol": pos.symbol,
                    "side": pos.side,
                    "entry_time": pos.entry_time.isoformat(),
                    "entry_price": float(pos.entry_price),
                    "quantity": float(pos.quantity),
                    "exit_time": pos.exit_time.isoformat() if pos.exit_time else None,
                    "exit_price": float(pos.exit_price) if pos.exit_price else None,
                    "status": pos.status,
                    "pnl": float(pos.pnl) if pos.pnl else None,
                    "pnl_percentage": float(pos.pnl_percentage) if pos.pnl_percentage else None,
                    "notes": pos.notes
                })

            return {"positions": positions_data}

    except Exception as e:
        logger.error(f"[PORTFOLIO] Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assets/{symbol}/status")
async def get_asset_status(symbol: str):
    """Get initialization status for a specific asset.

    Returns current status, progress, data source, and any error messages.
    """
    try:
        from atomicx.data.asset_manager import get_asset_manager

        asset_manager = get_asset_manager()
        status = await asset_manager.get_initialization_status(symbol)

        if status.get("status") == "not_found":
            raise HTTPException(
                status_code=404,
                detail=f"Asset {symbol} not found in portfolio"
            )

        return status

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[PORTFOLIO] Failed to get status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_portfolio_stats(symbol: str | None = None):
    """Get portfolio statistics (global or per-asset)."""
    try:
        from sqlalchemy import select
        from atomicx.data.storage.database import get_session
        from atomicx.data.storage.models import PortfolioStats

        async with get_session() as session:
            result = await session.execute(
                select(PortfolioStats).where(PortfolioStats.symbol == symbol)
            )
            stats = result.scalar_one_or_none()

            if not stats:
                # Return empty stats if not found
                return {
                    "symbol": symbol,
                    "total_invested": 0.0,
                    "total_pnl": 0.0,
                    "total_pnl_percentage": 0.0,
                    "total_positions": 0,
                    "open_positions": 0,
                    "closed_positions": 0,
                    "winning_positions": 0,
                    "losing_positions": 0,
                    "win_rate": 0.0,
                    "avg_win": None,
                    "avg_loss": None,
                    "largest_win": None,
                    "largest_loss": None
                }

            return {
                "symbol": stats.symbol,
                "total_invested": float(stats.total_invested),
                "total_pnl": float(stats.total_pnl),
                "total_pnl_percentage": float(stats.total_pnl_percentage),
                "total_positions": stats.total_positions,
                "open_positions": stats.open_positions,
                "closed_positions": stats.closed_positions,
                "winning_positions": stats.winning_positions,
                "losing_positions": stats.losing_positions,
                "win_rate": float(stats.win_rate),
                "avg_win": float(stats.avg_win) if stats.avg_win else None,
                "avg_loss": float(stats.avg_loss) if stats.avg_loss else None,
                "largest_win": float(stats.largest_win) if stats.largest_win else None,
                "largest_loss": float(stats.largest_loss) if stats.largest_loss else None,
                "updated_at": stats.updated_at.isoformat()
            }

    except Exception as e:
        logger.error(f"[PORTFOLIO] Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════


async def _update_portfolio_stats(session: Any, symbol: str | None = None):
    """Recalculate and update portfolio statistics."""
    from sqlalchemy import select, func
    from atomicx.data.storage.models import Position, PortfolioStats

    # Get all positions for this symbol (or global if symbol is None)
    query = select(Position)
    if symbol:
        query = query.where(Position.symbol == symbol)

    result = await session.execute(query)
    positions = result.scalars().all()

    # Calculate stats
    total_invested = sum(pos.entry_price * pos.quantity for pos in positions)
    open_positions = sum(1 for pos in positions if pos.status == "open")
    closed_positions = sum(1 for pos in positions if pos.status == "closed")
    total_pnl = sum(pos.pnl for pos in positions if pos.pnl is not None)

    winning_positions = sum(1 for pos in positions if pos.pnl and pos.pnl > 0)
    losing_positions = sum(1 for pos in positions if pos.pnl and pos.pnl < 0)

    win_rate = (winning_positions / closed_positions * 100) if closed_positions > 0 else 0.0

    wins = [float(pos.pnl) for pos in positions if pos.pnl and pos.pnl > 0]
    losses = [float(pos.pnl) for pos in positions if pos.pnl and pos.pnl < 0]

    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    largest_win = max(wins) if wins else None
    largest_loss = min(losses) if losses else None

    total_pnl_percentage = (float(total_pnl) / float(total_invested) * 100) if total_invested > 0 else 0.0

    # Update or create stats record
    stats_result = await session.execute(
        select(PortfolioStats).where(PortfolioStats.symbol == symbol)
    )
    stats = stats_result.scalar_one_or_none()

    if stats:
        stats.total_invested = total_invested
        stats.total_pnl = total_pnl
        stats.total_pnl_percentage = Decimal(str(total_pnl_percentage))
        stats.total_positions = len(positions)
        stats.open_positions = open_positions
        stats.closed_positions = closed_positions
        stats.winning_positions = winning_positions
        stats.losing_positions = losing_positions
        stats.win_rate = Decimal(str(win_rate))
        stats.avg_win = Decimal(str(avg_win)) if avg_win else None
        stats.avg_loss = Decimal(str(avg_loss)) if avg_loss else None
        stats.largest_win = Decimal(str(largest_win)) if largest_win else None
        stats.largest_loss = Decimal(str(largest_loss)) if largest_loss else None
    else:
        stats = PortfolioStats(
            symbol=symbol,
            total_invested=total_invested,
            total_pnl=total_pnl,
            total_pnl_percentage=Decimal(str(total_pnl_percentage)),
            total_positions=len(positions),
            open_positions=open_positions,
            closed_positions=closed_positions,
            winning_positions=winning_positions,
            losing_positions=losing_positions,
            win_rate=Decimal(str(win_rate)),
            avg_win=Decimal(str(avg_win)) if avg_win else None,
            avg_loss=Decimal(str(avg_loss)) if avg_loss else None,
            largest_win=Decimal(str(largest_win)) if largest_win else None,
            largest_loss=Decimal(str(largest_loss)) if largest_loss else None
        )
        session.add(stats)

    await session.commit()
    logger.debug(f"[PORTFOLIO] Updated stats for {symbol or 'global'}: P&L={float(total_pnl):.2f}, WR={win_rate:.1f}%")
