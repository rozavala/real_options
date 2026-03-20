import asyncio
import random
from typing import Dict, Optional
from datetime import datetime, timezone
from ib_insync import IB
import logging

logger = logging.getLogger(__name__)


# Compressed client ID range for remote/dev Gateway connections.
# Avoids collision with production IDs (100-279) when sharing a Gateway.
# Each purpose gets base + random(0, DEV_CLIENT_ID_JITTER) + commodity_offset.
DEV_CLIENT_ID_BASE = {
    "main": 10,
    "sentinel": 16,
    "emergency": 22,
    "monitor": 28,
    "orchestrator_orders": 34,
    "dashboard_orders": 40,
    "dashboard_close": 46,
    "microstructure": 52,
    "test_utilities": 58,
    "audit": 64,
    "equity_logger": 70,
    "reconciliation": 76,
    "cleanup": 82,
    "drawdown_check": 88,
    "deferred": 94,
    "equity_service": 106,  # Master-level equity polling (no commodity offset)
}
DEV_CLIENT_ID_JITTER = 5   # random(0, 5) — widened from 4 to reduce collision probability
DEV_CLIENT_ID_DEFAULT = 100  # Unknown purposes in dev: 100-105


# Multi-commodity client ID offsets.
# KC uses base 100-279, position_monitor uses 300-399.
# CC starts at 400 to give range 500-679.
COMMODITY_ID_OFFSET = {"KC": 0, "CC": 400, "SB": 800, "NG": 1200}


def _is_remote_gateway(config: dict) -> bool:
    """Check if the configured IB Gateway host is remote (not localhost)."""
    host = config.get('connection', {}).get('host', '127.0.0.1')
    return host not in ('127.0.0.1', 'localhost', '::1')


def _resolve_purpose(purpose: str) -> str:
    """Prefix purpose with engine ticker if running in multi-engine mode.

    In single-engine mode (no EngineRuntime set), returns purpose as-is.
    In multi-engine mode, returns 'KC_sentinel', 'CC_orders', etc.
    Already-prefixed purposes (e.g., from CommodityEngine._get_ib()) pass through.
    """
    # If purpose already contains an underscore with a known ticker prefix, pass through
    if '_' in purpose:
        prefix = purpose.split('_')[0]
        if prefix in COMMODITY_ID_OFFSET:
            return purpose

    try:
        from trading_bot.data_dir_context import get_engine_runtime
        rt = get_engine_runtime()
        if rt and rt.ticker:
            return f"{rt.ticker}_{purpose}"
    except Exception:
        pass
    return purpose


class IBConnectionPool:
    _instances: Dict[str, IB] = {}
    _locks: Dict[str, asyncio.Lock] = {}
    _reconnect_backoff: Dict[str, float] = {}
    _last_reconnect_attempt: Dict[str, float] = {}

    # Track connection health
    _consecutive_failures: Dict[str, int] = {}
    FORCE_RESET_THRESHOLD = 5  # Force reset after 5 consecutive failures

    # Base client IDs - will add random offset on reconnect
    CLIENT_ID_BASE = {
        "main": 100,
        "sentinel": 110,
        "emergency": 120,
        "monitor": 130,
        "orchestrator_orders": 140,    # Orchestrator's order-related tasks
        "dashboard_orders": 160,       # Dashboard's order generation
        "dashboard_close": 180,        # Dashboard's manual position close
        "microstructure": 200,    # Range: 200-209
        "test_utilities": 220,    # Range: 220-229 (Dashboard IB test button)
        "audit": 230,             # Range: 230-239 (Position audit cycle)
        "drawdown_check": 240,    # Range: 240-249
        "equity_logger": 250,     # Range: 250-259
        "reconciliation": 260,    # Range: 260-269
        "cleanup": 270,           # Range: 270-279
        "deferred": 290,          # Range: 290-299 (deferred trigger processing)
        "equity_service": 3000,   # Range: 3000-3009 (master-level equity polling)
        # Note: position_monitor.py uses 300-399
        # Note: reconciliation uses random 5000-9999
        # DEFAULT for unknown purposes: 280-289 (avoid adding new purposes without explicit ID)
    }

    MAX_RECONNECT_BACKOFF = 600  # 10 minutes
    DISCONNECT_SETTLE_TIME = 3.0  # Seconds to wait after disconnect (allows Gateway cleanup)

    @classmethod
    async def _force_reset_connection(cls, purpose: str):
        """Force reset a specific connection - nuclear option."""
        logger.warning(f"FORCE RESETTING connection: {purpose}")

        # Remove from instances
        old_ib = cls._instances.pop(purpose, None)
        if old_ib:
            try:
                old_ib.disconnect()
                # === NEW: Give Gateway time to cleanup ===
                await asyncio.sleep(3.0)
            except Exception:
                pass

        # Reset backoff
        cls._reconnect_backoff[purpose] = 5.0
        cls._last_reconnect_attempt[purpose] = 0
        cls._consecutive_failures[purpose] = 0

        # Note: We don't reset the lock - that could cause issues

    @classmethod
    async def get_connection(cls, purpose: str, config: dict) -> IB:
        # Save unprefixed purpose for client ID lookup (DEV_CLIENT_ID_BASE keys
        # are unprefixed: "drawdown_check", "sentinel", etc.)
        purpose_base = purpose
        purpose = _resolve_purpose(purpose)
        if purpose not in cls._locks:
            cls._locks[purpose] = asyncio.Lock()
            cls._reconnect_backoff[purpose] = 5.0
            cls._last_reconnect_attempt[purpose] = 0
            cls._consecutive_failures[purpose] = 0

        async with cls._locks[purpose]:
            ib = cls._instances.get(purpose)

            # Check if existing connection is alive
            if ib and ib.isConnected():
                # === ZOMBIE DETECTION: Verify connection is actually alive ===
                try:
                    await asyncio.wait_for(ib.reqCurrentTimeAsync(), timeout=5.0)  # Increased from 2.0
                    return ib
                except asyncio.TimeoutError:
                    logger.warning(f"IB ({purpose}) zombie connection detected: reqCurrentTime timed out after 5s")
                    # Force proper cleanup before reconnecting
                    try:
                        ib.disconnect()
                        await asyncio.sleep(3.0)  # Give Gateway time to cleanup
                    except Exception:
                        pass
                    cls._instances.pop(purpose, None)
                    # Fall through to reconnect
                except Exception as e:
                    logger.warning(f"IB ({purpose}) connection health check failed: {e}")
                    # Force proper cleanup before reconnecting
                    try:
                        ib.disconnect()
                        await asyncio.sleep(3.0)
                    except Exception:
                        pass
                    cls._instances.pop(purpose, None)
                    # Fall through to reconnect

            # === BACKOFF CHECK ===
            import time
            now = time.time()
            time_since_last = now - cls._last_reconnect_attempt.get(purpose, 0)
            backoff = cls._reconnect_backoff.get(purpose, 5.0)

            if time_since_last < backoff:
                wait_remaining = backoff - time_since_last
                logger.debug(f"IB ({purpose}) in backoff, {wait_remaining:.0f}s remaining")
                raise ConnectionError(f"Connection {purpose} in backoff period")

            cls._last_reconnect_attempt[purpose] = now

            # === DIRECT CONNECTION (No TCP Pre-Flight) ===
            host = config.get('connection', {}).get('host', '127.0.0.1')
            port = config.get('connection', {}).get('port', 7497)

            # === CLEAN DISCONNECT ===
            if ib:
                try:
                    logger.info(f"Disconnecting stale IB ({purpose}) before reconnect...")
                    ib.disconnect()
                    await asyncio.sleep(cls.DISCONNECT_SETTLE_TIME)
                except Exception as e:
                    logger.warning(f"Disconnect error for {purpose}: {e}")

            # === CONNECT WITH RANDOMIZED CLIENT ID ===
            # Multi-commodity offset: each commodity gets its own client ID range
            ticker = config.get('commodity', {}).get('ticker', 'KC')
            commodity_offset = COMMODITY_ID_OFFSET.get(ticker, 0)
            is_paper = config.get('connection', {}).get('paper', False)
            remote_tag = (" [REMOTE/PAPER]" if is_paper else " [REMOTE]") if _is_remote_gateway(config) else ""

            # Retry with different client IDs on collision (Error 326)
            max_id_retries = 3
            last_connect_error = None
            for id_attempt in range(max_id_retries):
                ib = IB()
                if _is_remote_gateway(config):
                    base_id = DEV_CLIENT_ID_BASE.get(purpose_base, DEV_CLIENT_ID_DEFAULT)
                    client_id = base_id + random.randint(0, DEV_CLIENT_ID_JITTER) + commodity_offset
                else:
                    base_id = cls.CLIENT_ID_BASE.get(purpose_base, 280)
                    client_id = base_id + random.randint(0, 9) + commodity_offset

                logger.info(f"Connecting IB ({purpose}) ID: {client_id}{remote_tag}...")

                try:
                    await ib.connectAsync(
                        host=host,
                        port=port,
                        clientId=client_id,
                        timeout=15  # Increased from 10 to match verify_system_readiness
                    )
                    cls._instances[purpose] = ib
                    cls._reconnect_backoff[purpose] = 5.0
                    cls._consecutive_failures[purpose] = 0  # Reset on success
                    logger.info(f"IB ({purpose}) connected successfully with ID {client_id}")

                    # Log to state
                    try:
                        from trading_bot.state_manager import StateManager
                        StateManager.save_state({
                            f"{purpose}_ib_status": "CONNECTED",
                            "last_ib_success": datetime.now(timezone.utc).isoformat()
                        }, namespace="sensors")
                    except Exception:
                        pass

                    return ib

                except Exception as e:
                    last_connect_error = e
                    error_str = str(e).lower()
                    # Client ID collision — retry with a different ID
                    if "already in use" in error_str or "client id" in error_str:
                        logger.warning(
                            f"IB ({purpose}) client ID {client_id} collision "
                            f"(attempt {id_attempt + 1}/{max_id_retries})"
                        )
                        try:
                            ib.disconnect()
                            await asyncio.sleep(1.0)
                        except Exception:
                            pass
                        continue
                    # Non-collision error — don't retry IDs
                    break

            e = last_connect_error
            if e is not None:
                # === FIX: Clean up failed connection to prevent CLOSE-WAIT accumulation ===
                # The IB() instance may have opened a TCP socket even if the API handshake
                # failed (TimeoutError). Without explicit disconnect(), the socket enters
                # CLOSE-WAIT state and accumulates over repeated retry attempts.
                try:
                    logger.debug(f"Cleaning up failed connection attempt for {purpose}")
                    ib.disconnect()
                    await asyncio.sleep(0.5)  # Brief settle for socket cleanup
                except Exception as cleanup_err:
                    logger.debug(f"Cleanup disconnect for {purpose} raised: {cleanup_err}")
                    pass  # Ignore cleanup errors - socket may not have been fully established

                cls._reconnect_backoff[purpose] = min(
                    cls._reconnect_backoff[purpose] * 2,
                    cls.MAX_RECONNECT_BACKOFF
                )
                cls._consecutive_failures[purpose] = cls._consecutive_failures.get(purpose, 0) + 1

                # === ALERTING LOGIC ===
                # Alert exactly once when threshold is reached
                if cls._consecutive_failures[purpose] == cls.FORCE_RESET_THRESHOLD:
                    try:
                        from notifications import send_pushover_notification
                        send_pushover_notification(
                            config.get('notifications', {}),
                            "🚨 IB GATEWAY ZOMBIE DETECTED",
                            f"Gateway accepting TCP but failing API handshake "
                            f"{cls.FORCE_RESET_THRESHOLD} times in a row. "
                            f"Manual restart may be required.",
                            ticker="ALL"
                        )
                    except Exception:
                        pass

                if cls._consecutive_failures.get(purpose, 0) >= cls.FORCE_RESET_THRESHOLD:
                    await cls._force_reset_connection(purpose)

                # === NEW: Log DISCONNECTED status ===
                try:
                    from trading_bot.state_manager import StateManager
                    StateManager.save_state({
                        f"{purpose}_ib_status": "DISCONNECTED",
                        f"{purpose}_last_error": str(e)[:100],
                        "last_ib_failure": datetime.now(timezone.utc).isoformat()
                    }, namespace="sensors")
                except Exception:
                    pass

                logger.error(f"IB ({purpose}) connection failed: {e}. "
                           f"Next retry backoff: {cls._reconnect_backoff[purpose]}s")
                raise e

    # Connections that place/cancel orders — only these need the disconnect guard.
    # Non-order connections (sentinel, microstructure, audit, etc.) skip the guard.
    ORDER_BEARING_PURPOSES = {
        'orchestrator_orders', 'dashboard_orders', 'dashboard_close',
        'emergency', 'deferred', 'monitor',
    }
    DISCONNECT_GUARD_TIMEOUT = 120  # Hard ceiling: 2 minutes max wait
    TERMINAL_ORDER_STATES = {'Filled', 'Cancelled', 'Inactive', 'ApiCancelled', 'Error'}

    @classmethod
    async def _drain_pending_orders(cls, ib: IB, purpose: str):
        """Ensure no orders are in non-terminal state before disconnecting.

        For order-bearing connections only. Sends cancel requests for any
        pending orders and waits (up to DISCONNECT_GUARD_TIMEOUT) for them
        to reach terminal state. This prevents the bug where IBKR fills an
        order after the system has already disconnected, causing missed
        fill callbacks and corrupted local state.
        """
        try:
            my_client_id = ib.client.clientId
            pending = []
            for trade in ib.openTrades():
                # Only drain orders placed by THIS connection's clientId.
                # IB Gateway returns trades from ALL client connections on the
                # same account; attempting to cancel another engine's orders
                # causes Error 10147 and 120s blocking stalls.
                if trade.order.clientId != my_client_id:
                    continue
                status = trade.orderStatus.status
                if status not in cls.TERMINAL_ORDER_STATES:
                    pending.append(trade)
                    logger.warning(
                        f"DISCONNECT GUARD [{purpose}]: Order {trade.order.orderId} "
                        f"still in state '{status}'. Sending cancel before disconnect."
                    )
                    ib.cancelOrder(trade.order)

            if not pending:
                logger.info(f"DISCONNECT GUARD [{purpose}]: No pending orders. Safe to disconnect.")
                return

            logger.info(
                f"DISCONNECT GUARD [{purpose}]: Waiting for {len(pending)} order(s) "
                f"to reach terminal state (timeout={cls.DISCONNECT_GUARD_TIMEOUT}s)..."
            )

            # Pushover removed — routine when orders are active at disconnect.
            # The CRITICAL log at the end of timeout still fires for real failures.

            loop = asyncio.get_running_loop()
            deadline = loop.time() + cls.DISCONNECT_GUARD_TIMEOUT
            while pending and loop.time() < deadline:
                await asyncio.sleep(0.5)
                still_pending = []
                for trade in pending:
                    current_status = trade.orderStatus.status
                    if current_status in cls.TERMINAL_ORDER_STATES:
                        logger.info(
                            f"DISCONNECT GUARD [{purpose}]: Order "
                            f"{trade.order.orderId} -> '{current_status}'."
                        )
                    else:
                        still_pending.append(trade)
                pending = still_pending

            if pending:
                logger.critical(
                    f"DISCONNECT GUARD [{purpose}]: {len(pending)} order(s) STILL "
                    f"non-terminal after {cls.DISCONNECT_GUARD_TIMEOUT}s! "
                    f"Disconnecting anyway. "
                    f"Orders: {[t.order.orderId for t in pending]}. "
                    f"RECONCILIATION WILL BE NEEDED."
                )
            else:
                logger.info(f"DISCONNECT GUARD [{purpose}]: All orders terminal. Safe to disconnect.")

        except Exception as e:
            logger.error(
                f"DISCONNECT GUARD [{purpose}]: Exception during guard check: {e}. "
                f"Proceeding with disconnect."
            )

    @classmethod
    def _is_order_bearing(cls, purpose: str) -> bool:
        """Check if a connection purpose is order-bearing (needs disconnect guard)."""
        # Strip commodity prefix (e.g., "KC_orchestrator_orders" -> "orchestrator_orders")
        base = purpose
        if '_' in purpose:
            prefix = purpose.split('_')[0]
            if prefix in COMMODITY_ID_OFFSET:
                base = purpose[len(prefix) + 1:]
        return base in cls.ORDER_BEARING_PURPOSES

    @classmethod
    async def release_connection(cls, purpose: str):
        """Releases a specific connection from the pool.

        For order-bearing connections, drains pending orders before disconnecting
        to prevent missed fill callbacks.
        """
        purpose = _resolve_purpose(purpose)
        if purpose not in cls._locks:
            cls._locks[purpose] = asyncio.Lock()

        async with cls._locks[purpose]:
            ib = cls._instances.pop(purpose, None)
            if ib:
                if ib.isConnected():
                    # Disconnect guard for order-bearing connections
                    if cls._is_order_bearing(purpose):
                        await cls._drain_pending_orders(ib, purpose)
                    else:
                        logger.debug(
                            f"DISCONNECT GUARD: Connection '{purpose}' is not "
                            f"order-bearing. Direct disconnect."
                        )
                    logger.info(f"Disconnecting IB ({purpose})...")
                    ib.disconnect()
                    await asyncio.sleep(3.0)
                # Reset backoff when explicitly released
                cls._reconnect_backoff[purpose] = 5.0

    @classmethod
    async def release_all(cls):
        """Release all pooled connections with proper cleanup.

        Order-bearing connections get the disconnect guard; others disconnect directly.
        """
        for name, ib in list(cls._instances.items()):
            try:
                if ib and ib.isConnected():
                    if cls._is_order_bearing(name):
                        await cls._drain_pending_orders(ib, name)
                    ib.disconnect()
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Connection cleanup failed for {name}: {e}")
            finally:
                cls._instances.pop(name, None)

        cls._instances.clear()
        cls._reconnect_backoff.clear()

        import gc
        gc.collect()
