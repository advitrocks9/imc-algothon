"""Async ETF-First order executor.

Executes the LON_ETF leg first; aborts if it fails. Only then fires
the 3 component legs concurrently via asyncio.gather. Uses a persistent
aiohttp.ClientSession on a background event loop for true async I/O.
"""

import asyncio
import logging
import threading
import time
from dataclasses import asdict, dataclass

import aiohttp

from bot_template import OrderRequest, OrderResponse

log = logging.getLogger("executor")

MAX_429_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds

ETF_SYMBOL = "LON_ETF"

STANDARD_HEADERS = {"Content-Type": "application/json; charset=utf-8"}


@dataclass
class LegResult:
    """Result of executing a single IOC leg."""

    order: OrderRequest
    response: OrderResponse | None
    filled: int
    unfilled: int
    error: str | None


@dataclass
class ExecutionReport:
    """Result of executing all legs of an arb."""

    legs: list[LegResult]
    target_volume: int
    min_filled: int
    max_filled: int
    all_filled_equal: bool
    timestamp: float


class AsyncExecutor:
    """ETF-First async executor with persistent aiohttp session.

    Architecture:
      - A daemon thread runs an asyncio event loop.
      - An aiohttp.ClientSession lives on that loop and persists across ticks.
      - The sync execute_ioc_batch() bridges into the async loop via
        run_coroutine_threadsafe(), blocking until the result is ready.

    ETF-First logic:
      1. Execute the LON_ETF leg alone.
      2. If it fills 0 or errors, abort — do not send component legs.
      3. If it fills > 0, fire the 3 component legs concurrently.
      4. If no LON_ETF in the batch (e.g. scratch), fire all concurrently.
    """

    def __init__(self, cmi_url: str, auth_token: str):
        self._cmi_url = cmi_url.rstrip("/")
        self._auth_headers = {**STANDARD_HEADERS, "Authorization": auth_token}

        # Persistent event loop in a daemon thread
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="async-executor",
        )
        self._thread.start()

        # Create session on the loop (blocks until ready)
        self._session: aiohttp.ClientSession = asyncio.run_coroutine_threadsafe(
            self._create_session(), self._loop,
        ).result()

    async def _create_session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(
            base_url=self._cmi_url,
            headers=self._auth_headers,
        )

    # --- Public sync API (same signature as old ConcurrentExecutor) ---

    def execute_ioc_batch(self, orders: list[OrderRequest]) -> ExecutionReport:
        """Send orders with ETF-First logic, cancel unfilled remainders.

        Blocks the calling thread until all legs complete.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._async_execute_ioc_batch(orders), self._loop,
        )
        return future.result()

    def shutdown(self) -> None:
        """Close the aiohttp session and stop the event loop."""
        try:
            asyncio.run_coroutine_threadsafe(
                self._session.close(), self._loop,
            ).result(timeout=5)
        except Exception:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    # --- Core async logic ---

    async def _async_execute_ioc_batch(
        self, orders: list[OrderRequest],
    ) -> ExecutionReport:
        # Partition into ETF leg and component legs
        etf_order = None
        comp_orders: list[OrderRequest] = []
        for o in orders:
            if o.product == ETF_SYMBOL:
                etf_order = o
            else:
                comp_orders.append(o)

        # No ETF in batch (e.g. scratch orders) — fire all concurrently
        if etf_order is None:
            results = await asyncio.gather(
                *[self._send_ioc(o) for o in orders],
            )
            return self._build_report(list(results), orders)

        # Step 1: Execute ETF leg first, alone
        etf_result = await self._send_ioc(etf_order)

        # Step 2: Abort if ETF failed
        if etf_result.filled == 0:
            log.warning(
                f"ETF-First: abort — {ETF_SYMBOL} filled 0 "
                f"(error={etf_result.error})"
            )
            return self._build_report([etf_result], orders)

        log.info(
            f"ETF-First: {ETF_SYMBOL} filled={etf_result.filled}, "
            f"proceeding with {len(comp_orders)} component legs"
        )

        # Step 3: Fire component legs concurrently
        comp_results = await asyncio.gather(
            *[self._send_ioc(o) for o in comp_orders],
        )

        all_results = [etf_result] + list(comp_results)
        return self._build_report(all_results, orders)

    async def _send_ioc(self, order: OrderRequest) -> LegResult:
        """Execute a single IOC: place order, cancel unfilled remainder."""
        resp = await self._send_with_retry(order)
        filled = resp.filled if resp else 0
        unfilled = resp.volume if resp else 0
        error = None if resp else "placement_failed"

        # Cancel resting remainder (IOC simulation)
        if resp and unfilled > 0:
            await self._cancel_with_retry(resp.id)
            unfilled = 0

        return LegResult(
            order=order,
            response=resp,
            filled=filled,
            unfilled=unfilled,
            error=error,
        )

    # --- HTTP with retry ---

    async def _send_with_retry(
        self, order: OrderRequest,
    ) -> OrderResponse | None:
        """POST /api/order with exponential backoff on 429 / errors."""
        for attempt in range(MAX_429_RETRIES + 1):
            try:
                async with self._session.post(
                    "/api/order", json=asdict(order),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return OrderResponse(**data)

                    text = await resp.text()
                    if resp.status == 429 and attempt < MAX_429_RETRIES:
                        backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                        log.warning(
                            f"429 on {order.product} (attempt {attempt + 1}), "
                            f"retry in {backoff:.1f}s"
                        )
                        await asyncio.sleep(backoff)
                        continue

                    log.error(
                        f"Order {order.product} failed: "
                        f"{resp.status} {text}"
                    )
            except Exception as e:
                log.error(f"Order {order.product} exception: {e}")
                if attempt < MAX_429_RETRIES:
                    backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(backoff)
                    continue

        return None

    async def _cancel_with_retry(self, order_id: str) -> None:
        """DELETE /api/order/{id} with exponential backoff on 429 / errors."""
        for attempt in range(MAX_429_RETRIES + 1):
            try:
                async with self._session.delete(
                    f"/api/order/{order_id}",
                ) as resp:
                    if resp.status in (200, 204):
                        return
                    if resp.status == 429 and attempt < MAX_429_RETRIES:
                        backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                        await asyncio.sleep(backoff)
                        continue
                    log.error(
                        f"Cancel {order_id} failed: {resp.status}"
                    )
            except Exception as e:
                log.error(f"Cancel {order_id} exception: {e}")
                if attempt < MAX_429_RETRIES:
                    backoff = RETRY_BACKOFF_BASE * (2 ** attempt)
                    await asyncio.sleep(backoff)
                    continue

    # --- Report builder ---

    @staticmethod
    def _build_report(
        results: list[LegResult], original_orders: list[OrderRequest],
    ) -> ExecutionReport:
        fills = [lr.filled for lr in results]
        target_vol = original_orders[0].volume if original_orders else 0
        return ExecutionReport(
            legs=results,
            target_volume=target_vol,
            min_filled=min(fills) if fills else 0,
            max_filled=max(fills) if fills else 0,
            all_filled_equal=len(set(fills)) <= 1,
            timestamp=time.monotonic(),
        )
