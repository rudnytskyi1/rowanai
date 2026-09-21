"""WebSocket transport for the room client (SPEC §4, §7).

Thin wrapper around the ``websockets`` library: connect with auto-reconnect
(3 s backoff, ``hello`` re-sent on every (re)connect), send JSON control
frames and binary payloads (microphone PCM, screenshot JPEG), receive
messages. Text frames come back as parsed ``dict``, binary frames as ``bytes``.

Since v1.4 exactly one task — the reader in :mod:`client.main` — calls
:meth:`WSClient.recv` for the lifetime of a connection (the server may talk
between utterances: proactive greetings, ``camera_request``), so it reads with
:data:`WAIT_FOREVER` and the per-reply deadline lives in that reader's consumer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Union

try:  # websockets >= 13 (new asyncio implementation)
    from websockets.asyncio.client import connect as ws_connect
except ImportError:  # pragma: no cover - older websockets
    from websockets.client import connect as ws_connect  # type: ignore

from websockets.exceptions import ConnectionClosed, WebSocketException

from common.protocol import MSG_READY

log = logging.getLogger(__name__)

#: Reconnect backoff required by SPEC §7 step 1.
RECONNECT_DELAY_S = 3.0
#: How long to wait for ``ready`` after sending ``hello``.
READY_TIMEOUT_S = 10.0
#: How long to wait for the next server message inside one utterance.
#: The server stays silent while it thinks, so this single wait must cover its
#: worst case between two messages: one vision call (120 s in ``server/vision.py``)
#: followed by one LLM completion (180 s in ``server/llm.py``), plus margin.
RECV_TIMEOUT_S = 420.0
#: ``recv(timeout=WAIT_FOREVER)`` blocks until a frame arrives or the socket
#: dies. v1.4 reads the socket permanently (proactive greetings, camera
#: requests), and an idle room is silent for hours: the deadline that guards one
#: utterance belongs to the caller waiting for that reply, not to the reader.
#: A dead peer is still noticed — the library's ping keepalive (20 s) closes the
#: connection, which makes the pending ``recv`` raise.
WAIT_FOREVER = 0.0

Message = Union[Dict[str, Any], bytes]


class WSDisconnected(Exception):
    """Raised when the connection is gone; the caller should retry later."""


class WSClient:
    """Single-connection WebSocket client with reconnect and hello handshake."""

    def __init__(
        self,
        url: str,
        hello: Dict[str, Any],
        reconnect_delay: float = RECONNECT_DELAY_S,
        ready_timeout: float = READY_TIMEOUT_S,
        recv_timeout: float = RECV_TIMEOUT_S,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.url = str(url)
        self.hello = dict(hello)
        self.reconnect_delay = float(reconnect_delay)
        self.ready_timeout = float(ready_timeout)
        self.recv_timeout = float(recv_timeout)
        self._should_stop = should_stop or (lambda: False)
        self._conn: Any = None
        self._send_lock = asyncio.Lock()
        self._announced_failure = False

    # -- state -----------------------------------------------------------
    @property
    def connected(self) -> bool:
        conn = self._conn
        if conn is None:
            return False
        state = getattr(conn, "state", None)
        if state is not None:
            return getattr(state, "name", "") == "OPEN"
        return not getattr(conn, "closed", False)  # pragma: no cover - legacy

    def drop(self) -> None:
        """Close the current connection so the next call reconnects.

        Used when the caller — not the socket — decides the link is dead, e.g.
        the reply to an utterance never arrived: closing here makes the reader
        task end and ``ensure_connected`` build a fresh connection.
        """
        if self._conn is not None:
            log.debug("Dropping the connection on request")
        self._drop()

    def _drop(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            closer = conn.close()
            if asyncio.iscoroutine(closer):
                asyncio.get_running_loop().create_task(_swallow(closer))
        except Exception as exc:  # pragma: no cover - teardown
            log.debug("Error while closing the socket: %s", exc)

    # -- connecting ------------------------------------------------------
    async def ensure_connected(self) -> None:
        """Block (retrying every ``reconnect_delay``) until connected and greeted."""
        while not self._should_stop():
            if self.connected:
                return
            self._conn = None
            try:
                conn = await ws_connect(
                    self.url,
                    open_timeout=self.reconnect_delay,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=3,
                    max_size=None,
                )
            except (OSError, WebSocketException, asyncio.TimeoutError, TimeoutError) as exc:
                if not self._announced_failure:
                    log.warning(
                        "No connection to the server %s (%s). Retrying every %.0f s...",
                        self.url, exc, self.reconnect_delay,
                    )
                    self._announced_failure = True
                else:
                    log.debug("Reconnect attempt failed: %s", exc)
                await self._sleep(self.reconnect_delay)
                continue

            self._conn = conn
            try:
                await self.send_json(self.hello)
                await self._await_ready()
            except (WSDisconnected, asyncio.TimeoutError, TimeoutError) as exc:
                log.warning("Handshake with the server failed: %s", exc)
                self._drop()
                await self._sleep(self.reconnect_delay)
                continue

            self._announced_failure = False
            log.info("Connected to the server %s", self.url)
            return
        raise WSDisconnected("the client is shutting down")

    async def _await_ready(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.ready_timeout
        while True:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise WSDisconnected("the server did not answer with ready")
            msg = await self.recv(timeout=remaining)
            if isinstance(msg, dict):
                if msg.get("type") == MSG_READY:
                    return
                log.debug("Message %s received before ready", msg.get("type"))
            else:
                log.debug("Binary frame received before ready (%d bytes)", len(msg))

    async def _sleep(self, seconds: float) -> None:
        """Interruptible sleep that reacts to the stop flag quickly."""
        step = 0.25
        waited = 0.0
        while waited < seconds:
            if self._should_stop():
                return
            await asyncio.sleep(min(step, seconds - waited))
            waited += step

    # -- I/O -------------------------------------------------------------
    async def send_json(self, payload: Dict[str, Any]) -> None:
        conn = self._conn
        if conn is None:
            raise WSDisconnected("no connection")
        data = json.dumps(payload, ensure_ascii=False)
        try:
            async with self._send_lock:
                await conn.send(data)
        except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as exc:
            self._drop()
            raise WSDisconnected(f"send failed: {exc}") from exc

    async def send_bytes(self, data: bytes) -> None:
        conn = self._conn
        if conn is None:
            raise WSDisconnected("no connection")
        try:
            async with self._send_lock:
                await conn.send(bytes(data))
        except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as exc:
            self._drop()
            raise WSDisconnected(f"send failed: {exc}") from exc

    async def recv(self, timeout: Optional[float] = None) -> Message:
        """Next server message: parsed dict for text frames, bytes for binary.

        ``timeout`` is seconds, ``None`` means :data:`RECV_TIMEOUT_S` and
        :data:`WAIT_FOREVER` (0 or less) waits for as long as the socket lives.
        """
        conn = self._conn
        if conn is None:
            raise WSDisconnected("no connection")
        wait = self.recv_timeout if timeout is None else float(timeout)
        while True:
            try:
                if wait <= 0:
                    raw = await conn.recv()
                else:
                    raw = await asyncio.wait_for(conn.recv(), timeout=wait)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                self._drop()
                raise WSDisconnected("the server is not responding") from exc
            except (ConnectionClosed, WebSocketException, OSError, RuntimeError) as exc:
                self._drop()
                raise WSDisconnected(f"connection closed: {exc}") from exc

            if isinstance(raw, (bytes, bytearray, memoryview)):
                return bytes(raw)
            try:
                payload = json.loads(raw)
            except (ValueError, TypeError):
                log.warning("Unreadable JSON received from the server: %r", str(raw)[:200])
                continue
            if not isinstance(payload, dict):
                log.warning("Expected a JSON object, got: %r", str(raw)[:200])
                continue
            return payload

    async def close(self) -> None:
        conn, self._conn = self._conn, None
        if conn is None:
            return
        try:
            await conn.close()
        except Exception as exc:  # pragma: no cover - teardown
            log.debug("Error while closing the connection: %s", exc)


async def _swallow(awaitable: Any) -> None:
    try:
        await awaitable
    except Exception as exc:  # pragma: no cover - teardown
        log.debug("Error while closing the socket in the background: %s", exc)
