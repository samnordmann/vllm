# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
ZMQ-based transport layer for P2P KV cache sharing.

Provides ZmqConnection (per-peer messaging) and ZmqTransport (connection
management). Message-content agnostic.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass

import msgspec
import zmq
import zmq.utils.monitor

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.p2p.control.base import (
    ControlConnection,
    ControlTransport,
)

logger = init_logger(__name__)

_HEARTBEAT_IVL_MS = 2000
_HEARTBEAT_TIMEOUT_MS = 10000
_HEARTBEAT_TTL_MS = 10000
MAX_CONTROL_MESSAGE_BYTES = 64 * 1024 * 1024

# Shared sentinels returned when there is nothing to report.
_EMPTY_INBOX: tuple[dict, ...] = ()
_EMPTY_NEW_CONNECTIONS: tuple[ControlConnection, ...] = ()


def _cleanup_call(
    action: Callable[[], object], description: str
) -> BaseException | None:
    """Run one teardown step without letting logging mask its failure."""
    try:
        action()
    except BaseException as exc:
        with suppress(BaseException):
            logger.warning("%s failed during ZMQ cleanup: %s", description, exc)
        return exc
    return None


def _tcp_addr(host: str, port: int | str) -> str:
    return f"tcp://{host}:{port}"


def _apply_heartbeat(sock: zmq.Socket) -> None:
    sock.setsockopt(zmq.HEARTBEAT_IVL, _HEARTBEAT_IVL_MS)
    sock.setsockopt(zmq.HEARTBEAT_TIMEOUT, _HEARTBEAT_TIMEOUT_MS)
    sock.setsockopt(zmq.HEARTBEAT_TTL, _HEARTBEAT_TTL_MS)
    sock.setsockopt(zmq.MAXMSGSIZE, MAX_CONTROL_MESSAGE_BYTES)


@dataclass
class _Sockets:
    dealer: zmq.Socket
    monitor: zmq.Socket


class ZmqConnection(ControlConnection):
    """Bidirectional message channel to a single remote peer."""

    def __init__(self, peer_id: str, sockets: _Sockets) -> None:
        super().__init__(peer_id)
        self._sockets = sockets
        # _dead: the peer is gone. _closed: the sockets have been released.
        # Distinct, so mark_dead() cannot turn close() into a no-op and leak
        # the DEALER and its monitor socket.
        self._dead = False
        self._closed = False
        self._monitor_closed = False
        self._dealer_closed = False
        self._inbox: list[dict] = []

    def send(self, msg: dict) -> None:
        """Send a msgpack-encoded message to this peer."""
        if not self.alive:
            raise RuntimeError(
                f"ZmqConnection: send on closed connection to {self.peer_id}"
            )
        data = msgspec.msgpack.encode(msg)
        if len(data) > MAX_CONTROL_MESSAGE_BYTES:
            raise ValueError(
                f"control message exceeds {MAX_CONTROL_MESSAGE_BYTES} bytes"
            )
        self._sockets.dealer.send(data)

    def recv(self) -> Sequence[dict]:
        """Drain and return all buffered incoming messages."""
        if not self._inbox:
            return _EMPTY_INBOX
        msgs = self._inbox
        self._inbox = []
        return msgs

    @property
    def alive(self) -> bool:
        return not (self._dead or self._closed)

    def close(self) -> None:
        if self._closed:
            return
        self._dead = True
        with suppress(BaseException):
            logger.info("ZmqConnection: closing connection to %s", self.peer_id)
        first_error: BaseException | None = None
        if not self._monitor_closed:
            error = _cleanup_call(
                self._sockets.monitor.close,
                f"monitor close for peer {self.peer_id}",
            )
            if error is None:
                self._monitor_closed = True
            else:
                first_error = error
        if not self._dealer_closed:
            error = _cleanup_call(
                self._sockets.dealer.close,
                f"dealer close for peer {self.peer_id}",
            )
            if error is None:
                self._dealer_closed = True
            elif first_error is None:
                first_error = error
        self._closed = self._monitor_closed and self._dealer_closed
        if first_error is not None:
            raise first_error

    def enqueue(self, msg: dict) -> None:
        """Buffer an incoming message."""
        self._inbox.append(msg)

    def mark_dead(self) -> None:
        """Mark connection as disconnected.

        Only flips liveness: the sockets stay open until close() releases
        them, so the owner can still drain recv() before tearing down.
        """
        self._dead = True

    @property
    def monitor_socket(self) -> zmq.Socket:
        """Monitor socket for disconnect detection (used by ZmqTransport)."""
        return self._sockets.monitor


class ZmqTransport(ControlTransport):
    """ZMQ implementation of ControlTransport.

    Manages a ROUTER socket for accepting connections and DEALER sockets
    for outbound connections. Message-content agnostic.
    """

    def __init__(self, local_id: str, host: str, port: int) -> None:
        self._local_id = local_id
        self._closed = False

        self._connections: dict[str, ZmqConnection] = {}
        self._pending_inbound: list[tuple[str, dict]] = []
        # Monotonic suffix for inproc monitor endpoints — see
        # _open_connection() for why peer_id alone is not enough.
        self._monitor_seq = 0
        self._deferred_sockets: list[zmq.Socket] = []

        context = zmq.Context()
        router: zmq.Socket | None = None
        bind_addr = _tcp_addr(host, port)
        try:
            router = context.socket(zmq.ROUTER)
            _apply_heartbeat(router)
            router.bind(bind_addr)
        except BaseException:
            if router is not None:
                _cleanup_call(
                    lambda: router.close(linger=0),
                    "partially constructed ROUTER close",
                )
            _cleanup_call(
                lambda: context.destroy(linger=0),
                "partially constructed context destroy",
            )
            raise
        with suppress(BaseException):
            logger.info(
                "ZmqTransport %s: ROUTER bound on %s", self._local_id, bind_addr
            )
        self._zmq_ctx: zmq.Context | None = context
        self._router: zmq.Socket | None = router

    # ------------------------------------------------------------------
    # ZmqConnection lifecycle
    # ------------------------------------------------------------------

    def connect(self, peer_id: str) -> ZmqConnection:
        """Open an outbound connection to a remote peer.

        A dead connection can still be registered: its owning session may mark
        it dead after this tick's sweep already ran, and poll() only
        unregisters it on the next pass. Retire such an entry instead of
        asserting, so a reconnect landing in that window succeeds. A live
        entry is still a genuine duplicate.
        """
        existing = self._connections.get(peer_id)
        if existing is not None:
            assert not existing.alive, f"ZmqConnection to {peer_id} already exists"
            with suppress(BaseException):
                logger.info(
                    "ZmqTransport %s: retiring dead connection to %s before reconnect",
                    self._local_id,
                    peer_id,
                )
            existing.close()
            if self._connections.get(peer_id) is existing:
                del self._connections[peer_id]
        with suppress(BaseException):
            logger.info(
                "ZmqTransport %s: opening OUTBOUND connection to %s",
                self._local_id,
                peer_id,
            )
        return self._open_connection(peer_id, direction="outbound")

    def poll(self) -> Sequence[ControlConnection]:
        """Process all pending I/O. Returns newly accepted connections.

        - Receives messages (buffered in each connection's inbox)
        - Creates connections for new inbound peers (connect msg in inbox)
        - Checks monitors for disconnections
        - Removes and closes dead connections
        """
        # Retire connections a session killed since the last poll() before
        # routing traffic: otherwise a reconnecting peer's first message is
        # enqueued into the dead connection and discarded along with it.
        # This closes only that between-polls window. A peer that dies while
        # poll() is running is not seen until the next _check_monitors(), and
        # one that dies silently not until the ZMQ heartbeat expires; both
        # still take a message into a doomed connection and are out of scope.
        self._sweep_dead_connections()

        self._recv_router()
        self._check_monitors()
        self._sweep_dead_connections()

        # Create connections for new inbound peers
        new_connections: list[ControlConnection] | None = None
        for sender_id, msg in self._pending_inbound:
            conn = self._connections.get(sender_id)
            if conn is None:
                logger.info(
                    "ZmqTransport %s: accepting INBOUND connection from %s",
                    self._local_id,
                    sender_id,
                )
                conn = self._open_connection(sender_id, direction="inbound")
                if new_connections is None:
                    new_connections = []
                new_connections.append(conn)
            conn.enqueue(msg)
        self._pending_inbound.clear()

        return (
            new_connections if new_connections is not None else _EMPTY_NEW_CONNECTIONS
        )

    def close(self) -> None:
        if self._closed:
            return
        first_error: BaseException | None = None

        for peer_id, conn in tuple(self._connections.items()):
            error = _cleanup_call(conn.close, f"connection close for peer {peer_id}")
            if error is None:
                if self._connections.get(peer_id) is conn:
                    del self._connections[peer_id]
            elif first_error is None:
                first_error = error

        for sock in tuple(self._deferred_sockets):
            error = _cleanup_call(
                lambda sock=sock: sock.close(linger=0),
                "deferred socket close",
            )
            if error is None:
                self._deferred_sockets = [
                    candidate
                    for candidate in self._deferred_sockets
                    if candidate is not sock
                ]
            elif first_error is None:
                first_error = error

        router = self._router
        if router is not None:
            error = _cleanup_call(lambda: router.close(linger=0), "ROUTER close")
            if error is None:
                self._router = None
            elif first_error is None:
                first_error = error

        context = self._zmq_ctx
        if context is not None:
            error = _cleanup_call(lambda: context.destroy(linger=0), "context destroy")
            if error is None or getattr(context, "closed", False) is True:
                # Context destruction is the root cleanup and closes every
                # remaining socket even if an earlier per-socket close failed.
                self._zmq_ctx = None
                self._router = None
                self._connections.clear()
                self._deferred_sockets.clear()
            elif first_error is None:
                first_error = error

        self._closed = self._zmq_ctx is None
        if first_error is not None:
            raise first_error

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _open_connection(
        self, peer_id: str, direction: str = "outbound"
    ) -> ZmqConnection:
        """Create a DEALER socket + monitor and register the connection."""
        host, port_str = peer_id.rsplit(":", 1)
        dealer_addr = _tcp_addr(host, port_str)

        logger.debug(
            "ZmqTransport %s: creating DEALER for %s peer %s -> %s",
            self._local_id,
            direction,
            peer_id,
            dealer_addr,
        )

        context = self._zmq_ctx
        if context is None:
            raise RuntimeError("ZmqTransport is closed")
        dealer: zmq.Socket | None = None
        monitor_sock: zmq.Socket | None = None
        conn: ZmqConnection | None = None
        try:
            dealer = context.socket(zmq.DEALER)
            _apply_heartbeat(dealer)
            dealer.identity = self._local_id.encode()

            # Unique per connection, not per peer: libzmq releases an inproc
            # endpoint on its reaper thread after the DEALER's close() has
            # already returned, so reconnect must use a fresh address.
            safe_id = peer_id.replace(":", "-").replace("/", "-")
            monitor_addr = f"inproc://p2p-monitor-{safe_id}-{self._monitor_seq}"
            self._monitor_seq += 1
            dealer.monitor(monitor_addr, zmq.EVENT_DISCONNECTED)

            monitor_sock = context.socket(zmq.PAIR)
            monitor_sock.connect(monitor_addr)
            dealer.connect(dealer_addr)

            sockets = _Sockets(dealer=dealer, monitor=monitor_sock)
            conn = ZmqConnection(peer_id, sockets)
            self._connections[peer_id] = conn
        except BaseException:
            if conn is not None and self._connections.get(peer_id) is conn:
                del self._connections[peer_id]
            if monitor_sock is not None:
                error = _cleanup_call(
                    lambda: monitor_sock.close(linger=0),
                    f"partial monitor close for peer {peer_id}",
                )
                if error is not None:
                    self._defer_socket(monitor_sock)
            if dealer is not None:
                error = _cleanup_call(
                    lambda: dealer.close(linger=0),
                    f"partial dealer close for peer {peer_id}",
                )
                if error is not None:
                    self._defer_socket(dealer)
            raise

        assert conn is not None
        with suppress(BaseException):
            logger.info(
                "ZmqTransport %s: %s connection established to %s "
                "(active connections: %d)",
                self._local_id,
                direction,
                peer_id,
                len(self._connections),
            )
        return conn

    def _sweep_dead_connections(self) -> None:
        """Unregister and release every connection that is no longer alive."""
        for pid in [p for p, c in self._connections.items() if not c.alive]:
            conn = self._connections[pid]
            conn.close()
            if self._connections.get(pid) is conn:
                del self._connections[pid]

    def _defer_socket(self, sock: zmq.Socket) -> None:
        if not any(candidate is sock for candidate in self._deferred_sockets):
            self._deferred_sockets.append(sock)

    def _recv_router(self) -> None:
        """Non-blocking: receive all pending messages from ROUTER."""
        assert self._router is not None
        while True:
            try:
                frames = self._router.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
            except zmq.ZMQError as exc:
                logger.warning("ZmqTransport %s: recv error: %s", self._local_id, exc)
                break

            if len(frames) != 2:
                logger.warning(
                    "ZmqTransport %s: dropping message with %d frames (expected 2)",
                    self._local_id,
                    len(frames),
                )
                continue

            identity, data = frames
            try:
                sender_id = identity.decode()
            except UnicodeDecodeError:
                logger.warning(
                    "ZmqTransport %s: dropping non-UTF-8 routing identity",
                    self._local_id,
                )
                continue

            if len(data) > MAX_CONTROL_MESSAGE_BYTES:
                logger.warning(
                    "ZmqTransport %s: dropping oversized message from %s "
                    "(%d > %d bytes)",
                    self._local_id,
                    sender_id,
                    len(data),
                    MAX_CONTROL_MESSAGE_BYTES,
                )
                continue

            logger.debug(
                "ZmqTransport %s: ROUTER recv from %s (%d bytes)",
                self._local_id,
                sender_id,
                len(data),
            )

            try:
                msg = msgspec.msgpack.decode(data)
            except Exception as exc:
                logger.warning(
                    "ZmqTransport %s: failed to decode message from %s: %s",
                    self._local_id,
                    sender_id,
                    exc,
                )
                continue

            conn = self._connections.get(sender_id)

            if conn is not None:
                conn.enqueue(msg)
            else:
                self._pending_inbound.append((sender_id, msg))

    def _check_monitors(self) -> None:
        """Non-blocking: check all monitor sockets for disconnection."""
        for conn in self._connections.values():
            if not conn.alive:
                continue
            try:
                event = zmq.utils.monitor.recv_monitor_message(
                    conn.monitor_socket, zmq.NOBLOCK
                )
            except zmq.Again:
                continue
            except zmq.ZMQError as exc:
                logger.warning(
                    "ZmqTransport %s: monitor error for peer %s: %s",
                    self._local_id,
                    conn.peer_id,
                    exc,
                )
                continue

            if event["event"] == zmq.EVENT_DISCONNECTED:
                logger.debug(
                    "ZmqTransport %s: peer %s disconnected",
                    self._local_id,
                    conn.peer_id,
                )
                conn.mark_dead()
