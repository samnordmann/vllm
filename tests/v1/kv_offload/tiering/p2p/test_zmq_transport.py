# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for vllm.v1.kv_offload.tiering.p2p.control.zmq."""

from __future__ import annotations

import socket
import time
from unittest.mock import MagicMock

import pytest
import zmq

from vllm.v1.kv_offload.tiering.p2p.control import zmq as zmq_module
from vllm.v1.kv_offload.tiering.p2p.control.zmq import (
    ZmqConnection,
    ZmqTransport,
    _Sockets,
)


class _ZmqLifecycleFailure(BaseException):
    pass


def _free_port() -> int:
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_transport(host: str = "127.0.0.1", attempts: int = 8):
    """Construct a ZmqTransport on a fresh port, retrying on bind collisions.

    Why: _free_port() releases the probe socket before ZmqTransport binds the
    same port — a parallel test run can steal it in between. Retrying on
    ZMQError/OSError closes that race without a production change.
    """
    last_err: Exception | None = None
    for _ in range(attempts):
        port = _free_port()
        try:
            return ZmqTransport(f"{host}:{port}", host, port), port
        except (zmq.ZMQError, OSError) as e:
            last_err = e
    assert last_err is not None
    raise last_err


def _wait_for_inbound(transport: ZmqTransport, deadline: float = 2.0):
    """Poll until at least one new inbound connection is accepted, or fail."""
    end = time.monotonic() + deadline
    while time.monotonic() < end:
        new = transport.poll()
        if new:
            return new
        time.sleep(0.005)
    raise AssertionError(f"no inbound connection within {deadline}s")


def _wait_for_messages(
    transport: ZmqTransport,
    conn: ZmqConnection,
    n: int,
    deadline: float = 2.0,
) -> list[dict]:
    """Poll until `conn` has received at least `n` messages, then return them."""
    end = time.monotonic() + deadline
    msgs: list[dict] = []
    while time.monotonic() < end:
        transport.poll()
        msgs.extend(conn.recv())
        if len(msgs) >= n:
            return msgs
        time.sleep(0.005)
    raise AssertionError(f"got {len(msgs)}/{n} messages within {deadline}s")


def _make_mock_connection(peer_id: str = "test:1234") -> ZmqConnection:
    """Create a ZmqConnection with mock sockets for unit testing."""
    sockets = _Sockets(dealer=MagicMock(), monitor=MagicMock())
    return ZmqConnection(peer_id, sockets)


class TestZmqConnection:
    """Tests for ZmqConnection in isolation (no real sockets)."""

    def test_enqueue_and_recv(self):
        """Messages enqueued are returned by recv() in order."""
        conn = _make_mock_connection()

        conn.enqueue({"type": "a"})
        conn.enqueue({"type": "b"})

        msgs = conn.recv()
        assert list(msgs) == [{"type": "a"}, {"type": "b"}]
        # Second recv is empty
        assert not conn.recv()

    def test_recv_returns_empty_initially(self):
        conn = _make_mock_connection()
        assert not conn.recv()

    def test_alive_initially_true(self):
        conn = _make_mock_connection()
        assert conn.alive is True

    def test_mark_dead(self):
        conn = _make_mock_connection()
        conn.mark_dead()
        assert conn.alive is False

    def test_send_raises_when_closed(self):
        conn = _make_mock_connection()
        conn.mark_dead()

        with pytest.raises(RuntimeError, match="closed connection"):
            conn.send({"type": "test"})

    def test_send_rejects_oversized_control_frame(self, monkeypatch):
        conn = _make_mock_connection()
        monkeypatch.setattr(zmq_module, "MAX_CONTROL_MESSAGE_BYTES", 16)

        with pytest.raises(ValueError, match="control message exceeds"):
            conn.send({"payload": "x" * 64})

        conn._sockets.dealer.send.assert_not_called()

    def test_close_retries_only_the_incomplete_socket(self):
        primary = _ZmqLifecycleFailure("monitor close")
        conn = _make_mock_connection()
        conn._sockets.monitor.close.side_effect = [primary, None]

        with pytest.raises(_ZmqLifecycleFailure) as raised:
            conn.close()

        assert raised.value is primary
        assert conn._sockets.monitor.close.call_count == 1
        assert conn._sockets.dealer.close.call_count == 1
        assert conn._closed is False
        conn.close()
        conn.close()
        assert conn._sockets.monitor.close.call_count == 2
        assert conn._sockets.dealer.close.call_count == 1
        assert conn._closed is True


class TestZmqTransactionalLifecycle:
    def test_successful_constructor_ignores_logging_failure(self, monkeypatch):
        router = MagicMock()
        context = MagicMock()
        context.socket.return_value = router
        monkeypatch.setattr(zmq_module.zmq, "Context", MagicMock(return_value=context))
        monkeypatch.setattr(
            zmq_module.logger,
            "info",
            MagicMock(side_effect=_ZmqLifecycleFailure("logging")),
        )

        transport = ZmqTransport("local:1", "127.0.0.1", 1)

        assert transport._router is router
        assert transport._zmq_ctx is context
        transport.close()

    def test_constructor_bind_failure_releases_router_and_context(self, monkeypatch):
        primary = _ZmqLifecycleFailure("bind")
        cleanup = _ZmqLifecycleFailure("cleanup")
        events: list[str] = []
        router = MagicMock()
        context = MagicMock()
        context.socket.return_value = router
        router.bind.side_effect = primary

        def close_router(*, linger):
            assert linger == 0
            events.append("router")
            raise cleanup

        def destroy_context(*, linger):
            assert linger == 0
            events.append("context")
            raise cleanup

        router.close.side_effect = close_router
        context.destroy.side_effect = destroy_context
        monkeypatch.setattr(zmq_module.zmq, "Context", MagicMock(return_value=context))

        with pytest.raises(_ZmqLifecycleFailure) as raised:
            ZmqTransport("local:1", "127.0.0.1", 1)

        assert raised.value is primary
        assert events == ["router", "context"]

    def test_open_failure_defers_sockets_whose_cleanup_failed(self, monkeypatch):
        primary = _ZmqLifecycleFailure("dealer connect")
        cleanup = _ZmqLifecycleFailure("socket cleanup")
        router = MagicMock()
        dealer = MagicMock()
        monitor = MagicMock()
        context = MagicMock()
        context.socket.side_effect = [router, dealer, monitor]
        monkeypatch.setattr(zmq_module.zmq, "Context", MagicMock(return_value=context))
        transport = ZmqTransport("local:1", "127.0.0.1", 1)
        dealer.connect.side_effect = primary
        dealer.close.side_effect = cleanup
        monitor.close.side_effect = cleanup

        with pytest.raises(_ZmqLifecycleFailure) as raised:
            transport.connect("peer:2")

        assert raised.value is primary
        assert transport._connections == {}
        assert transport._deferred_sockets == [monitor, dealer]
        monitor.close.side_effect = None
        dealer.close.side_effect = None
        transport.close()
        assert transport._closed is True

    def test_successful_open_ignores_logging_failure(self, monkeypatch):
        router = MagicMock()
        dealer = MagicMock()
        monitor = MagicMock()
        context = MagicMock()
        context.socket.side_effect = [router, dealer, monitor]
        monkeypatch.setattr(zmq_module.zmq, "Context", MagicMock(return_value=context))
        monkeypatch.setattr(
            zmq_module.logger,
            "info",
            MagicMock(side_effect=_ZmqLifecycleFailure("logging")),
        )
        transport = ZmqTransport("local:1", "127.0.0.1", 1)

        connection = transport.connect("peer:2")

        assert transport._connections == {"peer:2": connection}
        transport.close()

    def test_dead_connection_remains_owned_until_close_retry_succeeds(self):
        transport = ZmqTransport.__new__(ZmqTransport)
        transport._connections = {}
        conn = _make_mock_connection("peer:2")
        primary = _ZmqLifecycleFailure("monitor close")
        conn._sockets.monitor.close.side_effect = [primary, None]
        conn.mark_dead()
        transport._connections[conn.peer_id] = conn

        with pytest.raises(_ZmqLifecycleFailure) as raised:
            transport._sweep_dead_connections()

        assert raised.value is primary
        assert transport._connections == {conn.peer_id: conn}
        transport._sweep_dead_connections()
        assert transport._connections == {}

    def test_transport_close_attempts_every_resource_and_retries(self):
        primary = _ZmqLifecycleFailure("connection close")
        socket_error = _ZmqLifecycleFailure("deferred socket close")
        context_error = _ZmqLifecycleFailure("context destroy")
        failed_connection = MagicMock()
        failed_connection.close.side_effect = [primary, None]
        healthy_connection = MagicMock()
        deferred = MagicMock()
        deferred.close.side_effect = [socket_error, None]
        router = MagicMock()
        context = MagicMock()
        context.closed = False
        context.destroy.side_effect = [context_error, None]
        transport = ZmqTransport.__new__(ZmqTransport)
        transport._local_id = "local:1"
        transport._closed = False
        transport._connections = {
            "failed:1": failed_connection,
            "healthy:2": healthy_connection,
        }
        transport._deferred_sockets = [deferred]
        transport._router = router
        transport._zmq_ctx = context

        with pytest.raises(_ZmqLifecycleFailure) as raised:
            transport.close()

        assert raised.value is primary
        assert transport._connections == {"failed:1": failed_connection}
        assert transport._deferred_sockets == [deferred]
        assert transport._router is None
        assert transport._zmq_ctx is context
        assert transport._closed is False
        healthy_connection.close.assert_called_once_with()
        router.close.assert_called_once_with(linger=0)

        transport.close()
        assert transport._closed is True
        assert transport._connections == {}
        assert transport._deferred_sockets == []
        assert context.destroy.call_count == 2
        transport.close()
        assert failed_connection.close.call_count == 2
        assert healthy_connection.close.call_count == 1
        assert deferred.close.call_count == 2
        assert context.destroy.call_count == 2


class TestZmqTransportConnectivity:
    """Integration tests for ZmqTransport with real ZMQ sockets."""

    def test_connect_and_send_message(self):
        """Two transports can connect and exchange messages."""
        transport_a, port_a = _make_transport()
        transport_b, port_b = _make_transport()

        try:
            peer_a_id = f"127.0.0.1:{port_a}"
            conn_b_to_a = transport_b.connect(peer_a_id)
            conn_b_to_a.send({"type": "hello", "data": 42})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1

            conn_a_from_b = new_conns[0]
            assert conn_a_from_b.peer_id == f"127.0.0.1:{port_b}"

            msgs = _wait_for_messages(transport_a, conn_a_from_b, 1)
            assert msgs == [{"type": "hello", "data": 42}]
        finally:
            transport_a.close()
            transport_b.close()

    def test_bidirectional_messaging(self):
        """Both sides can send and receive after connection."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "connect", "from": "b"})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1
            conn_a = new_conns[0]

            conn_a.send({"type": "reply", "from": "a"})

            msgs = _wait_for_messages(transport_b, conn_b, 1)
            assert msgs == [{"type": "reply", "from": "a"}]
        finally:
            transport_a.close()
            transport_b.close()

    def test_poll_returns_empty_when_no_connections(self):
        transport, _ = _make_transport()
        try:
            assert not transport.poll()
        finally:
            transport.close()

    def test_multiple_messages(self):
        """Multiple messages are buffered and returned together."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"seq": 1})
            conn_b.send({"seq": 2})
            conn_b.send({"seq": 3})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1
            conn_a = new_conns[0]

            msgs = _wait_for_messages(transport_a, conn_a, 3)
            assert [m["seq"] for m in msgs] == [1, 2, 3]
        finally:
            transport_a.close()
            transport_b.close()

    def test_duplicate_connect_asserts(self):
        """Connecting to the same peer twice raises AssertionError."""
        # port_a is never bound — we just need a syntactically-valid peer id.
        port_a = _free_port()
        transport_b, _ = _make_transport()
        try:
            transport_b.connect(f"127.0.0.1:{port_a}")
            with pytest.raises(AssertionError, match="already exists"):
                transport_b.connect(f"127.0.0.1:{port_a}")
        finally:
            transport_b.close()

    def test_dead_connection_removed_on_poll(self):
        """Dead connections are cleaned up during poll."""
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "hello"})

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1

            # Mark the inbound connection dead manually.
            new_conns[0].mark_dead()

            # Pruning is synchronous within poll().
            transport_a.poll()
            assert len(transport_a._connections) == 0
            assert new_conns[0]._sockets.dealer.closed
        finally:
            transport_a.close()
            transport_b.close()

    def test_close_is_idempotent(self):
        """Calling close() twice doesn't raise."""
        transport, _ = _make_transport()
        transport.close()
        transport.close()  # should not raise


class TestZmqReconnect:
    """Reconnecting to a peer whose connection died (real ZMQ sockets).

    A session marks its connection dead while handling messages, which happens
    after the transport's own sweep has run for that tick — so a dead
    connection stays registered until the next poll(). Reconnecting in that
    window must succeed, and the retired connection must release its sockets.
    Real sockets are required: a mock reports every attribute as closed.
    """

    def test_close_after_mark_dead_releases_sockets(self):
        """close() releases sockets even when mark_dead() ran first.

        mark_dead() must not set the flag close() guards on, or every peer
        disconnect leaks a DEALER and a monitor socket.
        """
        transport, _ = _make_transport()
        try:
            conn = transport.connect(f"127.0.0.1:{_free_port()}")
            dealer, monitor = conn._sockets.dealer, conn._sockets.monitor

            conn.mark_dead()
            assert not conn.alive
            assert not dealer.closed

            conn.close()
            assert dealer.closed
            assert monitor.closed
        finally:
            transport.close()

    def test_connect_retires_dead_connection(self):
        """connect() replaces a registered-but-dead connection."""
        transport, _ = _make_transport()
        try:
            # The peer port is never bound — only the peer id matters here.
            peer_id = f"127.0.0.1:{_free_port()}"
            dead = transport.connect(peer_id)
            dead.mark_dead()

            conn = transport.connect(peer_id)

            assert conn is not dead
            assert conn.alive
            assert transport._connections[peer_id] is conn
            assert dead._sockets.dealer.closed
        finally:
            transport.close()

    def test_repeated_reconnect_to_same_peer(self):
        """A flapping peer stays reconnectable.

        Monitor endpoints are inproc addresses that libzmq releases
        asynchronously, so deriving one from peer_id alone makes each
        reconnect race the previous teardown and fail with EADDRINUSE.
        """
        transport, _ = _make_transport()
        try:
            peer_id = f"127.0.0.1:{_free_port()}"
            for _ in range(10):
                conn = transport.connect(peer_id)
                conn.mark_dead()
                transport.poll()
                assert conn._sockets.dealer.closed

            assert not transport._connections
        finally:
            transport.close()

    def test_inbound_message_survives_dead_registration(self):
        """A reconnecting peer's first message is not dropped.

        poll() must retire connections killed by their session before routing
        traffic, otherwise the message is enqueued into the dead connection and
        discarded when it is swept — and a session announces itself only once.
        Covers only that between-polls window: a peer dying while poll() runs,
        or dying silently until the heartbeat expires, is out of scope.
        """
        transport_a, port_a = _make_transport()
        transport_b, _ = _make_transport()

        try:
            conn_b = transport_b.connect(f"127.0.0.1:{port_a}")
            conn_b.send({"type": "connect", "seq": 1})

            inbound = _wait_for_inbound(transport_a)[0]
            _wait_for_messages(transport_a, inbound, 1)

            inbound.mark_dead()
            conn_b.send({"type": "connect", "seq": 2})

            # Wait until the frame is readable on the ROUTER, so the message is
            # known to have arrived rather than merely being slow.
            poller = zmq.Poller()
            poller.register(transport_a._router, zmq.POLLIN)
            assert poller.poll(2000), "message never reached the ROUTER"

            new_conns = _wait_for_inbound(transport_a)
            assert len(new_conns) == 1
            assert new_conns[0] is not inbound
            msgs = _wait_for_messages(transport_a, new_conns[0], 1)
            assert msgs == [{"type": "connect", "seq": 2}]
        finally:
            transport_a.close()
            transport_b.close()
