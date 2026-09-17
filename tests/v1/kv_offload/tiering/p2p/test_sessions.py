# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the unified bidirectional P2PSession.

A P2PSession owns a single ControlConnection and dispatches every message
type — both client-role (FetchMsg / TransferDoneMsg / AbortAck) and
server-role (FetchMsg / TransferDoneMsg / AbortAck from the peer's
perspective). These tests exercise both flows independently and the
bidirectional case where one session simultaneously serves a fetch and
completes its own load.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.p2p.session import (
    LoadResult,
    P2PSession,
    StoreResult,
)
from vllm.v1.kv_offload.tiering.p2p.session import protocol as protocol_module
from vllm.v1.kv_offload.tiering.p2p.session.client import (
    _ABORT_ACK_TIMEOUT_S,
    _LOAD_TIMEOUT_S,
)
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    MAX_BLOCK_INDEX,
    MAX_ROUND_SEQ,
    MAX_WIRE_KEY_BYTES,
    MAX_WIRE_LIST_ITEMS,
    SESSION_EPOCH_NBYTES,
    SOURCE_EPOCH_KEY,
    TARGET_EPOCH_KEY,
    TYPE_KEY,
    UNSPECIFIED_EPOCH,
    WIRE_MAJOR_KEY,
    WIRE_MINOR_KEY,
    WIRE_PROTOCOL_MAJOR,
    WIRE_PROTOCOL_MINOR,
    AbortAckMsg,
    AbortFetchMsg,
    ConnectAckMsg,
    ConnectMsg,
    DisconnectMsg,
    FetchMsg,
    LookupMsg,
    LookupRespMsg,
    TransferDoneMsg,
)
from vllm.v1.kv_offload.tiering.p2p.session.server import (
    _CANCEL_DRAIN_TIMEOUT_S,
    _InflightXfer,
    _OutboundRequestState,
)
from vllm.v1.kv_offload.tiering.p2p.session.session import (
    _MAX_CONSECUTIVE_DISPATCH_ERRORS,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# Shared PYTHONHASHSEED used by the session under test and the fake peer's
# ConnectMsg so the handshake succeeds unless a test overrides one side.
_DEFAULT_HASH_SEED = "0"
_DEFAULT_PEER_EPOCH = b"P" * SESSION_EPOCH_NBYTES

_STATE_MESSAGE_TYPES = {
    AbortAckMsg.TYPE,
    AbortFetchMsg.TYPE,
    DisconnectMsg.TYPE,
    FetchMsg.TYPE,
    LookupMsg.TYPE,
    LookupRespMsg.TYPE,
    TransferDoneMsg.TYPE,
}


def _channel_fields(
    source_epoch: bytes = _DEFAULT_PEER_EPOCH,
    target_epoch: bytes = b"L" * SESSION_EPOCH_NBYTES,
) -> dict:
    return {
        SOURCE_EPOCH_KEY: source_epoch,
        TARGET_EPOCH_KEY: target_epoch,
    }


class _CloseFailure(BaseException):
    pass


class _SubmissionFailure(BaseException):
    pass


class FakeDataTransport:
    """Minimal fake DataTransport for testing sessions."""

    def __init__(
        self,
        base_addr: int = 0x1000,
        num_blocks: int = 16,
        block_len: int = 4096,
        config_fingerprint: str = "",
    ) -> None:
        self._base_addr = base_addr
        self._num_blocks = num_blocks
        self._block_len = block_len
        self._config_fingerprint = config_fingerprint
        self._remote_peers: dict[str, dict] = {}
        self._transfers: dict[int, tuple] = {}
        self._next_id = 0
        self._poll_done: list[int] = []
        self._poll_failed: list[int] = []
        self._cancel_still_inflight: set[int] = set()
        self._cancel_calls: list[tuple[list[int], str]] = []
        self._ack_calls: list[tuple[str | None, tuple[int, ...]]] = []
        self._add_remote_calls: list[str] = []

    @property
    def base_addr(self) -> int:
        return self._base_addr

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    @property
    def block_len(self) -> int:
        return self._block_len

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    def get_agent_metadata(self) -> bytes:
        return b"fake-metadata"

    def add_remote_peer(
        self, peer_id, agent_metadata, base_addr, num_blocks, block_len
    ) -> None:
        self._add_remote_calls.append(peer_id)
        self._remote_peers[peer_id] = {
            ConnectMsg.AGENT_METADATA: agent_metadata,
            ConnectMsg.BASE_ADDR: base_addr,
            "num_blocks": num_blocks,
            "block_len": block_len,
        }

    def remove_remote_peer(self, peer_id: str) -> None:
        self._remote_peers.pop(peer_id, None)

    def write_blocks(self, peer_id, local_idxs, remote_idxs) -> int | None:
        if peer_id not in self._remote_peers:
            return None
        tid = self._next_id
        self._next_id += 1
        self._transfers[tid] = (peer_id, local_idxs, remote_idxs)
        return tid

    def poll(self, peer_id=None):
        from vllm.v1.kv_offload.tiering.p2p.data.base import PollResult

        result = PollResult(done=list(self._poll_done), failed=list(self._poll_failed))
        self._poll_done.clear()
        self._poll_failed.clear()
        return result

    def cancel(self, transfer_ids, mode: str = "immediate") -> list[int]:
        ids = list(transfer_ids)
        self._cancel_calls.append((ids, mode))
        if mode == "wait":
            still: list[int] = []
            for tid in ids:
                if tid in self._cancel_still_inflight:
                    still.append(tid)
                else:
                    self._transfers.pop(tid, None)
            return still
        for tid in ids:
            self._transfers.pop(tid, None)
            self._cancel_still_inflight.discard(tid)
        return []

    def ack_completions(self, peer_id, transfer_ids) -> None:
        self._ack_calls.append((peer_id, tuple(transfer_ids)))

    def close(self) -> None:
        pass


_DEFAULT_RECOVERY_RESULT = object()


class _RecoveringDataTransport(FakeDataTransport):
    """Owned-submit fake with exact-identity, peer-scoped recovery."""

    def __init__(self) -> None:
        super().__init__()
        self._owned: dict[int, tuple[str, object]] = {}
        self.submitted_tokens: list[object] = []
        self.recovery_calls: list[tuple[str, object]] = []
        self.submit_error: BaseException | None = None
        self.recovery_error: BaseException | None = None
        self.recovery_result: object = _DEFAULT_RECOVERY_RESULT

    def write_blocks_owned(
        self,
        peer_id,
        local_idxs,
        remote_idxs,
        *,
        recovery_token,
    ) -> int | None:
        transfer_id = super().write_blocks(peer_id, local_idxs, remote_idxs)
        if transfer_id is None:
            return None
        self._owned[transfer_id] = (peer_id, recovery_token)
        self.submitted_tokens.append(recovery_token)
        if self.submit_error is not None:
            error = self.submit_error
            self.submit_error = None
            raise error
        return transfer_id

    def recover_transfer_id(self, peer_id, recovery_token) -> int | None:
        self.recovery_calls.append((peer_id, recovery_token))
        if self.recovery_error is not None:
            raise self.recovery_error
        if self.recovery_result is not _DEFAULT_RECOVERY_RESULT:
            return self.recovery_result  # type: ignore[return-value]
        matches = [
            (transfer_id, owner_peer)
            for transfer_id, (owner_peer, token) in self._owned.items()
            if token is recovery_token
        ]
        if len(matches) > 1:
            raise RuntimeError("ambiguous recovery token")
        if not matches:
            return None
        transfer_id, owner_peer = matches[0]
        if owner_peer != peer_id:
            raise RuntimeError("recovery token belongs to another peer")
        return transfer_id

    def cancel(self, transfer_ids, mode: str = "immediate") -> list[int]:
        ids = list(transfer_ids)
        still = super().cancel(ids, mode)
        still_set = set(still)
        for transfer_id in ids:
            if transfer_id not in still_set:
                self._owned.pop(transfer_id, None)
        return still


class FakeConnection:
    """Fake ControlConnection that captures sent messages."""

    def __init__(
        self,
        peer_id: str = "peer:8000",
        peer_epoch: bytes = _DEFAULT_PEER_EPOCH,
    ) -> None:
        self.peer_id = peer_id
        self.peer_epoch = peer_epoch
        self._inbox: list[dict] = []
        self._sent: list[dict] = []
        self._closed = False
        # When True, send() raises to simulate a broken/dead connection.
        self.fail_send = False

    @property
    def alive(self) -> bool:
        return not self._closed

    def send(self, msg: dict) -> None:
        if self.fail_send:
            raise ConnectionError("simulated dead connection")
        self._sent.append(msg)

    def recv(self) -> list[dict]:
        msgs = self._inbox
        self._inbox = []
        return msgs

    def enqueue(self, msg: dict, *, stamp_channel: bool = True) -> None:
        inbound = dict(msg)
        msg_type = inbound.get(TYPE_KEY)
        if stamp_channel and msg_type == ConnectMsg.TYPE:
            inbound[TARGET_EPOCH_KEY] = self._local_epoch()
        elif stamp_channel and msg_type == ConnectAckMsg.TYPE:
            inbound[WIRE_MAJOR_KEY] = WIRE_PROTOCOL_MAJOR
            inbound[WIRE_MINOR_KEY] = WIRE_PROTOCOL_MINOR
            inbound[SOURCE_EPOCH_KEY] = self.peer_epoch
            inbound[TARGET_EPOCH_KEY] = self._local_epoch()
        elif stamp_channel and msg_type in _STATE_MESSAGE_TYPES:
            inbound[SOURCE_EPOCH_KEY] = self.peer_epoch
            inbound[TARGET_EPOCH_KEY] = self._local_epoch()
        self._inbox.append(inbound)

    def _local_epoch(self) -> bytes:
        connect = next(
            msg for msg in self._sent if msg.get(TYPE_KEY) == ConnectMsg.TYPE
        )
        return connect[ConnectMsg.SOURCE_EPOCH]

    def mark_dead(self) -> None:
        self._closed = True

    def close(self) -> None:
        self._closed = True


def _peer_connect_msg(
    peer_id: str = "peer:8000",
    block_len: int = 4096,
    fingerprint: str | None = "",
    hash_seed: str = _DEFAULT_HASH_SEED,
    source_epoch: bytes = _DEFAULT_PEER_EPOCH,
    target_epoch: bytes = UNSPECIFIED_EPOCH,
    num_blocks: int = 16,
) -> dict:
    """Build a ConnectMsg as if the peer sent it."""
    msg = {
        TYPE_KEY: ConnectMsg.TYPE,
        WIRE_MAJOR_KEY: WIRE_PROTOCOL_MAJOR,
        WIRE_MINOR_KEY: WIRE_PROTOCOL_MINOR,
        ConnectMsg.SOURCE_EPOCH: source_epoch,
        ConnectMsg.TARGET_EPOCH: target_epoch,
        ConnectMsg.PEER_ID: peer_id,
        ConnectMsg.AGENT_METADATA: b"peer-metadata",
        ConnectMsg.BASE_ADDR: 0x2000,
        ConnectMsg.NUM_BLOCKS: num_blocks,
        ConnectMsg.BLOCK_LEN: block_len,
        ConnectMsg.HASH_SEED: hash_seed,
    }
    if fingerprint is not None:
        msg[ConnectMsg.CONFIG_FINGERPRINT] = fingerprint
    return msg


def _peer_ack_msg(
    *,
    peer_id: str = "peer:8000",
    source_epoch: bytes = _DEFAULT_PEER_EPOCH,
    target_epoch: bytes,
) -> dict:
    return {
        TYPE_KEY: ConnectAckMsg.TYPE,
        WIRE_MAJOR_KEY: WIRE_PROTOCOL_MAJOR,
        WIRE_MINOR_KEY: WIRE_PROTOCOL_MINOR,
        ConnectAckMsg.PEER_ID: peer_id,
        ConnectAckMsg.SOURCE_EPOCH: source_epoch,
        ConnectAckMsg.TARGET_EPOCH: target_epoch,
    }


class FakeParent:
    """Configurable :class:`ParentManager` for server-role tests.

    ``stored`` is the dict of ready blocks (key → primary block_id).
    ``pending`` and ``retry`` script the first lookup() result for those
    keys; subsequent lookups behave normally (a key that promised
    HIT_PENDING / RETRY can later be promoted to HIT by adding it to
    ``stored`` and removing it from ``pending``/``retry``). ``calls``
    captures every parent invocation in order for assertions.

    Injected per-step via ``session.serve_external_requests(parent)`` —
    not held by the session, matching how ``TieringOffloadingManager``
    hands the tier a handle valid only for that call.
    """

    def __init__(
        self,
        stored: dict[OffloadKey, int] | None = None,
        pending: set[OffloadKey] | None = None,
        retry: set[OffloadKey] | None = None,
    ) -> None:
        self.stored: dict[OffloadKey, int] = dict(stored or {})
        self.pending: set[OffloadKey] = set(pending or ())
        self.retry: set[OffloadKey] = set(retry or ())
        self._next_job_id: int = 1000
        self.calls: list[tuple] = []

    def on_new_request(self, ctx: ReqContext) -> RequestOffloadingContext:
        self.calls.append(("on_new_request", ctx.req_id))
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, ctx: ReqContext) -> LookupResult:
        self.calls.append(("lookup", key, ctx.req_id))
        if key in self.pending:
            return LookupResult.HIT_PENDING
        if key in self.retry:
            return LookupResult.RETRY
        if key in self.stored:
            return LookupResult.HIT
        return LookupResult.MISS

    def create_store_job(
        self,
        keys: Sequence[OffloadKey],
        ctx: ReqContext,
    ) -> TransferJob:
        keys_list = list(keys)
        self.calls.append(("create_store_job", tuple(keys_list), ctx.req_id))
        block_ids = np.array([self.stored[k] for k in keys_list], dtype=np.int32)
        job_id = self._next_job_id
        self._next_job_id += 1
        return TransferJob(
            job_id=job_id,
            keys=keys_list,
            block_ids=block_ids,
            is_promotion=False,
            req_context=ctx,
        )

    def on_request_finished(self, ctx: ReqContext) -> None:
        self.calls.append(("on_request_finished", ctx.req_id))


def _make_session(
    conn: FakeConnection | None = None,
    transport: FakeDataTransport | None = None,
    peer_id: str = "peer:8000",
    local_id: str = "local:9000",
    local_hash_seed: str = _DEFAULT_HASH_SEED,
) -> tuple[P2PSession, FakeConnection, FakeDataTransport]:
    if conn is None:
        conn = FakeConnection(peer_id=peer_id)
    if transport is None:
        transport = FakeDataTransport()
    session = P2PSession(
        peer_id=peer_id,
        local_id=local_id,
        transport=transport,  # type: ignore[arg-type]
        local_block_len=transport.block_len,
        local_hash_seed=local_hash_seed,
        conn=conn,  # type: ignore[arg-type]
    )
    return session, conn, transport


def _serve(session: P2PSession, parent: FakeParent) -> None:
    """Resolve enqueued inbound lookups, as the manager does each step."""
    session.serve_external_requests(parent)  # type: ignore[arg-type]


def _activate(
    session: P2PSession,
    conn: FakeConnection,
    peer_id: str = "peer:8000",
    peer_epoch: bytes = _DEFAULT_PEER_EPOCH,
    num_blocks: int = 16,
) -> None:
    """Drive the handshake: peer sends ConnectMsg + ConnectAckMsg."""
    conn.peer_epoch = peer_epoch
    conn.enqueue(
        _peer_connect_msg(
            peer_id=peer_id,
            source_epoch=peer_epoch,
            num_blocks=num_blocks,
        )
    )
    conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: peer_id})
    session.poll()


# --- Accessors for the server role's consolidated per-request state.
# ServerRole keeps one _ServerRequestState per kv_request_id; entries are
# garbage-collected once fully idle, so "no outbound"/"no abort" reads as
# either a missing entry or a None field. These helpers paper over that.


def _client_load(session: P2PSession, kv_request_id: str):
    """The single in-flight load of a kv_request_id (loads are per-round)."""
    loads = session._client._requests[kv_request_id].loads
    assert len(loads) == 1
    return next(iter(loads.values()))


def _srv_outbound(session: P2PSession, kv_request_id: str):
    """Serve-side round for a kv_request_id, or None (idle / GC'd).

    Rounds are keyed by wire round_seq; surfaces the demanded round when
    a fetch has bound one, else any parked supply round.
    """
    st = session._server._requests.get(kv_request_id)
    if st is None or not st.outbound:
        return None
    for rnd in st.outbound.values():
        if rnd.demand_received:
            return rnd
    return next(iter(st.outbound.values()))


def _srv_lookups(session: P2PSession) -> list:
    """Every parked inbound _ActiveLookup across all requests."""
    return [
        lu for st in session._server._requests.values() for lu in st.lookups.values()
    ]


def _srv_abort_started(session: P2PSession, kv_request_id: str) -> float | None:
    """Pending-abort start time for a kv_request_id, or None."""
    for (kv, _), started in session._server._pending_aborts.items():
        if kv == kv_request_id:
            return started
    return None


def _srv_inflight_count(session: P2PSession, kv_request_id: str) -> int:
    """Inflight-transfer count tracked for a kv_request_id (0 if idle)."""
    st = session._server._requests.get(kv_request_id)
    return len(st.inflight_tids) if st is not None else 0


def _srv_total_inflight(session: P2PSession) -> int:
    """Sum of per-request inflight counts across all requests."""
    return sum(len(st.inflight_tids) for st in session._server._requests.values())


# ---------------------------------------------------------------------------
# Connect / handshake
# ---------------------------------------------------------------------------


class TestConnectHandshake:
    def test_connect_msg_sent_on_creation(self):
        """Session sends its own ConnectMsg on connection."""
        session, conn, _ = _make_session()
        assert len(conn._sent) == 1
        msg = conn._sent[0]
        assert msg[TYPE_KEY] == ConnectMsg.TYPE
        assert msg[ConnectMsg.PEER_ID] == "local:9000"
        assert msg[ConnectMsg.NUM_BLOCKS] == 16
        assert msg[ConnectMsg.BLOCK_LEN] == 4096
        assert ConnectMsg.AGENT_METADATA in msg
        assert msg[WIRE_MAJOR_KEY] == WIRE_PROTOCOL_MAJOR
        assert msg[WIRE_MINOR_KEY] == WIRE_PROTOCOL_MINOR
        assert type(msg[ConnectMsg.SOURCE_EPOCH]) is bytes
        assert len(msg[ConnectMsg.SOURCE_EPOCH]) == SESSION_EPOCH_NBYTES
        assert msg[ConnectMsg.SOURCE_EPOCH] != UNSPECIFIED_EPOCH
        assert msg[ConnectMsg.TARGET_EPOCH] == UNSPECIFIED_EPOCH

    def test_discovery_and_ack_wait_for_targeted_connect_before_import(self):
        session, conn, transport = _make_session()
        conn.enqueue(_peer_connect_msg(), stamp_channel=False)
        conn.enqueue(
            _peer_ack_msg(target_epoch=session._local_epoch),
            stamp_channel=False,
        )

        session.poll()

        assert session.alive
        assert not session.ready
        assert transport._add_remote_calls == []
        assert any(
            msg.get(TYPE_KEY) == ConnectMsg.TYPE
            and msg[ConnectMsg.TARGET_EPOCH] == _DEFAULT_PEER_EPOCH
            for msg in conn._sent
        )

        conn.enqueue(
            _peer_connect_msg(target_epoch=session._local_epoch),
            stamp_channel=False,
        )
        session.poll()

        assert session.ready
        assert transport._add_remote_calls == ["peer:8000"]

    def test_unready_session_retries_identical_discovery_connect(self):
        session, conn, _ = _make_session()
        first = conn._sent[0]

        session.poll()
        assert conn._sent == [first]

        session._last_discovery_send_at -= 1.0
        session.poll()

        discoveries = [
            msg
            for msg in conn._sent
            if msg.get(TYPE_KEY) == ConnectMsg.TYPE
            and msg[ConnectMsg.TARGET_EPOCH] == UNSPECIFIED_EPOCH
        ]
        assert len(discoveries) == 2
        assert discoveries[0] is discoveries[1]

    def test_ready_session_requires_two_frame_successor_proof(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        successor = b"N" * SESSION_EPOCH_NBYTES

        conn.enqueue(
            _peer_connect_msg(source_epoch=successor),
            stamp_channel=False,
        )
        session.poll()

        assert session.alive
        assert session.ready
        assert transport._add_remote_calls == ["peer:8000"]
        assert not any(
            msg.get(TYPE_KEY) == ConnectAckMsg.TYPE
            and msg[ConnectAckMsg.TARGET_EPOCH] == successor
            for msg in conn._sent
        )

        conn.enqueue(
            _peer_ack_msg(
                source_epoch=successor,
                target_epoch=session._local_epoch,
            ),
            stamp_channel=False,
        )
        session.poll()
        assert session.alive

        conn.enqueue(
            _peer_connect_msg(
                source_epoch=successor,
                target_epoch=session._local_epoch,
            ),
            stamp_channel=False,
        )
        session.poll()

        assert not session.alive
        assert transport._add_remote_calls == ["peer:8000"]

    def test_invalid_different_epoch_handshake_cannot_kill_ready_session(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        retired = b"R" * SESSION_EPOCH_NBYTES

        bad_connect = _peer_connect_msg(
            source_epoch=retired,
            target_epoch=session._local_epoch,
            fingerprint="retired-config",
        )
        conn.enqueue(bad_connect, stamp_channel=False)
        bad_ack = _peer_ack_msg(
            source_epoch=retired,
            target_epoch=session._local_epoch,
        )
        bad_ack[WIRE_MINOR_KEY] = WIRE_PROTOCOL_MINOR + 1
        conn.enqueue(bad_ack, stamp_channel=False)

        session.poll()

        assert session.alive
        assert session.ready
        assert retired not in session._connect_candidates
        assert retired not in session._acked_remote_epochs
        assert transport._add_remote_calls == ["peer:8000"]

    def test_peer_connect_triggers_add_remote_and_ack(self):
        """Connect is acknowledged, but import waits for matching peer Ack."""
        session, conn, transport = _make_session()
        conn.enqueue(_peer_connect_msg())
        session.poll()
        assert "peer:8000" not in transport._remote_peers
        ack = next(m for m in conn._sent if m[TYPE_KEY] == ConnectAckMsg.TYPE)
        assert ack[ConnectAckMsg.PEER_ID] == "local:9000"
        assert ack[ConnectAckMsg.SOURCE_EPOCH] == session._local_epoch
        assert ack[ConnectAckMsg.TARGET_EPOCH] == _DEFAULT_PEER_EPOCH

        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        assert "peer:8000" in transport._remote_peers

    def test_connect_ack_makes_session_ready(self):
        """Ack alone is insufficient; matching Connect completes readiness."""
        session, conn, _ = _make_session()
        assert not session.ready
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        assert not session.ready
        conn.enqueue(_peer_connect_msg())
        session.poll()
        assert session.ready

    def test_messages_queued_before_ack_flush_on_ack(self):
        """Outgoing messages sent before ConnectAck are flushed after."""
        session, conn, _ = _make_session()
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        # Before ack: only our ConnectMsg was sent.
        assert len(conn._sent) == 1
        assert conn._sent[0][TYPE_KEY] == ConnectMsg.TYPE
        # Matching Connect and Ack arrive.
        conn.enqueue(_peer_connect_msg())
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        # Queued fetch is now sent.
        assert any(m[TYPE_KEY] == FetchMsg.TYPE for m in conn._sent)

    def test_queued_abort_lost_return_replays_on_next_poll(self):
        """The handshake ack is one-shot; a retained Abort must self-replay."""
        session, conn, _ = _make_session()
        session.request_blocks(
            job_id=1, kv_request_id="req-abort-cut", keys=[b"k"], block_ids=[0]
        )
        session.finish_request("req-abort-cut")
        assert [msg[TYPE_KEY] for msg in session._queued] == [
            FetchMsg.TYPE,
            AbortFetchMsg.TYPE,
        ]

        original_send = conn.send
        lost_returns = 1

        def send_then_interrupt(msg):
            nonlocal lost_returns
            original_send(msg)
            if msg[TYPE_KEY] == AbortFetchMsg.TYPE and lost_returns:
                lost_returns -= 1
                raise _CloseFailure("queued abort lost return")

        conn.send = send_then_interrupt  # type: ignore[method-assign]
        conn.enqueue(_peer_connect_msg())
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        with pytest.raises(_CloseFailure, match="queued abort lost return"):
            session.poll()

        assert [msg[TYPE_KEY] for msg in session._queued] == [AbortFetchMsg.TYPE]
        assert ("req-abort-cut", 0) in session._client._abort_intents
        session.poll()
        assert session._queued == []
        aborts = [msg for msg in conn._sent if msg[TYPE_KEY] == AbortFetchMsg.TYPE]
        assert [msg[AbortFetchMsg.ROUND_SEQ] for msg in aborts] == [0, 0]

    def test_queued_abort_ack_lost_return_replays_on_next_poll(self):
        """A proven-quiescent queued AbortAck remains owned across a cut."""
        session, conn, _ = _make_session()
        session._server.on_abort_fetch("req-ack-cut", 7)
        assert [msg[TYPE_KEY] for msg in session._queued] == [AbortAckMsg.TYPE]

        original_send = conn.send
        lost_returns = 1

        def send_then_interrupt(msg):
            nonlocal lost_returns
            original_send(msg)
            if msg[TYPE_KEY] == AbortAckMsg.TYPE and lost_returns:
                lost_returns -= 1
                raise _CloseFailure("queued abort ack lost return")

        conn.send = send_then_interrupt  # type: ignore[method-assign]
        conn.enqueue(_peer_connect_msg())
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        with pytest.raises(_CloseFailure, match="queued abort ack lost return"):
            session.poll()

        assert [msg[TYPE_KEY] for msg in session._queued] == [AbortAckMsg.TYPE]
        assert ("req-ack-cut", 7) in session._server._abort_ack_intents
        session.poll()
        assert session._queued == []
        acks = [msg for msg in conn._sent if msg[TYPE_KEY] == AbortAckMsg.TYPE]
        assert [msg[AbortAckMsg.ROUND_SEQ] for msg in acks] == [7, 7]

    def test_block_len_mismatch_marks_dead(self):
        """Mismatched block_len rejects peer and marks connection dead."""
        session, conn, transport = _make_session()
        conn.enqueue(_peer_connect_msg(block_len=8192))  # mismatch
        session.poll()
        assert "peer:8000" not in transport._remote_peers
        assert not session.alive

    def test_config_fingerprint_mismatch_marks_dead(self):
        """Mismatched config fingerprint rejects peer."""
        transport = FakeDataTransport(config_fingerprint="abc123")
        session, conn, _ = _make_session(transport=transport)
        conn.enqueue(_peer_connect_msg(fingerprint="different"))
        session.poll()
        assert "peer:8000" not in transport._remote_peers
        assert not session.alive

    def test_config_fingerprint_match_succeeds(self):
        """Matching fingerprints register the peer."""
        transport = FakeDataTransport(config_fingerprint="same_fp")
        session, conn, _ = _make_session(transport=transport)
        conn.enqueue(_peer_connect_msg(fingerprint="same_fp"))
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        assert "peer:8000" in transport._remote_peers
        assert session.alive

    def test_missing_fingerprint_rejected(self):
        """Fingerprint is a mandatory exact string, including empty."""
        transport = FakeDataTransport(config_fingerprint="abc123")
        session, conn, _ = _make_session(transport=transport)
        conn.enqueue(_peer_connect_msg(fingerprint=None))
        session.poll()
        assert "peer:8000" not in transport._remote_peers
        assert not session.alive

    def test_hash_seed_mismatch_marks_dead(self):
        """Mismatched PYTHONHASHSEED rejects peer and marks connection dead."""
        session, conn, transport = _make_session(local_hash_seed="0")
        conn.enqueue(_peer_connect_msg(hash_seed="12345"))  # mismatch
        session.poll()
        assert "peer:8000" not in transport._remote_peers
        assert not session.alive
        assert not any(m[TYPE_KEY] == ConnectAckMsg.TYPE for m in conn._sent)

    def test_hash_seed_match_succeeds(self):
        """Matching PYTHONHASHSEED registers the peer and acks."""
        session, conn, transport = _make_session(local_hash_seed="12345")
        conn.enqueue(_peer_connect_msg(hash_seed="12345"))
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        assert "peer:8000" in transport._remote_peers
        assert session.alive
        assert any(m[TYPE_KEY] == ConnectAckMsg.TYPE for m in conn._sent)

    def test_hash_seed_advertised_in_connect_msg(self):
        """Session advertises its own PYTHONHASHSEED in the ConnectMsg."""
        _, conn, _ = _make_session(local_hash_seed="777")
        assert conn._sent[0][ConnectMsg.HASH_SEED] == "777"

    def test_connect_and_ack_from_different_epochs_do_not_provision(self):
        session, conn, transport = _make_session()
        old_epoch = b"O" * SESSION_EPOCH_NBYTES
        current_epoch = b"N" * SESSION_EPOCH_NBYTES
        session.request_blocks(1, "req-held", [b"k"], [0])

        conn.enqueue(_peer_connect_msg(source_epoch=old_epoch))
        conn.enqueue(
            _peer_ack_msg(
                source_epoch=current_epoch,
                target_epoch=session._local_epoch,
            ),
            stamp_channel=False,
        )
        session.poll()

        assert not session.ready
        assert transport._add_remote_calls == []
        assert [msg[TYPE_KEY] for msg in session._queued] == [FetchMsg.TYPE]

        conn.enqueue(_peer_connect_msg(source_epoch=current_epoch))
        session.poll()

        assert session.ready
        assert session._remote_epoch == current_epoch
        assert transport._add_remote_calls == ["peer:8000"]
        assert session._queued == []

    def test_ack_for_retired_local_epoch_does_not_complete_or_kill(self):
        session, conn, transport = _make_session()
        conn.enqueue(_peer_connect_msg())
        conn.enqueue(
            _peer_ack_msg(target_epoch=b"X" * SESSION_EPOCH_NBYTES),
            stamp_channel=False,
        )
        session.poll()

        assert session.alive
        assert not session.ready
        assert transport._add_remote_calls == []

        conn.enqueue(
            _peer_ack_msg(target_epoch=session._local_epoch),
            stamp_channel=False,
        )
        session.poll()
        assert session.ready
        assert transport._add_remote_calls == ["peer:8000"]

    def test_retired_connect_and_ack_do_not_kill_ready_session(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        old_epoch = b"O" * SESSION_EPOCH_NBYTES

        conn.enqueue(_peer_connect_msg(source_epoch=old_epoch))
        conn.enqueue(
            _peer_ack_msg(
                source_epoch=old_epoch,
                target_epoch=b"X" * SESSION_EPOCH_NBYTES,
            ),
            stamp_channel=False,
        )
        session.poll()

        assert session.alive
        assert session.ready
        assert session._remote_epoch == _DEFAULT_PEER_EPOCH
        assert transport._add_remote_calls == ["peer:8000"]

    def test_ack_lost_return_replays_and_completes_without_peer_retry(self):
        session, conn, transport = _make_session()
        conn.enqueue(
            _peer_ack_msg(target_epoch=session._local_epoch),
            stamp_channel=False,
        )
        session.poll()
        assert not session.ready

        original_send = conn.send
        interrupted = True

        def send_then_interrupt(msg):
            nonlocal interrupted
            original_send(msg)
            if msg[TYPE_KEY] == ConnectAckMsg.TYPE and interrupted:
                interrupted = False
                raise _CloseFailure("handshake Ack lost return")

        conn.send = send_then_interrupt  # type: ignore[method-assign]
        conn.enqueue(_peer_connect_msg())
        with pytest.raises(_CloseFailure, match="handshake Ack lost return"):
            session.poll()
        assert not session.ready
        assert transport._add_remote_calls == []
        assert _DEFAULT_PEER_EPOCH in session._pending_connect_acks

        session.poll()
        assert session.ready
        assert transport._add_remote_calls == ["peer:8000"]
        acks = [msg for msg in conn._sent if msg[TYPE_KEY] == ConnectAckMsg.TYPE]
        assert len(acks) == 2
        assert acks[0] == acks[1]

    @pytest.mark.parametrize(
        ("field", "value"),
        (
            (WIRE_MAJOR_KEY, True),
            (WIRE_MAJOR_KEY, WIRE_PROTOCOL_MAJOR + 1),
            (WIRE_MINOR_KEY, WIRE_PROTOCOL_MINOR + 1),
        ),
    )
    def test_bad_connect_version_fails_before_import_or_flush(self, field, value):
        session, conn, transport = _make_session()
        session.request_blocks(1, "req-held", [b"k"], [0])
        msg = _peer_connect_msg()
        msg[field] = value
        conn.enqueue(msg)
        session.poll()

        assert not session.alive
        assert transport._add_remote_calls == []
        assert [msg[TYPE_KEY] for msg in session._queued] == [FetchMsg.TYPE]

    def test_bad_ack_version_fails_before_import_or_flush(self):
        session, conn, transport = _make_session()
        session.request_blocks(1, "req-held", [b"k"], [0])
        conn.enqueue(_peer_connect_msg())
        ack = _peer_ack_msg(target_epoch=session._local_epoch)
        ack[WIRE_MINOR_KEY] = True
        conn.enqueue(ack, stamp_channel=False)
        session.poll()

        assert not session.alive
        assert transport._add_remote_calls == []
        assert [msg[TYPE_KEY] for msg in session._queued] == [FetchMsg.TYPE]

    @pytest.mark.parametrize(
        ("local_fingerprint", "remote_fingerprint"),
        (("", "remote"), ("local", "")),
    )
    def test_empty_and_nonempty_fingerprints_never_match(
        self, local_fingerprint, remote_fingerprint
    ):
        transport = FakeDataTransport(config_fingerprint=local_fingerprint)
        session, conn, _ = _make_session(transport=transport)
        conn.enqueue(_peer_connect_msg(fingerprint=remote_fingerprint))
        session.poll()

        assert not session.alive
        assert transport._add_remote_calls == []


class TestSessionEpochIsolation:
    @staticmethod
    def _state_cases() -> tuple[tuple[dict, str, str], ...]:
        return (
            (
                {
                    TYPE_KEY: TransferDoneMsg.TYPE,
                    TransferDoneMsg.KV_REQUEST_ID: "req",
                    TransferDoneMsg.SUCCESS: True,
                    TransferDoneMsg.ROUND_SEQ: 0,
                },
                "_client",
                "on_transfer_done",
            ),
            (
                {
                    TYPE_KEY: AbortAckMsg.TYPE,
                    AbortAckMsg.KV_REQUEST_ID: "req",
                    AbortAckMsg.ROUND_SEQ: 0,
                },
                "_client",
                "on_abort_ack",
            ),
            (
                {
                    TYPE_KEY: LookupRespMsg.TYPE,
                    LookupRespMsg.KV_REQUEST_ID: "req",
                    LookupRespMsg.KEYS: [b"k"],
                    LookupRespMsg.HITS: [True],
                    LookupRespMsg.ROUND_SEQ: 0,
                },
                "_client",
                "on_lookup_resp",
            ),
            (
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.KV_REQUEST_ID: "req",
                    FetchMsg.KEYS: [b"k"],
                    FetchMsg.BLOCK_INDEXES: [0],
                    FetchMsg.ROUND_SEQ: 0,
                },
                "_server",
                "on_fetch",
            ),
            (
                {
                    TYPE_KEY: LookupMsg.TYPE,
                    LookupMsg.KV_REQUEST_ID: "req",
                    LookupMsg.KEYS: [b"k"],
                    LookupMsg.ROUND_SEQ: 0,
                },
                "_server",
                "on_lookup",
            ),
            (
                {
                    TYPE_KEY: AbortFetchMsg.TYPE,
                    AbortFetchMsg.KV_REQUEST_ID: "req",
                    AbortFetchMsg.ROUND_SEQ: 0,
                },
                "_server",
                "on_abort_fetch",
            ),
        )

    @pytest.mark.parametrize(("msg", "role_name", "handler_name"), _state_cases())
    @pytest.mark.parametrize(
        "epoch_case",
        ("retired_pair", "retired_source", "retired_target"),
    )
    def test_stale_state_frame_is_dropped_before_role_or_routing_mutation(
        self, monkeypatch, msg, role_name, handler_name, epoch_case
    ):
        old, old_conn, _ = _make_session()
        old_peer_epoch = b"O" * SESSION_EPOCH_NBYTES
        _activate(old, old_conn, peer_epoch=old_peer_epoch)
        old_local_epoch = old._local_epoch
        old.close()
        assert old.close_complete

        current_peer_epoch = b"N" * SESSION_EPOCH_NBYTES
        session, conn, _ = _make_session()
        _activate(session, conn, peer_epoch=current_peer_epoch)
        calls: list[tuple] = []
        role = getattr(session, role_name)
        monkeypatch.setattr(role, handler_name, lambda *args: calls.append(args))
        source_epoch = (
            current_peer_epoch if epoch_case == "retired_target" else old_peer_epoch
        )
        target_epoch = (
            session._local_epoch if epoch_case == "retired_source" else old_local_epoch
        )
        stale = {
            **msg,
            SOURCE_EPOCH_KEY: source_epoch,
            TARGET_EPOCH_KEY: target_epoch,
        }

        conn.enqueue(stale, stamp_channel=False)
        result = session.poll()

        assert session.alive
        assert calls == []
        assert result.new_fetch_ids == []

    def test_stale_fetch_cannot_create_manager_binding_signal(self):
        old, old_conn, _ = _make_session()
        old_peer_epoch = b"O" * SESSION_EPOCH_NBYTES
        _activate(old, old_conn, peer_epoch=old_peer_epoch)
        old_local_epoch = old._local_epoch
        old.close()

        session, conn, _ = _make_session()
        _activate(session, conn, peer_epoch=b"N" * SESSION_EPOCH_NBYTES)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req-stale-bind",
                FetchMsg.KEYS: [b"k"],
                FetchMsg.BLOCK_INDEXES: [0],
                FetchMsg.ROUND_SEQ: 0,
                SOURCE_EPOCH_KEY: old_peer_epoch,
                TARGET_EPOCH_KEY: old_local_epoch,
            },
            stamp_channel=False,
        )

        result = session.poll()

        # P2PSecondaryTierManager binds only IDs in new_fetch_ids. The stale
        # frame is rejected before that journal or ServerRole is touched.
        assert result.new_fetch_ids == []
        assert "req-stale-bind" not in session._server._requests

    def test_stale_disconnect_does_not_kill_replacement_session(self):
        old, old_conn, _ = _make_session()
        old_peer_epoch = b"O" * SESSION_EPOCH_NBYTES
        _activate(old, old_conn, peer_epoch=old_peer_epoch)
        old_local_epoch = old._local_epoch
        old.close()

        new, new_conn, _ = _make_session()
        _activate(new, new_conn, peer_epoch=b"N" * SESSION_EPOCH_NBYTES)
        new_conn.enqueue(
            {
                TYPE_KEY: DisconnectMsg.TYPE,
                SOURCE_EPOCH_KEY: old_peer_epoch,
                TARGET_EPOCH_KEY: old_local_epoch,
            },
            stamp_channel=False,
        )

        new.poll()

        assert new.alive
        assert new.ready

    @pytest.mark.parametrize("missing", (SOURCE_EPOCH_KEY, TARGET_EPOCH_KEY))
    def test_missing_epoch_fails_without_resolving_active_destination(self, missing):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(7, "req", [b"k"], [0])
        msg = {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: "req",
            TransferDoneMsg.SUCCESS: True,
            TransferDoneMsg.ROUND_SEQ: 0,
            SOURCE_EPOCH_KEY: _DEFAULT_PEER_EPOCH,
            TARGET_EPOCH_KEY: session._local_epoch,
        }
        del msg[missing]

        conn.enqueue(msg, stamp_channel=False)
        result = session.poll()

        assert not session.alive
        assert result.loads == []
        assert session._client.has_active_loads

    @pytest.mark.parametrize("wrong", (SOURCE_EPOCH_KEY, TARGET_EPOCH_KEY))
    def test_wrong_epoch_is_stale_not_a_protocol_kill(self, wrong):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(7, "req", [b"k"], [0])
        msg = {
            TYPE_KEY: TransferDoneMsg.TYPE,
            TransferDoneMsg.KV_REQUEST_ID: "req",
            TransferDoneMsg.SUCCESS: True,
            TransferDoneMsg.ROUND_SEQ: 0,
            SOURCE_EPOCH_KEY: _DEFAULT_PEER_EPOCH,
            TARGET_EPOCH_KEY: session._local_epoch,
        }
        msg[wrong] = b"X" * SESSION_EPOCH_NBYTES

        conn.enqueue(msg, stamp_channel=False)
        result = session.poll()

        assert session.alive
        assert result.loads == []
        assert session._client.has_active_loads

    def test_old_terminals_cannot_resolve_new_token_zero_after_retirement(self):
        old, old_conn, _ = _make_session()
        old_peer_epoch = b"O" * SESSION_EPOCH_NBYTES
        _activate(old, old_conn, peer_epoch=old_peer_epoch)
        old_local_epoch = old._local_epoch
        old.close()
        assert old.close_complete

        new, new_conn, _ = _make_session()
        new_peer_epoch = b"N" * SESSION_EPOCH_NBYTES
        _activate(new, new_conn, peer_epoch=new_peer_epoch)
        new.request_blocks(7, "req-reused", [b"k"], [0])

        for msg_type in (TransferDoneMsg, AbortAckMsg):
            stale = {
                TYPE_KEY: msg_type.TYPE,
                msg_type.KV_REQUEST_ID: "req-reused",
                msg_type.ROUND_SEQ: 0,
                SOURCE_EPOCH_KEY: old_peer_epoch,
                TARGET_EPOCH_KEY: old_local_epoch,
            }
            if msg_type is TransferDoneMsg:
                stale[TransferDoneMsg.SUCCESS] = True
            new_conn.enqueue(stale, stamp_channel=False)

        assert new.poll().loads == []
        assert new._client.has_active_loads

        new_conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.KV_REQUEST_ID: "req-reused",
                TransferDoneMsg.SUCCESS: True,
                TransferDoneMsg.ROUND_SEQ: 0,
            }
        )
        assert new.poll().loads == [LoadResult(7, "req-reused", True)]

    def test_fetch_index_must_fit_matched_requester_block_count(self):
        session, conn, transport = _make_session()
        _activate(session, conn, num_blocks=2)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.KV_REQUEST_ID: "req",
                FetchMsg.KEYS: [b"k"],
                FetchMsg.BLOCK_INDEXES: [2],
                FetchMsg.ROUND_SEQ: 0,
            }
        )

        result = session.poll()

        assert not session.alive
        assert result.new_fetch_ids == []
        assert transport._transfers == {}


# ---------------------------------------------------------------------------
# Client-role flows
# ---------------------------------------------------------------------------


class TestClientFlows:
    def test_request_blocks_sends_fetch(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k1", b"k2"], block_ids=[0, 1]
        )
        lookup = conn._sent[-1]
        assert lookup[TYPE_KEY] == FetchMsg.TYPE
        assert lookup[FetchMsg.KV_REQUEST_ID] == "req-1"
        assert lookup[FetchMsg.KEYS] == [b"k1", b"k2"]
        assert lookup[FetchMsg.BLOCK_INDEXES] == [0, 1]
        assert lookup[SOURCE_EPOCH_KEY] == session._local_epoch
        assert lookup[TARGET_EPOCH_KEY] == _DEFAULT_PEER_EPOCH

    def test_round_token_exhaustion_fails_closed_without_wrap(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session._client._next_round_seq = MAX_ROUND_SEQ

        session.request_blocks(1, "req-last", [b"last"], [0])
        assert conn._sent[-1][FetchMsg.ROUND_SEQ] == MAX_ROUND_SEQ
        sent_before = len(conn._sent)

        with pytest.raises(OverflowError, match="operation-token space exhausted"):
            session.request_blocks(2, "req-overflow", [b"overflow"], [1])

        assert len(conn._sent) == sent_before
        assert "req-overflow" not in session._client._requests
        assert session._client._next_round_seq == MAX_ROUND_SEQ + 1

    def test_transfer_done_success(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.ROUND_SEQ: 0,
                TransferDoneMsg.KV_REQUEST_ID: "req-1",
                TransferDoneMsg.SUCCESS: True,
            }
        )
        loads = session.poll().loads
        assert loads == [LoadResult(job_id=1, kv_request_id="req-1", success=True)]

    def test_transfer_done_failure(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.ROUND_SEQ: 0,
                TransferDoneMsg.KV_REQUEST_ID: "req-1",
                TransferDoneMsg.SUCCESS: False,
            }
        )
        loads = session.poll().loads
        assert loads == [LoadResult(job_id=1, kv_request_id="req-1", success=False)]

    def test_finish_request_sends_abort(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        session.finish_request("req-1")
        abort = conn._sent[-1]
        assert abort[TYPE_KEY] == AbortFetchMsg.TYPE
        assert abort[AbortFetchMsg.KV_REQUEST_ID] == "req-1"

    def test_finish_preserves_active_load_until_terminal_result(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=7, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )

        session.finish_request("req-1")

        load = _client_load(session, "req-1")
        assert load.job_id == 7
        assert load.aborted_at is not None
        aborted_at = load.aborted_at
        assert session._client.has_active_loads is True
        session.finish_request("req-1")
        assert _client_load(session, "req-1").aborted_at == aborted_at
        aborts = [m for m in conn._sent if m[TYPE_KEY] == AbortFetchMsg.TYPE]
        assert len(aborts) == 1
        assert aborts[0][AbortFetchMsg.ROUND_SEQ] == 0

    def test_finish_abort_ack_emits_one_failure(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=8, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        session.finish_request("req-1")

        conn.enqueue(
            {
                TYPE_KEY: AbortAckMsg.TYPE,
                AbortAckMsg.ROUND_SEQ: 0,
                AbortAckMsg.KV_REQUEST_ID: "req-1",
            }
        )
        loads = session.poll().loads
        assert loads == [LoadResult(job_id=8, kv_request_id="req-1", success=False)]
        assert session._client.has_active_loads is False
        assert "req-1" not in session._client._requests
        session.ack_results(load_job_ids=(8,))
        assert session.poll().loads == []

    def test_stale_abort_ack_tombstone_cannot_finish_reused_request(self):
        """An old server tombstone cannot prove a new generation quiescent."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        request_id = "req-reused"

        session.request_blocks(1, request_id, [b"old"], [0])
        old_token = conn._sent[-1][FetchMsg.ROUND_SEQ]
        session.finish_request(request_id)
        conn.enqueue(
            {
                TYPE_KEY: AbortAckMsg.TYPE,
                AbortAckMsg.KV_REQUEST_ID: request_id,
                AbortAckMsg.ROUND_SEQ: old_token,
            }
        )
        assert session.poll().loads == [LoadResult(1, request_id, False)]
        session.ack_results(load_job_ids=(1,))
        assert request_id not in session._client._requests

        # Materialize the peer-side lost-return tombstone, then reuse the same
        # request ID. The new Fetch also drives a live fake DMA on the server
        # role so replaying the old Ack exercises the unsafe ABA boundary.
        session._server.on_abort_fetch(request_id, old_token)
        assert (request_id, old_token) in session._server._abort_ack_intents
        session.request_blocks(2, request_id, [b"new"], [1])
        new_fetch = conn._sent[-1]
        new_token = new_fetch[FetchMsg.ROUND_SEQ]
        assert new_token > old_token

        session.add_stored_blocks(request_id, [b"new"], [3], job_id=10)
        conn.enqueue(dict(new_fetch))
        session.poll()
        assert session._server._inflight

        sent_before = len(conn._sent)
        session._server.on_abort_fetch(request_id, old_token)
        stale_ack = next(
            msg for msg in conn._sent[sent_before:] if msg[TYPE_KEY] == AbortAckMsg.TYPE
        )
        conn.enqueue(dict(stale_ack))

        assert session.poll().loads == []
        assert session._client._requests[request_id].loads[new_token].job_id == 2
        assert session._client.has_active_loads is True
        assert transport._transfers

    def test_finish_abort_timeout_retains_destination_and_retries(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=9, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        session.finish_request("req-1")
        _client_load(session, "req-1").aborted_at = (
            time.monotonic() - _ABORT_ACK_TIMEOUT_S - 1.0
        )
        session._client._abort_intents[("req-1", 0)].last_sent_at = (
            time.monotonic() - _ABORT_ACK_TIMEOUT_S - 1.0
        )
        abort_count = sum(m[TYPE_KEY] == AbortFetchMsg.TYPE for m in conn._sent)

        loads = session.poll().loads
        assert loads == []
        assert session._client.has_active_loads is True
        assert "req-1" in session._client._requests
        assert sum(m[TYPE_KEY] == AbortFetchMsg.TYPE for m in conn._sent) == (
            abort_count + 1
        )

    def test_abort_lost_return_retains_intent_and_retries_same_round(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=91, kv_request_id="req-lost", keys=[b"k"], block_ids=[0]
        )
        original_send = conn.send
        lost_returns = 1

        def send_then_interrupt(msg):
            nonlocal lost_returns
            original_send(msg)
            if msg[TYPE_KEY] == AbortFetchMsg.TYPE and lost_returns:
                lost_returns -= 1
                raise _CloseFailure("abort lost return")

        conn.send = send_then_interrupt  # type: ignore[method-assign]
        with pytest.raises(_CloseFailure, match="abort lost return"):
            session.finish_request("req-lost")

        load = _client_load(session, "req-lost")
        assert ("req-lost", 0) in session._client._abort_intents
        assert load.aborted_at is None
        session.finish_request("req-lost")
        assert load.aborted_at is not None
        aborts = [m for m in conn._sent if m[TYPE_KEY] == AbortFetchMsg.TYPE]
        assert [m[AbortFetchMsg.ROUND_SEQ] for m in aborts] == [0, 0]

    def test_abort_send_exception_marks_dead_without_false_commit(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=92, kv_request_id="req-dead", keys=[b"k"], block_ids=[0]
        )
        conn.fail_send = True

        session.finish_request("req-dead")

        load = _client_load(session, "req-dead")
        assert ("req-dead", 0) in session._client._abort_intents
        assert load.aborted_at is None
        assert session.alive is False

    def test_late_transfer_done_after_abort_ack_is_ignored(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=10, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        session.finish_request("req-1")

        conn.enqueue(
            {
                TYPE_KEY: AbortAckMsg.TYPE,
                AbortAckMsg.ROUND_SEQ: 0,
                AbortAckMsg.KV_REQUEST_ID: "req-1",
            }
        )
        assert session.poll().loads == [
            LoadResult(job_id=10, kv_request_id="req-1", success=False)
        ]
        session.ack_results(load_job_ids=(10,))

        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.ROUND_SEQ: 0,
                TransferDoneMsg.KV_REQUEST_ID: "req-1",
                TransferDoneMsg.SUCCESS: True,
            }
        )
        assert session.poll().loads == []
        assert session._client.has_active_loads is False

    def test_active_loads_work_list_tracks_in_flight(self):
        """collect_results / has_active_loads use the _active_loads work-list,
        armed when a fetch is issued and discarded exactly when its load
        clears — a probe-only request never enters it, and completion empties
        it while the entry may briefly linger for GC."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        client = session._client

        # A probe-only request has no in-flight load: not in _active_loads.
        session.register_lookup("req-probe", b"hp")
        assert client._active_loads == set()
        assert client.has_active_loads is False

        # Issuing a fetch arms the work-list.
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        round_seq = conn._sent[-1][FetchMsg.ROUND_SEQ]
        assert client._active_loads == {"req-1"}
        assert client.has_active_loads is True

        # Completion clears the load and discards it from the work-list.
        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.ROUND_SEQ: round_seq,
                TransferDoneMsg.KV_REQUEST_ID: "req-1",
                TransferDoneMsg.SUCCESS: True,
            }
        )
        session.poll()
        assert client._active_loads == set()
        assert client.has_active_loads is False

    def test_load_timeout_sends_abort(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"k"], block_ids=[0]
        )
        _client_load(session, "req-1").submitted_at = time.monotonic() - 60.0
        session.poll()
        abort = conn._sent[-1]
        assert abort[TYPE_KEY] == AbortFetchMsg.TYPE

    def test_load_abort_ack_timeout_retries_without_publishing_failure(self):
        """An AbortAck timeout cannot authorize destination reuse."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=7, kv_request_id="req-7", keys=[b"k"], block_ids=[0]
        )
        # 1) Trip the load timeout to send AbortFetch and stamp aborted_at.
        _client_load(session, "req-7").submitted_at = (
            time.monotonic() - _LOAD_TIMEOUT_S - 1.0
        )
        loads = session.poll().loads
        assert loads == []
        assert any(
            m.get(TYPE_KEY) == AbortFetchMsg.TYPE
            and m[AbortFetchMsg.KV_REQUEST_ID] == "req-7"
            for m in conn._sent
        )
        assert _client_load(session, "req-7").aborted_at is not None

        # 2) Now backdate aborted_at past the abort-ack timeout. No ack ever
        # arrived from the peer.
        _client_load(session, "req-7").aborted_at = (
            time.monotonic() - _ABORT_ACK_TIMEOUT_S - 1.0
        )
        session._client._abort_intents[("req-7", 0)].last_sent_at = (
            time.monotonic() - _ABORT_ACK_TIMEOUT_S - 1.0
        )
        abort_count = sum(m[TYPE_KEY] == AbortFetchMsg.TYPE for m in conn._sent)
        loads = session.poll().loads
        assert loads == []
        assert "req-7" in session._client._requests
        assert session._client.has_active_loads is True
        assert sum(m[TYPE_KEY] == AbortFetchMsg.TYPE for m in conn._sent) == (
            abort_count + 1
        )

    def test_load_abort_ack_clears_request(self):
        """After load timeout sends AbortFetch, an arriving AbortAckMsg from
        the peer surfaces the failure cleanly and removes the request from
        _requests — covers the on_abort_ack arrival path."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(
            job_id=8, kv_request_id="req-8", keys=[b"k"], block_ids=[0]
        )
        _client_load(session, "req-8").submitted_at = (
            time.monotonic() - _LOAD_TIMEOUT_S - 1.0
        )
        # First poll: AbortFetch goes out.
        session.poll()
        assert _client_load(session, "req-8").aborted_at is not None

        # Peer acks the abort.
        conn.enqueue(
            {
                TYPE_KEY: AbortAckMsg.TYPE,
                AbortAckMsg.ROUND_SEQ: 0,
                AbortAckMsg.KV_REQUEST_ID: "req-8",
            }
        )
        loads = session.poll().loads
        assert loads == [LoadResult(job_id=8, kv_request_id="req-8", success=False)]
        assert "req-8" not in session._client._requests


# ---------------------------------------------------------------------------
# Symmetric-P2P lookup flow (do_p2p_fetch)
# ---------------------------------------------------------------------------


class TestLookupFlow:
    """Consumer-side state machine for do_p2p_fetch lookups."""

    def test_aggregate_flush_resolve_round_trip(self):
        """register_lookup → flush sends one LookupMsg → response
        resolves entries → register_lookup returns the cached bool on
        every call, and repeat probes never re-issue a LookupMsg."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        # Aggregate two keys for the same kv_request_id; both return None.
        assert session.register_lookup("req-1", b"hA") is None
        assert session.register_lookup("req-1", b"hB") is None

        # Flush sends one LookupMsg with both keys.
        sent_before = len(conn._sent)
        session.flush_pending_lookups()
        new = conn._sent[sent_before:]
        assert len(new) == 1
        msg = new[0]
        assert msg[TYPE_KEY] == LookupMsg.TYPE
        assert msg[LookupMsg.KV_REQUEST_ID] == "req-1"
        assert sorted(msg[LookupMsg.KEYS]) == [b"hA", b"hB"]

        # Idempotent re-flush sends nothing — the entries are now in-flight.
        sent_before = len(conn._sent)
        session.flush_pending_lookups()
        assert conn._sent[sent_before:] == []

        # While in-flight, register_lookup keeps returning None.
        assert session.register_lookup("req-1", b"hA") is None

        # Peer answers: hA hit, hB miss.
        conn.enqueue(
            {
                TYPE_KEY: LookupRespMsg.TYPE,
                LookupRespMsg.ROUND_SEQ: 0,
                LookupRespMsg.KV_REQUEST_ID: "req-1",
                LookupRespMsg.KEYS: [b"hA", b"hB"],
                LookupRespMsg.HITS: [True, False],
            }
        )
        session.poll()

        # register_lookup returns the resolved bool.
        assert session.register_lookup("req-1", b"hA") is True
        assert session.register_lookup("req-1", b"hB") is False
        # The entry is cached, not popped: repeat probes keep returning the
        # same result and never re-queue the key, so a flush sends nothing.
        assert session.register_lookup("req-1", b"hA") is True
        assert session.register_lookup("req-1", b"hB") is False
        sent_before = len(conn._sent)
        session.flush_pending_lookups()
        assert [
            m for m in conn._sent[sent_before:] if m[TYPE_KEY] == LookupMsg.TYPE
        ] == []

    def test_request_blocks_clears_probe_cache(self):
        """A resolved HIT probe is popped when its fetch is issued, so a
        re-scheduled request re-probes instead of trusting the stale True
        (the served block is unpinned and may have been evicted)."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        # Probe hA, flush, and let the peer resolve it to a HIT.
        assert session.register_lookup("req-1", b"hA") is None
        session.flush_pending_lookups()
        conn.enqueue(
            {
                TYPE_KEY: LookupRespMsg.TYPE,
                LookupRespMsg.ROUND_SEQ: 0,
                LookupRespMsg.KV_REQUEST_ID: "req-1",
                LookupRespMsg.KEYS: [b"hA"],
                LookupRespMsg.HITS: [True],
            }
        )
        session.poll()
        assert session.register_lookup("req-1", b"hA") is True

        # Fetch consumes the probe.
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"hA"], block_ids=[0]
        )
        assert b"hA" not in session._client._requests["req-1"].probes

        # Re-scheduled probe of the same key is treated as brand-new: it
        # returns None and re-queues, so a flush emits a fresh LookupMsg.
        assert session.register_lookup("req-1", b"hA") is None
        sent_before = len(conn._sent)
        session.flush_pending_lookups()
        fresh = [m for m in conn._sent[sent_before:] if m[TYPE_KEY] == LookupMsg.TYPE]
        assert len(fresh) == 1
        assert fresh[0][LookupMsg.KEYS] == [b"hA"]

    def test_flush_uses_work_list_not_full_scan(self):
        """flush drains a work-list rather than scanning every live request.

        A request with no newly-registered keys is not revisited: after a
        flush the work-list is empty, an idle re-flush sends nothing, and a
        subsequent register re-arms exactly the one affected id — even while
        an unrelated request stays live in ``_requests``.
        """
        session, conn, _ = _make_session()
        _activate(session, conn)
        client = session._client

        # Two requests register keys; both are queued for flush.
        session.register_lookup("req-A", b"hA")
        session.register_lookup("req-B", b"hB")
        assert client._flush_pending == {"req-A", "req-B"}

        # Flush drains the work-list even though both requests stay live.
        session.flush_pending_lookups()
        assert client._flush_pending == set()
        assert set(client._requests) == {"req-A", "req-B"}

        # An idle re-flush visits nothing and sends no LookupMsg.
        sent_before = len(conn._sent)
        session.flush_pending_lookups()
        assert [
            m for m in conn._sent[sent_before:] if m[TYPE_KEY] == LookupMsg.TYPE
        ] == []

        # A new register re-arms only that id.
        session.register_lookup("req-A", b"hA2")
        assert client._flush_pending == {"req-A"}

    def test_separate_lookup_msg_per_kv_request_id(self):
        """Hashes for different kv_request_ids flush as separate LookupMsgs."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        session.register_lookup("req-A", b"h1")
        session.register_lookup("req-B", b"h2")
        session.register_lookup("req-A", b"h3")

        sent_before = len(conn._sent)
        session.flush_pending_lookups()
        sent = [m for m in conn._sent[sent_before:] if m[TYPE_KEY] == LookupMsg.TYPE]
        assert len(sent) == 2
        by_req = {m[LookupMsg.KV_REQUEST_ID]: m[LookupMsg.KEYS] for m in sent}
        assert sorted(by_req["req-A"]) == [b"h1", b"h3"]
        assert by_req["req-B"] == [b"h2"]

    def test_multiple_lookup_msgs_across_steps(self):
        """A request's block set may be discovered across scheduler steps:
        each step that registers new keys flushes its own LookupMsg for
        the same kv_request_id, carrying only the newly-probed keys."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        # Step 1: probe hA, hB.
        session.register_lookup("req-1", b"hA")
        session.register_lookup("req-1", b"hB")
        sent_before = len(conn._sent)
        session.flush_pending_lookups()
        first = [m for m in conn._sent[sent_before:] if m[TYPE_KEY] == LookupMsg.TYPE]
        assert len(first) == 1
        assert first[0][LookupMsg.KV_REQUEST_ID] == "req-1"
        assert sorted(first[0][LookupMsg.KEYS]) == [b"hA", b"hB"]

        # Step 2: a new key is discovered for the same request. The
        # in-flight keys from step 1 are not re-sent; a second LookupMsg
        # goes out carrying only the newly-probed key.
        assert session.register_lookup("req-1", b"hA") is None  # in-flight no-op
        session.register_lookup("req-1", b"hC")
        sent_before = len(conn._sent)
        session.flush_pending_lookups()
        second = [m for m in conn._sent[sent_before:] if m[TYPE_KEY] == LookupMsg.TYPE]
        assert len(second) == 1
        assert second[0][LookupMsg.KV_REQUEST_ID] == "req-1"
        assert second[0][LookupMsg.KEYS] == [b"hC"]

    def test_terminal_tokens_survive_prune_and_request_id_reuse(self):
        session, conn, _ = _make_session()
        _activate(session, conn)

        round_tokens = []
        for key in (b"old", b"new"):
            session.register_lookup("req-reused", key)
            session.flush_pending_lookups()
            lookup = next(
                msg for msg in reversed(conn._sent) if msg[TYPE_KEY] == LookupMsg.TYPE
            )
            session.finish_request("req-reused")
            terminal = conn._sent[-1]
            assert terminal[TYPE_KEY] == FetchMsg.TYPE
            assert terminal[FetchMsg.KEYS] == []
            assert terminal[FetchMsg.ROUND_SEQ] == lookup[LookupMsg.ROUND_SEQ]
            round_tokens.append(terminal[FetchMsg.ROUND_SEQ])
            assert "req-reused" not in session._client._requests

        assert round_tokens == [0, 1]

    def test_delayed_old_lookup_hit_cannot_resolve_reused_request_key(self):
        session, conn, _ = _make_session()
        _activate(session, conn)

        session.register_lookup("req-reused", b"same-key")
        session.flush_pending_lookups()
        old_token = conn._sent[-1][LookupMsg.ROUND_SEQ]
        session.finish_request("req-reused")
        assert "req-reused" not in session._client._requests

        assert session.register_lookup("req-reused", b"same-key") is None
        session.flush_pending_lookups()
        new_token = conn._sent[-1][LookupMsg.ROUND_SEQ]
        assert new_token > old_token

        conn.enqueue(
            {
                TYPE_KEY: LookupRespMsg.TYPE,
                LookupRespMsg.ROUND_SEQ: old_token,
                LookupRespMsg.KV_REQUEST_ID: "req-reused",
                LookupRespMsg.KEYS: [b"same-key"],
                LookupRespMsg.HITS: [True],
            }
        )
        session.poll()
        assert session.register_lookup("req-reused", b"same-key") is None

        conn.enqueue(
            {
                TYPE_KEY: LookupRespMsg.TYPE,
                LookupRespMsg.ROUND_SEQ: new_token,
                LookupRespMsg.KV_REQUEST_ID: "req-reused",
                LookupRespMsg.KEYS: [b"same-key"],
                LookupRespMsg.HITS: [False],
            }
        )
        session.poll()
        assert session.register_lookup("req-reused", b"same-key") is False

    def test_split_response_resolves_across_messages(self):
        """Producer may answer one LookupMsg's keys across multiple
        LookupRespMsgs — pairs are self-describing so each lands."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        session.register_lookup("req-1", b"hA")
        session.register_lookup("req-1", b"hB")
        session.flush_pending_lookups()

        # Two responses, each carrying one of the two keys.
        conn.enqueue(
            {
                TYPE_KEY: LookupRespMsg.TYPE,
                LookupRespMsg.ROUND_SEQ: 0,
                LookupRespMsg.KV_REQUEST_ID: "req-1",
                LookupRespMsg.KEYS: [b"hA"],
                LookupRespMsg.HITS: [True],
            }
        )
        conn.enqueue(
            {
                TYPE_KEY: LookupRespMsg.TYPE,
                LookupRespMsg.ROUND_SEQ: 0,
                LookupRespMsg.KV_REQUEST_ID: "req-1",
                LookupRespMsg.KEYS: [b"hB"],
                LookupRespMsg.HITS: [False],
            }
        )
        session.poll()

        assert session.register_lookup("req-1", b"hA") is True
        assert session.register_lookup("req-1", b"hB") is False

    def test_finish_request_cancels_pending_lookups(self):
        """finish_request drops every pending lookup for the kv_request_id."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        session.register_lookup("req-1", b"hA")
        session.register_lookup("req-1", b"hB")
        session.register_lookup("req-2", b"hC")
        session.finish_request("req-1")

        # req-1 entries gone, req-2 untouched.
        assert "req-1" not in session._client._requests
        assert b"hC" in session._client._requests["req-2"].probes

    def test_finish_after_flushed_lookup_sends_empty_fetch(self):
        """LookupMsg flushed but no FetchMsg sent (all-miss case) →
        finish_request emits an empty FetchMsg so the peer can drop its
        lookup state and call parent.on_request_finished."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        session.register_lookup("req-1", b"hA")
        session.flush_pending_lookups()
        sent_before = len(conn._sent)

        session.finish_request("req-1")

        fetches = [m for m in conn._sent[sent_before:] if m[TYPE_KEY] == FetchMsg.TYPE]
        assert len(fetches) == 1
        assert fetches[0][FetchMsg.KV_REQUEST_ID] == "req-1"
        assert fetches[0][FetchMsg.KEYS] == []
        assert fetches[0][FetchMsg.BLOCK_INDEXES] == []

    def test_finish_without_flushed_lookup_sends_no_fetch(self):
        """No LookupMsg was ever sent → finish_request must not emit an
        empty FetchMsg (the peer has no state to release)."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        # Register but never flush.
        session.register_lookup("req-1", b"hA")
        sent_before = len(conn._sent)

        session.finish_request("req-1")

        fetches = [m for m in conn._sent[sent_before:] if m[TYPE_KEY] == FetchMsg.TYPE]
        assert fetches == []

    def test_finish_after_real_fetch_sends_no_second_fetch(self):
        """A real FetchMsg was already sent for the id → finish_request
        must not emit a second (empty) FetchMsg."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        session.register_lookup("req-1", b"hA")
        session.flush_pending_lookups()
        # Resolve the probe to a HIT before fetching, as the manager only
        # loads confirmed hits (an unresolved probe yields RETRY).
        conn.enqueue(
            {
                TYPE_KEY: LookupRespMsg.TYPE,
                LookupRespMsg.ROUND_SEQ: 0,
                LookupRespMsg.KV_REQUEST_ID: "req-1",
                LookupRespMsg.KEYS: [b"hA"],
                LookupRespMsg.HITS: [True],
            }
        )
        session.poll()
        session.request_blocks(
            job_id=1, kv_request_id="req-1", keys=[b"hA"], block_ids=[7]
        )
        sent_before = len(conn._sent)

        session.finish_request("req-1")

        fetches = [m for m in conn._sent[sent_before:] if m[TYPE_KEY] == FetchMsg.TYPE]
        assert fetches == []

    def test_server_lookup_deferred_until_serve_then_all_misses(self):
        """``poll()`` only enqueues an inbound LookupMsg — no response is
        sent until ``serve_external_requests``. With an all-miss parent
        the aggregated LookupRespMsg carries the same keys and
        ``hits=[False, ...]``."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        sent_before = len(conn._sent)
        conn.enqueue(
            {
                TYPE_KEY: LookupMsg.TYPE,
                LookupMsg.ROUND_SEQ: 0,
                LookupMsg.KV_REQUEST_ID: "req-1",
                LookupMsg.KEYS: [b"hX", b"hY", b"hZ"],
            }
        )
        session.poll()

        # Dispatch alone must not answer — the parent handle is only valid
        # during serve_external_requests.
        assert [
            m for m in conn._sent[sent_before:] if m[TYPE_KEY] == LookupRespMsg.TYPE
        ] == []

        _serve(session, FakeParent())

        resps = [
            m for m in conn._sent[sent_before:] if m[TYPE_KEY] == LookupRespMsg.TYPE
        ]
        assert len(resps) == 1
        resp = resps[0]
        assert resp[LookupRespMsg.KV_REQUEST_ID] == "req-1"
        assert resp[LookupRespMsg.ROUND_SEQ] == 0
        assert resp[LookupRespMsg.KEYS] == [b"hX", b"hY", b"hZ"]
        assert resp[LookupRespMsg.HITS] == [False, False, False]


# ---------------------------------------------------------------------------
# Server-side handling of inbound LookupMsg (ParentManager-driven)
#
# poll() only enqueues the LookupMsg; serve_external_requests(parent)
# resolves it. Tests follow the poll() → _serve() pattern.
# ---------------------------------------------------------------------------


def _send_lookup(conn: FakeConnection, kv_request_id: str, keys: list[bytes]):
    conn.enqueue(
        {
            TYPE_KEY: LookupMsg.TYPE,
            LookupMsg.ROUND_SEQ: 0,
            LookupMsg.KV_REQUEST_ID: kv_request_id,
            LookupMsg.KEYS: list(keys),
        }
    )


def _lookup_resps(conn: FakeConnection, since: int = 0) -> list[dict]:
    return [m for m in conn._sent[since:] if m[TYPE_KEY] == LookupRespMsg.TYPE]


class TestServerLookupHandling:
    def test_immediate_hits_create_one_store_job(self):
        """All-HIT batch: one create_store_job call with all keys, one
        LookupRespMsg with hits=[True]*N, on_request_finished fires at the
        end of serve, and `available` is populated for the eventual fetch."""
        cb = FakeParent(stored={b"hA": 1, b"hB": 2, b"hC": 3})
        session, conn, _ = _make_session()
        _activate(session, conn)

        sent_before = len(conn._sent)
        _send_lookup(conn, "req-1", [b"hA", b"hB", b"hC"])
        session.poll()
        _serve(session, cb)

        resps = _lookup_resps(conn, sent_before)
        assert len(resps) == 1
        assert resps[0][LookupRespMsg.KEYS] == [b"hA", b"hB", b"hC"]
        assert resps[0][LookupRespMsg.HITS] == [True, True, True]

        kinds = [c[0] for c in cb.calls]
        assert kinds.count("create_store_job") == 1
        cs = next(c for c in cb.calls if c[0] == "create_store_job")
        assert cs[1] == (b"hA", b"hB", b"hC")
        assert cb.calls[-1][0] == "on_request_finished"

        # Hits are pinned in outbound state for the upcoming FetchMsg match.
        assert set(_srv_outbound(session, "req-1").available) == {
            b"hA",
            b"hB",
            b"hC",
        }

    def test_all_misses_no_store_job_finish_fires(self):
        """All-MISS batch: no create_store_job call; one LookupRespMsg
        with hits=[False]*N; on_request_finished fires at end of serve."""
        cb = FakeParent()
        session, conn, _ = _make_session()
        _activate(session, conn)

        sent_before = len(conn._sent)
        _send_lookup(conn, "req-1", [b"hA", b"hB"])
        session.poll()
        _serve(session, cb)

        resps = _lookup_resps(conn, sent_before)
        assert len(resps) == 1
        assert resps[0][LookupRespMsg.HITS] == [False, False]
        assert all(c[0] != "create_store_job" for c in cb.calls)
        assert cb.calls[-1][0] == "on_request_finished"

    def test_mixed_hit_miss_pending_defers_response_until_aggregate(self):
        """HIT/MISS resolutions do not go out on first sight when any
        key is still HIT_PENDING / RETRY. The lookup parks until every
        key has settled (or the deadline fires), then one
        LookupRespMsg carries all keys in wire order."""
        cb = FakeParent(
            stored={b"hA": 1},
            pending={b"hB"},
            retry={b"hD"},
        )
        session, conn, _ = _make_session()
        _activate(session, conn)

        sent_before = len(conn._sent)
        _send_lookup(conn, "req-1", [b"hA", b"hB", b"hC", b"hD"])
        session.poll()
        _serve(session, cb)

        # No LookupRespMsg yet — hB and hD are still pending.
        assert _lookup_resps(conn, sent_before) == []
        # HIT is still pinned immediately so the eventual FetchMsg matches.
        cs_calls = [c for c in cb.calls if c[0] == "create_store_job"]
        assert len(cs_calls) == 1
        assert cs_calls[0][1] == (b"hA",)
        # Lookup is parked; on_request_finished not yet called.
        assert all(c[0] != "on_request_finished" for c in cb.calls)
        assert len(_srv_lookups(session)) == 1

    def test_pending_resolves_then_aggregate_response_fires(self):
        """A HIT_PENDING key that becomes HIT on a later poll releases
        the deferred aggregate response: one LookupRespMsg carrying
        both keys in wire order, and one create_store_job call per
        HIT (the second HIT is pinned when it resolves, not when the
        response goes out)."""
        cb = FakeParent(stored={b"hA": 1}, pending={b"hB"})
        session, conn, _ = _make_session()
        _activate(session, conn)

        sent_before = len(conn._sent)
        _send_lookup(conn, "req-1", [b"hA", b"hB"])
        session.poll()
        _serve(session, cb)
        # No response yet — hB still pending.
        assert _lookup_resps(conn, sent_before) == []

        # Promote hB.
        cb.pending.discard(b"hB")
        cb.stored[b"hB"] = 2

        # Drive resolver via a second serve_external_requests.
        _serve(session, cb)

        resps = _lookup_resps(conn, sent_before)
        assert len(resps) == 1
        assert resps[0][LookupRespMsg.KEYS] == [b"hA", b"hB"]
        assert resps[0][LookupRespMsg.HITS] == [True, True]

        cs_calls = [c for c in cb.calls if c[0] == "create_store_job"]
        assert len(cs_calls) == 2
        assert cs_calls[0][1] == (b"hA",)
        assert cs_calls[1][1] == (b"hB",)
        # on_request_finished fires once after the aggregate resolve.
        assert sum(1 for c in cb.calls if c[0] == "on_request_finished") == 1
        assert b"hA" in _srv_outbound(session, "req-1").available
        assert b"hB" in _srv_outbound(session, "req-1").available

    def test_pending_timeout_replies_miss_no_store_job(self):
        """A HIT_PENDING key that stays pending past the batch
        ``deadline`` is force-MISS and never pinned; the deferred
        aggregate response fires with hits=[False]."""
        cb = FakeParent(pending={b"hA"})
        session, conn, _ = _make_session()
        _activate(session, conn)

        sent_before = len(conn._sent)
        _send_lookup(conn, "req-1", [b"hA"])
        session.poll()
        _serve(session, cb)
        # Initial serve: nothing immediate, lookup parked, no LookupRespMsg.
        assert _lookup_resps(conn, sent_before) == []

        # Forge the deadline into the past to trigger the timeout branch.
        lookup = _srv_lookups(session)[0]
        lookup.deadline = time.monotonic() - 0.1

        _serve(session, cb)

        resps = _lookup_resps(conn, sent_before)
        assert len(resps) == 1
        assert resps[0][LookupRespMsg.KEYS] == [b"hA"]
        assert resps[0][LookupRespMsg.HITS] == [False]
        assert all(c[0] != "create_store_job" for c in cb.calls)
        assert sum(1 for c in cb.calls if c[0] == "on_request_finished") == 1

    def test_finish_request_called_per_lookup_msg_not_per_kv_request_id(self):
        """Two LookupMsgs for the same kv_request_id get distinct ctxs
        and two on_request_finished calls (one per batch)."""
        cb = FakeParent(stored={b"hA": 1, b"hB": 2})
        session, conn, _ = _make_session()
        _activate(session, conn)

        _send_lookup(conn, "req-1", [b"hA"])
        session.poll()
        _serve(session, cb)
        _send_lookup(conn, "req-1", [b"hB"])
        session.poll()
        _serve(session, cb)

        finish_calls = [c for c in cb.calls if c[0] == "on_request_finished"]
        assert len(finish_calls) == 2
        # Distinct synthetic req_ids
        assert finish_calls[0][1] != finish_calls[1][1]
        # Both namespaced under the same kv_request_id
        assert ":req-1:" in finish_calls[0][1]
        assert ":req-1:" in finish_calls[1][1]

    def test_close_returns_open_batch_ctxs_as_failed_serves(self):
        """Tearing the session down with a parked batch returns the
        synthetic ctx as a failed serve (no parent handle at teardown) so
        the manager can release the TieringManager's state on its next
        serve."""
        cb = FakeParent(pending={b"hA"})
        session, conn, _ = _make_session()
        _activate(session, conn)

        _send_lookup(conn, "req-1", [b"hA"])
        session.poll()
        _serve(session, cb)
        assert len(_srv_lookups(session)) == 1
        assert all(c[0] != "on_request_finished" for c in cb.calls)

        result = session.close()

        assert len(result.failed_serves) == 1
        assert ":req-1:" in result.failed_serves[0].req_id
        # close() itself must not call the parent.
        assert all(c[0] != "on_request_finished" for c in cb.calls)

    def test_wire_finish_drops_pending_batches_for_kv_request_id(self):
        """``ServerRole.finish(kv_request_id)`` drops every parked batch
        whose kv_request_id matches and queues its ctx for the next
        serve's on_request_finished."""
        cb = FakeParent(pending={b"hA", b"hB"})
        session, conn, _ = _make_session()
        _activate(session, conn)

        _send_lookup(conn, "req-1", [b"hA"])
        session.poll()
        _serve(session, cb)
        _send_lookup(conn, "req-2", [b"hB"])
        session.poll()
        _serve(session, cb)
        assert len(_srv_lookups(session)) == 2

        session._server.finish("req-1")

        # req-1 batch dropped from parked lookups; its ctx queued for release.
        remaining_kv_request_ids = {b.kv_request_id for b in _srv_lookups(session)}
        assert remaining_kv_request_ids == {"req-2"}
        queued = session._server._finished_lookup_ctxs
        assert len(queued) == 1
        assert ":req-1:" in queued[0].req_id

        # The next serve fires on_request_finished exactly once for req-1.
        _serve(session, cb)
        finish_calls = [c for c in cb.calls if c[0] == "on_request_finished"]
        assert len(finish_calls) == 1
        assert ":req-1:" in finish_calls[0][1]

    def test_incoming_fetch_drops_pending_lookups_for_kv_request_id(self):
        """A peer FetchMsg terminates the lookup phase for its id: parked
        lookups with matching kv_request_id are dropped and their
        ctx queued for on_request_finished; other kv_request_ids untouched."""
        cb = FakeParent(pending={b"hA", b"hB"})
        session, conn, _ = _make_session()
        _activate(session, conn)

        _send_lookup(conn, "req-1", [b"hA"])
        session.poll()
        _serve(session, cb)
        _send_lookup(conn, "req-2", [b"hB"])
        session.poll()
        _serve(session, cb)
        assert len(_srv_lookups(session)) == 2

        # Empty FetchMsg: peer signals "lookup phase done" without asking
        # for any blocks (the all-miss case).
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [],
                FetchMsg.BLOCK_INDEXES: [],
            }
        )
        session.poll()

        remaining_kv_request_ids = {lu.kv_request_id for lu in _srv_lookups(session)}
        assert remaining_kv_request_ids == {"req-2"}
        # Dispatch queues the ctx but does not call the parent yet.
        queued = session._server._finished_lookup_ctxs
        assert len(queued) == 1
        assert ":req-1:" in queued[0].req_id

        # The next serve fires on_request_finished exactly once for req-1.
        _serve(session, cb)
        finish_calls = [c for c in cb.calls if c[0] == "on_request_finished"]
        assert len(finish_calls) == 1
        assert ":req-1:" in finish_calls[0][1]

    def test_lookup_then_fetch_round_trip_emits_store_result(self):
        """End-to-end: lookup pins primary slots → fetch matches them →
        NIXL transfer completes → StoreResult surfaces with the
        create_store_job's job_id (the engine releases the pin)."""
        cb = FakeParent(stored={b"hA": 7, b"hB": 8})
        session, conn, transport = _make_session()
        _activate(session, conn)

        _send_lookup(conn, "req-1", [b"hA", b"hB"])
        session.poll()
        _serve(session, cb)
        cs = next(c for c in cb.calls if c[0] == "create_store_job")
        # FakeParent issues monotonic job_ids starting at 1000.
        expected_job_id = 1000

        # Consumer issues FetchMsg on the resolved hits.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"hA", b"hB"],
                FetchMsg.BLOCK_INDEXES: [14, 15],
            }
        )
        session.poll()

        # NIXL write_blocks called with our pinned local block_ids.
        assert len(transport._transfers) == 1
        _, (_peer, local, remote) = next(iter(transport._transfers.items()))
        assert local == [7, 8]
        assert remote == [14, 15]

        # Drive the transport completion.
        transport._poll_done.append(0)
        result = session.poll()

        store_results = [s for s in result.stores if s.success]
        assert any(s.job_id == expected_job_id for s in store_results)
        # Sanity: kv mention in synthetic ctx.
        assert cs[2].startswith("p2p:")


# ---------------------------------------------------------------------------
# Server-role flows
# ---------------------------------------------------------------------------


class TestServerFlows:
    def test_owned_submit_success_uses_exact_guard_without_recovery_scan(self):
        transport = _RecoveringDataTransport()
        session, conn, _ = _make_session(transport=transport)
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-owned",
                FetchMsg.KEYS: [b"k"],
                FetchMsg.BLOCK_INDEXES: [9],
            }
        )
        session.poll()

        session.add_stored_blocks("req-owned", [b"k"], [3], job_id=70)

        assert session._server._submitting_xfer is None
        xfer = session._server._inflight[0]
        assert transport.submitted_tokens == [xfer]
        assert transport.recovery_calls == []
        assert session._server._requests["req-owned"].inflight_tids == {0}
        assert xfer.round.inflight == 1

    def test_none_recovery_proves_prepublication_and_clears_guard(self):
        transport = _RecoveringDataTransport()
        failure = _SubmissionFailure("before provider publication")

        def interrupt_before_publication(
            peer_id, local_idxs, remote_idxs, *, recovery_token
        ):
            del peer_id, local_idxs, remote_idxs
            transport.submitted_tokens.append(recovery_token)
            raise failure

        transport.write_blocks_owned = interrupt_before_publication  # type: ignore[method-assign]
        session, conn, _ = _make_session(transport=transport)
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-prepublish",
                FetchMsg.KEYS: [b"k"],
                FetchMsg.BLOCK_INDEXES: [9],
            }
        )
        session.poll()

        with pytest.raises(_SubmissionFailure) as raised:
            session.add_stored_blocks("req-prepublish", [b"k"], [3], job_id=69)
        assert raised.value is failure
        assert session._server._submitting_xfer is None
        assert session._server._inflight == {}
        assert len(transport.recovery_calls) == 1
        assert transport.recovery_calls[0][1] is transport.submitted_tokens[0]
        assert session.poll().stores == [StoreResult(job_id=69, success=False)]

    def test_store_then_fetch_matches(self):
        """Blocks stored before fetch demand are matched on demand arrival."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1", b"k2"], [0, 1], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [10, 11],
            }
        )
        session.poll()
        assert len(transport._transfers) == 1
        _, (peer, local, remote) = next(iter(transport._transfers.items()))
        assert local == [0, 1]
        assert remote == [10, 11]

    def test_pd_store_then_fetch_binds_session_global_token(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-global", [b"k"], [4], job_id=17)

        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 91,
                FetchMsg.KV_REQUEST_ID: "req-global",
                FetchMsg.KEYS: [b"k"],
                FetchMsg.BLOCK_INDEXES: [12],
            }
        )
        session.poll()

        assert len(transport._transfers) == 1
        xfer = next(iter(session._server._inflight.values()))
        assert xfer.round_key == 91
        assert -1 not in session._server._requests["req-global"].outbound

    def test_fetch_then_store_matches(self):
        """Fetch demand registered before store; store fulfills it."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        assert len(transport._transfers) == 0
        session.add_stored_blocks("req-1", [b"k1"], [3], job_id=1)
        assert len(transport._transfers) == 1

    def test_transfer_completion_emits_store_result_and_done(self):
        """Completed transfer reports StoreResult and sends TransferDoneMsg."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        stores = session.poll().stores
        assert StoreResult(job_id=1, success=True) in stores
        assert any(m[TYPE_KEY] == TransferDoneMsg.TYPE for m in conn._sent)

    def test_abort_fetch_replies_with_ack(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.ROUND_SEQ: 0,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        ack = next(m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE)
        assert ack[AbortAckMsg.KV_REQUEST_ID] == "req-1"
        assert _srv_abort_started(session, "req-1") is None

    def test_abort_fetch_defers_ack_when_cancel_pending(self):
        """If cancel(mode='wait') reports still-inflight tids, the ack is
        deferred and the abort is parked (abort_started_at set)."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        # Seed an inflight transfer for req-1 that the transport pretends
        # cannot be canceled yet.
        tid = 42
        session._server._inflight_add(
            tid,
            _InflightXfer(
                kv_request_id="req-1",
                block_count=1,
                job_ids={1},
                round=_OutboundRequestState(inflight=1),
            ),
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.ROUND_SEQ: 0,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()

        assert not any(m[TYPE_KEY] == AbortAckMsg.TYPE for m in conn._sent)
        assert _srv_abort_started(session, "req-1") is not None
        # First attempt happens inside _on_abort_fetch; the per-tick
        # drain runs again at the end of poll() — both are wait-mode.
        assert all(mode == "wait" for _, mode in transport._cancel_calls)
        assert tid in session._server._inflight  # still tracked

    def test_abort_fetch_acks_after_drain(self):
        """Once the transport reports the tid as DONE the parked abort
        completes and AbortAckMsg is sent."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        tid = 42
        session._server._inflight_add(
            tid,
            _InflightXfer(
                kv_request_id="req-1",
                block_count=1,
                job_ids={1},
                round=_OutboundRequestState(inflight=1),
            ),
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.ROUND_SEQ: 0,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        assert _srv_abort_started(session, "req-1") is not None

        # Backend finishes draining: transport.poll() will return tid as
        # DONE, and the next cancel(mode='wait') call sees it's gone.
        transport._cancel_still_inflight.discard(tid)
        transport._poll_done.append(tid)

        session.poll()

        ack = next(m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE)
        assert ack[AbortAckMsg.KV_REQUEST_ID] == "req-1"
        assert _srv_abort_started(session, "req-1") is None
        assert tid not in session._server._inflight

    def test_abort_fetch_timeout_retains_pin_and_defers_ack(self):
        """A drain deadline is diagnostic, never permission to release."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        tid = 42
        session._server._inflight_add(
            tid,
            _InflightXfer(
                kv_request_id="req-1",
                block_count=1,
                job_ids={1},
                round=_OutboundRequestState(inflight=1),
            ),
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.ROUND_SEQ: 0,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        assert _srv_abort_started(session, "req-1") is not None
        # Backdate past the drain deadline.
        session._server._pending_aborts[("req-1", 0)] = (
            time.monotonic() - _CANCEL_DRAIN_TIMEOUT_S - 1.0
        )
        # Even after the warning deadline, retain ownership and keep waiting.
        transport._cancel_calls.clear()

        session.poll()

        assert not any(m[TYPE_KEY] == AbortAckMsg.TYPE for m in conn._sent)
        assert _srv_abort_started(session, "req-1") is not None
        assert tid in session._server._inflight
        assert transport._cancel_calls
        assert all(mode == "wait" for _, mode in transport._cancel_calls)

        # Only proven quiescence permits the owner to be released and acked.
        transport._cancel_still_inflight.remove(tid)
        session.poll()
        ack = next(m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE)
        assert ack[AbortAckMsg.KV_REQUEST_ID] == "req-1"
        assert _srv_abort_started(session, "req-1") is None
        assert tid not in session._server._inflight

    def test_abort_fetch_idempotent_while_draining(self):
        """Receiving AbortFetchMsg twice for the same kv_request_id
        keeps a single pending entry and produces a single ack."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        tid = 42
        session._server._inflight_add(
            tid,
            _InflightXfer(
                kv_request_id="req-1",
                block_count=1,
                job_ids={1},
                round=_OutboundRequestState(inflight=1),
            ),
        )
        transport._cancel_still_inflight.add(tid)

        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.ROUND_SEQ: 0,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        first_started_at = _srv_abort_started(session, "req-1")

        # Second AbortFetchMsg for the same kv_request_id while still
        # draining must not reset the deadline.
        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.ROUND_SEQ: 0,
                AbortFetchMsg.KV_REQUEST_ID: "req-1",
            }
        )
        session.poll()
        assert _srv_abort_started(session, "req-1") == first_started_at

        # Now let the drain succeed and confirm exactly one ack ever.
        transport._cancel_still_inflight.discard(tid)
        transport._poll_done.append(tid)
        session.poll()

        acks = [m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE]
        assert len(acks) == 1
        assert acks[0][AbortAckMsg.KV_REQUEST_ID] == "req-1"

    def test_abort_ack_lost_return_retains_intent_and_replays(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        original_send = conn.send
        lost_returns = 1

        def send_then_interrupt(msg):
            nonlocal lost_returns
            original_send(msg)
            if msg[TYPE_KEY] == AbortAckMsg.TYPE and lost_returns:
                lost_returns -= 1
                raise _CloseFailure("abort ack lost return")

        conn.send = send_then_interrupt  # type: ignore[method-assign]
        conn.enqueue(
            {
                TYPE_KEY: AbortFetchMsg.TYPE,
                AbortFetchMsg.ROUND_SEQ: 0,
                AbortFetchMsg.KV_REQUEST_ID: "req-ack-cut",
            }
        )
        with pytest.raises(_CloseFailure, match="abort ack lost return"):
            session.poll()

        key = ("req-ack-cut", 0)
        assert key in session._server._abort_ack_intents
        assert key in session._server._pending_aborts
        session._server.drain_pending_aborts()
        assert key not in session._server._pending_aborts
        acks = [m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE]
        assert [m[AbortAckMsg.ROUND_SEQ] for m in acks] == [0, 0]

        session._server.on_abort_fetch("req-ack-cut", 0)
        acks = [m for m in conn._sent if m[TYPE_KEY] == AbortAckMsg.TYPE]
        assert len(acks) == 3

    def test_store_timeout(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        # Backdate.
        session._server._store_jobs[1] = time.monotonic() - 60.0
        stores = session.poll().stores
        assert StoreResult(job_id=1, success=False) in stores

    def test_store_timeout_then_late_completion_no_duplicate(self):
        """A job timed out by _timeout_pending_store_jobs must not also
        emit a contradictory StoreResult(success=True) when the transport
        later reports the same transfer as done."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        tid = next(iter(transport._transfers))

        # Backdate the store job so the next poll times it out.
        session._server._store_jobs[1] = time.monotonic() - 60.0
        stores = session.poll().stores
        assert StoreResult(job_id=1, success=False) in stores
        assert StoreResult(job_id=1, success=True) not in stores
        session.ack_results(store_job_ids=(1,))

        # Transport later reports the same transfer as done — must not
        # emit a second (contradictory) StoreResult for job_id=1.
        transport._poll_done.append(tid)
        stores = session.poll().stores
        assert all(s.job_id != 1 for s in stores), (
            f"unexpected duplicate StoreResult after timeout: {stores}"
        )

    def test_store_timeout_then_late_failure_no_duplicate(self):
        """Symmetric guard: a timed-out job must not also emit a second
        StoreResult(success=False) when the transport later reports the
        same transfer as failed."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        tid = next(iter(transport._transfers))

        session._server._store_jobs[1] = time.monotonic() - 60.0
        stores = session.poll().stores
        assert [s for s in stores if s.job_id == 1] == [
            StoreResult(job_id=1, success=False)
        ]
        session.ack_results(store_job_ids=(1,))

        transport._poll_failed.append(tid)
        stores = session.poll().stores
        assert all(s.job_id != 1 for s in stores), (
            f"unexpected duplicate StoreResult after timeout: {stores}"
        )

    def test_active_store_timeout_waits_for_dma_quiescence(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        tid = next(iter(transport._transfers))
        transport._cancel_still_inflight.add(tid)
        session._server._store_jobs[1] = time.monotonic() - 60.0

        stores = session.poll().stores
        assert all(result.job_id != 1 for result in stores)
        assert 1 in session._server._store_jobs
        assert tid in session._server._inflight
        assert not any(msg[TYPE_KEY] == TransferDoneMsg.TYPE for msg in conn._sent)
        assert all(mode == "wait" for _, mode in transport._cancel_calls)

        transport._cancel_still_inflight.remove(tid)
        stores = session.poll().stores
        assert StoreResult(job_id=1, success=False) in stores
        assert 1 not in session._server._store_jobs
        assert tid not in session._server._inflight
        assert _srv_outbound(session, "req-1") is None
        done = [msg for msg in conn._sent if msg[TYPE_KEY] == TransferDoneMsg.TYPE]
        assert len(done) == 1
        assert done[0][TransferDoneMsg.SUCCESS] is False

    def test_pre_demand_timeout_leaves_failed_tombstone(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"stale"], [0], job_id=1)
        session._server._store_jobs[1] = time.monotonic() - 60.0

        assert session.poll().stores == [StoreResult(job_id=1, success=False)]
        session.ack_results(store_job_ids=(1,))
        failed_round = _srv_outbound(session, "req-1")
        assert failed_round.failed is True
        assert failed_round.available == {}

        # A producer batch arriving before the delayed fetch is rejected by
        # the tombstone and never exposes its block to DMA.
        session.add_stored_blocks("req-1", [b"late"], [1], job_id=2)
        assert session.poll().stores == [StoreResult(job_id=2, success=False)]
        session.ack_results(store_job_ids=(2,))
        assert transport._transfers == {}

        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"stale"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        assert transport._transfers == {}
        done = next(msg for msg in conn._sent if msg[TYPE_KEY] == TransferDoneMsg.TYPE)
        assert done[TransferDoneMsg.SUCCESS] is False
        assert transport._transfers == {}
        assert _srv_outbound(session, "req-1") is None

    def test_fetch_dispatch_fact_survives_server_call_to_store_cut(self):
        """A consumed FetchMsg remains visible when server dispatch is cut."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        original = session._server.on_fetch

        def dispatch_then_interrupt(*args, **kwargs):
            original(*args, **kwargs)
            raise _SubmissionFailure("after server fetch state committed")

        session._server.on_fetch = dispatch_then_interrupt
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-cut",
                FetchMsg.KEYS: [b"k"],
                FetchMsg.BLOCK_INDEXES: [2],
            }
        )
        with pytest.raises(_SubmissionFailure, match="fetch state committed"):
            session.poll()

        assert session.pending_results().new_fetch_ids == ["req-cut"]
        assert _srv_outbound(session, "req-cut").demand_received is True
        session._server.on_fetch = original
        assert session.poll().new_fetch_ids == ["req-cut"]
        session.ack_results(new_fetch_ids=("req-cut",))
        assert session.poll().new_fetch_ids == []

    def test_empty_fetch_finalization_resumes_after_result_publish_cut(self):
        """The poll boundary resumes a finalizer created by empty FetchMsg."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-cut", [b"leftover"], [0], job_id=41)
        original = session._server._publish_store_result
        interrupted = False

        def publish_then_interrupt(*args, **kwargs):
            nonlocal interrupted
            result = original(*args, **kwargs)
            if not interrupted:
                interrupted = True
                raise _SubmissionFailure("after store result publication")
            return result

        session._server._publish_store_result = publish_then_interrupt
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-cut",
                FetchMsg.KEYS: [],
                FetchMsg.BLOCK_INDEXES: [],
            }
        )
        with pytest.raises(_SubmissionFailure, match="result publication"):
            session.poll()

        assert ("req-cut", 0) in session._server._finalizing_rounds
        assert session._server._pending_store_results == {
            41: StoreResult(job_id=41, success=True)
        }
        session._server._publish_store_result = original
        result = session.poll()
        assert result.stores == [StoreResult(job_id=41, success=True)]
        assert not any(msg[TYPE_KEY] == TransferDoneMsg.TYPE for msg in conn._sent)
        session.ack_results(store_job_ids=(41,), new_fetch_ids=("req-cut",))
        assert session._server._finalizing_rounds == {}

    def test_finish_finalization_resumes_after_result_publish_cut(self):
        """A one-shot local finish cannot strand its outbound finalizer."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-cut", [b"available"], [0], job_id=42)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-cut",
                FetchMsg.KEYS: [b"missing"],
                FetchMsg.BLOCK_INDEXES: [3],
            }
        )
        session.poll()
        original = session._server._publish_store_result
        interrupted = False

        def publish_then_interrupt(*args, **kwargs):
            nonlocal interrupted
            result = original(*args, **kwargs)
            if not interrupted:
                interrupted = True
                raise _SubmissionFailure("after finish result publication")
            return result

        session._server._publish_store_result = publish_then_interrupt
        with pytest.raises(_SubmissionFailure, match="finish result publication"):
            session.finish_request("req-cut")

        assert ("req-cut", 0) in session._server._finalizing_rounds
        session._server._publish_store_result = original
        result = session.poll()
        assert result.stores == [StoreResult(job_id=42, success=False)]
        terminals = [
            msg
            for msg in conn._sent
            if msg[TYPE_KEY] == TransferDoneMsg.TYPE
            and msg[TransferDoneMsg.KV_REQUEST_ID] == "req-cut"
        ]
        assert len(terminals) == 1
        assert terminals[0][TransferDoneMsg.SUCCESS] is False
        session.ack_results(store_job_ids=(42,), new_fetch_ids=("req-cut",))
        assert session._server._finalizing_rounds == {}

    def test_close_rejects_every_public_mutation_and_poll(self):
        session, _, _ = _make_session()
        session.close()

        operations = (
            lambda: session.request_blocks(1, "req", [b"k"], [0]),
            lambda: session.add_stored_blocks("req", [b"k"], [0], 2),
            lambda: session.finish_request("req"),
            lambda: session.register_lookup("req", b"k"),
            session.flush_pending_lookups,
            lambda: session.serve_external_requests(FakeParent()),
            session.poll,
        )
        for operation in operations:
            with pytest.raises(RuntimeError, match="closing or closed"):
                operation()

    def test_failed_round_marks_done_sibling_failed_before_drain(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1", b"k2", b"k3"],
                FetchMsg.BLOCK_INDEXES: [5, 6, 7],
            }
        )
        session.poll()
        for job_id, key, block_id in (
            (1, b"k1", 0),
            (2, b"k2", 1),
            (3, b"k3", 2),
        ):
            session.add_stored_blocks("req-1", [key], [block_id], job_id=job_id)
        tids = list(session._server._inflight)
        transport._poll_done.append(tids[0])
        transport._poll_failed.append(tids[1])
        transport._cancel_still_inflight.add(tids[2])

        stores = session.poll().stores
        assert StoreResult(job_id=1, success=False) in stores
        assert StoreResult(job_id=1, success=True) not in stores
        assert StoreResult(job_id=2, success=False) in stores
        assert all(result.job_id != 3 for result in stores)
        assert tids[2] in session._server._inflight
        assert not any(msg[TYPE_KEY] == TransferDoneMsg.TYPE for msg in conn._sent)

        transport._cancel_still_inflight.remove(tids[2])
        stores = session.poll().stores
        assert StoreResult(job_id=3, success=False) in stores
        assert session._server._inflight == {}
        done = [msg for msg in conn._sent if msg[TYPE_KEY] == TransferDoneMsg.TYPE]
        assert len(done) == 1
        assert done[0][TransferDoneMsg.SUCCESS] is False
        assert _srv_outbound(session, "req-1") is None

    def test_finalize_refuses_to_release_active_round(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        tid = next(iter(session._server._inflight))

        with pytest.raises(RuntimeError, match="DMA is still active"):
            session._server._finalize_outbound("req-1", 0, success=False)

        assert tid in session._server._inflight
        assert 1 in session._server._store_jobs
        assert _srv_outbound(session, "req-1") is not None


# ---------------------------------------------------------------------------
# finish_request server-role early-fail flow
# ---------------------------------------------------------------------------


class TestFinishRequestServerSide:
    def _last_transfer_done(self, conn: FakeConnection) -> dict | None:
        for msg in reversed(conn._sent):
            if msg[TYPE_KEY] == TransferDoneMsg.TYPE:
                return msg
        return None

    def test_no_inflight_unmatched_demand_sends_failure(self):
        """finish_request with unmatched demand and no inflight ->
        immediate TransferDoneMsg(success=False); _outbound cleared."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        # Decoder demanded a block we never stored.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        assert _srv_outbound(session, "req-1") is not None

        session.finish_request("req-1")

        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert _srv_outbound(session, "req-1") is None

    def test_with_inflight_defers_then_fires_on_last_transfer(self):
        """finish_request with inflight defers; last transfer fires the
        early-fail message and clears _outbound."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        # Demand 2 blocks; we store 1 (kicks one inflight transfer).
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [10, 11],
            }
        )
        session.poll()
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        assert len(transport._transfers) == 1

        # finish_request while inflight: no early-fail yet.
        before = len(conn._sent)
        session.finish_request("req-1")
        assert len(conn._sent) == before
        assert _srv_outbound(session, "req-1") is not None
        assert _srv_outbound(session, "req-1").finishing

        # Last inflight settles -> early-fail fires.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        session.poll()

        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert _srv_outbound(session, "req-1") is None

    def test_full_demand_satisfied_still_sends_success(self):
        """finish_request must not override a fully-satisfied transfer:
        when remaining hits 0, success=True still fires."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [10],
            }
        )
        session.poll()
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        # Mark finishing (e.g., on_request_finished racing with the last
        # store) — but all demand is satisfied.
        session.finish_request("req-1")

        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        session.poll()

        msg = next(m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE)
        assert msg[TransferDoneMsg.SUCCESS] is True

    def test_prefiller_first_finish_before_fetch(self):
        """Prefiller-first: finish_request runs before the decoder's
        fetch arrives. State is held until fetch, then
        finalized — success=True if all demand was matched against
        available blocks, else success=False."""
        # Case A: all demand satisfied by available blocks.
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        # finish_request first — no demand received yet -> defer.
        session.finish_request("req-1")
        assert _srv_outbound(session, "req-1") is not None
        # Fetch arrives now: demand fully satisfied by available.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [10],
            }
        )
        session.poll()
        # The transfer was inflight; finalize it.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        session.poll()
        msg = next(m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE)
        assert msg[TransferDoneMsg.SUCCESS] is True

        # Case B: demand exceeds available -> early-fail fires from fetch.
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.add_stored_blocks("req-2", [b"k1"], [0], job_id=2)
        session.finish_request("req-2")
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-2",
                FetchMsg.KEYS: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [10, 11],
            }
        )
        session.poll()
        # Inflight for k1 still in flight; nothing yet for the early-fail
        # — the same code path will fire from _collect_store_results.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        session.poll()
        msg = next(m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE)
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert _srv_outbound(session, "req-2") is None

    def test_unknown_request_is_noop(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        before = len(conn._sent)
        session.finish_request("never-existed")
        assert len(conn._sent) == before

    def test_finish_request_no_inflight_emits_store_failure(self):
        """finish_request with a stored-but-unmatched job and no inflight ->
        TransferDoneMsg(success=False) AND deferred
        StoreResult(success=False) for the submit_store'd job, instead of
        the 30s _STORE_TIMEOUT_S path."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        # Decoder demanded b"demand"; we never stored it.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"demand"],
                FetchMsg.BLOCK_INDEXES: [5],
            }
        )
        session.poll()
        # We did submit_store a different block — goes to available, never
        # matches demand. Without the shortcut, job 42 sits in _store_jobs
        # for _STORE_TIMEOUT_S.
        session.add_stored_blocks("req-1", [b"unrelated"], [0], job_id=42)
        assert _srv_outbound(session, "req-1").pending_job_ids == {42}

        session.finish_request("req-1")

        # Peer notified immediately with success=False (remaining > 0).
        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.SUCCESS] is False
        assert _srv_outbound(session, "req-1") is None

        # Local store job surfaces on the next poll, success=False.
        stores = session.poll().stores
        assert StoreResult(job_id=42, success=False) in stores
        assert 42 not in session._server._store_jobs

    def test_finish_request_remaining_zero_emits_success_via_inflight(self):
        """Deferred-via-inflight path: finish_request with inflight, last
        transfer drains remaining to 0 -> TransferDoneMsg(success=True)
        AND StoreResult(success=True)."""
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [10],
            }
        )
        session.poll()
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=7)
        # finish_request races with the inflight transfer.
        session.finish_request("req-1")
        assert _srv_outbound(session, "req-1") is not None  # deferred

        # Last inflight completes -> _finalize_outbound(success=True) fires.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        stores = session.poll().stores

        msg = self._last_transfer_done(conn)
        assert msg is not None
        assert msg[TransferDoneMsg.SUCCESS] is True
        assert StoreResult(job_id=7, success=True) in stores
        assert _srv_outbound(session, "req-1") is None

    def test_write_blocks_failure_finalizes_with_failure(self):
        """write_blocks returning None must not leave the request hanging.

        The matched blocks are gone from req.demanded but no inflight
        will satisfy them, so remaining > 0 forever. Setting finishing
        and calling _finalize_outbound(success=False) immediately (no
        other inflight) tells the peer + emits StoreResult(success=False)
        without waiting on _STORE_TIMEOUT_S or _LOAD_TIMEOUT_S.
        """
        session, conn, transport = _make_session()
        _activate(session, conn)
        # Decoder demands b"k1".
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [10],
            }
        )
        session.poll()
        # Force write_blocks to fail on the next call.
        transport.write_blocks = lambda *a, **kw: None  # type: ignore[assignment]

        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=42)

        # Demand already exists, so quiescence plus the terminal failure fully
        # retires the round. Only pre-demand failures need a late-fetch tombstone.
        assert _srv_outbound(session, "req-1") is None
        # Peer notified with success=False.
        msg = next(m for m in conn._sent if m[TYPE_KEY] == TransferDoneMsg.TYPE)
        assert msg[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert msg[TransferDoneMsg.SUCCESS] is False
        # Local store job surfaces on the next poll.
        stores = session.poll().stores
        assert StoreResult(job_id=42, success=False) in stores
        assert 42 not in session._server._store_jobs

    def test_partial_match_completes_in_two_rounds(self):
        """Peer demand for [k1, k2, k3]; first round only k1 is available,
        second round adds k2 and k3. Each round transfers what's matched
        and the request finalizes with success once remaining hits zero.
        """
        session, conn, transport = _make_session()
        _activate(session, conn)

        # Peer fetches three blocks.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1", b"k2", b"k3"],
                FetchMsg.BLOCK_INDEXES: [10, 11, 12],
            }
        )
        session.poll()
        # Demand registered, no matches yet.
        assert session._server._inflight == {}
        outbound = _srv_outbound(session, "req-1")
        assert outbound.remaining == 3
        assert set(outbound.demanded.keys()) == {b"k1", b"k2", b"k3"}

        # Round 1: only k1 is stored locally.
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=100)
        # One inflight transfer for k1.
        assert len(session._server._inflight) == 1
        tid_1 = next(iter(session._server._inflight))
        assert transport._transfers[tid_1][1] == [0]
        assert transport._transfers[tid_1][2] == [10]
        # k2 and k3 still demanded.
        assert set(outbound.demanded.keys()) == {b"k2", b"k3"}

        # Transfer 1 completes.
        transport._poll_done.append(tid_1)
        stores = session.poll().stores
        assert StoreResult(job_id=100, success=True) in stores
        assert session._server._inflight == {}
        assert outbound.remaining == 2
        # Not yet finalized — still 2 blocks demanded.
        assert _srv_outbound(session, "req-1") is not None

        # Round 2: k2 and k3 arrive together.
        session.add_stored_blocks("req-1", [b"k2", b"k3"], [1, 2], job_id=200)
        assert len(session._server._inflight) == 1
        tid_2 = next(iter(session._server._inflight))
        assert tid_2 != tid_1
        assert sorted(transport._transfers[tid_2][1]) == [1, 2]

        # Transfer 2 completes — request now fully satisfied.
        transport._poll_done.append(tid_2)
        stores = session.poll().stores
        assert StoreResult(job_id=200, success=True) in stores
        # _finalize_outbound fired — request gone, peer notified with success.
        assert _srv_outbound(session, "req-1") is None
        done = next(m for m in conn._sent if m.get(TYPE_KEY) == TransferDoneMsg.TYPE)
        assert done[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert done[TransferDoneMsg.SUCCESS] is True

    def test_write_blocks_failure_wait_cancels_existing_sibling(self):
        """A failed second submit invalidates and drains the whole round."""
        session, conn, transport = _make_session()
        _activate(session, conn)

        # Peer demands two blocks.
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [10, 11],
            }
        )
        session.poll()

        # Round 1: k1 transfers cleanly.
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=100)
        assert len(session._server._inflight) == 1
        tid_1 = next(iter(session._server._inflight))
        outbound = _srv_outbound(session, "req-1")
        assert outbound.remaining == 2  # decrement happens on completion
        assert outbound.finishing is False

        # Round 2: write_blocks fails for k2 while transfer_1 is still inflight.
        transport.write_blocks = lambda *a, **kw: None  # type: ignore[assignment]
        session.add_stored_blocks("req-1", [b"k2"], [1], job_id=200)
        # The fake transport proves wait-cancel complete immediately. Both
        # source jobs therefore fail together and the round becomes a
        # tombstone; the first sibling is never reported successful.
        assert tid_1 not in session._server._inflight
        assert outbound.failed is True
        assert _srv_outbound(session, "req-1") is None
        done = next(m for m in conn._sent if m.get(TYPE_KEY) == TransferDoneMsg.TYPE)
        assert done[TransferDoneMsg.KV_REQUEST_ID] == "req-1"
        assert done[TransferDoneMsg.SUCCESS] is False

        stores = session.poll().stores
        assert StoreResult(job_id=100, success=False) in stores
        assert StoreResult(job_id=100, success=True) not in stores
        assert StoreResult(job_id=200, success=False) in stores

    @pytest.mark.parametrize(
        "cut",
        ["transport_return", "primary_map", "request_index", "round_counter"],
    )
    def test_submit_interruption_recovers_every_adoption_cut(self, cut):
        """Every lost-return cut converges from identity and primary state."""
        transport = _RecoveringDataTransport()
        session, conn, _ = _make_session(transport=transport)
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-cut",
                FetchMsg.KEYS: [b"k"],
                FetchMsg.BLOCK_INDEXES: [9],
            }
        )
        session.poll()

        failure = _SubmissionFailure(cut)
        server = session._server
        # Preserve the recovered transfer so assertions can inspect the fully
        # rebuilt graph before a later poll proves quiescence.
        transport._cancel_still_inflight.add(0)
        if cut == "transport_return":
            transport.submit_error = failure
        else:

            def interrupt_inflight_add(transfer_id, xfer):
                server._inflight[transfer_id] = xfer
                if cut == "primary_map":
                    raise failure
                server._get_or_create_request(xfer.kv_request_id).inflight_tids.add(
                    transfer_id
                )
                if cut == "request_index":
                    raise failure
                xfer.round.inflight += 1
                raise failure

            server._inflight_add = interrupt_inflight_add  # type: ignore[method-assign]

        with pytest.raises(_SubmissionFailure) as raised:
            session.add_stored_blocks("req-cut", [b"k"], [3], job_id=71)
        assert raised.value is failure

        assert server._submitting_xfer is None
        assert list(server._inflight) == [0]
        xfer = server._inflight[0]
        assert transport.submitted_tokens[-1] is xfer
        assert server._requests["req-cut"].inflight_tids == {0}
        assert xfer.round.inflight == 1
        assert xfer.round.failed is True
        assert server._failed_rounds[("req-cut", 0)] is xfer.round
        assert len(transport.recovery_calls) == (1 if cut == "transport_return" else 0)
        if transport.recovery_calls:
            assert transport.recovery_calls[0][1] is xfer

        transport._cancel_still_inflight.clear()
        stores = session.poll().stores
        assert stores == [StoreResult(job_id=71, success=False)]
        assert server._inflight == {}
        assert server._failed_rounds == {}

    def test_ambiguous_recovery_retains_guard_pins_and_quiescence_barriers(self):
        transport = _RecoveringDataTransport()
        session, conn, _ = _make_session(transport=transport)
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-ambiguous",
                FetchMsg.KEYS: [b"k"],
                FetchMsg.BLOCK_INDEXES: [9],
            }
        )
        session.poll()
        transport.submit_error = _SubmissionFailure("lost return")
        ambiguity = RuntimeError("ambiguous recovery")
        transport.recovery_error = ambiguity

        with pytest.raises(RuntimeError) as raised:
            session.add_stored_blocks("req-ambiguous", [b"k"], [3], job_id=72)
        assert raised.value is ambiguity

        server = session._server
        guarded = server._submitting_xfer
        assert guarded is not None
        assert transport.submitted_tokens[-1] is guarded
        assert 72 in server._store_jobs
        assert server.has_inflight_transfers
        assert server._has_inflight_for("req-ambiguous")
        server._maybe_prune("req-ambiguous")
        assert "req-ambiguous" in server._requests

        with pytest.raises(RuntimeError, match="DMA is still active"):
            server._finalize_outbound("req-ambiguous", 0, success=False)
        server._mark_round_failed("req-ambiguous", 0, guarded.round)
        with pytest.raises(RuntimeError, match="ownership is unresolved"):
            server._drain_failed_rounds()
        with pytest.raises(RuntimeError, match="ownership is unresolved"):
            server._finalize_abort("req-ambiguous", 0)
        with pytest.raises(RuntimeError) as close_error:
            server.close()
        assert close_error.value is ambiguity
        assert server._close_journal is None
        assert server._submitting_xfer is guarded
        assert 72 in server._store_jobs

    def test_interrupted_legacy_transport_without_recovery_fails_closed(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-legacy",
                FetchMsg.KEYS: [b"k"],
                FetchMsg.BLOCK_INDEXES: [9],
            }
        )
        session.poll()

        def lose_legacy_return(peer_id, local_idxs, remote_idxs):
            FakeDataTransport.write_blocks(transport, peer_id, local_idxs, remote_idxs)
            raise _SubmissionFailure("legacy lost return")

        transport.write_blocks = lose_legacy_return  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="does not support"):
            session.add_stored_blocks("req-legacy", [b"k"], [3], job_id=73)

        assert session._server._submitting_xfer is not None
        assert 73 in session._server._store_jobs
        assert session._server._inflight == {}
        assert transport._cancel_calls == []

    def test_recovery_rejects_bool_id_and_primary_collision(self):
        for invalid in (True, 0):
            transport = _RecoveringDataTransport()
            session, conn, _ = _make_session(transport=transport)
            _activate(session, conn)
            conn.enqueue(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.ROUND_SEQ: 0,
                    FetchMsg.KV_REQUEST_ID: "req-invalid",
                    FetchMsg.KEYS: [b"k"],
                    FetchMsg.BLOCK_INDEXES: [9],
                }
            )
            session.poll()
            transport.submit_error = _SubmissionFailure("lost return")
            transport.recovery_result = invalid
            if invalid == 0:
                other_round = _OutboundRequestState(inflight=1)
                session._server._inflight[0] = _InflightXfer(
                    kv_request_id="other",
                    block_count=1,
                    job_ids={999},
                    round=other_round,
                )

            expected = TypeError if invalid is True else RuntimeError
            with pytest.raises(expected):
                session.add_stored_blocks("req-invalid", [b"k"], [3], job_id=74)
            assert session._server._submitting_xfer is not None
            assert 74 in session._server._store_jobs
            assert transport._cancel_calls == []


# ---------------------------------------------------------------------------
# Bidirectional — the case the unification is meant to fix
# ---------------------------------------------------------------------------


class TestBidirectional:
    def test_session_handles_both_roles_concurrently(self):
        """Single session simultaneously serves a fetch and completes a load.

        This is the regression test for the unification: with the old
        split design the inbound FetchMsg would be dispatched to a
        client-only session and dropped (or a server-only session would
        miss the TransferDoneMsg). One unified session handles both.
        """
        session, conn, transport = _make_session()
        _activate(session, conn)

        # Server role: the peer fetches a block from us.
        session.add_stored_blocks("req-srv", [b"served"], [0], job_id=100)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-srv",
                FetchMsg.KEYS: [b"served"],
                FetchMsg.BLOCK_INDEXES: [7],
            }
        )

        # Client role: we ask the peer for a different block.
        session.request_blocks(
            job_id=200, kv_request_id="req-cli", keys=[b"loaded"], block_ids=[3]
        )

        # Both flows progress in the same poll.
        session.poll()

        # Server side: write_blocks was submitted.
        assert len(transport._transfers) == 1
        # Client side: the lookup was sent.
        assert any(
            m[TYPE_KEY] == FetchMsg.TYPE and m[FetchMsg.KV_REQUEST_ID] == "req-cli"
            for m in conn._sent
        )

        # Peer now signals: server-side transfer completes AND a
        # TransferDoneMsg arrives for our client-side request, all in
        # one batch on the same connection.
        tid = next(iter(transport._transfers))
        transport._poll_done.append(tid)
        conn.enqueue(
            {
                TYPE_KEY: TransferDoneMsg.TYPE,
                TransferDoneMsg.ROUND_SEQ: 0,
                TransferDoneMsg.KV_REQUEST_ID: "req-cli",
                TransferDoneMsg.SUCCESS: True,
            }
        )
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores

        assert LoadResult(job_id=200, kv_request_id="req-cli", success=True) in loads
        assert StoreResult(job_id=100, success=True) in stores


# ---------------------------------------------------------------------------
# Pending sessions (no connection yet)
# ---------------------------------------------------------------------------


class TestPendingSession:
    def test_pending_session_buffers_stored_blocks(self):
        """Pending session accepts add_stored_blocks but cannot send."""
        transport = FakeDataTransport()
        session = P2PSession(
            peer_id="peer:8000",
            local_id="local:9000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
            local_hash_seed=_DEFAULT_HASH_SEED,
            conn=None,
        )
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores
        assert loads == []
        assert stores == []
        assert not session.connected
        assert session.alive

    def test_attach_connection_sends_connect(self):
        """attach_connection triggers our ConnectMsg send."""
        transport = FakeDataTransport()
        session = P2PSession(
            peer_id="peer:8000",
            local_id="local:9000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
            local_hash_seed=_DEFAULT_HASH_SEED,
            conn=None,
        )
        conn = FakeConnection(peer_id="peer:8000")
        session.attach_connection(conn)  # type: ignore[arg-type]
        assert conn._sent
        assert conn._sent[0][TYPE_KEY] == ConnectMsg.TYPE

    def test_attach_connection_twice_raises(self):
        """attach_connection on an already-connected session raises."""
        session, conn, _ = _make_session()
        with pytest.raises(ValueError, match="already connected"):
            session.attach_connection(FakeConnection())  # type: ignore[arg-type]

    def test_pending_close_returns_pending_stores(self):
        """Closing a pending session reports buffered stores as failed."""
        transport = FakeDataTransport()
        session = P2PSession(
            peer_id="peer:8000",
            local_id="local:9000",
            transport=transport,  # type: ignore[arg-type]
            local_block_len=4096,
            local_hash_seed=_DEFAULT_HASH_SEED,
            conn=None,
        )
        session.add_stored_blocks("req-1", [b"k1"], [0], job_id=1)
        session.add_stored_blocks("req-2", [b"k2"], [1], job_id=2)
        result = session.close()
        assert result.failed_jobs == []
        assert result.failed_req_ids == []
        assert set(result.failed_stores) == {1, 2}
        assert result.failed_serves == []


# ---------------------------------------------------------------------------
# Disconnect
# ---------------------------------------------------------------------------


class TestDisconnect:
    def test_disconnect_marks_session_dead(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: DisconnectMsg.TYPE})
        session.poll()
        assert not session.alive

    def test_close_returns_pending_loads_and_stores(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(1, "req-1", [b"k"], [0])
        session.request_blocks(2, "req-2", [b"k"], [0])
        session.add_stored_blocks("req-srv", [b"k"], [0], job_id=10)
        result = session.close()
        assert set(result.failed_jobs) == {1, 2}
        assert set(result.failed_req_ids) == {"req-1", "req-2"}
        assert set(result.failed_stores) == {10}
        assert result.failed_serves == []
        assert session.close_complete is False
        assert session._client.has_active_loads is True

    def test_dead_session_close_quarantines_unacknowledged_destination(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(101, "req-quarantine", [b"k"], [0])
        conn.mark_dead()

        result = session.close()

        assert result.failed_jobs == [101]
        assert session.close_complete is False
        assert session._client.has_active_loads is True
        assert _client_load(session, "req-quarantine").job_id == 101
        assert session.close() is result

    def test_close_retry_preserves_results_after_connection_failure(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.request_blocks(1, "req-1", [b"k"], [0])
        session.add_stored_blocks("req-srv", [b"k"], [0], job_id=10)
        primary = RuntimeError("connection close")
        close_calls = 0

        def close_connection():
            nonlocal close_calls
            close_calls += 1
            if close_calls == 1:
                raise primary
            conn._closed = True

        conn.close = close_connection  # type: ignore[method-assign]
        with pytest.raises(RuntimeError) as raised:
            session.close()

        assert raised.value is primary
        result = session.close()
        assert result.failed_jobs == [1]
        assert result.failed_req_ids == ["req-1"]
        assert result.failed_stores == [10]
        assert session.close() is result
        assert close_calls == 2

    def test_close_retry_after_cancel_preserves_all_result_fields(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        session.request_blocks(1, "req-load", [b"load"], [0])
        assert session.register_lookup("req-probe", b"probe") is None

        parent = FakeParent(pending={b"parked"})
        _send_lookup(conn, "req-serve", [b"parked"])
        session.poll()
        _serve(session, parent)

        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-store",
                FetchMsg.KEYS: [b"store"],
                FetchMsg.BLOCK_INDEXES: [7],
            }
        )
        session.poll()
        session.add_stored_blocks("req-store", [b"store"], [3], job_id=10)
        assert session._server._inflight

        primary = _CloseFailure("cancel interrupted")
        original_cancel = transport.cancel
        cancel_calls = 0

        def cancel_once(transfer_ids, mode="immediate"):
            nonlocal cancel_calls
            cancel_calls += 1
            if cancel_calls == 1:
                raise primary
            return original_cancel(transfer_ids, mode)

        transport.cancel = cancel_once  # type: ignore[method-assign]
        with pytest.raises(_CloseFailure) as raised:
            session.close()

        assert raised.value is primary
        assert not session.alive
        result = session.close()
        assert result.failed_jobs == [1]
        assert set(result.failed_req_ids) == {"req-load", "req-probe"}
        assert result.failed_stores == [10]
        assert len(result.failed_serves) == 1
        assert ":req-serve:" in result.failed_serves[0].req_id
        assert session.close() is result
        assert cancel_calls == 2

    def test_close_retains_store_failure_until_transfer_is_quiescent(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-store",
                FetchMsg.KEYS: [b"store"],
                FetchMsg.BLOCK_INDEXES: [7],
            }
        )
        session.poll()
        session.add_stored_blocks("req-store", [b"store"], [3], job_id=10)
        tid = next(iter(session._server._inflight))
        transport._cancel_still_inflight.add(tid)

        result = session.close()
        assert result.failed_stores == [10]
        assert session.close_complete is False
        assert tid in session._server._inflight
        assert 10 in session._server._store_jobs
        assert transport._cancel_calls[-1] == ([tid], "wait")

        transport._cancel_still_inflight.remove(tid)
        assert session.close() is result
        assert session.close_complete is True
        assert session._server._inflight == {}
        assert session._server._store_jobs == {}
        assert transport._cancel_calls[-1] == ([tid], "wait")

    def test_disconnect_send_baseexception_does_not_block_close(self):
        session, conn, _ = _make_session()
        _activate(session, conn)

        def fail_disconnect(_msg):
            raise _CloseFailure("disconnect send")

        conn.send = fail_disconnect  # type: ignore[method-assign]
        result = session.close()

        assert result.failed_jobs == []
        assert not session.connected
        assert not session.alive
        assert conn._closed

    def test_send_failure_marks_connection_dead(self):
        """A raising send must mark the connection dead, not silently drop
        the message — otherwise the session lingers alive, is never reaped,
        and in-flight lookups/loads toward the dead peer hang forever."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        assert session.alive

        conn.fail_send = True
        # request_blocks flushes a FetchMsg synchronously via _do_send.
        session.request_blocks(1, "req-1", [b"k"], [0])

        assert not session.alive

    def test_close_surfaces_inflight_lookups(self):
        """close() reports kv_request_ids whose symmetric-P2P probe is still
        unresolved; resolved probes are not reported (their answer is in)."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        session.register_lookup("req-hit", b"hA")
        session.register_lookup("req-inflight", b"hB")
        session.flush_pending_lookups()

        # Only req-hit is answered; req-inflight stays in flight.
        conn.enqueue(
            {
                TYPE_KEY: LookupRespMsg.TYPE,
                LookupRespMsg.ROUND_SEQ: 0,
                LookupRespMsg.KV_REQUEST_ID: "req-hit",
                LookupRespMsg.KEYS: [b"hA"],
                LookupRespMsg.HITS: [True],
            }
        )
        session.poll()

        result = session.close()
        assert result.failed_jobs == []
        assert result.failed_req_ids == ["req-inflight"]


# ---------------------------------------------------------------------------
# Adversarial / malformed messages
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_lookup_response_missing_token_disconnects_without_resolving(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        session.register_lookup("req", b"same-key")
        session.flush_pending_lookups()

        conn.enqueue(
            {
                TYPE_KEY: LookupRespMsg.TYPE,
                LookupRespMsg.KV_REQUEST_ID: "req",
                LookupRespMsg.KEYS: [b"same-key"],
                LookupRespMsg.HITS: [True],
            }
        )
        session.poll()

        assert session.alive is False
        assert session._client._requests["req"].probes[b"same-key"] is None

    def test_unknown_message_type_logged(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: "evil_command"})
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores
        assert loads == []
        assert stores == []

    def test_empty_message(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({})
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores
        assert loads == []
        assert stores == []

    def test_non_dict_message(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn._inbox.append(42)  # type: ignore[arg-type]
        result_ = session.poll()
        loads = result_.loads
        stores = result_.stores
        assert loads == []
        assert stores == []

    def test_fetch_mismatched_lengths(self):
        session, conn, transport = _make_session()
        _activate(session, conn)
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-bad",
                FetchMsg.KEYS: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [1],
            }
        )
        session.poll()
        assert len(transport._transfers) == 0
        # Protocol violation: session disconnects immediately so the peer
        # can't keep wedging us with malformed traffic.
        assert not session.alive
        assert any(m[TYPE_KEY] == DisconnectMsg.TYPE for m in conn._sent)

    def test_transfer_done_missing_kv_request_id(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: TransferDoneMsg.TYPE, TransferDoneMsg.SUCCESS: True})
        loads = session.poll().loads
        assert loads == []
        assert not session.alive
        assert any(m[TYPE_KEY] == DisconnectMsg.TYPE for m in conn._sent)

    def test_duplicate_connect_ack(self):
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: ConnectAckMsg.TYPE, ConnectAckMsg.PEER_ID: "peer:8000"})
        session.poll()
        assert session.ready


class TestDispatchErrorHandling:
    """Errors raised by message handlers split into two classes:

    - Protocol-contract violations from the peer (ValueError) → disconnect
      on the first occurrence; retrying won't help and may corrupt state.
    - Anything else is treated as an internal bug: log loudly, count, and
      only disconnect once errors arrive in a tight burst. A successful
      dispatch in between resets the counter.
    """

    def test_value_error_disconnects_on_first_occurrence(self):
        """A FetchMsg that fails validate() raises ValueError and must
        terminate the session immediately, with a DisconnectMsg sent."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        # length mismatch → FetchMsg.validate raises ValueError
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-bad",
                FetchMsg.KEYS: [b"k1", b"k2"],
                FetchMsg.BLOCK_INDEXES: [1],
            }
        )
        session.poll()
        assert not session.alive
        assert any(m[TYPE_KEY] == DisconnectMsg.TYPE for m in conn._sent)

    def test_transfer_done_missing_field_disconnects(self):
        """Same contract for a malformed TransferDoneMsg from the peer."""
        session, conn, _ = _make_session()
        _activate(session, conn)
        conn.enqueue({TYPE_KEY: TransferDoneMsg.TYPE, TransferDoneMsg.SUCCESS: True})
        session.poll()
        assert not session.alive

    def test_internal_error_does_not_disconnect_once(self):
        """A non-ValueError raised by a handler is treated as an internal
        bug: counter increments, session stays alive on a single hit."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated internal bug")

        session._server.on_fetch = _boom  # type: ignore[assignment]
        conn.enqueue(
            {
                TYPE_KEY: FetchMsg.TYPE,
                FetchMsg.ROUND_SEQ: 0,
                FetchMsg.KV_REQUEST_ID: "req-1",
                FetchMsg.KEYS: [b"k1"],
                FetchMsg.BLOCK_INDEXES: [0],
            }
        )
        session.poll()
        assert session.alive
        assert session._dispatch_error_count == 1

    def test_internal_error_threshold_disconnects(self):
        """Once consecutive non-protocol errors hit the threshold, the
        session tears down via _protocol_error."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated internal bug")

        session._server.on_fetch = _boom  # type: ignore[assignment]
        for _ in range(_MAX_CONSECUTIVE_DISPATCH_ERRORS):
            conn.enqueue(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.ROUND_SEQ: 0,
                    FetchMsg.KV_REQUEST_ID: "req-1",
                    FetchMsg.KEYS: [b"k1"],
                    FetchMsg.BLOCK_INDEXES: [0],
                }
            )
        session.poll()
        assert not session.alive
        assert any(m[TYPE_KEY] == DisconnectMsg.TYPE for m in conn._sent)

    def test_internal_error_counter_resets_on_success(self):
        """A successful dispatch between errors prevents the threshold
        from being reached."""
        session, conn, _ = _make_session()
        _activate(session, conn)

        original_on_fetch = session._server.on_fetch

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated internal bug")

        # Alternate (boom, success) (_MAX-1) times: counter rises to 1
        # then resets to 0 each cycle, never reaching the threshold.
        for _ in range(_MAX_CONSECUTIVE_DISPATCH_ERRORS - 1):
            session._server.on_fetch = _boom  # type: ignore[assignment]
            conn.enqueue(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.ROUND_SEQ: 0,
                    FetchMsg.KV_REQUEST_ID: "req-1",
                    FetchMsg.KEYS: [b"k1"],
                    FetchMsg.BLOCK_INDEXES: [0],
                }
            )
            session.poll()
            session._server.on_fetch = original_on_fetch  # type: ignore[assignment]
            # A benign no-op message (unknown type) dispatches cleanly
            # and resets the consecutive-error counter.
            conn.enqueue({TYPE_KEY: "unknown_for_test"})
            session.poll()

        assert session.alive
        assert session._dispatch_error_count == 0


class TestInflightPerReqInvariant:
    """Per-request `inflight_tids` is the O(1) replacement for the
    previous O(N) scan in `_has_inflight_for`. These tests check that
    every mutation site keeps the set in sync with `_inflight` and
    that the lookup is correct under high fan-out.
    """

    def test_invariant_holds_through_lifecycle(self):
        """Run a full submit→complete sequence for two concurrent
        kv_request_ids and assert the counter matches `_inflight` at
        every observable step, including the empty-after-finish case."""
        session, conn, transport = _make_session()
        _activate(session, conn)

        def _invariant_holds() -> bool:
            return _srv_total_inflight(session) == len(session._server._inflight)

        assert _invariant_holds()

        # Two requests, two blocks each, all dispatched in one batch.
        session.add_stored_blocks("req-A", [b"a1", b"a2"], [0, 1], job_id=10)
        session.add_stored_blocks("req-B", [b"b1", b"b2"], [2, 3], job_id=11)
        for kv_id, keys, indexes in (
            ("req-A", [b"a1", b"a2"], [4, 5]),
            ("req-B", [b"b1", b"b2"], [6, 7]),
        ):
            conn.enqueue(
                {
                    TYPE_KEY: FetchMsg.TYPE,
                    FetchMsg.ROUND_SEQ: 0,
                    FetchMsg.KV_REQUEST_ID: kv_id,
                    FetchMsg.KEYS: keys,
                    FetchMsg.BLOCK_INDEXES: indexes,
                }
            )
        session.poll()

        assert _invariant_holds()
        assert session._server._has_inflight_for("req-A")
        assert session._server._has_inflight_for("req-B")
        assert not session._server._has_inflight_for("req-C")

        # Complete req-A's transfer first; req-B should still be inflight.
        a_tids = [
            tid
            for tid, x in session._server._inflight.items()
            if x.kv_request_id == "req-A"
        ]
        for tid in a_tids:
            transport._poll_done.append(tid)
        session.poll()

        assert _invariant_holds()
        assert not session._server._has_inflight_for("req-A")
        assert _srv_inflight_count(session, "req-A") == 0  # entry drained
        assert session._server._has_inflight_for("req-B")

        # Complete req-B; counter must drain to empty.
        b_tids = [
            tid
            for tid, x in session._server._inflight.items()
            if x.kv_request_id == "req-B"
        ]
        for tid in b_tids:
            transport._poll_done.append(tid)
        session.poll()

        assert _invariant_holds()
        assert session._server._inflight == {}
        assert _srv_total_inflight(session) == 0

    def test_has_inflight_for_correct_with_many_requests(self):
        """Populate many inflight xfers across many ids; lookup must
        match the actual presence in `_inflight` for both hits and
        misses. The whole point of the counter is that this lookup is
        constant-time, but we assert correctness, not timing."""
        session, _, _ = _make_session()
        for kv_id_idx in range(100):
            kv_id = f"req-{kv_id_idx}"
            for j in range(10):
                tid = kv_id_idx * 10 + j
                session._server._inflight_add(
                    tid,
                    _InflightXfer(
                        kv_request_id=kv_id,
                        block_count=1,
                        job_ids={tid},
                        round=_OutboundRequestState(inflight=1),
                    ),
                )
        assert _srv_total_inflight(session) == len(session._server._inflight)
        assert session._server._has_inflight_for("req-0")
        assert session._server._has_inflight_for("req-99")
        assert not session._server._has_inflight_for("req-missing")

        # Drain all entries for req-50 and confirm the entry disappears.
        tids_50 = [
            tid
            for tid, x in session._server._inflight.items()
            if x.kv_request_id == "req-50"
        ]
        for tid in tids_50:
            session._server._inflight_pop(tid)
        assert _srv_inflight_count(session, "req-50") == 0
        assert not session._server._has_inflight_for("req-50")
        # Other ids unaffected.
        assert session._server._has_inflight_for("req-49")


# ---------------------------------------------------------------------------
# Protocol validation tests (unchanged from the prior file)
# ---------------------------------------------------------------------------


class TestConnectMsgValidation:
    def _valid_msg(self) -> dict:
        return {
            TYPE_KEY: ConnectMsg.TYPE,
            WIRE_MAJOR_KEY: WIRE_PROTOCOL_MAJOR,
            WIRE_MINOR_KEY: WIRE_PROTOCOL_MINOR,
            ConnectMsg.SOURCE_EPOCH: _DEFAULT_PEER_EPOCH,
            ConnectMsg.TARGET_EPOCH: UNSPECIFIED_EPOCH,
            ConnectMsg.PEER_ID: "peer:1",
            ConnectMsg.AGENT_METADATA: b"meta",
            ConnectMsg.BASE_ADDR: 0x1000,
            ConnectMsg.NUM_BLOCKS: 8,
            ConnectMsg.BLOCK_LEN: 4096,
            ConnectMsg.CONFIG_FINGERPRINT: "",
            ConnectMsg.HASH_SEED: "0",
        }

    def test_valid_message_passes(self):
        ConnectMsg.validate(self._valid_msg())

    def test_missing_peer_id(self):
        msg = self._valid_msg()
        del msg[ConnectMsg.PEER_ID]
        with pytest.raises(ValueError, match="peer_id"):
            ConnectMsg.validate(msg)

    def test_missing_agent_metadata(self):
        msg = self._valid_msg()
        del msg[ConnectMsg.AGENT_METADATA]
        with pytest.raises(ValueError, match="agent_metadata"):
            ConnectMsg.validate(msg)

    def test_agent_metadata_wrong_type(self):
        msg = self._valid_msg()
        msg[ConnectMsg.AGENT_METADATA] = "not bytes"
        with pytest.raises(ValueError, match="agent_metadata"):
            ConnectMsg.validate(msg)

    def test_base_addr_negative(self):
        msg = self._valid_msg()
        msg[ConnectMsg.BASE_ADDR] = -1
        with pytest.raises(ValueError, match="base_addr"):
            ConnectMsg.validate(msg)

    def test_num_blocks_zero(self):
        msg = self._valid_msg()
        msg[ConnectMsg.NUM_BLOCKS] = 0
        with pytest.raises(ValueError, match="num_blocks"):
            ConnectMsg.validate(msg)

    def test_block_len_zero(self):
        msg = self._valid_msg()
        msg[ConnectMsg.BLOCK_LEN] = 0
        with pytest.raises(ValueError, match="block_len"):
            ConnectMsg.validate(msg)

    def test_missing_hash_seed(self):
        msg = self._valid_msg()
        del msg[ConnectMsg.HASH_SEED]
        with pytest.raises(ValueError, match="hash_seed"):
            ConnectMsg.validate(msg)

    def test_hash_seed_wrong_type(self):
        msg = self._valid_msg()
        msg[ConnectMsg.HASH_SEED] = 12345  # int, not str
        with pytest.raises(ValueError, match="hash_seed"):
            ConnectMsg.validate(msg)

    def test_missing_config_fingerprint(self):
        msg = self._valid_msg()
        del msg[ConnectMsg.CONFIG_FINGERPRINT]
        with pytest.raises(ValueError, match="config_fingerprint"):
            ConnectMsg.validate(msg)

    def test_config_fingerprint_wrong_type(self):
        msg = self._valid_msg()
        msg[ConnectMsg.CONFIG_FINGERPRINT] = b"fingerprint"
        with pytest.raises(ValueError, match="config_fingerprint"):
            ConnectMsg.validate(msg)

    @pytest.mark.parametrize(
        "field",
        (ConnectMsg.BASE_ADDR, ConnectMsg.NUM_BLOCKS, ConnectMsg.BLOCK_LEN),
    )
    def test_boolean_integer_rejected(self, field):
        msg = self._valid_msg()
        msg[field] = True
        with pytest.raises(ValueError, match=field):
            ConnectMsg.validate(msg)

    def test_num_blocks_bound(self):
        msg = self._valid_msg()
        msg[ConnectMsg.NUM_BLOCKS] = MAX_BLOCK_INDEX + 2
        with pytest.raises(ValueError, match="num_blocks"):
            ConnectMsg.validate(msg)

    def test_memory_span_overflow(self):
        msg = self._valid_msg()
        msg[ConnectMsg.BASE_ADDR] = MAX_ROUND_SEQ
        msg[ConnectMsg.NUM_BLOCKS] = 2
        msg[ConnectMsg.BLOCK_LEN] = 1
        with pytest.raises(ValueError, match="address space"):
            ConnectMsg.validate(msg)

    def test_agent_metadata_bound(self, monkeypatch):
        monkeypatch.setattr(protocol_module, "MAX_AGENT_METADATA_BYTES", 3)
        msg = self._valid_msg()
        msg[ConnectMsg.AGENT_METADATA] = b"four"
        with pytest.raises(ValueError, match="agent_metadata"):
            ConnectMsg.validate(msg)

    @pytest.mark.parametrize("field", (WIRE_MAJOR_KEY, WIRE_MINOR_KEY))
    def test_missing_version_field(self, field):
        msg = self._valid_msg()
        del msg[field]
        with pytest.raises(ValueError, match=field):
            ConnectMsg.validate(msg)

    def test_missing_target_epoch(self):
        msg = self._valid_msg()
        del msg[ConnectMsg.TARGET_EPOCH]
        with pytest.raises(ValueError, match="target_epoch"):
            ConnectMsg.validate(msg)

    def test_zero_source_epoch_is_reserved(self):
        msg = self._valid_msg()
        msg[ConnectMsg.SOURCE_EPOCH] = UNSPECIFIED_EPOCH
        with pytest.raises(ValueError, match="reserved"):
            ConnectMsg.validate(msg)


class TestConnectAckValidation:
    @staticmethod
    def _valid_msg() -> dict:
        return _peer_ack_msg(target_epoch=b"L" * SESSION_EPOCH_NBYTES)

    def test_valid_message_passes(self):
        ConnectAckMsg.validate(self._valid_msg())

    @pytest.mark.parametrize(
        ("field", "value"),
        (
            (WIRE_MAJOR_KEY, True),
            (WIRE_MINOR_KEY, WIRE_PROTOCOL_MINOR + 1),
            (ConnectAckMsg.SOURCE_EPOCH, b"short"),
            (ConnectAckMsg.SOURCE_EPOCH, UNSPECIFIED_EPOCH),
            (ConnectAckMsg.TARGET_EPOCH, None),
            (ConnectAckMsg.TARGET_EPOCH, UNSPECIFIED_EPOCH),
        ),
    )
    def test_malformed_or_unsupported_field_rejected(self, field, value):
        msg = self._valid_msg()
        msg[field] = value
        with pytest.raises(ValueError):
            ConnectAckMsg.validate(msg)


class TestFetchMsgValidation:
    def _valid_msg(self) -> dict:
        return {
            TYPE_KEY: FetchMsg.TYPE,
            **_channel_fields(),
            FetchMsg.ROUND_SEQ: 0,
            FetchMsg.KV_REQUEST_ID: "req-1",
            FetchMsg.KEYS: [b"k1", b"k2"],
            FetchMsg.BLOCK_INDEXES: [0, 1],
        }

    def test_valid_message_passes(self):
        FetchMsg.validate(self._valid_msg())

    @pytest.mark.parametrize("field", (SOURCE_EPOCH_KEY, TARGET_EPOCH_KEY))
    def test_zero_channel_epoch_is_reserved(self, field):
        msg = self._valid_msg()
        msg[field] = UNSPECIFIED_EPOCH
        with pytest.raises(ValueError, match="reserved"):
            FetchMsg.validate(msg)

    def test_length_mismatch(self):
        msg = self._valid_msg()
        msg[FetchMsg.BLOCK_INDEXES] = [0]
        with pytest.raises(ValueError, match="length mismatch"):
            FetchMsg.validate(msg)

    def test_negative_index(self):
        msg = self._valid_msg()
        msg[FetchMsg.BLOCK_INDEXES] = [0, -1]
        with pytest.raises(ValueError, match="invalid index"):
            FetchMsg.validate(msg)

    @pytest.mark.parametrize("index", (True, MAX_BLOCK_INDEX + 1))
    def test_non_exact_or_oversized_index(self, index):
        msg = self._valid_msg()
        msg[FetchMsg.BLOCK_INDEXES] = [0, index]
        with pytest.raises(ValueError, match="invalid index"):
            FetchMsg.validate(msg)

    def test_oversized_key(self):
        msg = self._valid_msg()
        msg[FetchMsg.KEYS] = [b"k", b"x" * (MAX_WIRE_KEY_BYTES + 1)]
        with pytest.raises(ValueError, match="keys"):
            FetchMsg.validate(msg)

    def test_oversized_list(self):
        msg = self._valid_msg()
        msg[FetchMsg.KEYS] = [b"k"] * (MAX_WIRE_LIST_ITEMS + 1)
        with pytest.raises(ValueError, match="exceeds"):
            FetchMsg.validate(msg)

    def test_tuple_is_not_a_wire_list(self):
        msg = self._valid_msg()
        msg[FetchMsg.KEYS] = (b"k1", b"k2")
        with pytest.raises(ValueError, match="expected list"):
            FetchMsg.validate(msg)

    @pytest.mark.parametrize("missing", (SOURCE_EPOCH_KEY, TARGET_EPOCH_KEY))
    def test_channel_epoch_is_required(self, missing):
        msg = self._valid_msg()
        del msg[missing]
        with pytest.raises(ValueError, match=missing):
            FetchMsg.validate(msg)


class TestRoundSeqValidation:
    @staticmethod
    def _messages(round_seq: object) -> tuple[tuple[type, dict], ...]:
        return (
            (
                FetchMsg,
                {
                    **_channel_fields(),
                    FetchMsg.KV_REQUEST_ID: "req",
                    FetchMsg.ROUND_SEQ: round_seq,
                    FetchMsg.KEYS: [],
                    FetchMsg.BLOCK_INDEXES: [],
                },
            ),
            (
                LookupMsg,
                {
                    **_channel_fields(),
                    LookupMsg.KV_REQUEST_ID: "req",
                    LookupMsg.ROUND_SEQ: round_seq,
                    LookupMsg.KEYS: [],
                },
            ),
            (
                LookupRespMsg,
                {
                    **_channel_fields(),
                    LookupRespMsg.KV_REQUEST_ID: "req",
                    LookupRespMsg.ROUND_SEQ: round_seq,
                    LookupRespMsg.KEYS: [],
                    LookupRespMsg.HITS: [],
                },
            ),
            (
                TransferDoneMsg,
                {
                    **_channel_fields(),
                    TransferDoneMsg.KV_REQUEST_ID: "req",
                    TransferDoneMsg.ROUND_SEQ: round_seq,
                    TransferDoneMsg.SUCCESS: True,
                },
            ),
            (
                AbortFetchMsg,
                {
                    **_channel_fields(),
                    AbortFetchMsg.KV_REQUEST_ID: "req",
                    AbortFetchMsg.ROUND_SEQ: round_seq,
                },
            ),
            (
                AbortAckMsg,
                {
                    **_channel_fields(),
                    AbortAckMsg.KV_REQUEST_ID: "req",
                    AbortAckMsg.ROUND_SEQ: round_seq,
                },
            ),
        )

    @pytest.mark.parametrize("round_seq", (0, MAX_ROUND_SEQ))
    def test_exact_uint64_tokens_are_accepted(self, round_seq):
        for message_type, msg in self._messages(round_seq):
            message_type.validate(msg)

    @pytest.mark.parametrize("round_seq", (None, True, -1, MAX_ROUND_SEQ + 1))
    def test_non_uint64_tokens_are_rejected(self, round_seq):
        for message_type, msg in self._messages(round_seq):
            with pytest.raises(ValueError, match="uint64 operation token"):
                message_type.validate(msg)


class TestTransferDoneMsgValidation:
    def test_valid_message_passes(self):
        msg = {
            TYPE_KEY: TransferDoneMsg.TYPE,
            **_channel_fields(),
            TransferDoneMsg.ROUND_SEQ: 0,
            TransferDoneMsg.KV_REQUEST_ID: "req-1",
            TransferDoneMsg.SUCCESS: True,
        }
        TransferDoneMsg.validate(msg)

    def test_success_wrong_type(self):
        msg = {
            TYPE_KEY: TransferDoneMsg.TYPE,
            **_channel_fields(),
            TransferDoneMsg.ROUND_SEQ: 0,
            TransferDoneMsg.KV_REQUEST_ID: "req-1",
            TransferDoneMsg.SUCCESS: 1,
        }
        with pytest.raises(ValueError, match="success"):
            TransferDoneMsg.validate(msg)
