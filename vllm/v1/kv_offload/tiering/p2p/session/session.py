# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PSession — bidirectional session combining client + server roles.

A single P2PSession per remote peer handles BOTH directions of the P2P
protocol on one ControlConnection: it can request blocks from the peer
("client" role, in :mod:`.client`) AND serve blocks to the peer
("server" role, in :mod:`.server`). This module is the thin coordinator
that owns the connection, the handshake, send-gating, and the message
dispatch — each parsed message is forwarded to the corresponding role.

Each incarnation owns a random wire epoch. Both sides advertise their metadata
and epoch in ConnectMsg and echo the peer epoch in ConnectAckMsg. Metadata is
imported and queued state is released only after those two independent facts
agree on the same remote epoch.
"""

from __future__ import annotations

import contextlib
import secrets
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey
from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection
from vllm.v1.kv_offload.tiering.p2p.session.client import ClientRole, LoadResult
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
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
    validate_channel,
)
from vllm.v1.kv_offload.tiering.p2p.session.server import (
    ServerRole,
    StoreResult,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.tiering.base import JobId, ParentManager
    from vllm.v1.kv_offload.tiering.p2p.data import DataTransport

logger = init_logger(__name__)

# Cap on consecutive non-protocol dispatch exceptions before we tear
# down the session. Protocol violations (ValueError) disconnect on the
# first occurrence; this threshold protects against repeated internal
# bugs that may indicate a peer-induced bad state. Reset on any
# successful dispatch.
_MAX_CONSECUTIVE_DISPATCH_ERRORS = 5
_MAX_HANDSHAKE_CANDIDATES = 8
_CONNECT_RETRY_INTERVAL_S = 0.5


class _ConnectCandidate(NamedTuple):
    epoch: bytes
    agent_metadata: bytes
    base_addr: int
    num_blocks: int
    block_len: int
    config_fingerprint: str
    hash_seed: str


class SessionPollResult(NamedTuple):
    """Result of one P2PSession.poll() tick.

    `loads`/`stores` are the same per-role results the manager has always
    consumed. `new_fetch_ids` reports kv_request_ids whose FetchMsg
    arrived this tick — the manager uses them to bind kv_request_id →
    session and replay any submit_store batches parked while no peer had
    asked yet. Reporting (rather than calling back into the manager
    mid-dispatch) keeps the dependency strictly top-down.
    """

    loads: list[LoadResult]
    stores: list[StoreResult]
    new_fetch_ids: list[str]


class SessionCloseResult(NamedTuple):
    """Result of tearing down a P2PSession.

    `failed_jobs`/`failed_stores` are the in-flight jobs (client loads /
    server stores) the manager must mark failed. `failed_req_ids` is every
    client-side kv_request_id whose lookup() must fail (in-flight loads plus
    unresolved probes); `failed_serves` is the server-side lookup state the
    dead peer can no longer resolve.
    """

    failed_jobs: list[int]  # client load job_ids
    failed_req_ids: list[str]  # client kv_request_ids (loads + probes)
    failed_stores: list[int]  # server store job_ids
    failed_serves: list[ReqContext]  # server-side lookup ctxs needing release


class P2PSession:
    """Bidirectional session — coordinator over ClientRole + ServerRole.

    Lifecycle:
      - Constructor with conn=None  ⇒ pending. Accepts add_stored_blocks
        but cannot send (used by the prefiller to buffer blocks before
        the decoder connects).
      - Constructor with conn != None ⇒ connected. Sends our own ConnectMsg
        immediately; the peer's ConnectMsg arrives in poll() and is
        dispatched to _on_connect (which calls transport.add_remote_peer
        and replies with ConnectAckMsg). Outgoing sends are queued until
        ConnectAckMsg confirms our metadata reached the peer.
      - attach_connection(conn) on a pending session ⇒ same as above,
        starting from pending.
    """

    def __init__(
        self,
        peer_id: str,
        local_id: str,
        transport: DataTransport,
        local_block_len: int,
        local_hash_seed: str,
        conn: ControlConnection | None = None,
    ) -> None:
        self.peer_id = peer_id
        self._local_id = local_id
        self._transport = transport
        self._local_block_len = local_block_len
        self._local_hash_seed = local_hash_seed
        self._conn: ControlConnection | None = None
        self._close_result: SessionCloseResult | None = None
        self._closing = False

        # Reserve the all-zero epoch for discovery Connect messages. The loop
        # has effectively one iteration, but makes that wire invariant exact.
        self._local_epoch = UNSPECIFIED_EPOCH
        while self._local_epoch == UNSPECIFIED_EPOCH:
            self._local_epoch = secrets.token_bytes(SESSION_EPOCH_NBYTES)
        self._remote_epoch: bytes | None = None
        self._remote_num_blocks: int | None = None
        self._connect_candidates: dict[bytes, _ConnectCandidate] = {}
        self._acked_remote_epochs: dict[bytes, None] = {}
        self._targeted_connect_epochs: set[bytes] = set()
        self._pending_connect_acks: dict[bytes, dict] = {}
        self._remote_import_started = False
        self._successor_epoch: bytes | None = None
        self._connect_msg: dict | None = None
        self._last_discovery_send_at = float("-inf")

        # True only after Connect and ConnectAck agree on one remote epoch and
        # that candidate's metadata has been imported.
        self._send_ready = False
        # Msgs waiting to be sent on connection establishment
        self._queued: list[dict] = []

        # Consecutive non-protocol dispatch errors. Reset on success.
        self._dispatch_error_count: int = 0

        # Every role outcome and fetch notification remains owned here until
        # the manager explicitly acknowledges it. Dict insertion order keeps
        # the historical result order without a second queue.
        self._load_results: dict[int, LoadResult] = {}
        self._store_results: dict[int, StoreResult] = {}
        self._new_fetch_ids: dict[str, None] = {}

        self._client = ClientRole(peer_id=peer_id, send=self._send)
        self._server = ServerRole(
            peer_id=peer_id,
            transport=transport,
            send=self._send,
        )

        if conn is not None:
            self.attach_connection(conn)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def alive(self) -> bool:
        # Pending sessions (awaiting connection) are alive until teardown
        # starts. A closing session stays dead even after its connection has
        # been detached, allowing manager-owned retirement to be retried.
        return not self._closing and (self._conn is None or self._conn.alive)

    @property
    def connected(self) -> bool:
        return self._conn is not None

    @property
    def close_complete(self) -> bool:
        """True only after local and possible remote DMA are quiescent."""
        return (
            self._closing
            and self._client.close_complete
            and self._server.close_complete
            and self._conn is None
        )

    @property
    def ready(self) -> bool:
        """True after the peer acked our ConnectMsg (we may send freely)."""
        return self._send_ready

    @property
    def has_pending_work(self) -> bool:
        """True while inbound loads or outbound transfers are outstanding."""
        return self._client.has_active_loads or self._server.has_inflight_transfers

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def attach_connection(self, conn: ControlConnection) -> None:
        """Attach a connection to a pending session and announce ourselves.

        Symmetric: every side advertises its NIXL metadata on connect, so
        whichever peer receives a session first can register the other.
        """
        self._check_open()
        if self._conn is not None:
            raise ValueError(f"P2PSession {self.peer_id}: already connected")
        self._conn = conn
        self._send_connect(force=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def request_blocks(
        self,
        job_id: JobId,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
    ) -> None:
        """Send fetch to the peer."""
        self._check_open()
        self._client.request_blocks(
            job_id, kv_request_id, keys, block_ids, send_ready=self._send_ready
        )

    def add_stored_blocks(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: JobId,
    ) -> None:
        """New blocks stored locally — match against pending fetch demand."""
        self._check_open()
        self._server.add_stored_blocks(kv_request_id, keys, block_ids, job_id)

    def finish_request(self, kv_request_id: str) -> None:
        """Called when the request is finishing locally.

        Finishes the client role (aborts any inbound load and drops any
        pending symmetric-P2P lookup state) and finalizes any outbound
        serving (server role) for this id. Roles that aren't active for
        this id are silent no-ops.
        """
        self._check_open()
        self._client.finish(kv_request_id)
        self._server.finish(kv_request_id)

    def register_lookup(self, kv_request_id: str, key: bytes) -> bool | None:
        """Register or resolve one (kv_request_id, key) probe.

        Called from the manager's lookup() for symmetric-P2P consumers
        (``remote_kv_source`` sub-dict in kv_transfer_params). See
        ``ClientRole.register_lookup`` for the state-machine contract.
        """
        self._check_open()
        return self._client.register_lookup(kv_request_id, key)

    def flush_pending_lookups(self) -> None:
        """Flush any aggregated symmetric-P2P lookups for this peer.

        Called once per scheduler step from the manager's
        ``on_schedule_end()``. Send-gating is handled inside the
        client's ``_send`` callback (queues until ConnectAckMsg).
        """
        self._check_open()
        self._client.flush_pending_lookups()

    def serve_external_requests(self, parent: ParentManager) -> None:
        """Resolve inbound peer lookups against the tiering manager.

        Delegates to the server role; the ``parent`` handle is valid
        only for the duration of this call.
        """
        self._check_open()
        self._server.serve_external_requests(parent)

    def poll(self) -> SessionPollResult:
        """Process incoming messages, drive transfers, apply timeouts."""
        self._check_open()
        if self._conn is None:
            # Pending session — store-job timeouts still apply so buffered
            # jobs that never get picked up are surfaced as failures.
            self._adopt_role_results((), self._server.collect_idle_timeouts())
            return self.pending_results()

        # Until the exact two-sided handshake completes, periodically replay
        # the immutable discovery Connect. This repairs a first frame rejected
        # while the peer was still retiring an older same-ID session.
        if not self._send_ready:
            self._send_connect()

        # A BaseException can surface after a handshake Ack was accepted by
        # the socket. Retain and replay its exact epoch pair until send returns.
        self._flush_pending_connect_acks()
        self._maybe_complete_handshake()

        # A BaseException can interrupt a handshake queue flush after the
        # underlying send committed. The queue deliberately retains that head;
        # every later poll retries its immutable wire identity without waiting
        # for another ConnectAck that the peer has already sent.
        if self._send_ready and self._queued:
            self._flush_queued()

        for msg in self._conn.recv():
            self._on_message(msg)

        self._adopt_role_results(
            self._client.collect_results(), self._server.collect_results()
        )
        self._server.drain_pending_aborts()
        return self.pending_results()

    def pending_results(self) -> SessionPollResult:
        """Peek at outcomes and fetch notices owned by this session."""
        return SessionPollResult(
            loads=list(self._load_results.values()),
            stores=list(self._store_results.values()),
            new_fetch_ids=list(self._new_fetch_ids),
        )

    def ack_results(
        self,
        load_job_ids: Sequence[int] = (),
        store_job_ids: Sequence[int] = (),
        new_fetch_ids: Sequence[str] = (),
    ) -> None:
        """Forget only entries durably adopted by the manager."""
        for job_id in load_job_ids:
            if type(job_id) is not int:
                raise TypeError("load job_id must be an exact int")
            self._load_results.pop(job_id, None)
        for job_id in store_job_ids:
            if type(job_id) is not int:
                raise TypeError("store job_id must be an exact int")
            self._store_results.pop(job_id, None)
        for kv_request_id in new_fetch_ids:
            if type(kv_request_id) is not str:
                raise TypeError("kv_request_id must be an exact str")
            self._new_fetch_ids.pop(kv_request_id, None)

    def owns_store_job(self, job_id: int) -> bool:
        """Whether replaying an unbound manager batch would duplicate it."""
        return job_id in self._store_results or self._server.owns_store_job(job_id)

    def close(self) -> SessionCloseResult:
        """Shut down.

        failed_jobs: stranded client load job_ids. Publish them only after
            ``close_complete`` proves destination reuse safe.
        failed_req_ids: client kv_request_ids to fail — in-flight loads plus
            requests with an unresolved symmetric-P2P probe toward the
            now-dead peer. The manager fails these so the consumer's lookup()
            falls back to local prefill instead of deferring forever on an
            answer that can never arrive.
        failed_stores: server store job_ids to fail.
        failed_serves: synthetic lookup ctxs still owing
            ``parent.on_request_finished`` (the manager flushes these on
            its next ``serve_external_requests``).
        """
        if not self._closing:
            # Move already-terminal role outcomes up before role teardown can
            # clear live request maps. Role acknowledgement happens only after
            # the session dictionaries above own every returned result.
            self._adopt_role_results(
                self._client.collect_results(), self._server.collect_results()
            )
        self._closing = True
        client_result = self._client.close()
        failed_stores, failed_serves = self._server.close()
        if self._close_result is None:
            self._close_result = SessionCloseResult(
                failed_jobs=client_result.failed_jobs,
                failed_req_ids=client_result.failed_req_ids,
                failed_stores=failed_stores,
                failed_serves=failed_serves,
            )

        if self._conn is not None:
            if self._remote_epoch is not None:
                with contextlib.suppress(BaseException):
                    self._do_send({TYPE_KEY: DisconnectMsg.TYPE})
            self._conn.close()
            self._conn = None

        assert self._close_result is not None
        return self._close_result

    def _adopt_role_results(
        self,
        loads: Sequence[LoadResult],
        stores: Sequence[StoreResult],
    ) -> None:
        """Publish role outcomes locally, then acknowledge their old owners."""
        for result in loads:
            existing = self._load_results.setdefault(result.job_id, result)
            if existing != result:
                raise RuntimeError(f"conflicting load outcomes for job {result.job_id}")
        for result in stores:
            existing = self._store_results.setdefault(result.job_id, result)
            if existing != result:
                raise RuntimeError(
                    f"conflicting store outcomes for job {result.job_id}"
                )
        if loads:
            self._client.ack_results(tuple(result.job_id for result in loads))
        if stores:
            self._server.ack_results(tuple(result.job_id for result in stores))

    def _check_open(self) -> None:
        if self._closing:
            raise RuntimeError(f"P2PSession {self.peer_id} is closing or closed")

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _on_message(self, msg: dict) -> None:
        msg_type = msg.get(TYPE_KEY) if isinstance(msg, dict) else msg
        try:
            self._dispatch_message(msg)
        except ValueError as exc:
            # Protocol contract violation from the peer — *Msg.validate()
            # and handler-level checks raise ValueError. Retrying won't
            # help and may corrupt session state, so disconnect now.
            self._protocol_error(f"malformed {msg_type!r}: {exc}")
            return
        except Exception as exc:
            # Anything else is most likely an internal bug rather than a
            # peer fault. Log loudly with a traceback so it doesn't
            # disappear, but don't kill the session on a single hiccup.
            # Disconnect only if errors keep arriving — that pattern is
            # consistent with a peer wedging us into a broken state.
            self._dispatch_error_count += 1
            logger.exception(
                "P2PSession %s: error handling message %r (count=%d): %s",
                self.peer_id,
                msg_type,
                self._dispatch_error_count,
                exc,
            )
            if self._dispatch_error_count >= _MAX_CONSECUTIVE_DISPATCH_ERRORS:
                self._protocol_error(
                    f"too many consecutive dispatch errors "
                    f"({self._dispatch_error_count})"
                )
            return
        self._dispatch_error_count = 0

    def _protocol_error(self, reason: str) -> None:
        """Log a protocol violation and disconnect.

        Best-effort sends ``DisconnectMsg`` so the peer learns why we're
        going away, then marks the connection dead. The manager reaps
        the session on the next poll via ``alive``.
        """
        logger.error(
            "P2PSession %s: protocol error: %s — disconnecting",
            self.peer_id,
            reason,
        )
        if self._conn is not None:
            if self._remote_epoch is not None:
                with contextlib.suppress(Exception):
                    self._do_send({TYPE_KEY: DisconnectMsg.TYPE})
            self._conn.mark_dead()

    def _dispatch_message(self, msg: dict) -> None:
        # Drop messages buffered before disconnect: a poll batch can
        # contain msg-after-DisconnectMsg, and dispatching them would
        # mutate state on a dead session.
        if self._conn is not None and not self._conn.alive:
            return
        if type(msg) is not dict:
            raise ValueError(f"message must be exact dict, got {type(msg).__name__}")
        msg_type = msg.get(TYPE_KEY)
        if msg_type == ConnectMsg.TYPE:
            self._on_connect(msg)
        elif msg_type == ConnectAckMsg.TYPE:
            self._on_connect_ack(msg)
        elif msg_type == FetchMsg.TYPE:
            if not self._is_current_channel(msg):
                return
            FetchMsg.validate(msg, channel_validated=True)
            kv_request_id = msg[FetchMsg.KV_REQUEST_ID]
            keys = [
                OffloadKey(bh if isinstance(bh, bytes) else bytes(bh))
                for bh in msg[FetchMsg.KEYS]
            ]
            block_indexes = msg[FetchMsg.BLOCK_INDEXES]
            assert self._remote_num_blocks is not None
            for index in block_indexes:
                if index >= self._remote_num_blocks:
                    raise ValueError(
                        "block_indexes contains an index outside the requester's "
                        f"advertised {self._remote_num_blocks} blocks"
                    )
            round_seq = msg[FetchMsg.ROUND_SEQ]
            # Run the server-role state machine inline as today —
            # add_fetch_demand records demand against any blocks we've
            # already seen in `available`. Report the kv_request_id so
            # the manager (after poll() returns) can replay any parked
            # submit_store batches; their add_stored_blocks calls hit
            # the demand recorded here and submit transfers immediately.
            # Publish the validated routing fact before the fallible server
            # transition. If a signal lands after demand or DMA submission,
            # the next poll can still hand the consumed message to the manager.
            self._new_fetch_ids.setdefault(kv_request_id, None)
            self._server.on_fetch(kv_request_id, keys, block_indexes, round_seq)
        elif msg_type == AbortFetchMsg.TYPE:
            if not self._is_current_channel(msg):
                return
            AbortFetchMsg.validate(msg, channel_validated=True)
            self._server.on_abort_fetch(
                msg[AbortFetchMsg.KV_REQUEST_ID],
                msg[AbortFetchMsg.ROUND_SEQ],
            )
        elif msg_type == TransferDoneMsg.TYPE:
            if not self._is_current_channel(msg):
                return
            TransferDoneMsg.validate(msg, channel_validated=True)
            self._client.on_transfer_done(
                msg[TransferDoneMsg.KV_REQUEST_ID],
                msg[TransferDoneMsg.SUCCESS],
                msg[TransferDoneMsg.ROUND_SEQ],
            )
        elif msg_type == AbortAckMsg.TYPE:
            if not self._is_current_channel(msg):
                return
            AbortAckMsg.validate(msg, channel_validated=True)
            self._client.on_abort_ack(
                msg[AbortAckMsg.KV_REQUEST_ID],
                msg[AbortAckMsg.ROUND_SEQ],
            )
        elif msg_type == LookupMsg.TYPE:
            if not self._is_current_channel(msg):
                return
            LookupMsg.validate(msg, channel_validated=True)
            kv_request_id = msg[LookupMsg.KV_REQUEST_ID]
            keys = [
                OffloadKey(bh if isinstance(bh, bytes) else bytes(bh))
                for bh in msg[LookupMsg.KEYS]
            ]
            self._server.on_lookup(kv_request_id, keys, msg[LookupMsg.ROUND_SEQ])
        elif msg_type == LookupRespMsg.TYPE:
            if not self._is_current_channel(msg):
                return
            LookupRespMsg.validate(msg, channel_validated=True)
            kv_request_id = msg[LookupRespMsg.KV_REQUEST_ID]
            keys = [
                OffloadKey(bh if isinstance(bh, bytes) else bytes(bh))
                for bh in msg[LookupRespMsg.KEYS]
            ]
            hits = msg[LookupRespMsg.HITS]
            self._client.on_lookup_resp(
                kv_request_id, keys, hits, msg[LookupRespMsg.ROUND_SEQ]
            )
        elif msg_type == DisconnectMsg.TYPE:
            if not self._is_current_channel(msg):
                return
            DisconnectMsg.validate(msg, channel_validated=True)
            if self._conn is not None:
                self._conn.mark_dead()
        else:
            logger.warning(
                "P2PSession %s: unknown message type %r", self.peer_id, msg_type
            )

    def _is_current_channel(self, msg: dict) -> bool:
        """Reject stale epochs before semantic validation or state mutation."""
        validate_channel(msg)
        source_epoch = msg[SOURCE_EPOCH_KEY]
        target_epoch = msg[TARGET_EPOCH_KEY]
        if (
            not self._send_ready
            or source_epoch != self._remote_epoch
            or target_epoch != self._local_epoch
        ):
            logger.debug(
                "P2PSession %s: dropping %s from stale/unready epoch",
                self.peer_id,
                msg.get(TYPE_KEY),
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def _on_connect(self, msg: dict) -> None:
        # A Connect either discovers us (zero target) or proves it observed
        # this exact local incarnation. Any other target is unambiguously stale
        # and is dropped before metadata, role, or data-plane mutation.
        source_epoch = msg.get(ConnectMsg.SOURCE_EPOCH)
        if type(source_epoch) is not bytes or len(source_epoch) != SESSION_EPOCH_NBYTES:
            raise ValueError(
                f"connect source_epoch must be exactly {SESSION_EPOCH_NBYTES} bytes"
            )
        target_epoch = msg.get(ConnectMsg.TARGET_EPOCH)
        if type(target_epoch) is not bytes or len(target_epoch) != SESSION_EPOCH_NBYTES:
            raise ValueError(
                f"connect target_epoch must be exactly {SESSION_EPOCH_NBYTES} bytes"
            )
        if target_epoch not in (UNSPECIFIED_EPOCH, self._local_epoch):
            logger.debug(
                "P2PSession %s: dropping Connect targeting retired local epoch",
                self.peer_id,
            )
            return

        is_successor = self._send_ready and source_epoch != self._remote_epoch
        try:
            candidate = self._validate_connect_candidate(msg, source_epoch)
        except ValueError as exc:
            if is_successor:
                logger.warning(
                    "P2PSession %s: dropping invalid successor Connect: %s",
                    self.peer_id,
                    exc,
                )
                return
            raise
        self._remember_connect_candidate(
            candidate, targeted=target_epoch == self._local_epoch
        )

        if is_successor:
            # Do not Ack a possible successor while the current session still
            # owns imported metadata and data-plane work. Challenge it instead.
            # Only its matching targeted Connect + Ack can retire this owner;
            # a lone stale discovery or Ack cannot kill a healthy session.
            self._send_connect(target_epoch=source_epoch, force=True)
            self._maybe_request_successor()
            return

        self._send_connect_ack(source_epoch)
        if not self._send_ready:
            self._send_connect(target_epoch=source_epoch, force=True)
            self._maybe_complete_handshake()

    def _validate_connect_candidate(
        self, msg: dict, source_epoch: bytes
    ) -> _ConnectCandidate:
        ConnectMsg.validate(msg)
        if msg[ConnectMsg.PEER_ID] != self.peer_id:
            raise ValueError(
                f"peer_id mismatch: expected {self.peer_id!r}, "
                f"got {msg[ConnectMsg.PEER_ID]!r}"
            )
        if msg[ConnectMsg.BLOCK_LEN] != self._local_block_len:
            raise ValueError(
                f"block_len mismatch from {self.peer_id}: "
                f"remote={msg[ConnectMsg.BLOCK_LEN]}, "
                f"local={self._local_block_len}"
            )
        remote_fp = msg[ConnectMsg.CONFIG_FINGERPRINT]
        local_fp = self._transport.config_fingerprint
        if type(local_fp) is not str or remote_fp != local_fp:
            raise ValueError(
                f"config fingerprint mismatch from {self.peer_id}: "
                f"remote={remote_fp!r}, local={local_fp!r}"
            )
        if msg[ConnectMsg.HASH_SEED] != self._local_hash_seed:
            raise ValueError(
                f"hash seed mismatch from {self.peer_id}: "
                f"remote={msg[ConnectMsg.HASH_SEED]!r}, "
                f"local={self._local_hash_seed!r}. Ensure PYTHONHASHSEED "
                "(if set) matches on all P2P peers."
            )

        return _ConnectCandidate(
            epoch=source_epoch,
            agent_metadata=msg[ConnectMsg.AGENT_METADATA],
            base_addr=msg[ConnectMsg.BASE_ADDR],
            num_blocks=msg[ConnectMsg.NUM_BLOCKS],
            block_len=msg[ConnectMsg.BLOCK_LEN],
            config_fingerprint=remote_fp,
            hash_seed=msg[ConnectMsg.HASH_SEED],
        )

    def _remember_connect_candidate(
        self, candidate: _ConnectCandidate, *, targeted: bool
    ) -> None:
        source_epoch = candidate.epoch
        existing = self._connect_candidates.get(source_epoch)
        if existing is not None and existing != candidate:
            raise ValueError("conflicting Connect metadata for one source epoch")
        if existing is None:
            self._ensure_handshake_capacity(source_epoch)
            self._connect_candidates[source_epoch] = candidate
        if targeted:
            self._targeted_connect_epochs.add(source_epoch)

    def _on_connect_ack(self, msg: dict) -> None:
        # An Ack for a retired local session is safely distinguishable before
        # version/peer validation and must not kill a healthy reconnect.
        target_epoch = msg.get(ConnectAckMsg.TARGET_EPOCH)
        if type(target_epoch) is not bytes or len(target_epoch) != SESSION_EPOCH_NBYTES:
            raise ValueError(
                f"connect_ack target_epoch must be exactly {SESSION_EPOCH_NBYTES} bytes"
            )
        if target_epoch != self._local_epoch:
            logger.debug(
                "P2PSession %s: dropping Ack targeting retired local epoch",
                self.peer_id,
            )
            return
        source_epoch = msg.get(ConnectAckMsg.SOURCE_EPOCH)
        if type(source_epoch) is not bytes or len(source_epoch) != SESSION_EPOCH_NBYTES:
            raise ValueError(
                f"connect_ack source_epoch must be exactly {SESSION_EPOCH_NBYTES} bytes"
            )
        is_successor = self._send_ready and source_epoch != self._remote_epoch
        try:
            ConnectAckMsg.validate(msg)
            if msg[ConnectAckMsg.PEER_ID] != self.peer_id:
                raise ValueError(
                    f"peer_id mismatch: expected {self.peer_id!r}, "
                    f"got {msg[ConnectAckMsg.PEER_ID]!r}"
                )
        except ValueError as exc:
            if is_successor:
                logger.warning(
                    "P2PSession %s: dropping invalid successor Ack: %s",
                    self.peer_id,
                    exc,
                )
                return
            raise
        self._remember_connect_ack(source_epoch)
        if is_successor:
            self._send_connect(target_epoch=source_epoch, force=True)
            self._maybe_request_successor()
            return
        self._maybe_complete_handshake()

    def _remember_connect_ack(self, source_epoch: bytes) -> None:
        if source_epoch not in self._acked_remote_epochs:
            self._ensure_handshake_capacity(source_epoch)
            self._acked_remote_epochs[source_epoch] = None

    def _ensure_handshake_capacity(self, epoch: bytes) -> None:
        """Bound the union of candidate/Ack epochs, preserving the live one."""
        if epoch in self._connect_candidates or epoch in self._acked_remote_epochs:
            return
        known_count = len(self._connect_candidates) + sum(
            known not in self._connect_candidates for known in self._acked_remote_epochs
        )
        if known_count < _MAX_HANDSHAKE_CANDIDATES:
            return
        for known in (*self._connect_candidates, *self._acked_remote_epochs):
            if known != self._remote_epoch:
                self._connect_candidates.pop(known, None)
                self._acked_remote_epochs.pop(known, None)
                self._targeted_connect_epochs.discard(known)
                self._pending_connect_acks.pop(known, None)
                return
        raise ValueError("handshake candidate capacity exhausted")

    def _maybe_complete_handshake(self) -> None:
        if self._send_ready:
            return
        matches = [
            epoch
            for epoch in self._connect_candidates
            if epoch in self._acked_remote_epochs
            and epoch in self._targeted_connect_epochs
        ]
        if not matches:
            return
        if len(matches) != 1:
            raise ValueError("ambiguous remote session epochs in handshake")
        remote_epoch = matches[0]
        candidate = self._connect_candidates[remote_epoch]

        self._remote_import_started = True
        try:
            self._transport.add_remote_peer(
                self.peer_id,
                agent_metadata=candidate.agent_metadata,
                base_addr=candidate.base_addr,
                num_blocks=candidate.num_blocks,
                block_len=candidate.block_len,
            )
            self._remote_epoch = remote_epoch
            self._remote_num_blocks = candidate.num_blocks
            self._connect_candidates = {remote_epoch: candidate}
            self._acked_remote_epochs = {remote_epoch: None}
            self._targeted_connect_epochs = {remote_epoch}
            self._pending_connect_acks.clear()
            self._send_ready = True
        except BaseException:
            if self._conn is not None:
                self._conn.mark_dead()
            raise

        if self._queued:
            logger.debug(
                "P2PSession %s: handshake complete, flushing %d queued msg(s)",
                self.peer_id,
                len(self._queued),
            )
        self._flush_queued()

    def _maybe_request_successor(self) -> None:
        """Retire only after a successor proves this local incarnation.

        Successor metadata is never imported in place. Manager-owned close
        first quiesces the old data plane and removes its peer; the successor's
        discovery retry is then accepted by a fresh session/local epoch.
        """
        if not self._send_ready or self._successor_epoch is not None:
            return
        for epoch in self._connect_candidates:
            if (
                epoch != self._remote_epoch
                and epoch in self._acked_remote_epochs
                and epoch in self._targeted_connect_epochs
            ):
                self._successor_epoch = epoch
                logger.info(
                    "P2PSession %s: successor handshake proven; retiring old epoch",
                    self.peer_id,
                )
                if self._conn is not None:
                    self._conn.mark_dead()
                return

    def _send_connect_ack(self, target_epoch: bytes) -> None:
        msg = {
            TYPE_KEY: ConnectAckMsg.TYPE,
            WIRE_MAJOR_KEY: WIRE_PROTOCOL_MAJOR,
            WIRE_MINOR_KEY: WIRE_PROTOCOL_MINOR,
            ConnectAckMsg.PEER_ID: self._local_id,
            ConnectAckMsg.SOURCE_EPOCH: self._local_epoch,
            ConnectAckMsg.TARGET_EPOCH: target_epoch,
        }
        self._pending_connect_acks[target_epoch] = msg
        self._flush_pending_connect_acks()

    def _flush_pending_connect_acks(self) -> None:
        if self._conn is None:
            return
        for target_epoch, msg in tuple(self._pending_connect_acks.items()):
            try:
                self._conn.send(msg)
            except Exception:
                self._conn.mark_dead()
                return
            self._pending_connect_acks.pop(target_epoch, None)

    def _flush_queued(self) -> None:
        """Retry accepted pre-handshake messages until one send is uncertain."""
        # Dequeue one message only after its send succeeds. Exception and
        # BaseException cuts therefore retain the current owner and queue tail.
        while self._queued:
            if not self._do_send(self._queued[0]):
                return
            self._queued.pop(0)

    # ------------------------------------------------------------------
    # Send helpers
    # ------------------------------------------------------------------

    def _send_connect(
        self,
        *,
        target_epoch: bytes = UNSPECIFIED_EPOCH,
        force: bool = False,
    ) -> bool:
        """Send a discovery/reply Connect with bounded discovery retries."""
        assert self._conn is not None
        if self._connect_msg is None:
            self._connect_msg = {
                TYPE_KEY: ConnectMsg.TYPE,
                WIRE_MAJOR_KEY: WIRE_PROTOCOL_MAJOR,
                WIRE_MINOR_KEY: WIRE_PROTOCOL_MINOR,
                ConnectMsg.SOURCE_EPOCH: self._local_epoch,
                ConnectMsg.TARGET_EPOCH: UNSPECIFIED_EPOCH,
                ConnectMsg.PEER_ID: self._local_id,
                ConnectMsg.AGENT_METADATA: self._transport.get_agent_metadata(),
                ConnectMsg.BASE_ADDR: self._transport.base_addr,
                ConnectMsg.NUM_BLOCKS: self._transport.num_blocks,
                ConnectMsg.BLOCK_LEN: self._transport.block_len,
                ConnectMsg.CONFIG_FINGERPRINT: self._transport.config_fingerprint,
                ConnectMsg.HASH_SEED: self._local_hash_seed,
            }
            ConnectMsg.validate(self._connect_msg)

        now = time.monotonic()
        if (
            target_epoch == UNSPECIFIED_EPOCH
            and not force
            and now - self._last_discovery_send_at < _CONNECT_RETRY_INTERVAL_S
        ):
            return True
        msg = self._connect_msg
        if target_epoch != UNSPECIFIED_EPOCH:
            msg = dict(msg)
            msg[ConnectMsg.TARGET_EPOCH] = target_epoch
        ConnectMsg.validate(msg)
        try:
            self._conn.send(msg)
        except Exception:
            self._conn.mark_dead()
            return False
        if target_epoch == UNSPECIFIED_EPOCH:
            self._last_discovery_send_at = now
        return True

    def _send(self, msg: dict) -> bool:
        if self._conn is None or not self._send_ready:
            logger.debug(
                "P2PSession %s: queueing %s (ready=%s queue_depth=%d)",
                self.peer_id,
                msg.get(TYPE_KEY),
                self._send_ready,
                len(self._queued) + 1,
            )
            self._queued.append(msg)
            return True
        return self._do_send(msg)

    def _do_send(self, msg: dict) -> bool:
        if self._conn is None:
            return False
        if self._remote_epoch is None:
            return False
        source_epoch = msg.setdefault(SOURCE_EPOCH_KEY, self._local_epoch)
        target_epoch = msg.setdefault(TARGET_EPOCH_KEY, self._remote_epoch)
        if source_epoch != self._local_epoch or target_epoch != self._remote_epoch:
            raise RuntimeError("queued message carries a foreign session epoch")
        try:
            self._conn.send(msg)
            logger.debug("P2PSession %s: sent %s", self.peer_id, msg.get(TYPE_KEY))
        except Exception:
            # A send failure means the connection is broken. Swallowing it
            # silently strands every in-flight lookup/load toward this peer:
            # the session stays alive, is never reaped, and the consumer's
            # lookup() keeps returning RETRY until the HTTP client times out.
            # Mark the connection dead so the manager reaps the session on
            # its next poll and surfaces the stranded work as failures.
            logger.warning(
                "P2PSession %s: send of %s failed — marking connection dead",
                self.peer_id,
                msg.get(TYPE_KEY),
            )
            self._conn.mark_dead()
            return False
        return True
