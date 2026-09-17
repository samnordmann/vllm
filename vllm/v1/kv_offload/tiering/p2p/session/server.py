# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server-role state machine for a single peer session.

Owns block matching (supply vs. demand), inflight RDMA transfers, store-job
timeouts, abort-drain, and produces ``StoreResult`` for completed stores.
The session coordinator parses wire messages and dispatches typed
arguments here; this module never touches ``ControlConnection`` directly
— it emits via the ``send`` callback injected by the coordinator (which
gates on ConnectAck).

Protocol violations the role can detect (today: duplicate ``FetchMsg``
for the same ``kv_request_id``) are surfaced as ``ValueError`` so the
coordinator's ``_dispatch_message`` can reuse its existing
``_protocol_error`` path.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, NamedTuple

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LookupResult, OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
    TYPE_KEY,
    AbortAckMsg,
    LookupRespMsg,
    TransferDoneMsg,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobId, ParentManager
    from vllm.v1.kv_offload.tiering.p2p.data import DataTransport

logger = init_logger(__name__)

_STORE_TIMEOUT_S = 30.0
_CANCEL_DRAIN_TIMEOUT_S = 10.0
# Cap on time the server holds a HIT_PENDING / RETRY key from an inbound
# LookupMsg before falling back to MISS. Long enough that an in-flight
# primary write or a just-started promotion typically completes; short
# enough that the consumer doesn't sit idle on a stuck producer.
_LOOKUP_PENDING_TIMEOUT_S = 5.0
# Local-only owner for PD supply that arrived before its FetchMsg exposed the
# peer's session-global operation token. Negative values are invalid on wire.
_UNBOUND_PD_ROUND = -1


class StoreResult(NamedTuple):
    """Result from a session poll, server side."""

    job_id: int
    success: bool


class _MatchResult(NamedTuple):
    """Result of block matching: pairs ready for transfer."""

    local_idxs: list[int]
    remote_idxs: list[int]
    # The set of store job IDs that contributed blocks
    job_ids: set[int]


@dataclass
class _OutboundRequestState:
    """Server-role state for a single fetch round of a peer request.

    A kv_request_id may run several lookup→fetch rounds; rounds live in
    ``_ServerRequestState.outbound`` keyed by the wire ``round_seq``, so
    terminals touch only their own round.
    """

    # Supply came from inbound lookup pins (symmetric): no late
    # submit_store can arrive, so unmatched fetch demand fails fast. PD
    # rounds park demand for stores instead.
    lookup_supplied: bool = False
    demand_received: bool = False
    available: dict[OffloadKey, tuple[int, int]] = field(
        default_factory=dict
    )  # key → (job_id, local_block_idx): blocks we have, awaiting demand
    demanded: dict[OffloadKey, int] = field(
        default_factory=dict
    )  # key → remote_block_idx: blocks peer wants, awaiting supply
    remaining: int = 0  # blocks that need to be transferred to client
    total_blocks: int = 0
    # Successful terminal transfer IDs are the authoritative decrement facts.
    # ``remaining`` is derived from this map after an interruption.
    completed_blocks_by_tid: dict[int, int] = field(default_factory=dict)
    finishing: bool = False  # Signal finish request ASAP
    inflight: int = 0  # transfers submitted for this round, not yet polled
    # Job IDs that submit_store'd blocks for this round and have not
    # yet emitted a StoreResult. The terminal-finalize helper drains
    # this set; poll-done and poll-failed discard entries as their
    # StoreResults fire.
    pending_job_ids: set[int] = field(default_factory=set)
    # A failed round is a tombstone: it never submits more DMA. The
    # failure journal drains every already-submitted transfer before
    # publishing the remaining StoreResults or the wire terminal.
    failed: bool = False
    failure_settled: bool = False
    failure_sent: bool = False
    failure_send_done: bool = True

    def add_stored_blocks(
        self,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: int,
    ) -> _MatchResult:
        """Add locally-stored blocks. Returns matched pairs."""
        self.pending_job_ids.add(job_id)
        local_idxs: list[int] = []
        remote_idxs: list[int] = []
        for key, local_idx in zip(keys, block_ids):
            remote_idx = self.demanded.pop(key, None)
            if remote_idx is not None:
                local_idxs.append(local_idx)
                remote_idxs.append(remote_idx)
            else:
                self.available[key] = (job_id, local_idx)
        return _MatchResult(
            local_idxs=local_idxs,
            remote_idxs=remote_idxs,
            job_ids={job_id} if local_idxs else set(),
        )

    def add_fetch_demand(
        self,
        keys: Sequence[OffloadKey],
        block_indexes: Sequence[int],
    ) -> _MatchResult:
        """Register the peer's fetch demand. Returns matched pairs."""
        self.demand_received = True
        self.total_blocks = len(keys)
        self.remaining = len(keys)

        local_idxs: list[int] = []
        remote_idxs: list[int] = []
        job_ids: set[int] = set()
        for key, remote_idx in zip(keys, block_indexes):
            stored_entry = self.available.pop(key, None)
            if stored_entry is not None:
                stored_job_id, local_idx = stored_entry
                local_idxs.append(local_idx)
                remote_idxs.append(remote_idx)
                job_ids.add(stored_job_id)
            else:
                self.demanded[key] = remote_idx
        return _MatchResult(
            local_idxs=local_idxs,
            remote_idxs=remote_idxs,
            job_ids=job_ids,
        )


@dataclass
class _InflightXfer:
    """Metadata for a single inflight RDMA transfer, keyed by transfer_id."""

    kv_request_id: str
    block_count: int
    # The set of store job IDs that contributed blocks to this transfer.
    job_ids: set[int]
    # Round this transfer serves and its key in ``st.outbound``;
    # remaining/finalize apply only while the round is still registered.
    # Dummy default for test-seeded entries.
    round: _OutboundRequestState = field(default_factory=_OutboundRequestState)
    round_key: int = 0


@dataclass
class _TerminalTransfer:
    """Session adoption receipt for one replayable transport terminal."""

    xfer: _InflightXfer
    outcome: str
    applied: bool = False


@dataclass
class _FinalizingRound:
    """Strong owner across result publication, wire send, and unlink."""

    round: _OutboundRequestState
    success: bool
    send_done: bool
    job_ids: frozenset[int]
    results_published: bool = False
    message_sent: bool = False
    unlinked: bool = False


@dataclass
class _ActiveLookup:
    """In-flight state for one inbound LookupMsg.

    Aggregates per-key HIT/MISS resolutions and defers the single
    outbound LookupRespMsg until every key has been resolved or the
    ``deadline`` fires (remaining ``pending`` keys then force-MISS).
    Exactly one LookupRespMsg is emitted per LookupMsg, carrying every
    key in the original wire order.
    """

    lookup_id: int
    kv_request_id: str
    ctx: ReqContext
    # Wire round these probes belong to; pins park under it.
    round_seq: int = 0
    # Keys from the inbound LookupMsg, preserved in wire order so
    # the aggregated response goes back in the same order.
    keys: list[OffloadKey] = field(default_factory=list)
    # Per-key resolution: True = HIT, False = MISS. A key is present
    # here once definitively resolved; still-pending keys are only
    # in ``pending``.
    resolved: dict[OffloadKey, bool] = field(default_factory=dict)
    # Keys still awaiting resolution (HIT_PENDING / RETRY from
    # ``parent.lookup``); re-polled by ``_resolve_pending_lookups``.
    pending: set[OffloadKey] = field(default_factory=set)
    # Absolute deadline (``time.monotonic``). Once reached, remaining
    # ``pending`` keys are force-resolved to MISS so the consumer
    # can fall back instead of waiting on a stuck producer.
    deadline: float = 0.0


class _PendingLookup(NamedTuple):
    """A raw inbound LookupMsg awaiting resolution.

    Enqueued by ``on_lookup`` during dispatch and drained by the next
    ``serve_external_requests``. The deadline for any resulting
    HIT_PENDING / RETRY key is measured from ``enqueued_at``.
    """

    keys: list[OffloadKey]
    enqueued_at: float
    round_seq: int = 0


@dataclass
class _ServerRequestState:
    """Per-kv_request_id server-side state.

    Consolidates the outbound serve/transfer state, the symmetric-P2P
    inbound-lookup state, abort bookkeeping, and the inflight-transfer
    count for one kv_request_id. The owning id is the ``_requests`` dict
    key and is not duplicated here. An entry is dropped once every field
    is idle — see ``ServerRole._maybe_prune``.
    """

    # Fetch rounds keyed by wire round_seq. PD supply received before Fetch is
    # held under the local-only _UNBOUND_PD_ROUND owner, then rebound to the
    # peer's token. A round is removed at terminal/failure/abort.
    outbound: dict[int, _OutboundRequestState] = field(default_factory=dict)
    # Raw inbound LookupMsgs not yet processed against the ParentManager.
    pending_lookups: list[_PendingLookup] = field(default_factory=list)
    # Per-LookupMsg state parked with HIT_PENDING / RETRY keys, keyed by
    # the (globally unique) lookup_id and re-polled each serve.
    lookups: dict[int, _ActiveLookup] = field(default_factory=dict)
    # Transfer ids in ``ServerRole._inflight`` for this id. Kept in sync
    # via _inflight_add / _inflight_pop so a non-empty set is an exact
    # "has any inflight transfer" predicate and the abort drain can
    # enumerate this request's transfers without scanning all of
    # ``_inflight``.
    inflight_tids: set[int] = field(default_factory=set)


class ServerRole:
    """Server-side store/serve state machine for one peer session.

    The coordinator owns the connection and the send-gating; this role
    is given a ``send`` callback, the ``DataTransport``, and the
    ``peer_id`` for transport calls and log messages.
    """

    def __init__(
        self,
        peer_id: str,
        transport: DataTransport,
        send: Callable[[dict], bool | None],
    ) -> None:
        self._peer_id = peer_id
        self._transport = transport
        self._send = send
        owned_submit = getattr(transport, "write_blocks_owned", None)
        if callable(owned_submit):
            # Cache the bound method once: production submits pay no repeated
            # feature lookup or compatibility branch.
            self._submit_blocks_owned = owned_submit
        else:

            def legacy_submit(
                target_peer: str,
                local_idxs: list[int],
                remote_idxs: list[int],
                *,
                recovery_token: object,
            ) -> int | None:
                del recovery_token
                # Resolve dynamically so test/legacy transports that replace
                # their old entry point retain the historical behavior.
                return transport.write_blocks(target_peer, local_idxs, remote_idxs)

            self._submit_blocks_owned = legacy_submit

        # All per-kv_request_id state lives here. Entries are created
        # lazily and dropped by _maybe_prune once every field is idle.
        self._requests: dict[str, _ServerRequestState] = {}
        # kv_request_ids with lookup work (unprocessed pending_lookups or
        # parked lookups) for the next serve to visit — the work-list that
        # keeps serve_external_requests from scanning every request.
        self._serve_pending: set[str] = set()
        # transfer_id → xfer. Mutate ONLY via _inflight_add / _inflight_pop
        # so the per-request inflight_tids stays in sync.
        self._inflight: dict[int, _InflightXfer] = {}
        self._terminal_transfers: dict[int, _TerminalTransfer] = {}
        # Commit-last owner for the one transfer whose transport return has
        # not yet been durably adopted into the session graph.
        self._submitting_xfer: _InflightXfer | None = None
        self._store_jobs: dict[int, float] = {}  # job_id → submitted_at
        # Jobs whose deadline expired but whose source memory must remain
        # pinned until every transfer in their failed round is quiescent.
        self._timed_out_store_jobs: set[int] = set()
        # Cold-path failure journal. An entry remains until wait-mode
        # cancellation has released every sibling transfer and any required
        # TransferDone failure has been emitted. The failed round itself may
        # remain in ``outbound`` as a tombstone for a late FetchMsg.
        self._failed_rounds: dict[tuple[str, int], _OutboundRequestState] = {}
        # A result remains here until P2PSession acknowledges adoption.
        self._pending_store_results: dict[int, StoreResult] = {}
        self._finalizing_rounds: dict[tuple[str, int], _FinalizingRound] = {}
        # Synthetic lookup ctxs whose ``on_request_finished`` still needs to
        # fire but which were closed outside a serve window (FetchMsg / local
        # finish popped their parked lookup). Drained via
        # ``parent.on_request_finished`` in ``serve_external_requests``.
        self._finished_lookup_ctxs: list[ReqContext] = []
        self._lookup_id_counter: int = 0
        # Parked aborts awaiting drain, keyed by (kv_request_id, round)
        # with the abort start time.
        self._pending_aborts: dict[tuple[str, int], float] = {}
        self._abort_timeout_warned: set[tuple[str, int]] = set()
        # Round identities whose local DMA drain was proven. Retaining this
        # tombstone makes AbortAck replay safe after a lost return or duplicate.
        self._abort_ack_intents: set[tuple[str, int]] = set()
        # One durable close journal publishes results and the exact cancel set
        # in a single assignment before any state is cleared or native code is
        # entered. Unknown transfer IDs are idempotent in DataTransport.cancel,
        # so retrying this set is safe after an outcome-ambiguous interruption.
        self._close_journal: (
            tuple[list[int], list[ReqContext], tuple[int, ...]] | None
        ) = None
        self._close_complete = False

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _get_or_create_request(self, kv_request_id: str) -> _ServerRequestState:
        """Get or create the state entry for a kv_request_id."""
        st = self._requests.get(kv_request_id)
        if st is None:
            st = _ServerRequestState()
            self._requests[kv_request_id] = st
        return st

    def _maybe_prune(self, kv_request_id: str) -> None:
        """Drop the entry once it holds no live state."""
        st = self._requests.get(kv_request_id)
        if (
            st is not None
            and not st.outbound
            and not st.inflight_tids
            and not st.lookups
            and not st.pending_lookups
            and not any(kv == kv_request_id for kv, _ in self._pending_aborts)
            and (
                self._submitting_xfer is None
                or self._submitting_xfer.kv_request_id != kv_request_id
            )
        ):
            del self._requests[kv_request_id]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_stored_blocks(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_ids: Sequence[int],
        job_id: JobId,
        round_seq: int | None = None,
        *,
        from_lookup: bool = False,
    ) -> None:
        """New blocks stored locally — match within their fetch round.

        Lookup pins carry their explicit operation token. A PD submit_store has
        no token in the manager API: it binds to the unique matching demanded
        round, or remains under a local unbound owner until Fetch supplies one.
        """
        self._reconcile_and_drain_submitting_transfer()
        st = self._get_or_create_request(kv_request_id)
        if round_seq is None:
            matched_round: int | None = None
            sole_round: int | None = None
            round_count = 0
            for candidate_seq, candidate in st.outbound.items():
                if candidate_seq == _UNBOUND_PD_ROUND or candidate.lookup_supplied:
                    continue
                round_count += 1
                sole_round = candidate_seq
                if any(key in candidate.demanded for key in keys):
                    if matched_round is not None and matched_round != candidate_seq:
                        raise RuntimeError(
                            "PD store batch matches multiple fetch generations"
                        )
                    matched_round = candidate_seq
            if matched_round is not None:
                round_seq = matched_round
            elif round_count == 1:
                round_seq = sole_round
            elif round_count > 1:
                raise RuntimeError(
                    "PD store batch has ambiguous fetch-generation ownership"
                )
            else:
                round_seq = _UNBOUND_PD_ROUND
        assert round_seq is not None
        rnd = st.outbound.get(round_seq)
        if rnd is not None and rnd.failed:
            # A prior timeout/failure may already have released older source
            # slots. Keep the round as a tombstone so a late producer batch
            # cannot resurrect it and DMA from recycled memory.
            self._publish_store_result(job_id, False, rnd)
            logger.warning(
                "P2PSession %s: rejecting store job %d for failed "
                "kv_request_id=%s round=%s",
                self._peer_id,
                job_id,
                kv_request_id,
                round_seq,
            )
            return
        self._store_jobs[job_id] = time.monotonic()
        if rnd is None:
            rnd = st.outbound[round_seq] = _OutboundRequestState()
        if from_lookup:
            rnd.lookup_supplied = True
        result = rnd.add_stored_blocks(keys, block_ids, job_id)
        if result.local_idxs and rnd.demand_received:
            self._submit_transfer(kv_request_id, result, rnd, round_seq)

    def owns_store_job(self, job_id: JobId) -> bool:
        """Whether this role still owns or has journaled ``job_id``."""
        return job_id in self._store_jobs or job_id in self._pending_store_results

    def on_fetch(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        block_indexes: Sequence[int],
        round_seq: int = 0,
    ) -> None:
        """Handle a FetchMsg from the peer.

        A non-empty fetch binds and closes its round, leaving lookup
        state alone (the next round's LookupMsg may already be in
        flight). The terminal empty fetch closes the id: parked lookups
        are popped and every remaining round drained. A second fetch for
        a round already holding demand raises ValueError
        (protocol-error disconnect).
        """
        self._reconcile_and_drain_submitting_transfer()
        logger.debug(
            "P2PSession %s: fetch RECEIVED kv_request_id=%s round=%s blocks=%d",
            self._peer_id,
            kv_request_id,
            round_seq,
            len(keys),
        )
        st = self._requests.get(kv_request_id)
        existing = st.outbound.get(round_seq) if st is not None else None
        if existing is not None and existing.demand_received:
            raise ValueError(
                f"duplicate fetch for kv_request_id={kv_request_id} round={round_seq}"
            )
        st = self._get_or_create_request(kv_request_id)
        req = st.outbound.get(round_seq)
        if req is None:
            # PD supply can precede Fetch and therefore cannot know the peer's
            # session-global token. Publish the wire-key owner before removing
            # its local placeholder so an interrupted re-entry converges.
            req = st.outbound.get(_UNBOUND_PD_ROUND)
            if req is None:
                req = _OutboundRequestState()
            st.outbound[round_seq] = req
        unbound = st.outbound.get(_UNBOUND_PD_ROUND)
        if unbound is not None and unbound is req:
            st.outbound.pop(_UNBOUND_PD_ROUND, None)
        result = req.add_fetch_demand(keys, block_indexes)
        if not keys:
            # Terminal empty fetch: close the lookup phase and drain
            # every round with no TransferDoneMsg (nothing waits on it).
            self._finish_inbound_lookups(kv_request_id)
            for key, active_round in list(st.outbound.items()):
                if active_round.inflight or active_round.failed:
                    self._mark_round_failed(
                        kv_request_id,
                        key,
                        active_round,
                        send_done=False,
                    )
                else:
                    self._finalize_outbound(kv_request_id, key, send_done=False)
            self._drain_failed_rounds()
            return
        if req.failed:
            # The tombstone may predate demand. Record that a terminal is now
            # owed, but let the failure journal send it only after all prior
            # DMA for this round is known quiescent.
            req.demanded.clear()
            self._failed_rounds[(kv_request_id, round_seq)] = req
            return
        if req.lookup_supplied and req.demanded:
            # A symmetric round's supply always precedes its fetch, so
            # unmatched demand is unservable — fail now, not at the load
            # timeout. PD rounds keep parking demand for stores that
            # arrive later.
            logger.warning(
                "P2PSession %s: fetch kv_request_id=%s round=%s demanded %d "
                "blocks but %d have no pinned supply; failing fetch "
                "immediately",
                self._peer_id,
                kv_request_id,
                round_seq,
                len(keys),
                len(req.demanded),
            )
            self._finalize_outbound(kv_request_id, round_seq, success=False)
            return
        if result.local_idxs:
            self._submit_transfer(kv_request_id, result, req, round_seq)
        # Prefiller-first mode: finish_request may have run before
        # fetch arrived. If so, finalize once we know what was
        # demanded — fully satisfied → success, else early-fail.
        if req.finishing and req.inflight == 0:
            self._finalize_outbound(kv_request_id, round_seq)

    def on_abort_fetch(self, kv_request_id: str, round_seq: int = 0) -> None:
        """Handle an AbortFetchMsg from the peer, cancelling one round."""
        self._reconcile_and_drain_submitting_transfer()
        abort_key = (kv_request_id, round_seq)
        if abort_key in self._abort_ack_intents:
            accepted = self._send(
                {
                    TYPE_KEY: AbortAckMsg.TYPE,
                    AbortAckMsg.KV_REQUEST_ID: kv_request_id,
                    AbortAckMsg.ROUND_SEQ: round_seq,
                }
            )
            if accepted is not False:
                self._pending_aborts.pop(abort_key, None)
                self._abort_timeout_warned.discard(abort_key)
                self._maybe_prune(kv_request_id)
            return
        # Abort for an unknown id may be a benign race/duplicate or a
        # real protocol violation; we don't track completed ids, so warn.
        st = self._requests.get(kv_request_id)
        if (st is None or not st.outbound) and not self._has_inflight_for(
            kv_request_id
        ):
            logger.warning(
                "P2PSession %s: abort_fetch for unknown kv_request_id=%s "
                "(no outbound or inflight state); benign race or stale",
                self._peer_id,
                kv_request_id,
            )
        # Idempotent: receiving AbortFetchMsg again before we've sent the
        # ack just triggers another drain attempt without resetting the
        # deadline.
        self._get_or_create_request(kv_request_id)
        self._pending_aborts.setdefault((kv_request_id, round_seq), time.monotonic())
        self._drain_abort(kv_request_id, round_seq)

    def on_lookup(
        self,
        kv_request_id: str,
        keys: Sequence[OffloadKey],
        round_seq: int = 0,
    ) -> None:
        """Enqueue a LookupMsg from a symmetric-P2P consumer.

        Dispatch runs during ``session.poll()`` where the
        :class:`ParentManager` handle is not available, so this only
        records the raw request. It is resolved — querying the tiering
        manager and emitting the aggregated ``LookupRespMsg`` — by the
        next :meth:`serve_external_requests`, the sole window in which
        parent calls are valid.
        """
        logger.debug(
            "P2P LOOKUP server %s: RECV LookupMsg kv_request_id=%s round=%s keys=%d",
            self._peer_id,
            kv_request_id,
            round_seq,
            len(keys),
        )
        self._get_or_create_request(kv_request_id).pending_lookups.append(
            _PendingLookup(
                keys=list(keys),
                enqueued_at=time.monotonic(),
                round_seq=round_seq,
            )
        )
        self._serve_pending.add(kv_request_id)

    def serve_external_requests(self, parent: ParentManager) -> None:
        """Resolve inbound peer lookups against the tiering manager.

        Called once per scheduler step with a ``parent`` handle valid
        only for this call. Drains newly-enqueued LookupMsgs, re-polls
        any parked HIT_PENDING / RETRY keys, and releases the
        bookkeeping for lookups closed since the last serve.
        """
        for kv_request_id in list(self._serve_pending):
            st = self._requests.get(kv_request_id)
            if st is None:
                self._serve_pending.discard(kv_request_id)
                continue
            if st.pending_lookups:
                pending = st.pending_lookups
                st.pending_lookups = []
                for pl in pending:
                    self._process_inbound_lookup(
                        kv_request_id, pl.keys, pl.enqueued_at, pl.round_seq, parent
                    )
            self._resolve_pending_lookups(kv_request_id, parent)
            st = self._requests.get(kv_request_id)
            if st is None or (not st.pending_lookups and not st.lookups):
                self._serve_pending.discard(kv_request_id)
                self._maybe_prune(kv_request_id)

        if self._finished_lookup_ctxs:
            for ctx in self._finished_lookup_ctxs:
                parent.on_request_finished(ctx)
            self._finished_lookup_ctxs = []

    def _poll_lookup_keys(
        self,
        lookup: _ActiveLookup,
        keys: Iterable[OffloadKey],
        parent: ParentManager,
    ) -> list[OffloadKey]:
        """Poll ``keys`` against the tiering manager and pin any HITs.

        For each key not already definitively resolved, query
        ``parent.lookup`` and record the outcome on ``lookup``: HIT / MISS
        land in ``resolved`` and clear ``pending``; HIT_PENDING / RETRY park
        in ``pending`` for a later serve. Newly-HIT keys are pinned in one
        batch via :meth:`_pin_and_register_hits` and also returned (for
        caller logging).

        Shared by the first-sighting pass (:meth:`_process_inbound_lookup`)
        and the re-poll pass (:meth:`_resolve_pending_lookups`); callers pass
        a de-duplicated ``keys`` collection.
        """
        new_hits: list[OffloadKey] = []
        for h in keys:
            if h in lookup.resolved:
                continue
            result = parent.lookup(h, lookup.ctx)
            if result is LookupResult.HIT:
                new_hits.append(h)
                lookup.resolved[h] = True
                lookup.pending.discard(h)
            elif result is LookupResult.MISS:
                lookup.resolved[h] = False
                lookup.pending.discard(h)
            else:
                lookup.pending.add(h)
        if new_hits:
            self._pin_and_register_hits(lookup, new_hits, parent)
        return new_hits

    def _process_inbound_lookup(
        self,
        kv_request_id: str,
        keys: list[OffloadKey],
        enqueued_at: float,
        round_seq: int,
        parent: ParentManager,
    ) -> None:
        """Resolve one enqueued LookupMsg against ``parent``.

        For each key, query the tiering manager via ``parent.lookup``
        and pin any HITs immediately via ``parent.create_store_job``
        (plumbed into the existing ``add_stored_blocks`` matching path so
        the eventual FetchMsg finds them). HIT_PENDING / RETRY keys park
        in :class:`_ActiveLookup` for re-polling by
        :meth:`_resolve_pending_lookups`.

        The outbound ``LookupRespMsg`` is deferred until every key has
        settled to HIT or MISS (or the batch-level ``deadline`` fires,
        forcing any stragglers to MISS). One LookupRespMsg goes out per
        LookupMsg — carrying every key in wire order — after which
        ``parent.on_request_finished`` fires and the entry is dropped.
        """
        self._lookup_id_counter += 1
        lookup_id = self._lookup_id_counter
        ctx = ReqContext(req_id=f"p2p:{self._peer_id}:{kv_request_id}:lu{lookup_id}")
        lookup = _ActiveLookup(
            lookup_id=lookup_id,
            kv_request_id=kv_request_id,
            ctx=ctx,
            round_seq=round_seq,
            keys=list(keys),
            deadline=enqueued_at + _LOOKUP_PENDING_TIMEOUT_S,
        )

        # Open per-request bookkeeping for this synthetic ctx before the
        # first lookup; released by ``on_request_finished`` once every
        # key has settled.
        parent.on_new_request(ctx)

        # dict.fromkeys de-duplicates keys within the LookupMsg while
        # preserving wire order — each unique key is polled once.
        hit_keys = self._poll_lookup_keys(lookup, dict.fromkeys(lookup.keys), parent)

        logger.debug(
            "P2P LOOKUP server %s: RESOLVED kv_request_id=%s hits=%d misses=%d "
            "pending=%d",
            self._peer_id,
            kv_request_id,
            len(hit_keys),
            sum(1 for v in lookup.resolved.values() if not v),
            len(lookup.pending),
        )

        if lookup.pending:
            self._get_or_create_request(kv_request_id).lookups[lookup_id] = lookup
        else:
            # Every key resolved on first sight — emit the aggregated
            # response now and close the synthetic request.
            self._finalize_lookup(lookup, parent)

    def _pin_and_register_hits(
        self,
        lookup: _ActiveLookup,
        keys: list[OffloadKey],
        parent: ParentManager,
    ) -> None:
        """Pin primary slots for HIT keys and park them as the lookup's
        round supply via ``add_stored_blocks``.

        Caller has already confirmed every key is HIT (single-threaded
        scheduler ⇒ no eviction race), so the JobMetadata returned by
        ``parent.create_store_job`` carries parallel ``keys``/``block_ids``
        of length ``len(keys)``.
        """
        meta = parent.create_store_job(keys, lookup.ctx)
        self.add_stored_blocks(
            lookup.kv_request_id,
            list(meta.keys),
            meta.block_ids.tolist(),
            meta.job_id,
            round_seq=lookup.round_seq,
            from_lookup=True,
        )

    def _resolve_pending_lookups(
        self, kv_request_id: str, parent: ParentManager
    ) -> None:
        """Re-poll a request's deferred LookupMsg keys; finalize when ready.

        Walks every parked :class:`_ActiveLookup` for ``kv_request_id`` and
        re-calls ``parent.lookup`` per still-pending key, moving HIT/MISS
        results into ``resolved``. If ``deadline`` has passed, remaining
        ``pending`` keys are force-resolved to MISS so the consumer
        can fall back instead of waiting on a stuck producer. Newly-HIT
        keys are pinned via ``parent.create_store_job`` in one call per
        affected lookup. Lookups whose ``pending`` empties out get their
        aggregated LookupRespMsg sent by :meth:`_finalize_lookup`.
        """
        st = self._requests.get(kv_request_id)
        if st is None or not st.lookups:
            return
        now = time.monotonic()
        finished_lookups: list[int] = []
        for lookup_id, lookup in st.lookups.items():
            self._poll_lookup_keys(lookup, list(lookup.pending), parent)

            if lookup.pending and now >= lookup.deadline:
                for h in lookup.pending:
                    lookup.resolved[h] = False
                lookup.pending.clear()

            if not lookup.pending:
                finished_lookups.append(lookup_id)

        for lookup_id in finished_lookups:
            lookup = st.lookups.pop(lookup_id)
            self._finalize_lookup(lookup, parent)

    def _finalize_lookup(self, lookup: _ActiveLookup, parent: ParentManager) -> None:
        """Emit the aggregated LookupRespMsg and close the synthetic request.

        Called once per lookup when ``pending`` is empty — either every
        key resolved to HIT or MISS, or the deadline forced remaining
        stragglers to MISS. Preserves the wire order of the inbound
        LookupMsg so the client can zip keys and hits positionally.
        """
        if lookup.keys:
            hits = [lookup.resolved[h] for h in lookup.keys]
            n_hit = sum(1 for v in hits if v)
            logger.debug(
                "P2P LOOKUP server %s: SEND LookupRespMsg kv_request_id=%s "
                "keys=%d hits=%d misses=%d",
                self._peer_id,
                lookup.kv_request_id,
                len(lookup.keys),
                n_hit,
                len(hits) - n_hit,
            )
            self._send(
                {
                    TYPE_KEY: LookupRespMsg.TYPE,
                    LookupRespMsg.KV_REQUEST_ID: lookup.kv_request_id,
                    LookupRespMsg.KEYS: list(lookup.keys),
                    LookupRespMsg.HITS: hits,
                    LookupRespMsg.ROUND_SEQ: lookup.round_seq,
                }
            )
        parent.on_request_finished(lookup.ctx)

    def _finish_inbound_lookups(self, kv_request_id: str) -> None:
        """Close the server-side lookup phase for ``kv_request_id``.

        Pops every parked ``_ActiveLookup`` for this id (so
        ``_resolve_pending_lookups`` cannot promote a HIT_PENDING /
        RETRY key into a fresh ``parent.create_store_job`` after this
        point) and queues each ``lookup.ctx`` for
        ``parent.on_request_finished`` (fired by the next
        ``serve_external_requests``, since no parent handle is available
        during dispatch) so the TieringManager can release per-lookup
        bookkeeping. Any still-unprocessed raw LookupMsg for this id is
        dropped — it never got ``on_new_request``, so nothing is owed.
        The aggregated LookupRespMsg is skipped — the client already
        knows the request is over (it just sent a terminal FetchMsg, or
        is finishing locally).

        Called on the two events that mean "no more lookup traffic for
        ``kv_request_id`` is expected on this session": the terminal
        empty FetchMsg from the peer and a local ``finish``. Whichever
        fires second is a no-op.
        """
        st = self._requests.get(kv_request_id)
        if st is None:
            return
        st.pending_lookups.clear()
        for lookup in st.lookups.values():
            self._finished_lookup_ctxs.append(lookup.ctx)
        st.lookups.clear()
        self._serve_pending.discard(kv_request_id)
        self._maybe_prune(kv_request_id)

    def finish(self, kv_request_id: str) -> None:
        """Mark an outbound request finishing.

        No more submit_store calls will arrive for this id. Any blocks
        the peer demanded but we never stored will never come; tell the
        peer to stop waiting (TransferDoneMsg success=False) instead of
        letting it hit _LOAD_TIMEOUT_S.

        Also drops any in-flight lookups for this kv_request_id and
        queues their ctxs for ``parent.on_request_finished`` so the
        TieringManager can release per-request bookkeeping. (For
        symmetric P2P this path is rarely hit since the producer has no
        local request lifecycle for the consumer's id; this cleanup is
        mostly active on the PD side.)

        If the decoder hasn't sent fetch yet (no demand received),
        defer — on_fetch will finalize once demand arrives.

        If inflight transfers exist for this id, defer — the last
        completing transfer in collect_results will fire the message.
        """
        self._reconcile_and_drain_submitting_transfer()
        self._finish_inbound_lookups(kv_request_id)

        st = self._requests.get(kv_request_id)
        if st is None:
            return
        for key, req in list(st.outbound.items()):
            req.finishing = True
            if req.failed:
                self._failed_rounds[(kv_request_id, key)] = req
                continue
            if not req.demand_received or req.inflight:
                # No demand yet (prefiller-first): on_fetch finalizes via
                # `finishing`. Inflight: the last completion finalizes.
                continue
            self._finalize_outbound(kv_request_id, key)

    def collect_results(self) -> list[StoreResult]:
        """Drain timeouts, deferred results, and transport completions.

        Inbound LookupMsg resolution (including re-polling HIT_PENDING /
        RETRY keys) is NOT done here — it runs in
        ``serve_external_requests`` where the ParentManager is available.
        """
        self._reconcile_and_drain_submitting_transfer()
        if self._finalizing_rounds:
            self._resume_finalizing_rounds()
        self._timeout_pending_store_jobs()
        # Scope the poll to this peer: the transport is shared across all peer
        # sessions of the engine, and poll() drains completed handles. An
        # unscoped poll here would consume sibling sessions' completions and
        # report them as "unknown transfer_id", starving those sessions.
        poll_result = self._transport.poll(self._peer_id)

        # Failure dominates success for a round. Publish every transport
        # terminal receipt before changing session ownership; replay after any
        # cut therefore sees the same xfer and outcome.
        for tid in poll_result.failed:
            xfer = self._inflight.get(tid)
            if xfer is not None:
                self._mark_round_failed(
                    xfer.kv_request_id,
                    xfer.round_key,
                    xfer.round,
                )
        for outcome, transfer_ids in (
            ("done", poll_result.done),
            ("failed", poll_result.failed),
        ):
            for tid in transfer_ids:
                self._adopt_transport_terminal(tid, outcome)
        self._drain_terminal_transfers()

        if self._failed_rounds:
            self._drain_failed_rounds()
        return list(self._pending_store_results.values())

    def collect_idle_timeouts(self) -> list[StoreResult]:
        """Run only the store-job timeout sweep.

        Used by the coordinator's no-conn poll path: a pending session
        cannot have inflight transfers (no peer registered yet), so we
        skip the transport poll and the deferred-result drain.
        """
        self._reconcile_and_drain_submitting_transfer()
        if self._finalizing_rounds:
            self._resume_finalizing_rounds()
        self._timeout_pending_store_jobs()
        return list(self._pending_store_results.values())

    def ack_results(self, job_ids: Sequence[int]) -> None:
        """Release results already adopted by P2PSession."""
        for job_id in job_ids:
            if type(job_id) is not int:
                raise TypeError("job_id must be an exact int")
            self._pending_store_results.pop(job_id, None)
        for finalizer_key in tuple(self._finalizing_rounds):
            self._maybe_retire_finalizer(finalizer_key)

    def drain_pending_aborts(self) -> None:
        """Re-attempt every parked abort once per poll tick."""
        self._reconcile_and_drain_submitting_transfer()
        for kv_request_id, round_seq in list(self._pending_aborts):
            self._drain_abort(kv_request_id, round_seq)

    def close(self) -> tuple[list[int], list[ReqContext]]:
        """Tear down. Cancels inflight.

        Returns ``(failed_store_job_ids, failed_serves)`` where
        ``failed_serves`` are synthetic lookup ctxs still owing a
        ``parent.on_request_finished``. The session is going away with no
        parent handle in hand, so the manager flushes these in its next
        ``serve_external_requests``.
        """
        # Close's own cancellation journal must see every recovered ID before
        # any failed result can be published.
        self._reconcile_submitting_transfer()
        self._drain_terminal_transfers()
        if self._close_journal is None:
            failed_stores = list(self._store_jobs.keys())
            # Surface every synthetic ctx still owing on_request_finished so
            # the manager can release the TieringManager's per-request
            # bookkeeping: parked lookups plus any already queued from a
            # FetchMsg / finish that closed them before this teardown.
            failed_serves = [
                lu.ctx for st in self._requests.values() for lu in st.lookups.values()
            ]
            failed_serves.extend(self._finished_lookup_ctxs)
            self._close_journal = (
                failed_stores,
                failed_serves,
                tuple(self._inflight),
            )

        failed_stores, failed_serves, cancel_ids = self._close_journal
        still_inflight = (
            self._transport.cancel(cancel_ids, mode="wait") if cancel_ids else []
        )
        # Publish the narrowed retry set before examining it. If cancellation
        # committed but an asynchronous exception arrived before this
        # assignment, retrying the original IDs remains safe by contract.
        self._close_journal = (
            failed_stores,
            failed_serves,
            tuple(still_inflight),
        )
        if still_inflight:
            return failed_stores, failed_serves

        self._store_jobs.clear()
        self._timed_out_store_jobs.clear()
        self._inflight.clear()
        self._failed_rounds.clear()
        self._requests.clear()
        self._serve_pending.clear()
        self._pending_aborts.clear()
        self._abort_timeout_warned.clear()
        self._abort_ack_intents.clear()
        self._finished_lookup_ctxs.clear()
        self._close_complete = True
        return failed_stores, failed_serves

    @property
    def close_complete(self) -> bool:
        """True only after every transport transfer is quiescent."""
        return self._close_complete

    @property
    def has_inflight_transfers(self) -> bool:
        """True if any outbound store transfer is still in flight."""
        return (
            bool(self._inflight)
            or bool(self._terminal_transfers)
            or self._submitting_xfer is not None
        )

    # ------------------------------------------------------------------
    # Internal — inflight bookkeeping
    # ------------------------------------------------------------------

    def _has_inflight_for(self, kv_request_id: str) -> bool:
        st = self._requests.get(kv_request_id)
        guarded = self._submitting_xfer
        return (st is not None and bool(st.inflight_tids)) or (
            guarded is not None and guarded.kv_request_id == kv_request_id
        )

    def _inflight_add(self, tid: int, xfer: _InflightXfer) -> None:
        """Insert an inflight transfer and record it on the request."""
        if type(tid) is not int:
            raise TypeError("transport transfer_id must be an exact int")
        if tid in self._inflight and self._inflight[tid] is not xfer:
            raise RuntimeError(f"duplicate transport transfer_id: {tid}")
        self._inflight[tid] = xfer
        self._get_or_create_request(xfer.kv_request_id).inflight_tids.add(tid)

    def reconcile_submitting_transfer(self) -> None:
        """Recover an interrupted transport return without releasing pins.

        This deliberately does not drain the resulting failed-round journal:
        manager shutdown needs the recovered primary IDs for its own bounded
        cancellation snapshot. Normal session entry points use the private
        reconcile-and-drain wrapper below.
        """
        self._reconcile_submitting_transfer()

    def _reconcile_and_drain_submitting_transfer(self) -> None:
        self._reconcile_submitting_transfer()
        if self._failed_rounds:
            self._drain_failed_rounds()

    def _reconcile_submitting_transfer(self) -> None:
        """Finish or conservatively fail the commit-last submit transaction.

        ``_submitting_xfer`` is the exact identity token shared with the data
        transport. The primary session map is authoritative; its request set
        and round counter are reconstructed derivatives. A missing transport
        recovery hook, ambiguous ownership, invalid ID, or collision raises
        while leaving the guard and every source pin intact.
        """
        xfer = self._submitting_xfer
        if xfer is None:
            return

        session_matches = [
            tid for tid, candidate in self._inflight.items() if candidate is xfer
        ]
        if len(session_matches) > 1:
            raise RuntimeError("submission guard has multiple session transfer IDs")

        if session_matches:
            transfer_id: int | None = session_matches[0]
        else:
            recover = getattr(self._transport, "recover_transfer_id", None)
            if not callable(recover):
                raise RuntimeError(
                    "data transport does not support transfer return recovery"
                )
            transfer_id = recover(self._peer_id, xfer)

        if transfer_id is not None and type(transfer_id) is not int:
            raise TypeError("recovered transport transfer_id must be an exact int")
        if transfer_id is not None:
            existing = self._inflight.get(transfer_id)
            if existing is not None and existing is not xfer:
                raise RuntimeError(
                    f"recovered transport transfer_id collision: {transfer_id}"
                )
            # Publish/repair the authoritative owner before any derived state.
            self._inflight[transfer_id] = xfer

        st = self._get_or_create_request(xfer.kv_request_id)
        round_owner = st.outbound.get(xfer.round_key)
        if round_owner is None:
            st.outbound[xfer.round_key] = xfer.round
        elif round_owner is not xfer.round:
            raise RuntimeError(
                "submission guard conflicts with outbound round owner: "
                f"kv_request_id={xfer.kv_request_id} round={xfer.round_key}"
            )

        # These two structures are indexes, not independent ownership. Rebuild
        # them from the primary map so every interrupted adoption cut converges.
        st.inflight_tids = {
            tid
            for tid, candidate in self._inflight.items()
            if candidate.kv_request_id == xfer.kv_request_id
        }
        xfer.round.inflight = sum(
            candidate.round is xfer.round for candidate in self._inflight.values()
        )

        # An interrupted attempt is failed conservatively even when its request
        # was recovered: wait-cancel must prove quiescence before source reuse.
        self._mark_round_failed(
            xfer.kv_request_id,
            xfer.round_key,
            xfer.round,
        )
        # Commit last. Any BaseException before this store leaves a retryable,
        # strongly owned token and therefore cannot expose source memory.
        self._submitting_xfer = None

    def _inflight_pop(self, tid: int) -> _InflightXfer | None:
        """Pop an inflight transfer and drop it from the request's set.

        Callers are responsible for the ``_maybe_prune`` that may follow once
        the request's other state has also cleared.
        """
        xfer = self._inflight.pop(tid, None)
        if xfer is None:
            return None
        xfer.round.inflight -= 1
        assert xfer.round.inflight >= 0
        st = self._requests.get(xfer.kv_request_id)
        if st is not None:
            st.inflight_tids.discard(tid)
        return xfer

    def _settle_xfer_jobs(
        self, xfer: _InflightXfer, success: bool
    ) -> list[StoreResult]:
        """Emit StoreResults for a completed transfer's store jobs.

        Pops each attached job from ``_store_jobs`` and clears it from
        its round's pending set. A job already popped (via timeout,
        cancel, etc.) is skipped so we never double-emit a contradictory
        result.
        """
        results: list[StoreResult] = []
        for job_id in xfer.job_ids:
            if (
                job_id not in self._store_jobs
                and job_id not in xfer.round.pending_job_ids
            ):
                continue
            results.append(self._publish_store_result(job_id, success, xfer.round))
        return results

    # ------------------------------------------------------------------
    # Internal — finalize / abort drain
    # ------------------------------------------------------------------

    def _fail_round_jobs(self, rnd: _OutboundRequestState) -> list[StoreResult]:
        """Fail a terminated round's still-pending store jobs (idempotent)."""
        results: list[StoreResult] = []
        for job_id in tuple(rnd.pending_job_ids):
            results.append(self._publish_store_result(job_id, False, rnd))
        return results

    def _publish_store_result(
        self,
        job_id: int,
        success: bool,
        rnd: _OutboundRequestState | None = None,
    ) -> StoreResult:
        """Publish the outcome before destructively unlinking its job."""
        result = StoreResult(job_id=job_id, success=success)
        existing = self._pending_store_results.setdefault(job_id, result)
        if existing != result:
            raise RuntimeError(f"conflicting terminal outcomes for store job {job_id}")
        self._store_jobs.pop(job_id, None)
        self._timed_out_store_jobs.discard(job_id)
        if rnd is not None:
            rnd.pending_job_ids.discard(job_id)
        return result

    def _adopt_transport_terminal(self, tid: int, outcome: str) -> None:
        existing = self._terminal_transfers.get(tid)
        if existing is not None:
            if existing.outcome != outcome:
                raise RuntimeError(f"conflicting outcomes for transfer {tid}")
            return
        xfer = self._inflight.get(tid)
        if xfer is None:
            logger.error(
                "P2PSession %s: transport reported %s for unknown transfer_id=%d",
                self._peer_id,
                outcome,
                tid,
            )
            return
        self._terminal_transfers[tid] = _TerminalTransfer(xfer, outcome)

    def _rebuild_inflight_state(self, xfer: _InflightXfer) -> None:
        """Derive request and round indexes from the authoritative owner map."""
        st = self._requests.get(xfer.kv_request_id)
        if st is not None:
            st.inflight_tids = {
                tid
                for tid, candidate in self._inflight.items()
                if candidate.kv_request_id == xfer.kv_request_id
            }
        xfer.round.inflight = sum(
            candidate.round is xfer.round for candidate in self._inflight.values()
        )

    def _drain_terminal_transfers(self) -> None:
        for tid, receipt in tuple(self._terminal_transfers.items()):
            xfer = receipt.xfer
            rnd = xfer.round
            if not receipt.applied:
                if receipt.outcome != "done":
                    self._mark_round_failed(xfer.kv_request_id, xfer.round_key, rnd)
                owner = self._inflight.get(tid)
                if owner is not None and owner is not xfer:
                    raise RuntimeError("terminal transfer changed session owner")
                self._inflight.pop(tid, None)
                self._rebuild_inflight_state(xfer)

                if receipt.outcome == "done" and not rnd.failed:
                    existing_count = rnd.completed_blocks_by_tid.setdefault(
                        tid, xfer.block_count
                    )
                    if existing_count != xfer.block_count:
                        raise RuntimeError("terminal transfer changed block count")
                    rnd.remaining = rnd.total_blocks - sum(
                        rnd.completed_blocks_by_tid.values()
                    )
                    if rnd.remaining < 0:
                        raise RuntimeError("completed blocks exceed fetch demand")
                    self._settle_xfer_jobs(xfer, success=True)
                    st = self._requests.get(xfer.kv_request_id)
                    owned = st is not None and st.outbound.get(xfer.round_key) is rnd
                    if owned and rnd.remaining == 0:
                        self._finalize_outbound(
                            xfer.kv_request_id, xfer.round_key, success=True
                        )
                    elif owned and rnd.finishing and rnd.inflight == 0:
                        self._finalize_outbound(
                            xfer.kv_request_id, xfer.round_key, success=False
                        )
                else:
                    self._settle_xfer_jobs(xfer, success=False)
                receipt.applied = True

            self._transport.ack_completions(self._peer_id, (tid,))
            self._terminal_transfers.pop(tid, None)
            self._maybe_retire_finalizer((xfer.kv_request_id, xfer.round_key))
            self._maybe_prune(xfer.kv_request_id)

    def _finalize_outbound(
        self,
        kv_request_id: str,
        round_key: int,
        success: bool | None = None,
        send_done: bool = True,
    ) -> None:
        """Pop one round and emit its terminal results.

        If ``success`` is None, derive it from ``req.remaining == 0``.
        ``send_done=False`` skips the TransferDoneMsg (terminal empty
        fetch). Other rounds of the id are untouched.
        """
        finalizer_key = (kv_request_id, round_key)
        finalizer = self._finalizing_rounds.get(finalizer_key)
        if finalizer is None:
            st = self._requests[kv_request_id]
            req = st.outbound[round_key]
        else:
            req = finalizer.round
            st = self._requests.get(kv_request_id)
        guarded = self._submitting_xfer
        if (
            req.inflight
            or any(xfer.round is req for xfer in self._inflight.values())
            or (guarded is not None and guarded.round is req)
        ):
            raise RuntimeError(
                "refusing to finalize outbound round while DMA is still "
                f"active: kv_request_id={kv_request_id} round={round_key}"
            )
        if req.failed and not req.failure_settled:
            raise RuntimeError(
                "refusing to finalize failed outbound round before its "
                f"quiescence journal settles: kv_request_id={kv_request_id} "
                f"round={round_key}"
            )
        if success is None:
            success = req.demand_received and req.remaining == 0
        if finalizer is None:
            finalizer = _FinalizingRound(
                round=req,
                success=success,
                send_done=send_done,
                job_ids=frozenset(req.pending_job_ids),
            )
            self._finalizing_rounds[finalizer_key] = finalizer
        elif finalizer.success != success or finalizer.send_done != send_done:
            raise RuntimeError("outbound finalization outcome changed during retry")

        if not finalizer.results_published:
            for job_id in tuple(req.pending_job_ids):
                self._publish_store_result(job_id, success, req)
            finalizer.results_published = True
        logger.debug(
            "P2PSession %s: finalize kv_request_id=%s round=%s success=%s "
            "remaining=%d leftover_available=%d send_done=%s",
            self._peer_id,
            kv_request_id,
            round_key,
            success,
            req.remaining,
            len(req.available),
            send_done,
        )
        if send_done and req.demand_received and not finalizer.message_sent:
            self._send(
                {
                    TYPE_KEY: TransferDoneMsg.TYPE,
                    TransferDoneMsg.KV_REQUEST_ID: kv_request_id,
                    TransferDoneMsg.SUCCESS: success,
                    TransferDoneMsg.ROUND_SEQ: round_key,
                }
            )
            finalizer.message_sent = True
        if not send_done or not req.demand_received:
            finalizer.message_sent = True
        if not finalizer.unlinked:
            current = st.outbound.get(round_key) if st is not None else None
            if current is not None and current is not req:
                raise RuntimeError("outbound round changed during finalization")
            if current is req:
                del st.outbound[round_key]
            finalizer.unlinked = True
        self._maybe_prune(kv_request_id)
        self._maybe_retire_finalizer(finalizer_key)

    def _maybe_retire_finalizer(self, finalizer_key: tuple[str, int]) -> None:
        finalizer = self._finalizing_rounds.get(finalizer_key)
        if finalizer is None or not finalizer.unlinked or not finalizer.message_sent:
            return
        if any(job_id in self._pending_store_results for job_id in finalizer.job_ids):
            return
        if any(
            receipt.xfer.round is finalizer.round
            for receipt in self._terminal_transfers.values()
        ):
            return
        self._finalizing_rounds.pop(finalizer_key, None)

    def _resume_finalizing_rounds(self) -> None:
        """Resume one-shot finalization intents at the next poll boundary."""
        for finalizer_key, finalizer in tuple(self._finalizing_rounds.items()):
            self._finalize_outbound(
                *finalizer_key,
                success=finalizer.success,
                send_done=finalizer.send_done,
            )

    def _drain_abort(self, kv_request_id: str, round_seq: int) -> None:
        """One drain attempt for a pending abort.

        Marks the aborted round failed, then asks the transport to cancel its
        inflight transfers in ``mode="wait"``. Sends ``AbortAckMsg`` only
        after nothing remains inflight. The timeout is diagnostic only: an
        acknowledgement must never authorize destination reuse while a remote
        write may still be active.
        """
        st = self._requests[kv_request_id]
        ids = [
            tid
            for tid, x in self._inflight.items()
            if x.kv_request_id == kv_request_id and x.round_key == round_seq
        ]
        rnd = st.outbound.get(round_seq)
        if rnd is None and ids:
            rounds = {
                id(self._inflight[tid].round): self._inflight[tid].round for tid in ids
            }
            if len(rounds) != 1:
                raise RuntimeError(
                    "multiple outbound states own one abort round: "
                    f"kv_request_id={kv_request_id} round={round_seq}"
                )
            rnd = next(iter(rounds.values()))
            st.outbound[round_seq] = rnd
        if rnd is not None:
            self._mark_round_failed(
                kv_request_id,
                round_seq,
                rnd,
                send_done=False,
            )
            self._drain_failed_rounds()

        started_at = self._pending_aborts[(kv_request_id, round_seq)]
        abort_key = (kv_request_id, round_seq)
        if (
            time.monotonic() - started_at >= _CANCEL_DRAIN_TIMEOUT_S
            and abort_key not in self._abort_timeout_warned
        ):
            self._abort_timeout_warned.add(abort_key)
            logger.warning(
                "P2PSession %s: cancel drain timed out for kv_request_id=%s,"
                " retaining %d transfers and continuing wait-mode drain",
                self._peer_id,
                kv_request_id,
                len(ids),
            )

        if abort_key not in self._failed_rounds and not any(
            x.kv_request_id == kv_request_id and x.round_key == round_seq
            for x in self._inflight.values()
        ):
            self._finalize_abort(kv_request_id, round_seq)

    def _finalize_abort(self, kv_request_id: str, round_seq: int) -> None:
        guarded = self._submitting_xfer
        if (
            guarded is not None
            and guarded.kv_request_id == kv_request_id
            and guarded.round_key == round_seq
        ):
            raise RuntimeError(
                "refusing to acknowledge abort while transfer submission "
                f"ownership is unresolved: kv_request_id={kv_request_id} "
                f"round={round_seq}"
            )
        abort_key = (kv_request_id, round_seq)
        # Publish retry ownership before the fallible send and commit removal
        # only after local control-transport acceptance.
        self._abort_ack_intents.add(abort_key)
        accepted = self._send(
            {
                TYPE_KEY: AbortAckMsg.TYPE,
                AbortAckMsg.KV_REQUEST_ID: kv_request_id,
                AbortAckMsg.ROUND_SEQ: round_seq,
            }
        )
        if accepted is False:
            return
        self._pending_aborts.pop(abort_key, None)
        self._abort_timeout_warned.discard(abort_key)
        self._maybe_prune(kv_request_id)

    # ------------------------------------------------------------------
    # Internal — transfers and store-job timeouts
    # ------------------------------------------------------------------

    def _submit_transfer(
        self,
        kv_request_id: str,
        result: _MatchResult,
        rnd: _OutboundRequestState,
        round_key: int,
    ) -> None:
        if self._submitting_xfer is not None:
            raise RuntimeError("another transfer submission remains unresolved")
        xfer = _InflightXfer(
            kv_request_id=kv_request_id,
            block_count=len(result.local_idxs),
            job_ids=result.job_ids,
            round=rnd,
            round_key=round_key,
        )
        logger.debug(
            "P2PSession %s: NIXL write_blocks CALL kv_request_id=%s "
            "local_idxs=%d remote_idxs=%d",
            self._peer_id,
            kv_request_id,
            len(result.local_idxs),
            len(result.remote_idxs),
        )
        self._submitting_xfer = xfer
        try:
            # Feature selection is cached at construction, keeping the normal
            # production path to one call plus the two guard-pointer stores.
            # A legacy wrapper discards the token; if it is interrupted,
            # recovery remains unsupported and fails closed.
            transfer_id = self._submit_blocks_owned(
                self._peer_id,
                result.local_idxs,
                result.remote_idxs,
                recovery_token=xfer,
            )

            if transfer_id is None:
                # An ordinary None return is the transport's proof that no
                # request from this attempt can still access the buffers.
                self._mark_round_failed(kv_request_id, round_key, rnd)
                self._submitting_xfer = None
                self._drain_failed_rounds()
                logger.warning(
                    "P2PSession %s: write_blocks failed for %s (%d blocks)",
                    self._peer_id,
                    kv_request_id,
                    len(result.local_idxs),
                )
                return
            if type(transfer_id) is not int:
                raise TypeError("transport transfer_id must be an exact int")

            # Primary owner, request index, and round count land in that order.
            # Recovery reconstructs the latter two if an interruption cuts the
            # sequence. The guard is the commit-last record.
            self._inflight_add(transfer_id, xfer)
            rnd.inflight += 1
            self._submitting_xfer = None
        except BaseException:
            if self._submitting_xfer is xfer:
                self._reconcile_submitting_transfer()
                if self._failed_rounds:
                    self._drain_failed_rounds()
            raise

        logger.debug(
            "P2PSession %s: NIXL write_blocks SUBMITTED kv_request_id=%s "
            "transfer_id=%d blocks=%d",
            self._peer_id,
            kv_request_id,
            transfer_id,
            len(result.local_idxs),
        )

    def _mark_round_failed(
        self,
        kv_request_id: str,
        round_key: int,
        rnd: _OutboundRequestState,
        *,
        send_done: bool = True,
    ) -> None:
        """Publish failure ownership before any fallible drain operation."""
        if not rnd.failed:
            rnd.failed = True
        # Clearing is deliberately idempotent: if an asynchronous exception
        # cut a prior call after publishing ``failed``, retry still completes
        # the tombstone before it can be drained.
        rnd.available.clear()
        rnd.demanded.clear()
        if not send_done:
            rnd.failure_send_done = False
        self._failed_rounds[(kv_request_id, round_key)] = rnd

    def _drain_failed_rounds(self) -> None:
        """Wait-cancel failed rounds and publish outcomes after quiescence."""
        for failure_key, rnd in tuple(self._failed_rounds.items()):
            kv_request_id, round_key = failure_key
            guarded = self._submitting_xfer
            if guarded is not None and guarded.round is rnd:
                raise RuntimeError(
                    "refusing to drain failed round while transfer submission "
                    f"ownership is unresolved: kv_request_id={kv_request_id} "
                    f"round={round_key}"
                )
            st = self._requests.get(kv_request_id)
            if st is None or st.outbound.get(round_key) is not rnd:
                raise RuntimeError(
                    "failed-round journal lost its authoritative owner: "
                    f"kv_request_id={kv_request_id} round={round_key}"
                )

            ids = [tid for tid, xfer in self._inflight.items() if xfer.round is rnd]
            still = self._transport.cancel(ids, mode="wait") if ids else []
            still_set = set(still)
            if not still_set.issubset(ids):
                raise RuntimeError(
                    "transport returned unexpected transfer IDs while draining "
                    f"kv_request_id={kv_request_id} round={round_key}"
                )
            for tid in ids:
                if tid in still_set:
                    continue
                # Cancellation proved this transfer quiescent. Publish the
                # same local terminal receipt used by poll before unlinking
                # any session owner; a cut can therefore replay safely.
                self._adopt_transport_terminal(tid, "failed")
            self._drain_terminal_transfers()
            if still:
                continue
            if rnd.inflight or any(
                xfer.round is rnd for xfer in self._inflight.values()
            ):
                raise RuntimeError(
                    "failed round lost transfer ownership before quiescence: "
                    f"kv_request_id={kv_request_id} round={round_key}"
                )

            if not rnd.failure_settled:
                self._fail_round_jobs(rnd)
                rnd.failure_settled = True
            should_finalize = (
                rnd.finishing
                or not rnd.failure_send_done
                or (rnd.demand_received and rnd.failure_send_done)
            )
            if should_finalize:
                self._finalize_outbound(
                    kv_request_id,
                    round_key,
                    success=False,
                    send_done=rnd.failure_send_done,
                )
                rnd.failure_sent = rnd.demand_received and rnd.failure_send_done
            del self._failed_rounds[failure_key]
            self._maybe_prune(kv_request_id)

    def _timeout_pending_store_jobs(self) -> None:
        if not self._store_jobs:
            return
        deadline = time.monotonic() - _STORE_TIMEOUT_S
        timed_out: list[int] | None = None
        for jid, submitted_at in self._store_jobs.items():
            if submitted_at <= deadline and jid not in self._timed_out_store_jobs:
                if timed_out is None:
                    timed_out = []
                timed_out.append(jid)
        if timed_out is None:
            return
        for jid in timed_out:
            owner: tuple[str, int, _OutboundRequestState] | None = None
            for kv_request_id, st in self._requests.items():
                for round_key, rnd in st.outbound.items():
                    if jid in rnd.pending_job_ids:
                        owner = (kv_request_id, round_key, rnd)
                        break
                if owner is not None:
                    break
            if owner is None:
                # We cannot prove that an unindexed provider submission is
                # quiescent. Retain the source pin and surface the invariant
                # violation instead of reporting a failure unsafely.
                logger.error(
                    "P2PSession %s: timed-out store job %d has no outbound "
                    "owner; retaining it because DMA quiescence is unknown",
                    self._peer_id,
                    jid,
                )
                self._timed_out_store_jobs.add(jid)
                continue
            self._timed_out_store_jobs.add(jid)
            kv_request_id, round_key, rnd = owner
            self._mark_round_failed(kv_request_id, round_key, rnd)
            logger.warning(
                "P2PSession %s: store job %d timed out; draining its "
                "failed round before releasing source memory",
                self._peer_id,
                jid,
            )
        if self._failed_rounds:
            self._drain_failed_rounds()
