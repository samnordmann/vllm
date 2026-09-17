# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2PSecondaryTierManager: Secondary tier for P2P KV cache sharing.

Owns transports and a single bidirectional P2PSession per remote peer.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from typing_extensions import override

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import get_none_hash_seed
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import (
    JobResult,
    SecondaryTierManager,
    TransferJob,
)
from vllm.v1.kv_offload.tiering.p2p.control import ControlTransport, ZmqTransport
from vllm.v1.kv_offload.tiering.p2p.data import (
    DataTransport,
    NixlTransport,
    TorchTransferTransport,
)
from vllm.v1.kv_offload.tiering.p2p.session import (
    P2PSession,
    SessionCloseResult,
    SessionPollResult,
)

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import OffloadingSpec
    from vllm.v1.kv_offload.tiering.base import ParentManager
    from vllm.v1.kv_offload.tiering.p2p.control.base import ControlConnection

logger = init_logger(__name__)

# Reap unbound store batches that have been parked without a FetchMsg
# binding them to a session for longer than this. Protects against the
# prefiller buffering blocks for a decoder that never asks (decoder died,
# network partition, lost kv_request_id). Must be longer than the per-store
# deadline so the store-timeout path fires first for individual jobs.
_UNBOUND_STORE_TIMEOUT_S = 60.0

# Time we wait during shutdown for inflight transfers to drain via
# cancel(mode="wait"). Expiry fails closed: the manager reports that
# resources were retained and never authorizes unsafe immediate release.
_SHUTDOWN_DRAIN_TIMEOUT_S = 3.0

# Sleep between iterations of the bounded drain loops in drain_jobs() and
# _drain_inflight_for_shutdown(). Short enough to keep latency low, long
# enough to avoid busy-spinning the scheduler thread.
_DRAIN_SLEEP_S = 0.001


def _close_initialization_resource(resource: object | None, description: str) -> None:
    """Best-effort constructor rollback that preserves the primary failure."""
    if resource is None:
        return
    try:
        resource.close()  # type: ignore[attr-defined]
    except BaseException as exc:
        with suppress(BaseException):
            logger.warning(
                "P2P %s cleanup failed during initialization: %s", description, exc
            )


def _remote_prefiller_params(kv_params: dict | None) -> dict | None:
    """Return the ``remote_prefiller`` sub-dict, or None if absent.

    Set on decoder requests to name the remote prefiller they pull from;
    carries kv_request_id, remote_host, remote_port.
    """
    if not kv_params:
        return None
    return kv_params.get("remote_prefiller")


def _remote_decoder_params(kv_params: dict | None) -> dict | None:
    """Return the ``remote_decoder`` sub-dict, or None if absent.

    Set on prefiller requests to name the remote decoder they serve;
    carries kv_request_id.
    """
    if not kv_params:
        return None
    return kv_params.get("remote_decoder")


def _remote_kv_source_params(kv_params: dict | None) -> dict | None:
    """Return the ``remote_kv_source`` sub-dict, or None if absent.

    Set on symmetric-P2P consumer requests to name the remote source they
    pull from; carries kv_request_id, remote_host, remote_port.
    """
    if not kv_params:
        return None
    return kv_params.get("remote_kv_source")


def _peer_id_from_params(role_params: dict) -> str | None:
    """Build ``host:port`` peer_id from a role-scoped sub-dict, or None."""
    host = role_params.get("remote_host")
    port = role_params.get("remote_port")
    if host and port:
        return f"{host}:{port}"
    return None


@dataclass(slots=True)
class P2PSourceInfo:
    """Consumer side: this request fetches from a remote (prefiller or peer)."""

    kv_request_id: str
    peer_id: str
    do_probe: bool  # False for remote_prefiller (PD), True for remote_kv_source


@dataclass(slots=True)
class P2PDestInfo:
    """Producer side: a remote fetches this request's blocks from us.

    ``kv_request_id`` is None when the ``remote_decoder`` block is present
    but malformed (no id); the block's presence still marks the request as
    remote-decode, so submit_store must fail rather than store locally.
    """

    kv_request_id: str | None


def _parse_source(kv_params: dict | None) -> P2PSourceInfo | None:
    """Parse the consumer sub-dict (PD ``remote_prefiller`` or symmetric
    ``remote_kv_source``) into a ``P2PSourceInfo``, or None if absent/incomplete."""
    role = _remote_prefiller_params(kv_params)
    do_probe = False
    if role is None:
        role = _remote_kv_source_params(kv_params)
        do_probe = True
    if not role:
        return None
    peer_id = _peer_id_from_params(role)
    kv_request_id = role.get("kv_request_id")
    if peer_id is None or not kv_request_id:
        return None
    return P2PSourceInfo(
        kv_request_id=kv_request_id,
        peer_id=peer_id,
        do_probe=do_probe,
    )


def _parse_dest(kv_params: dict | None) -> P2PDestInfo | None:
    """Parse the producer ``remote_decoder`` sub-dict into a ``P2PDestInfo``,
    or None if the block is absent (not a remote-decode request)."""
    role = _remote_decoder_params(kv_params)
    if role is None:
        return None
    return P2PDestInfo(kv_request_id=role.get("kv_request_id") or None)


def _annotate_req_context(req_context: ReqContext) -> None:
    """Parse kv_transfer_params once and cache the P2P routing state.

    Called from ``on_new_request``; later calls for the same request read
    the cached ``P2PSourceInfo``/``P2PDestInfo`` via ``get_state`` instead
    of re-parsing.
    """
    source = _parse_source(req_context.kv_transfer_params)
    if source is not None:
        req_context.set_state(source)
    dest = _parse_dest(req_context.kv_transfer_params)
    if dest is not None:
        req_context.set_state(dest)


@dataclass
class _UnboundStoreBatch:
    """A submit_store batch parked at the manager before any peer has fetched.

    Indexed by kv_request_id only — the prefiller no longer learns the peer
    identity at store time. When a FetchMsg(kv_request_id) arrives on some
    session, the manager binds the kv_request_id to that session and replays
    every parked batch into ServerRole via session.add_stored_blocks.
    """

    job_id: int
    keys: list[OffloadKey]
    block_ids: Sequence[int]
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass
class _RetiringSession:
    """Cold-path ownership journal for a dead session.

    The session leaves live routing before any fallible teardown. The journal
    then survives session-close or data-peer removal failures and is retried by
    the next scheduler poll without adding a branch to transfer submission or
    completion polling.
    """

    session: P2PSession
    close_result: SessionCloseResult | None = None
    peer_removed: bool = False
    job_results_published: bool = False
    failed_req_ids_published: bool = False
    failed_serves_published: bool = False
    quarantine_warned: bool = False


class P2PSecondaryTierManager(SecondaryTierManager):
    """Secondary tier for P2P KV cache sharing.

    A single P2PSession per remote peer handles both client-role (loading
    blocks from the peer) and server-role (serving blocks to the peer)
    over the same control connection.

    Single-threaded: every public method runs on the scheduler thread, and
    the engine drives polling via ``get_finished_jobs()`` once per step.
    ``has_pending_work()`` keeps the engine ticking so the control transport
    and existing sessions are polled even when no requests are scheduled.
    """

    def __init__(
        self,
        offloading_spec: OffloadingSpec,
        primary_kv_view: memoryview,
        tier_type: str = "p2p",
        host: str | None = None,
        port: int | None = None,
        backends: list[str] | None = None,
        num_threads: int = 4,
        data_transport: str = "nixl",
        transfer_backend: str = "nixl",
        transfer_progress_mode: str = "background",
        transfer_thread_mode: str = "single",
        transfer_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the P2P secondary tier manager.

        All keyword arguments after ``primary_kv_view`` come from the
        ``secondary_tiers`` entry in ``kv_connector_extra_config``. See
        ``docs/features/kv_offloading_usage.md`` for the user-facing
        configuration reference.

        Args:
            offloading_spec: Owning ``OffloadingSpec`` (provides normalized
                model, parallel, and cache layout configuration).
            primary_kv_view: Memoryview over the CPU primary tier; the
                NIXL agent registers this region for RDMA transfers.
            tier_type: Tier identifier (defaults to ``"p2p"``).
            host: Address the ZMQ control socket binds to, used verbatim
                as both the bind address and the identity peers dial back
                (mirrors the NIXL connector's ``VLLM_NIXL_SIDE_CHANNEL_HOST``;
                no auto-detection). Defaults to
                ``VLLM_P2P_SIDE_CHANNEL_HOST`` (``localhost``) when not set;
                must be set to the node's routable IP for cross-host P2P so
                remote peers can reach the socket.
            port: Base port for the ZMQ control socket. Must be
                reachable from peers. Defaults to
                ``VLLM_P2P_SIDE_CHANNEL_PORT`` (``5710``) when not set.
                The bound port is ``base + data_parallel_index`` so each
                DP replica gets a distinct port (one socket per replica,
                like NIXL); for DP=1 the offset is 0.
            backends: NIXL transport backends (e.g. ``["UCX"]``,
                ``["MOONCAKE"]``, ``["LIBFABRIC"]``). Defaults to
                ``["UCX"]``. When any non-UCX backend is requested, the
                NIXL agent is initialized with ``backends=...``;
                otherwise it falls back to a UCX-only agent with
                ``num_threads`` threads.
            num_threads: NIXL agent worker threads for the UCX-only
                branch. Ignored when ``backends`` contains a non-UCX
                entry.
            data_transport: Data-plane implementation. ``"nixl"`` keeps the
                native NIXL path; ``"torch"`` selects the experimental
                ``torch.distributed._transfer`` adapter.
            transfer_backend: PyTorch endpoint-transfer backend name when
                ``data_transport="torch"``.
            transfer_progress_mode: PyTorch endpoint progress mode. Defaults
                to background progress, matching the native NIXL agent.
            transfer_thread_mode: PyTorch endpoint thread-safety mode.
            transfer_options: Options forwarded to the PyTorch endpoint. The
                NIXL provider inherits native agent defaults for backends,
                thread count, and telemetry unless overridden here.
            **kwargs: Reserved for future tier-specific options.
        """
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        # Block hashes chain from NONE_HASH (see v1/core/kv_cache_utils.py).
        # Peers whose seeds differ compute different hashes for identical
        # content, so lookups silently miss and no KV crosses the wire. The
        # seed is advertised and verified against each peer during the
        # handshake, so a mismatch is rejected loudly instead of degrading
        # silently. Resolved lazily in _get_hash_seed: this tier is built
        # before init_none_hash runs, and a non-cryptographic hash algorithm
        # seeds NONE_HASH randomly, so the value is only known afterwards.
        self._hash_seed: str | None = None
        if host is None:
            host = envs.VLLM_P2P_SIDE_CHANNEL_HOST
        if port is None:
            port = envs.VLLM_P2P_SIDE_CHANNEL_PORT
        # One control socket per DP replica: offset the base by the global
        # data-parallel index so replicas on a host don't collide (mirrors
        # NIXL). For DP=1 the index is 0, leaving the base port unchanged.
        dp_index = offloading_spec.config.parallel.data_parallel_index
        port = int(port) + dp_index
        # Two decoupled identities:
        #   _local_id (``host:port``): the ZMQ control identity that peers
        #     dial back, used verbatim (the socket binds this host/port and
        #     the address is parsed back into host:port by the remote).
        #   _nixl_agent_name (uuid4): the NIXL agent name. It is never dialed
        #     — it travels opaquely inside the agent metadata blob — so it
        #     only needs to be globally unique. A per-process uuid guarantees
        #     that even for peers sharing a host:port (mirrors the NIXL
        #     connector; avoids the "remote agent name equals local" reject).
        self._local_id = f"{host}:{port}"
        self._nixl_agent_name = str(uuid.uuid4())

        config_fields = FileMapper.from_offloading_spec(
            root_dir="",
            offloading_spec=offloading_spec,
            blocks_per_file=offloading_spec.blocks_per_chunk,
            parallel_agnostic=True,
        ).get_run_config()
        # Allocate all fallible Python bookkeeping before native transports.
        # Once the data plane exists, ZMQ is the final acquisition and every
        # failure below has a short, explicit reverse-order rollback.
        self._sessions: dict[str, P2PSession] = {}
        self._retiring_sessions: dict[str, _RetiringSession] = {}
        # kv_request_id → session, set when the bound session has received
        # FetchMsg for that id. submit_store after binding routes directly
        # to the session; before binding, batches are parked in
        # _unbound_stores below. Stays in sync with _sessions: entries
        # pointing to a reaped session are purged in _reap_dead_sessions.
        self._kv_to_session: dict[str, P2PSession] = {}
        # kv_request_id → list of batches submit_store'd before any peer
        # asked for that id. Drained into a session by _on_session_fetch
        # when the corresponding FetchMsg arrives, or surfaced as failures
        # by _reap_unbound_stores after _UNBOUND_STORE_TIMEOUT_S.
        self._unbound_stores: dict[str, dict[int, _UnboundStoreBatch]] = {}

        # Results remain keyed and replayable until the top-level manager
        # explicitly acknowledges adoption.
        self._finished_jobs: dict[int, JobResult] = {}
        # kv_request_ids that hit a transport/session failure; On load lookup()
        # rejects them so the request falls back to local prefill.
        self._failed_req_ids: set[str] = set()
        # Synthetic lookup ctxs from reaped sessions still owing a
        # ``parent.on_request_finished`` (the session's failed_serves). The
        # dead session had no parent handle at teardown; these are flushed
        # at the top of the next ``serve_external_requests`` where the
        # handle is valid.
        self._failed_serve_ctxs: list[ReqContext] = []
        self._closing = False
        self._closed = False

        data: DataTransport | None = None
        control: ControlTransport | None = None
        try:
            if data_transport == "nixl":
                data = NixlTransport(
                    self._nixl_agent_name,
                    primary_kv_view,
                    config_fields=config_fields,
                    backends=backends,
                    num_threads=int(num_threads),
                )
            elif data_transport == "torch":
                resolved_transfer_options = dict(transfer_options or {})
                if transfer_backend == "nixl":
                    # Keep the NIXL provider on the same UCX setup as the native
                    # path. Explicit provider options remain authoritative.
                    resolved_transfer_options.setdefault(
                        "backends", list(backends) if backends else ["UCX"]
                    )
                    resolved_transfer_options.setdefault(
                        "num_threads", int(num_threads)
                    )
                    resolved_transfer_options.setdefault("capture_telemetry", True)
                data = TorchTransferTransport(
                    self._nixl_agent_name,
                    primary_kv_view,
                    config_fields=config_fields,
                    endpoint_id=self._local_id,
                    backend=transfer_backend,
                    progress_mode=transfer_progress_mode,
                    thread_mode=transfer_thread_mode,
                    options=resolved_transfer_options,
                )
                if not data.available:
                    raise RuntimeError(
                        "data_transport='torch' requires an available "
                        "torch.distributed._transfer API and backend"
                    )
            else:
                raise ValueError(
                    "data_transport must be either 'nixl' or 'torch', "
                    f"got {data_transport!r}"
                )
            control = ZmqTransport(self._local_id, host, port)
        except BaseException:
            _close_initialization_resource(control, "control transport")
            _close_initialization_resource(data, "data transport")
            raise

        assert data is not None and control is not None
        self._data = data
        self._control = control

    # ------------------------------------------------------------------
    # SecondaryTierManager interface
    # ------------------------------------------------------------------

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        self._check_open()
        source = req_context.get_state(P2PSourceInfo)
        if source is None:
            return LookupResult.MISS
        if source.kv_request_id in self._failed_req_ids:
            return LookupResult.MISS

        # Symmetric-P2P consumer (``remote_kv_source`` sub-dict): probe the
        # peer asynchronously. First call registers the (kv_request_id,
        # key) entry and returns RETRY; flush_pending_lookups()
        # in on_schedule_end batches the LookupMsg; a later step's
        # lookup() returns HIT/MISS once LookupRespMsg has arrived.
        # PD path (``remote_prefiller`` sub-dict only) keeps the eager HIT.
        if source.do_probe:
            session = self._sessions.get(source.peer_id)
            if session is None:
                return LookupResult.MISS
            result = session.register_lookup(source.kv_request_id, key)
            if result is True:
                return LookupResult.HIT
            if result is False:
                return LookupResult.MISS
            return LookupResult.RETRY

        # PD consumer (we are the decoder): all kv blocks should be on the
        # prefiller side. Return HIT immediately.
        return LookupResult.HIT

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        """Parse kv_transfer_params once and open the outbound session.

        Parses the P2P routing state onto ``req_context`` (cached for the
        later lookup/submit/finish calls). On the consumer side
        (``remote_prefiller`` for PD or ``remote_kv_source`` for symmetric
        P2P), open a session toward the producer at remote_host:remote_port
        so submit_load can issue FetchMsg as soon as it fires. On the
        prefiller side, sessions are created when the consumer's inbound
        connection arrives in _accept_new_peers — submit_store no longer
        pre-creates anything.

        Producer-leg requests (``remote_decoder`` present) ask for
        REQUEST_LEVEL: the peer needs every block of the request, not just
        the ones this request computed.
        """
        self._check_open()
        _annotate_req_context(req_context)
        source = req_context.get_state(P2PSourceInfo)
        if source is not None:
            self._get_or_create_session(source.peer_id)
        dest = req_context.get_state(P2PDestInfo)
        if dest is not None and dest.kv_request_id:
            return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        """Cancels pending loads and prunes session-scoped state.

        Consumer side (``remote_prefiller`` for PD or ``remote_kv_source``
        for symmetric-P2P): looks up the session by peer_id because the
        producer's address is what addresses the client-role load to
        cancel; also drops any pending symmetric-P2P lookup state via
        ``session.finish_request``.
        Prefiller side (``remote_decoder`` set): looks up via kv_request_id
        because peer_id is no longer carried on store-time
        kv_transfer_params; if a session has bound the id, finish it. If
        no session has bound the id yet, this is a no-op: parked batches
        in `_unbound_stores` are left in place and cleaned up only by
        `_reap_unbound_stores` after `_UNBOUND_STORE_TIMEOUT_S`.
        """
        self._check_open()
        source = req_context.get_state(P2PSourceInfo)
        dest = req_context.get_state(P2PDestInfo)
        kv_request_id = source.kv_request_id if source is not None else None
        if kv_request_id is None and dest is not None:
            kv_request_id = dest.kv_request_id
        if not kv_request_id:
            return
        self._failed_req_ids.discard(kv_request_id)

        if source is not None:
            session = self._sessions.get(source.peer_id)
            if session is not None:
                session.finish_request(kv_request_id)
            return

        # Prefiller-side finish: identify the session via kv_request_id.
        session = self._kv_to_session.get(kv_request_id)
        if session is not None:
            session.finish_request(kv_request_id)
            if self._kv_to_session.get(kv_request_id) is session:
                self._kv_to_session.pop(kv_request_id, None)
            return

    @override
    def submit_store(self, job_metadata: TransferJob) -> None:
        self._check_open()
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids.tolist()

        assert len(keys) == len(block_ids)

        dest = job_metadata.req_context.get_state(P2PDestInfo)
        logger.debug(
            "P2P %s: submit_store ENTRY job_id=%d blocks=%d "
            "remote_decoder=%s kv_request_id=%s",
            self._local_id,
            job_id,
            len(block_ids),
            dest is not None,
            dest.kv_request_id if dest is not None else None,
        )
        # Absent ``remote_decoder`` block => not a remote-decode request:
        # succeed locally without parking. An empty/malformed dict is still
        # a remote-decode signal and must fail the missing-id check below.
        if dest is None:
            self._publish_finished_job(JobResult(job_id=job_id, success=True))
            return

        kv_request_id = dest.kv_request_id
        if not kv_request_id:
            logger.warning(
                "P2P %s: submit_store missing kv_request_id",
                self._local_id,
            )
            self._publish_finished_job(JobResult(job_id=job_id, success=False))
            return

        # Fast path: a session has already received FetchMsg for this id,
        # so we can route the batch straight into its ServerRole.
        session = self._kv_to_session.get(kv_request_id)
        if session is not None:
            session.add_stored_blocks(kv_request_id, keys, block_ids, job_id)
            return

        # No session bound yet — park the batch keyed by kv_request_id.
        # _on_session_fetch drains it on the first FetchMsg; if no peer
        # ever asks, _reap_unbound_stores surfaces the job as failed.
        batch = _UnboundStoreBatch(
            job_id=job_id,
            keys=keys,
            block_ids=block_ids,
        )
        existing = self._unbound_stores.setdefault(kv_request_id, {}).setdefault(
            job_id, batch
        )
        if existing != batch:
            raise RuntimeError(f"conflicting unbound store job {job_id}")
        logger.debug(
            "P2P %s: parked submit_store kv_request_id=%s job_id=%d blocks=%d",
            self._local_id,
            kv_request_id,
            job_id,
            len(block_ids),
        )

    @override
    def submit_load(self, job_metadata: TransferJob) -> None:
        self._check_open()
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        block_ids = job_metadata.block_ids

        source = job_metadata.req_context.get_state(P2PSourceInfo)
        logger.debug(
            "P2P %s: submit_load ENTRY job_id=%d blocks=%d kv_request_id=%s peer=%s",
            self._local_id,
            job_id,
            len(block_ids),
            source.kv_request_id if source is not None else None,
            source.peer_id if source is not None else None,
        )
        if source is None:
            logger.debug(
                "P2P %s: submit_load job_id=%d FAILED missing consumer params",
                self._local_id,
                job_id,
            )
            self._publish_finished_job(JobResult(job_id=job_id, success=False))
            return

        kv_request_id = source.kv_request_id
        peer_id = source.peer_id

        if not keys:
            logger.debug(
                "P2P %s: submit_load job_id=%d short-circuit success (no keys)",
                self._local_id,
                job_id,
            )
            self._publish_finished_job(JobResult(job_id=job_id, success=True))
            return

        session = self._sessions.get(peer_id)
        if session is None:
            logger.warning(
                "P2P %s: submit_load job_id=%d NO SESSION for peer=%s",
                self._local_id,
                job_id,
                peer_id,
            )
            self._publish_finished_job(JobResult(job_id=job_id, success=False))
            self._failed_req_ids.add(kv_request_id)
            return
        logger.debug(
            "P2P %s: submit_load job_id=%d -> request_blocks peer=%s "
            "kv_request_id=%s blocks=%d session_ready=%s",
            self._local_id,
            job_id,
            peer_id,
            kv_request_id,
            len(block_ids),
            session.ready,
        )
        session.request_blocks(job_id, kv_request_id, keys, block_ids)

    @override
    def get_finished_jobs(self) -> Iterable[JobResult]:
        """Compatibility drain; interruption-safe callers use peek/ack."""
        result = list(self.peek_finished_jobs())
        self.ack_finished_jobs(tuple(item.job_id for item in result))
        return result

    @override
    def peek_finished_jobs(self) -> Iterable[JobResult]:
        self._check_open()
        self._poll_once()
        return list(self._finished_jobs.values())

    @override
    def ack_finished_jobs(self, job_ids: Iterable[int]) -> None:
        for job_id in job_ids:
            if type(job_id) is not int:
                raise TypeError("job_id must be an exact int")
            self._finished_jobs.pop(job_id, None)

    @override
    def has_pending_work(self) -> bool:
        # The engine tick is the only driver of _control.poll() and
        # session.poll(); without it we miss new peer connects and
        # inbound fetch messages on existing sessions. Keep the engine
        # ticking for the lifetime of this manager.
        return True

    @override
    def drain_jobs(self) -> None:
        """Block until every submitted load/store job has completed or failed.

        Loops calling ``_poll_once()`` until no session has outstanding
        inbound loads or in-flight outbound stores. Mid-flight transfers
        are NOT cancelled — the caller (``TieringOffloadingManager.reset_cache``)
        needs the primary memoryview to be quiescent, not aborted. Results
        accumulate in ``_finished_jobs`` and are surfaced by the next
        ``get_finished_jobs()`` call.
        """
        self._check_open()
        start = time.monotonic()
        warned = False
        while True:
            self._poll_once()
            pending = any(s.has_pending_work for s in self._sessions.values()) or any(
                retirement.session.has_pending_work
                for retirement in self._retiring_sessions.values()
            )
            if not pending:
                return
            if not warned and time.monotonic() - start > 5.0:
                logger.warning(
                    "P2PSecondaryTierManager.drain_jobs: still draining "
                    "after 5s; a stuck transfer will block the engine.",
                )
                warned = True
            time.sleep(_DRAIN_SLEEP_S)

    @override
    def serve_external_requests(self, parent: ParentManager) -> None:
        """Serve inbound peer lookups against the tiering manager.

        Called once per scheduler step (before this tier's
        ``on_schedule_end``) with a ``parent`` handle valid only for the
        duration of the call — the sole window in which the P2P server
        role may query the tiering manager. First release bookkeeping for
        the failed serves left by a reaped session, then let every live
        session resolve its enqueued inbound LookupMsgs.
        """
        self._check_open()
        if self._failed_serve_ctxs:
            for ctx in self._failed_serve_ctxs:
                parent.on_request_finished(ctx)
            self._failed_serve_ctxs = []
        for session in self._sessions.values():
            session.serve_external_requests(parent)

    @override
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._check_open()
        # Flush any p2p lookups aggregated during this step.
        # One LookupMsg per (peer, kv_request_id) with unsent entries;
        # send-gating happens inside the session if not yet ready.
        for session in self._sessions.values():
            session.flush_pending_lookups()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_hash_seed(self) -> str:
        """The seed NONE_HASH was derived from, resolved on first session."""
        if self._hash_seed is None:
            self._hash_seed = get_none_hash_seed()
        return self._hash_seed

    def _check_open(self) -> None:
        if self._closing or self._closed:
            raise RuntimeError("P2P secondary tier is closing or closed")

    def _publish_finished_job(self, result: JobResult) -> None:
        existing = self._finished_jobs.setdefault(result.job_id, result)
        if existing != result:
            raise RuntimeError(f"conflicting outcomes for job {result.job_id}")

    def _adopt_session_result(
        self,
        session: P2PSession,
        result: SessionPollResult,
        *,
        accept_fetch: bool = True,
    ) -> None:
        """Publish a session snapshot, apply fetch bindings, then ack it."""
        for load in result.loads:
            self._publish_finished_job(
                JobResult(job_id=load.job_id, success=load.success)
            )
            if not load.success:
                self._failed_req_ids.add(load.kv_request_id)
        for store in result.stores:
            self._publish_finished_job(
                JobResult(job_id=store.job_id, success=store.success)
            )

        for kv_request_id in result.new_fetch_ids:
            if not accept_fetch:
                batches = self._unbound_stores.get(kv_request_id)
                if batches is not None:
                    self._failed_req_ids.add(kv_request_id)
                    for job_id, batch in tuple(batches.items()):
                        self._publish_finished_job(
                            JobResult(job_id=batch.job_id, success=False)
                        )
                        batches.pop(job_id, None)
                    if not batches:
                        self._unbound_stores.pop(kv_request_id, None)
                continue
            owner = self._kv_to_session.setdefault(kv_request_id, session)
            if owner is not session:
                raise RuntimeError(
                    f"fetch {kv_request_id!r} changed owning peer session"
                )
            batches = self._unbound_stores.get(kv_request_id)
            if batches is None:
                continue
            for job_id, batch in tuple(batches.items()):
                if job_id not in self._finished_jobs and not session.owns_store_job(
                    job_id
                ):
                    session.add_stored_blocks(
                        kv_request_id, batch.keys, batch.block_ids, job_id
                    )
                batches.pop(job_id, None)
            if not batches:
                self._unbound_stores.pop(kv_request_id, None)

        if result.loads or result.stores or result.new_fetch_ids:
            session.ack_results(
                tuple(load.job_id for load in result.loads),
                tuple(store.job_id for store in result.stores),
                tuple(result.new_fetch_ids),
            )

    def _get_or_create_session(self, peer_id: str) -> P2PSession:
        """Return the existing session for peer_id, or open one outbound.

        Consumer-side helper for on_new_request: when ``remote_prefiller``
        (PD) or ``remote_kv_source`` (symmetric P2P) is set, the consumer must reach the
        producer at peer_id. If we already have a session toward that
        peer (from a prior load or a peer-initiated inbound), reuse it;
        otherwise open an outbound ControlConnection and build a
        connected session.
        """
        session = self._sessions.get(peer_id)
        if session is not None:
            return session
        if peer_id in self._retiring_sessions:
            raise RuntimeError(
                f"session for {peer_id} is still retiring after peer failure"
            )
        conn = self._control.connect(peer_id)
        session = P2PSession(
            peer_id=peer_id,
            local_id=self._local_id,
            transport=self._data,
            local_block_len=self._data.block_len,
            local_hash_seed=self._get_hash_seed(),
            conn=conn,
        )
        self._sessions[peer_id] = session
        return session

    def _accept_new_peers(self, new_connections: Sequence[ControlConnection]) -> None:
        for conn in new_connections:
            logger.info(
                "P2P %s: accepting incoming connection from %s",
                self._local_id,
                conn.peer_id,
            )
            try:
                existing = self._sessions.get(conn.peer_id)
                if existing is not None or conn.peer_id in self._retiring_sessions:
                    raise ValueError(f"duplicate connection from {conn.peer_id}")
                self._sessions[conn.peer_id] = P2PSession(
                    peer_id=conn.peer_id,
                    local_id=self._local_id,
                    transport=self._data,
                    local_block_len=self._data.block_len,
                    local_hash_seed=self._get_hash_seed(),
                    conn=conn,
                )
                logger.info(
                    "P2P %s: created connected session for %s",
                    self._local_id,
                    conn.peer_id,
                )
            except (ValueError, KeyError, TypeError, AssertionError) as exc:
                logger.error("P2P %s: rejecting peer: %s", self._local_id, exc)
                conn.close()

    def _reap_dead_sessions(self) -> None:
        # Move newly dead sessions out of live routing, then retry every
        # manager-owned retirement journal — peer is gone.
        # Stranded prefiller-side stores are no longer tracked through a
        # session (they live in _unbound_stores keyed by kv_request_id);
        # _reap_unbound_stores handles their timeout independently.
        dead: list[tuple[str, P2PSession]] | None = None
        for pid, s in self._sessions.items():
            if not s.alive:
                if dead is None:
                    dead = []
                dead.append((pid, s))
        for pid, session in dead or ():
            retirement = self._retiring_sessions.get(pid)
            if retirement is None:
                # Publish durable cleanup ownership before unlinking the live
                # route. An interrupted epilogue is repaired on the next pass.
                self._retiring_sessions[pid] = _RetiringSession(session=session)
            elif retirement.session is not session:
                raise RuntimeError(f"conflicting retiring session for peer {pid}")

        if not self._retiring_sessions:
            return

        failures: list[BaseException] = []
        for pid, retirement in tuple(self._retiring_sessions.items()):
            session = retirement.session
            # Remove every route before entering fallible cleanup. The separate
            # retirement journal remains the strong owner until completion.
            if self._sessions.get(pid) is session:
                del self._sessions[pid]
            stale_kv_ids = [
                kid for kid, s in self._kv_to_session.items() if s is session
            ]
            for kid in stale_kv_ids:
                del self._kv_to_session[kid]

            try:
                self._adopt_session_result(
                    session, session.pending_results(), accept_fetch=False
                )
                close_complete = getattr(session, "close_complete", True)
                if retirement.close_result is None or not close_complete:
                    close_result = session.close()
                    if retirement.close_result is None:
                        retirement.close_result = close_result
                close_result = retirement.close_result
                assert close_result is not None
                if not getattr(session, "close_complete", True):
                    if not retirement.failed_req_ids_published:
                        # Request routing may fail immediately without
                        # releasing its promotion job or CPU destination.
                        self._failed_req_ids.update(close_result.failed_req_ids)
                        retirement.failed_req_ids_published = True
                    if not retirement.quarantine_warned:
                        retirement.quarantine_warned = True
                        with suppress(BaseException):
                            logger.warning(
                                "P2P %s: control session %s died with %d "
                                "unacknowledged load(s); ZMQ disconnect is not "
                                "NIXL quiescence, retaining CPU destinations "
                                "until explicit proof or worker restart",
                                self._local_id,
                                pid,
                                len(close_result.failed_jobs),
                            )
                    # Nonblocking retirement: retain failed-store pins and try
                    # wait-mode cancellation again on the next scheduler tick.
                    continue
                if not retirement.peer_removed:
                    self._data.remove_remote_peer(pid)
                    retirement_complete = getattr(
                        self._data, "peer_retirement_complete", None
                    )
                    if callable(retirement_complete) and not retirement_complete(pid):
                        continue
                    retirement.peer_removed = True
                self._adopt_session_result(
                    session, session.pending_results(), accept_fetch=False
                )
                if not retirement.job_results_published:
                    # Reconstruct committed IDs from the destination on every
                    # retry. This closes the append-committed / flag-not-yet-set
                    # BaseException boundary without penalizing the live path.
                    for job_id in (
                        *close_result.failed_jobs,
                        *close_result.failed_stores,
                    ):
                        self._publish_finished_job(
                            JobResult(job_id=job_id, success=False)
                        )
                    retirement.job_results_published = True
                if not retirement.failed_req_ids_published:
                    self._failed_req_ids.update(close_result.failed_req_ids)
                    retirement.failed_req_ids_published = True
                if not retirement.failed_serves_published:
                    published_ctx_ids = {id(ctx) for ctx in self._failed_serve_ctxs}
                    for ctx in close_result.failed_serves:
                        if id(ctx) not in published_ctx_ids:
                            self._failed_serve_ctxs.append(ctx)
                            published_ctx_ids.add(id(ctx))
                    retirement.failed_serves_published = True
                del self._retiring_sessions[pid]
            except BaseException as exc:
                failures.append(exc)
                continue

            with suppress(BaseException):
                logger.warning("P2P %s: peer %s down", self._local_id, pid)
        if failures:
            raise failures[0]

    def _reap_unbound_stores(self) -> None:
        """Time out submit_store batches that no peer has ever fetched.

        Walks `_unbound_stores` for entries whose oldest batch is older
        than `_UNBOUND_STORE_TIMEOUT_S`. Drops the kv_request_id, surfaces
        every batched job as failed, and adds the id to `_failed_req_ids`
        so a late inbound FetchMsg short-circuits to a clean rejection.
        """
        if not self._unbound_stores:
            return
        deadline = time.monotonic() - _UNBOUND_STORE_TIMEOUT_S
        expired: list[str] | None = None
        for kid, batches in self._unbound_stores.items():
            # Batches are appended in arrival order, so the head is oldest.
            if (
                batches
                and min(batch.submitted_at for batch in batches.values()) <= deadline
            ):
                if expired is None:
                    expired = []
                expired.append(kid)
        if expired is None:
            return
        for kid in expired:
            batches = self._unbound_stores[kid]
            expired_count = len(batches)
            self._failed_req_ids.add(kid)
            for job_id, batch in tuple(batches.items()):
                self._publish_finished_job(
                    JobResult(job_id=batch.job_id, success=False)
                )
                batches.pop(job_id, None)
            if not batches:
                self._unbound_stores.pop(kid, None)
            logger.warning(
                "P2P %s: unbound store kv_request_id=%s timed out after %.0fs "
                "without a fetch — failing %d job(s)",
                self._local_id,
                kid,
                _UNBOUND_STORE_TIMEOUT_S,
                expired_count,
            )

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll_once(self) -> None:
        """One sweep of the polling work.

        Drains the control transport, polls every session, accumulates
        their results into ``_finished_jobs``, and reaps any dead sessions.
        Runs on the scheduler thread.
        """
        new_connections = self._control.poll()
        if new_connections:
            logger.info(
                "P2P %s: _poll_once got %d new connection(s): %s",
                self._local_id,
                len(new_connections),
                [c.peer_id for c in new_connections],
            )
            # A control poll can both surface a replacement and reveal that the
            # old same-ID session was already dead. Retire/remove the old owner
            # before deciding whether the newly accepted connection is duplicate.
            # If retirement is quarantined, reject safely; discovery retries.
            self._reap_dead_sessions()
            self._accept_new_peers(new_connections)

        for session in self._sessions.values():
            result = session.poll()
            self._adopt_session_result(session, result, accept_fetch=session.alive)

        self._reap_dead_sessions()
        reap_retired = getattr(self._data, "reap_retired_peers", None)
        if callable(reap_retired):
            reap_retired()
        self._reap_unbound_stores()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @override
    def shutdown(self) -> None:
        if self._closed:
            return
        self._closing = True
        failures: list[BaseException] = []
        resources_quiescent = True

        def attempt(action: Callable[[], object], description: str) -> bool:
            try:
                action()
                return True
            except BaseException as exc:
                failures.append(exc)
                with suppress(BaseException):
                    logger.warning(
                        "P2P %s failed during shutdown: %s", description, exc
                    )
                return False

        resources_quiescent = attempt(
            self._drain_inflight_for_shutdown, "inflight drain"
        )
        sessions = tuple(self._sessions.values()) + tuple(
            retirement.session for retirement in self._retiring_sessions.values()
        )
        for session in sessions:
            # Orphan ctxs from close() are intentionally dropped: the manager
            # is being torn down, so there is no next serve_external_requests
            # to flush them and no TieringManager left to release.
            closed = attempt(session.close, f"session {session.peer_id} close")
            resources_quiescent = (
                resources_quiescent
                and closed
                and getattr(session, "close_complete", True)
            )
        if resources_quiescent:
            self._sessions.clear()
            self._retiring_sessions.clear()
            self._kv_to_session.clear()
            # Surface buffered store jobs as failed only after every data-plane
            # owner is quiescent. Until then the manager remains their owner.
            for batches in self._unbound_stores.values():
                for batch in batches.values():
                    self._publish_finished_job(
                        JobResult(job_id=batch.job_id, success=False)
                    )
            self._unbound_stores.clear()
        elif not failures:
            failures.append(
                RuntimeError(
                    "P2P shutdown incomplete; active transfer resources retained"
                )
            )
        attempt(self._control.close, "control transport close")
        if resources_quiescent:
            attempt(self._data.close, "data transport close")
        if failures:
            raise failures[0]
        self._closed = True

    def _drain_inflight_for_shutdown(self) -> None:
        """Bounded fail-closed drain of inflight transfers before close.

        Mirrors session._drain_abort but as a single bounded loop. Collects
        inflight transfer_ids from each session, repeatedly calls
        _data.cancel(..., mode="wait") and _data.poll() so handles can
        surface as done/failed. If the deadline expires, retain every owner and
        raise; immediate cancellation cannot prove DMA quiescence and could
        release source memory while a device or NIC still reads it.
        """
        sessions = tuple(self._sessions.values()) + tuple(
            retirement.session for retirement in self._retiring_sessions.values()
        )
        # Recover any data-plane request whose transport return was interrupted
        # before taking the authoritative cancellation snapshot. Recovery does
        # not drain here: the bounded manager loop owns shutdown quiescence.
        for session in sessions:
            session._server.reconcile_submitting_transfer()
        ids = [tid for s in sessions for tid in s._server._inflight]
        if not ids:
            return
        deadline = time.monotonic() + _SHUTDOWN_DRAIN_TIMEOUT_S
        still: list[int] = ids
        while still and time.monotonic() < deadline:
            still = list(self._data.cancel(still, mode="wait"))
            if not still:
                break
            # poll() advances NIXL handle state so the next wait-cancel
            # has a chance to release the handles.
            self._data.poll()
            time.sleep(_DRAIN_SLEEP_S)
        if still:
            logger.warning(
                "P2P %s: shutdown drain timed out after %.1fs with %d "
                "transfers still inflight; resources retained",
                self._local_id,
                _SHUTDOWN_DRAIN_TIMEOUT_S,
                len(still),
            )
            raise RuntimeError(
                "P2P shutdown drain timed out with active transfers; resources retained"
            )
