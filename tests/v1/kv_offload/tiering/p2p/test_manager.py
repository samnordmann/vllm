# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for P2PSecondaryTierManager.

Tests the manager's job routing, session lifecycle, and result collection
using fake transport and session objects.
"""

from __future__ import annotations

import time
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import DEFAULT_NONE_HASH_SEED, init_none_hash
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadPolicy,
    ReqContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob
from vllm.v1.kv_offload.tiering.p2p import manager as manager_module
from vllm.v1.kv_offload.tiering.p2p.manager import (
    _UNBOUND_STORE_TIMEOUT_S,
    P2PSecondaryTierManager,
    _annotate_req_context,
)
from vllm.v1.kv_offload.tiering.p2p.session import (
    LoadResult,
    SessionCloseResult,
    SessionPollResult,
    StoreResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _remote_prefiller_kv_params(
    remote_host: str = "10.0.0.1",
    remote_port: int = 8000,
    kv_request_id: str = "req-1",
) -> dict:
    """Decoder-side kv_transfer_params: ``remote_prefiller`` sub-dict carries
    kv_request_id + remote_host + remote_port."""
    return {
        "remote_prefiller": {
            "kv_request_id": kv_request_id,
            "remote_host": remote_host,
            "remote_port": remote_port,
        },
    }


def _remote_kv_source_kv_params(
    remote_host: str = "10.0.0.1",
    remote_port: int = 8000,
    kv_request_id: str = "req-1",
) -> dict:
    """Symmetric-P2P consumer kv_transfer_params: ``remote_kv_source`` sub-dict has
    the same shape as ``remote_prefiller`` (kv_request_id + remote_host + port)."""
    return {
        "remote_kv_source": {
            "kv_request_id": kv_request_id,
            "remote_host": remote_host,
            "remote_port": remote_port,
        },
    }


def _remote_decoder_kv_params(kv_request_id: str = "req-1") -> dict:
    """Prefiller-side kv_transfer_params: ``remote_decoder`` sub-dict carries
    kv_request_id only."""
    return {"remote_decoder": {"kv_request_id": kv_request_id}}


def _req_context(kv_params: dict | None = None) -> ReqContext:
    ctx = ReqContext(req_id="test", kv_transfer_params=kv_params)
    # Mirror on_new_request: parse the P2P routing state once and cache it,
    # so lookup/submit_*/on_request_finished can read it back via get_state.
    _annotate_req_context(ctx)
    return ctx


def _job_metadata(
    job_id: int,
    keys: list[bytes] | None = None,
    block_ids: list[int] | None = None,
    kv_params: dict | None = None,
) -> TransferJob:
    if keys is None:
        keys = [b"key1"]
    if block_ids is None:
        block_ids = list(range(len(keys)))
    return TransferJob(
        job_id=job_id,
        keys=keys,
        block_ids=np.array(block_ids),
        is_promotion=False,
        req_context=_req_context(kv_params),
    )


def _make_manager() -> P2PSecondaryTierManager:
    """Create a manager with stubbed __init__."""
    mgr = P2PSecondaryTierManager.__new__(P2PSecondaryTierManager)
    mgr._local_id = "127.0.0.1:7777"
    mgr._hash_seed = "0"
    mgr._finished_jobs = {}
    mgr._failed_req_ids = set()
    mgr._sessions = {}
    mgr._retiring_sessions = {}
    mgr._kv_to_session = {}
    mgr._unbound_stores = {}
    mgr._failed_serve_ctxs = []
    mgr._data = None  # type: ignore[assignment]
    mgr._closing = False
    mgr._closed = False
    return mgr


def _finished_results(mgr: P2PSecondaryTierManager) -> list[JobResult]:
    return list(mgr._finished_jobs.values())


def _init_offloading_spec() -> SimpleNamespace:
    """Minimal offloading_spec for driving the real __init__."""
    return SimpleNamespace(
        config=SimpleNamespace(parallel=SimpleNamespace(data_parallel_index=0)),
        blocks_per_chunk=1,
    )


class _ConstructorFailure(BaseException):
    pass


class _ConstructorResource:
    def __init__(
        self,
        *,
        available: bool = True,
        close_error: BaseException | None = None,
    ) -> None:
        self.available = available
        self.close_error = close_error
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


# ---------------------------------------------------------------------------
# Tests for __init__ hash seed resolution
# ---------------------------------------------------------------------------


class TestInitHashSeed:
    def _build(self, monkeypatch) -> P2PSecondaryTierManager:
        monkeypatch.setattr(manager_module, "NixlTransport", lambda *a, **k: object())
        monkeypatch.setattr(manager_module, "ZmqTransport", lambda *a, **k: object())
        monkeypatch.setattr(
            manager_module.FileMapper,
            "from_offloading_spec",
            lambda **k: SimpleNamespace(get_run_config=lambda: {}),
        )
        return P2PSecondaryTierManager(
            offloading_spec=_init_offloading_spec(),
            primary_kv_view=memoryview(bytearray(16)),
        )

    def test_missing_pythonhashseed_uses_default(self, monkeypatch):
        """P2P falls back to the deterministic default seed when unset."""
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        mgr = self._build(monkeypatch)
        init_none_hash(sha256)
        assert mgr._get_hash_seed() == DEFAULT_NONE_HASH_SEED

    def test_pythonhashseed_set_succeeds(self, monkeypatch):
        """With PYTHONHASHSEED set, the handshake advertises it."""
        monkeypatch.setenv("PYTHONHASHSEED", "12345")
        mgr = self._build(monkeypatch)
        init_none_hash(sha256)
        assert mgr._get_hash_seed() == "12345"

    def test_seed_resolved_after_init_none_hash(self, monkeypatch):
        """The seed is read lazily, not at construction time.

        This tier is built before init_none_hash runs, and a non-cryptographic
        hash algorithm seeds NONE_HASH randomly, so resolving in __init__ would
        advertise a value that does not match the NONE_HASH actually in use.
        """
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        mgr = self._build(monkeypatch)
        assert mgr._hash_seed is None
        monkeypatch.setattr(
            manager_module, "get_none_hash_seed", lambda: "random-seed-abc"
        )
        assert mgr._get_hash_seed() == "random-seed-abc"


class TestConstructorRollback:
    @staticmethod
    def _patch_file_mapper(monkeypatch) -> None:
        monkeypatch.setattr(
            manager_module.FileMapper,
            "from_offloading_spec",
            lambda **_: SimpleNamespace(get_run_config=lambda: {}),
        )

    @pytest.mark.parametrize("data_transport", ["nixl", "torch"])
    def test_control_failure_closes_only_selected_data_transport(
        self, monkeypatch, data_transport
    ):
        self._patch_file_mapper(monkeypatch)
        primary = _ConstructorFailure("control construction")
        nixl = _ConstructorResource()
        torch = _ConstructorResource()
        nixl_factory = MagicMock(return_value=nixl)
        torch_factory = MagicMock(return_value=torch)
        monkeypatch.setattr(manager_module, "NixlTransport", nixl_factory)
        monkeypatch.setattr(manager_module, "TorchTransferTransport", torch_factory)

        def fail_control(*args, **kwargs):
            raise primary

        monkeypatch.setattr(manager_module, "ZmqTransport", fail_control)
        with pytest.raises(_ConstructorFailure) as raised:
            P2PSecondaryTierManager(
                _init_offloading_spec(),
                memoryview(bytearray(16)),
                data_transport=data_transport,
            )

        assert raised.value is primary
        selected, unselected = (
            (nixl, torch) if data_transport == "nixl" else (torch, nixl)
        )
        assert selected.close_calls == 1
        assert unselected.close_calls == 0
        assert nixl_factory.call_count == (data_transport == "nixl")
        assert torch_factory.call_count == (data_transport == "torch")

    def test_unavailable_torch_transport_is_closed_before_control(self, monkeypatch):
        self._patch_file_mapper(monkeypatch)
        data = _ConstructorResource(available=False)
        monkeypatch.setattr(
            manager_module, "TorchTransferTransport", MagicMock(return_value=data)
        )
        control_factory = MagicMock(
            side_effect=AssertionError("control must not be constructed")
        )
        monkeypatch.setattr(manager_module, "ZmqTransport", control_factory)

        with pytest.raises(RuntimeError, match="requires an available"):
            P2PSecondaryTierManager(
                _init_offloading_spec(),
                memoryview(bytearray(16)),
                data_transport="torch",
            )

        assert data.close_calls == 1
        control_factory.assert_not_called()

    def test_cleanup_failure_does_not_mask_control_failure(self, monkeypatch):
        self._patch_file_mapper(monkeypatch)
        primary = _ConstructorFailure("control construction")
        cleanup = _ConstructorFailure("data cleanup")
        data = _ConstructorResource(close_error=cleanup)
        monkeypatch.setattr(
            manager_module, "NixlTransport", MagicMock(return_value=data)
        )

        def fail_control(*args, **kwargs):
            raise primary

        monkeypatch.setattr(manager_module, "ZmqTransport", fail_control)
        with pytest.raises(_ConstructorFailure) as raised:
            P2PSecondaryTierManager(_init_offloading_spec(), memoryview(bytearray(16)))

        assert raised.value is primary
        assert data.close_calls == 1

    def test_success_does_not_prematurely_close_transports(self, monkeypatch):
        self._patch_file_mapper(monkeypatch)
        data = _ConstructorResource()
        control = _ConstructorResource()
        monkeypatch.setattr(
            manager_module, "NixlTransport", MagicMock(return_value=data)
        )
        monkeypatch.setattr(
            manager_module, "ZmqTransport", MagicMock(return_value=control)
        )

        manager = P2PSecondaryTierManager(
            _init_offloading_spec(), memoryview(bytearray(16))
        )

        assert manager._data is data
        assert manager._control is control
        assert data.close_calls == control.close_calls == 0


# ---------------------------------------------------------------------------
# Tests for _peer_id_from_params
# ---------------------------------------------------------------------------


class TestPeerIdFromParams:
    def test_valid_params(self):
        result = manager_module._peer_id_from_params(
            {"remote_host": "10.0.0.1", "remote_port": 8000}
        )
        assert result == "10.0.0.1:8000"

    def test_missing_host(self):
        result = manager_module._peer_id_from_params({"remote_port": 8000})
        assert result is None

    def test_missing_port(self):
        result = manager_module._peer_id_from_params({"remote_host": "10.0.0.1"})
        assert result is None

    def test_empty_dict(self):
        result = manager_module._peer_id_from_params({})
        assert result is None


# ---------------------------------------------------------------------------
# Tests for lookup
# ---------------------------------------------------------------------------


class TestLookup:
    def test_lookup_returns_miss_without_kv_params(self):
        mgr = _make_manager()
        ctx = _req_context(kv_params=None)
        assert mgr.lookup(b"key", ctx) is LookupResult.MISS

    def test_lookup_returns_miss_without_required_fields(self):
        mgr = _make_manager()
        ctx = _req_context(kv_params={"remote_prefiller": {"remote_host": "x"}})
        assert mgr.lookup(b"key", ctx) is LookupResult.MISS

    def test_lookup_returns_hit_for_valid_request(self):
        mgr = _make_manager()
        ctx = _req_context(kv_params=_remote_prefiller_kv_params())
        assert mgr.lookup(b"key", ctx) is LookupResult.HIT

    def test_lookup_returns_miss_for_failed_request(self):
        mgr = _make_manager()
        mgr._failed_req_ids.add("req-1")
        ctx = _req_context(kv_params=_remote_prefiller_kv_params(kv_request_id="req-1"))
        assert mgr.lookup(b"key", ctx) is LookupResult.MISS

    def test_lookup_returns_hit_for_different_request_id(self):
        mgr = _make_manager()
        mgr._failed_req_ids.add("req-1")
        ctx = _req_context(kv_params=_remote_prefiller_kv_params(kv_request_id="req-2"))
        assert mgr.lookup(b"key", ctx) is LookupResult.HIT

    def test_lookup_returns_miss_without_prefill_key(self):
        """No ``remote_prefiller`` sub-dict means the request was not routed for
        remote prefill — local prefill should run instead, so lookup()
        returns MISS even when a stale ``remote_decoder`` block is present."""
        mgr = _make_manager()
        ctx = _req_context(kv_params=_remote_decoder_kv_params())
        assert mgr.lookup(b"key", ctx) is LookupResult.MISS


# ---------------------------------------------------------------------------
# Tests for on_new_request offload policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kv_params,expected",
    [
        (_remote_decoder_kv_params(), OffloadPolicy.REQUEST_LEVEL),
        (None, OffloadPolicy.BLOCK_LEVEL),
        (_remote_prefiller_kv_params(), OffloadPolicy.BLOCK_LEVEL),
        ({"remote_decoder": {}}, OffloadPolicy.BLOCK_LEVEL),
    ],
    ids=["producer", "plain", "consumer", "producer_no_id"],
)
def test_on_new_request_policy(monkeypatch, kv_params, expected):
    """Only a producer leg carrying a kv_request_id widens to REQUEST_LEVEL."""
    mgr = _make_manager()
    monkeypatch.setattr(mgr, "_get_or_create_session", lambda peer_id: None)
    assert mgr.on_new_request(_req_context(kv_params=kv_params)).policy is expected


# ---------------------------------------------------------------------------
# Tests for serve_external_requests
# ---------------------------------------------------------------------------


class _RecordingParent:
    """Minimal ParentManager stub recording on_request_finished calls."""

    def __init__(self) -> None:
        self.finished: list[str] = []

    def on_new_request(self, ctx):
        from vllm.v1.kv_offload.base import RequestOffloadingContext

        return RequestOffloadingContext()

    def lookup(self, key, ctx):
        return LookupResult.MISS

    def create_store_job(self, keys, ctx):
        raise AssertionError("unreachable")

    def on_request_finished(self, ctx) -> None:
        self.finished.append(ctx.req_id)


class _RecordingSession:
    """Fake P2PSession that records the parent it was served with."""

    def __init__(self) -> None:
        self.served_with: list[object] = []

    def serve_external_requests(self, parent) -> None:
        self.served_with.append(parent)


class TestServeExternalRequests:
    def test_flushes_failed_serve_ctxs_then_serves_each_session(self):
        """serve_external_requests releases the failed serves left by
        reaped sessions via parent.on_request_finished (clearing the
        queue), then delegates to every live session with the same parent."""
        mgr = _make_manager()
        ctx = ReqContext(req_id="p2p:peer:req-1:lu1")
        mgr._failed_serve_ctxs = [ctx]
        sess_a = _RecordingSession()
        sess_b = _RecordingSession()
        mgr._sessions = {"a": sess_a, "b": sess_b}  # type: ignore[assignment]

        parent = _RecordingParent()
        mgr.serve_external_requests(parent)  # type: ignore[arg-type]

        # Failed serve released and queue cleared.
        assert parent.finished == ["p2p:peer:req-1:lu1"]
        assert mgr._failed_serve_ctxs == []
        # Every live session served with the same parent handle.
        assert sess_a.served_with == [parent]
        assert sess_b.served_with == [parent]

    def test_no_failed_serves_still_serves_sessions(self):
        mgr = _make_manager()
        sess = _RecordingSession()
        mgr._sessions = {"a": sess}  # type: ignore[assignment]

        parent = _RecordingParent()
        mgr.serve_external_requests(parent)  # type: ignore[arg-type]

        assert parent.finished == []
        assert sess.served_with == [parent]


# ---------------------------------------------------------------------------
# Tests for submit_store
# ---------------------------------------------------------------------------


class TestSubmitStore:
    def test_no_decode_succeeds_immediately(self):
        """Without a ``remote_decoder`` block, job succeeds immediately."""
        mgr = _make_manager()
        job = _job_metadata(job_id=1, kv_params={})
        mgr.submit_store(job)
        assert _finished_results(mgr) == [JobResult(job_id=1, success=True)]

    def test_missing_kv_request_id_fails(self):
        """Missing kv_request_id inside ``remote_decoder`` fails the job."""
        mgr = _make_manager()
        params: dict = {"remote_decoder": {}}
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_store(job)
        assert _finished_results(mgr) == [JobResult(job_id=1, success=False)]

    def test_no_binding_yet_parks_in_unbound_stores(self):
        """submit_store without a bound session buffers the batch keyed
        by kv_request_id; no session is pre-created (peer_id is unknown
        to the producer at store time)."""
        mgr = _make_manager()
        job = _job_metadata(
            job_id=1,
            keys=[b"k1", b"k2"],
            block_ids=[3, 4],
            kv_params=_remote_decoder_kv_params(kv_request_id="req-1"),
        )
        mgr.submit_store(job)

        assert mgr._sessions == {}
        assert mgr._finished_jobs == {}
        batches = mgr._unbound_stores["req-1"]
        assert len(batches) == 1
        batch = batches[1]
        assert (
            batch.job_id,
            list(batch.keys),
            list(batch.block_ids),
        ) == (
            1,
            [b"k1", b"k2"],
            [3, 4],
        )

    def test_routes_to_bound_session(self):
        """If a session has already received FetchMsg for this
        kv_request_id (so _kv_to_session is populated), submit_store
        forwards directly to that session rather than re-buffering."""
        mgr = _make_manager()
        bound = _FakeSession(peer_id="10.0.0.1:8000", connected=True)
        mgr._kv_to_session["req-1"] = bound  # type: ignore[assignment]
        # Note: _sessions is intentionally untouched — this test isolates
        # the kv_request_id → session fast path.
        job = _job_metadata(
            job_id=7,
            keys=[b"k1", b"k2"],
            block_ids=[3, 4],
            kv_params=_remote_decoder_kv_params(kv_request_id="req-1"),
        )
        mgr.submit_store(job)

        assert len(bound.stores_added) == 1
        kv_req_id, keys, _, job_id = bound.stores_added[0]
        assert (kv_req_id, keys, job_id) == ("req-1", [b"k1", b"k2"], 7)
        assert mgr._unbound_stores == {}
        assert mgr._finished_jobs == {}

    def test_extra_top_level_keys_are_ignored(self):
        """Producer-side kv_transfer_params should not pre-create a
        session even when a stale caller still passes a top-level
        ``remote_host``/``remote_port`` next to ``remote_decoder``."""
        mgr = _make_manager()
        params = _remote_decoder_kv_params()
        params["remote_host"] = "stale"
        params["remote_port"] = 12345
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_store(job)
        # No session pre-created, no peer-keyed state.
        assert mgr._sessions == {}
        assert "req-1" in mgr._unbound_stores


# ---------------------------------------------------------------------------
# Tests for submit_load
# ---------------------------------------------------------------------------


class TestSubmitLoad:
    def test_missing_params_fails(self):
        """Missing required kv_params fields fails the job."""
        mgr = _make_manager()
        job = _job_metadata(job_id=1, kv_params={})
        mgr.submit_load(job)
        assert _finished_results(mgr) == [JobResult(job_id=1, success=False)]

    def test_empty_keys_succeeds_immediately(self):
        """Empty key list succeeds immediately."""
        mgr = _make_manager()
        job = _job_metadata(
            job_id=1, keys=[], block_ids=[], kv_params=_remote_prefiller_kv_params()
        )
        mgr.submit_load(job)
        assert _finished_results(mgr) == [JobResult(job_id=1, success=True)]

    def test_no_session_fails(self):
        """No session for peer fails and marks request failed."""
        mgr = _make_manager()
        job = _job_metadata(job_id=1, kv_params=_remote_prefiller_kv_params())
        mgr.submit_load(job)
        assert _finished_results(mgr) == [JobResult(job_id=1, success=False)]
        assert "req-1" in mgr._failed_req_ids

    def test_happy_path_with_active_session(self):
        """When the peer's session exists, submit_load forwards to
        session.request_blocks and does NOT add a finished result yet."""
        mgr = _make_manager()
        peer_id = "10.0.0.1:8000"
        existing = _FakeSession(peer_id=peer_id, connected=True)
        mgr._sessions[peer_id] = existing  # type: ignore[assignment]
        job = _job_metadata(
            job_id=42,
            keys=[b"k1", b"k2"],
            block_ids=[5, 6],
            kv_params=_remote_prefiller_kv_params(kv_request_id="req-42"),
        )
        mgr.submit_load(job)

        assert existing.requests == [(42, "req-42")]
        assert mgr._finished_jobs == {}
        assert "req-42" not in mgr._failed_req_ids

    def test_missing_consumer_flag_fails(self):
        """Peer fields present but neither do_remote_prefill nor
        do_p2p_fetch is set — submit_load fails the job rather than
        emit a stray FetchMsg."""
        mgr = _make_manager()
        params = {
            "remote_host": "10.0.0.1",
            "remote_port": 8000,
            "kv_request_id": "req-1",
        }
        job = _job_metadata(job_id=1, kv_params=params)
        mgr.submit_load(job)
        assert _finished_results(mgr) == [JobResult(job_id=1, success=False)]


# ---------------------------------------------------------------------------
# Tests for on_request_finished
# ---------------------------------------------------------------------------


class TestOnRequestFinished:
    def _make_with_failed(self) -> P2PSecondaryTierManager:
        mgr = _make_manager()
        mgr._failed_req_ids = {"req-1"}
        return mgr

    def test_prunes_failed_req_ids(self):
        mgr = self._make_with_failed()
        ctx = _req_context(kv_params=_remote_prefiller_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert "req-1" not in mgr._failed_req_ids

    def test_no_kv_params_does_nothing(self):
        mgr = self._make_with_failed()
        ctx = _req_context(kv_params=None)
        mgr.on_request_finished(ctx)
        assert "req-1" in mgr._failed_req_ids

    def test_no_kv_request_id_does_nothing(self):
        mgr = self._make_with_failed()
        ctx = _req_context(kv_params={"remote_host": "x", "remote_port": 1})
        mgr.on_request_finished(ctx)
        assert "req-1" in mgr._failed_req_ids

    def test_decoder_side_calls_session_finish_request(self):
        """Decoder-side finish (``remote_prefiller`` set) still routes via peer_id
        because the consumer addresses the producer it loaded from. The
        session's finish_request cancels the client-role load."""
        mgr = _make_manager()
        peer_id = "10.0.0.1:8000"
        session = _FakeSession(peer_id=peer_id)
        mgr._sessions[peer_id] = session
        ctx = _req_context(kv_params=_remote_prefiller_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert session.finishes == ["req-1"]

    def test_p2p_consumer_side_calls_session_finish_request(self):
        """Symmetric-P2P consumer finish (``remote_kv_source`` set) routes via peer_id
        so the session drops any pending lookups (cancel_lookups) and
        cancels any inbound load."""
        mgr = _make_manager()
        peer_id = "10.0.0.1:8000"
        session = _FakeSession(peer_id=peer_id)
        mgr._sessions[peer_id] = session
        ctx = _req_context(kv_params=_remote_kv_source_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert session.finishes == ["req-1"]

    def test_prefiller_bound_id_routes_via_kv_to_session(self):
        """Prefiller-side finish for an id whose session is already bound
        (FetchMsg received) routes via _kv_to_session and pops the entry."""
        mgr = _make_manager()
        bound = _FakeSession(peer_id="some-peer:1", connected=True)
        mgr._kv_to_session["req-1"] = bound  # type: ignore[assignment]
        ctx = _req_context(kv_params=_remote_decoder_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert bound.finishes == ["req-1"]
        assert "req-1" not in mgr._kv_to_session

    def test_prefiller_route_unlinks_only_after_finish_returns(self):
        primary = _ConstructorFailure("finish interrupted")

        class InterruptedFinish(_FakeSession):
            def __init__(self) -> None:
                super().__init__(peer_id="some-peer:1", connected=True)
                self.interrupt = True

            def finish_request(self, kv_request_id):
                if self.interrupt:
                    self.interrupt = False
                    raise primary
                super().finish_request(kv_request_id)

        mgr = _make_manager()
        session = InterruptedFinish()
        mgr._kv_to_session["req-1"] = session  # type: ignore[assignment]
        ctx = _req_context(kv_params=_remote_decoder_kv_params(kv_request_id="req-1"))

        with pytest.raises(_ConstructorFailure) as raised:
            mgr.on_request_finished(ctx)
        assert raised.value is primary
        assert mgr._kv_to_session["req-1"] is session

        mgr.on_request_finished(ctx)
        assert session.finishes == ["req-1"]
        assert "req-1" not in mgr._kv_to_session

    def test_prefiller_unbound_id_leaves_batches_parked(self):
        """Prefiller-side finish for an id with parked unbound batches
        and no session binding is a no-op on `_unbound_stores`. The
        parked batches survive until a peer fetches them or the
        `_reap_unbound_stores` timeout fires — `on_request_finished`
        must not evict them."""
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        mgr = _make_manager()
        mgr._unbound_stores["req-1"] = {
            10: _UnboundStoreBatch(job_id=10, keys=[b"k"], block_ids=[0]),
            11: _UnboundStoreBatch(job_id=11, keys=[b"k2"], block_ids=[1]),
        }
        ctx = _req_context(kv_params=_remote_decoder_kv_params(kv_request_id="req-1"))
        mgr.on_request_finished(ctx)
        assert "req-1" in mgr._unbound_stores
        assert list(mgr._unbound_stores["req-1"]) == [10, 11]
        outcomes = {(r.job_id, r.success) for r in _finished_results(mgr)}
        assert (10, False) not in outcomes
        assert (11, False) not in outcomes


# ---------------------------------------------------------------------------
# Tests for get_finished_jobs
# ---------------------------------------------------------------------------


class _FakeServerHalf:
    def __init__(self) -> None:
        self._inflight: dict[int, object] = {}
        self.reconcile_calls = 0
        self.recover_on_reconcile: dict[int, object] = {}
        self.reconcile_error: BaseException | None = None

    def reconcile_submitting_transfer(self) -> None:
        self.reconcile_calls += 1
        if self.reconcile_error is not None:
            raise self.reconcile_error
        self._inflight.update(self.recover_on_reconcile)

    @property
    def has_inflight_transfers(self) -> bool:
        return bool(self._inflight)


class _FakeClientHalf:
    def __init__(self) -> None:
        self._inbound: dict[int, object] = {}

    @property
    def has_active_loads(self) -> bool:
        return bool(self._inbound)


class _FakeSession:
    """Fake bidirectional session that returns canned poll() results."""

    def __init__(
        self,
        peer_id: str = "fake:1",
        alive: bool = True,
        connected: bool = True,
        loads: list[LoadResult] | None = None,
        stores: list[StoreResult] | None = None,
        new_fetch_ids: list[str] | None = None,
        close_jobs: list[int] | None = None,
        close_req_ids: list[str] | None = None,
        close_stores: list[int] | None = None,
        close_failed_serves: list[ReqContext] | None = None,
        close_errors: list[BaseException] | None = None,
        close_complete: bool = True,
    ) -> None:
        self.peer_id = peer_id
        self.alive = alive
        self.connected = connected
        self.ready = True
        self._loads = {result.job_id: result for result in loads or ()}
        self._stores = {result.job_id: result for result in stores or ()}
        self._new_fetch_ids = dict.fromkeys(new_fetch_ids or ())
        self._close_jobs = close_jobs or []
        self._close_req_ids = close_req_ids or []
        self._close_stores = close_stores or []
        self._close_failed_serves = close_failed_serves or []
        self._close_errors = list(close_errors or [])
        self.close_complete = close_complete
        self.close_calls = 0
        self.requests: list[tuple[int, str]] = []
        self.stores_added: list[tuple[str, list, object, int]] = []
        self.attached: list[object] = []
        self.finishes: list[str] = []
        self.ack_calls: list[
            tuple[tuple[int, ...], tuple[int, ...], tuple[str, ...]]
        ] = []
        # Mirror P2PSession._server._inflight (transfer_id → handle) and
        # P2PSession._client.has_active_loads for the shutdown-drain and
        # drain_jobs paths. Tests populate _server._inflight when needed.
        self._server = _FakeServerHalf()
        self._client = _FakeClientHalf()

    @property
    def has_pending_work(self) -> bool:
        return self._client.has_active_loads or self._server.has_inflight_transfers

    def poll(self):
        return self.pending_results()

    def pending_results(self):
        return SessionPollResult(
            loads=list(self._loads.values()),
            stores=list(self._stores.values()),
            new_fetch_ids=list(self._new_fetch_ids),
        )

    def ack_results(self, load_job_ids=(), store_job_ids=(), new_fetch_ids=()) -> None:
        loads = tuple(load_job_ids)
        stores = tuple(store_job_ids)
        fetches = tuple(new_fetch_ids)
        self.ack_calls.append((loads, stores, fetches))
        for job_id in loads:
            self._loads.pop(job_id, None)
        for job_id in stores:
            self._stores.pop(job_id, None)
        for kv_request_id in fetches:
            self._new_fetch_ids.pop(kv_request_id, None)

    def owns_store_job(self, job_id: int) -> bool:
        return job_id in self._stores or any(
            stored_job_id == job_id for _, _, _, stored_job_id in self.stores_added
        )

    def request_blocks(self, job_id, kv_request_id, keys, block_ids):
        self.requests.append((job_id, kv_request_id))

    def add_stored_blocks(self, kv_request_id, keys, block_ids, job_id):
        self.stores_added.append((kv_request_id, list(keys), block_ids, job_id))

    def attach_connection(self, conn):
        self.attached.append(conn)
        self.connected = True

    def finish_request(self, kv_request_id):
        self.finishes.append(kv_request_id)

    def close(self):
        self.close_calls += 1
        if self._close_errors:
            raise self._close_errors.pop(0)
        return SessionCloseResult(
            failed_jobs=self._close_jobs,
            failed_req_ids=self._close_req_ids,
            failed_stores=self._close_stores,
            failed_serves=self._close_failed_serves,
        )


class TestGetFinished:
    def _make(self) -> P2PSecondaryTierManager:
        mgr = _make_manager()
        mgr._finished_jobs = {
            1: JobResult(job_id=1, success=True),
            2: JobResult(job_id=2, success=False),
        }

        class FakeControl:
            def poll(self):
                return []

        mgr._control = FakeControl()  # type: ignore[assignment]
        mgr._data = None  # type: ignore[assignment]
        return mgr

    def test_drains_finished_jobs(self):
        """get_finished_jobs returns and clears accumulated results."""
        mgr = self._make()
        results = list(mgr.get_finished_jobs())
        assert len(results) == 2
        assert JobResult(job_id=1, success=True) in results
        assert JobResult(job_id=2, success=False) in results
        # Second call returns empty.
        assert list(mgr.get_finished_jobs()) == []

    def test_reaps_dead_sessions(self):
        """Dead connected sessions are removed and their pending jobs failed."""

        class FakeData:
            def remove_remote_peer(self, pid):
                pass

        mgr = self._make()
        mgr._data = FakeData()  # type: ignore[assignment]
        dead = _FakeSession(
            peer_id="dead:1234",
            alive=False,
            connected=True,
            close_jobs=[20],
            close_req_ids=["req-load"],
            close_stores=[10, 11],
        )
        mgr._sessions["dead:1234"] = dead  # type: ignore[assignment]

        results = list(mgr.get_finished_jobs())
        # 2 baseline + 1 failed load + 2 failed stores
        assert len(results) == 5
        assert JobResult(job_id=10, success=False) in results
        assert JobResult(job_id=11, success=False) in results
        assert JobResult(job_id=20, success=False) in results
        assert "dead:1234" not in mgr._sessions
        assert "req-load" in mgr._failed_req_ids

    def test_reap_retains_dead_session_until_close_retry_succeeds(self):
        primary = _ConstructorFailure("connection close")

        class FakeData:
            def __init__(self):
                self.removed: list[str] = []

            def remove_remote_peer(self, peer_id):
                self.removed.append(peer_id)

        manager = self._make()
        data = FakeData()
        manager._data = data  # type: ignore[assignment]
        session = _FakeSession(
            peer_id="dead:1234",
            alive=False,
            connected=True,
            close_jobs=[20],
            close_errors=[primary],
        )
        manager._sessions[session.peer_id] = session  # type: ignore[assignment]

        with pytest.raises(_ConstructorFailure) as raised:
            manager._reap_dead_sessions()

        assert raised.value is primary
        assert manager._sessions == {}
        assert manager._retiring_sessions[session.peer_id].session is session
        assert data.removed == []
        assert JobResult(job_id=20, success=False) not in _finished_results(manager)

        manager._reap_dead_sessions()
        assert manager._sessions == {}
        assert manager._retiring_sessions == {}
        assert data.removed == [session.peer_id]
        assert (
            _finished_results(manager).count(JobResult(job_id=20, success=False)) == 1
        )
        assert session.close_calls == 2

    def test_reap_quarantines_load_until_session_proves_quiescence(self, monkeypatch):
        class FakeData:
            def __init__(self):
                self.removed: list[str] = []

            def remove_remote_peer(self, peer_id):
                self.removed.append(peer_id)

        manager = self._make()
        data = FakeData()
        manager._data = data  # type: ignore[assignment]
        session = _FakeSession(
            peer_id="dead:quarantine",
            alive=False,
            close_jobs=[73],
            close_req_ids=["req-73"],
            close_complete=False,
        )
        session._client._inbound[73] = object()
        manager._sessions[session.peer_id] = session  # type: ignore[assignment]
        warnings: list[str] = []
        monkeypatch.setattr(
            manager_module.logger,
            "warning",
            lambda message, *args: warnings.append(message % args),
        )

        manager._reap_dead_sessions()

        assert manager._sessions == {}
        assert manager._retiring_sessions[session.peer_id].session is session
        assert JobResult(73, False) not in _finished_results(manager)
        assert data.removed == []
        assert len(warnings) == 1
        assert "ZMQ disconnect is not NIXL quiescence" in warnings[0]
        manager._reap_dead_sessions()
        assert len(warnings) == 1

    def test_reap_retries_peer_removal_after_session_detaches(self):
        primary = _ConstructorFailure("peer removal")

        class FakeData:
            def __init__(self):
                self.removed: list[str] = []

            def remove_remote_peer(self, peer_id):
                self.removed.append(peer_id)
                if len(self.removed) == 1:
                    raise primary

        class DetachingSession(_FakeSession):
            def close(self):
                result = super().close()
                self.connected = False
                self.alive = False
                return result

        manager = self._make()
        data = FakeData()
        manager._data = data  # type: ignore[assignment]
        failed_serve = MagicMock(spec=ReqContext)
        session = DetachingSession(
            peer_id="dead:1234",
            alive=False,
            connected=True,
            close_jobs=[20],
            close_req_ids=["req-load"],
            close_stores=[10],
            close_failed_serves=[failed_serve],
        )
        manager._sessions[session.peer_id] = session  # type: ignore[assignment]
        manager._kv_to_session["req-store"] = session  # type: ignore[assignment]

        with pytest.raises(_ConstructorFailure) as raised:
            manager._reap_dead_sessions()

        assert raised.value is primary
        assert manager._sessions == {}
        assert manager._kv_to_session == {}
        retirement = manager._retiring_sessions[session.peer_id]
        assert retirement.session is session
        assert retirement.close_result is not None
        assert retirement.peer_removed is False
        assert JobResult(20, False) not in _finished_results(manager)
        assert JobResult(10, False) not in _finished_results(manager)

        manager._reap_dead_sessions()
        assert manager._retiring_sessions == {}
        assert data.removed == [session.peer_id, session.peer_id]
        assert _finished_results(manager).count(JobResult(20, False)) == 1
        assert _finished_results(manager).count(JobResult(10, False)) == 1
        assert manager._failed_req_ids == {"req-load"}
        assert manager._failed_serve_ctxs == [failed_serve]

    def test_reap_publication_retry_is_exactly_once_per_sink(self):
        req_failure = _ConstructorFailure("request publication")
        serve_failure = _ConstructorFailure("serve publication")

        class FakeData:
            def remove_remote_peer(self, _peer_id):
                pass

        class FailFirstUpdate(set):
            def __init__(self):
                super().__init__()
                self.failed = False

            def update(self, values):
                if not self.failed:
                    self.failed = True
                    raise req_failure
                return super().update(values)

        class FailAfterFirstAppend(list):
            def __init__(self):
                super().__init__()
                self.failed = False

            def append(self, value):
                super().append(value)
                if not self.failed:
                    self.failed = True
                    raise serve_failure

        manager = self._make()
        manager._data = FakeData()  # type: ignore[assignment]
        manager._failed_req_ids = FailFirstUpdate()
        manager._failed_serve_ctxs = FailAfterFirstAppend()
        ctx_a = MagicMock(spec=ReqContext)
        ctx_b = MagicMock(spec=ReqContext)
        session = _FakeSession(
            peer_id="dead:1234",
            alive=False,
            connected=True,
            close_jobs=[20],
            close_req_ids=["req-load"],
            close_stores=[10],
            close_failed_serves=[ctx_a, ctx_b],
        )
        manager._sessions[session.peer_id] = session  # type: ignore[assignment]

        with pytest.raises(_ConstructorFailure) as first:
            manager._reap_dead_sessions()
        assert first.value is req_failure
        assert _finished_results(manager).count(JobResult(20, False)) == 1
        assert _finished_results(manager).count(JobResult(10, False)) == 1

        with pytest.raises(_ConstructorFailure) as second:
            manager._reap_dead_sessions()
        assert second.value is serve_failure
        assert _finished_results(manager).count(JobResult(20, False)) == 1
        assert _finished_results(manager).count(JobResult(10, False)) == 1
        assert list(manager._failed_serve_ctxs) == [ctx_a]

        manager._reap_dead_sessions()
        assert manager._retiring_sessions == {}
        assert _finished_results(manager).count(JobResult(20, False)) == 1
        assert _finished_results(manager).count(JobResult(10, False)) == 1
        assert manager._failed_req_ids == {"req-load"}
        assert list(manager._failed_serve_ctxs) == [ctx_a, ctx_b]
        assert session.close_calls == 1

    def test_reap_retains_store_pins_until_session_is_quiescent(self):
        class FakeData:
            def __init__(self):
                self.removed: list[str] = []

            def remove_remote_peer(self, peer_id):
                self.removed.append(peer_id)

        class DrainingSession(_FakeSession):
            def __init__(self):
                super().__init__(
                    peer_id="dead:1234",
                    alive=False,
                    connected=True,
                    close_stores=[10],
                )
                self.close_complete = False

            def close(self):
                result = super().close()
                self.connected = False
                if self.close_calls >= 2:
                    self.close_complete = True
                return result

        manager = self._make()
        data = FakeData()
        manager._data = data  # type: ignore[assignment]
        session = DrainingSession()
        manager._sessions[session.peer_id] = session  # type: ignore[assignment]

        manager._reap_dead_sessions()
        assert manager._sessions == {}
        assert manager._retiring_sessions[session.peer_id].session is session
        assert data.removed == []
        assert JobResult(10, False) not in _finished_results(manager)

        manager._reap_dead_sessions()
        assert manager._retiring_sessions == {}
        assert data.removed == [session.peer_id]
        assert _finished_results(manager).count(JobResult(10, False)) == 1
        assert session.close_calls == 2

    def test_reap_fails_probes(self):
        """A reaped session's in-flight lookups land in _failed_req_ids so
        the consumer's lookup() returns MISS instead of RETRY forever."""

        class FakeData:
            def remove_remote_peer(self, pid):
                pass

        mgr = self._make()
        mgr._data = FakeData()  # type: ignore[assignment]
        dead = _FakeSession(
            peer_id="dead:1234",
            alive=False,
            connected=True,
            close_req_ids=["req-probe-1", "req-probe-2"],
        )
        mgr._sessions["dead:1234"] = dead  # type: ignore[assignment]

        list(mgr.get_finished_jobs())
        assert "dead:1234" not in mgr._sessions
        assert "req-probe-1" in mgr._failed_req_ids
        assert "req-probe-2" in mgr._failed_req_ids

    def test_unbound_store_kept_within_timeout(self):
        """Recently-parked unbound stores stay across a poll."""
        mgr = self._make()
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        mgr._unbound_stores["req-fresh"] = {
            99: _UnboundStoreBatch(job_id=99, keys=[b"k"], block_ids=[0])
        }
        list(mgr.get_finished_jobs())
        assert "req-fresh" in mgr._unbound_stores

    def test_unbound_store_reaped_after_timeout(self):
        """Unbound stores past _UNBOUND_STORE_TIMEOUT_S surface as failed
        and their kv_request_id lands in _failed_req_ids so a late
        FetchMsg/lookup doesn't try to satisfy them."""
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        mgr = self._make()
        stale = _UnboundStoreBatch(job_id=10, keys=[b"k"], block_ids=[0])
        # Backdate the submission so the head batch is past the deadline.
        stale.submitted_at = time.monotonic() - _UNBOUND_STORE_TIMEOUT_S - 1.0
        mgr._unbound_stores["req-stale"] = {
            10: stale,
            11: _UnboundStoreBatch(job_id=11, keys=[b"k2"], block_ids=[1]),
        }

        results = list(mgr.get_finished_jobs())

        assert "req-stale" not in mgr._unbound_stores
        # 2 baseline + 2 buffered stores
        assert JobResult(job_id=10, success=False) in results
        assert JobResult(job_id=11, success=False) in results
        assert "req-stale" in mgr._failed_req_ids

    def test_unbound_timeout_publication_cut_replays_before_unlink(self, monkeypatch):
        """A post-publication cut retains every authoritative batch."""
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        primary = _ConstructorFailure("after result publication")

        class InterruptingResults(dict):
            interrupt = True

            def setdefault(self, key, default=None):
                result = super().setdefault(key, default)
                if self.interrupt:
                    self.interrupt = False
                    raise primary
                return result

        mgr = _make_manager()
        stale = _UnboundStoreBatch(job_id=10, keys=[b"k"], block_ids=[0])
        stale.submitted_at = time.monotonic() - _UNBOUND_STORE_TIMEOUT_S - 1.0
        second = _UnboundStoreBatch(job_id=11, keys=[b"k2"], block_ids=[1])
        mgr._unbound_stores["req-cut"] = {10: stale, 11: second}
        mgr._finished_jobs = InterruptingResults()
        warnings: list[str] = []
        monkeypatch.setattr(
            manager_module.logger,
            "warning",
            lambda message, *args: warnings.append(message % args),
        )

        with pytest.raises(_ConstructorFailure) as raised:
            mgr._reap_unbound_stores()
        assert raised.value is primary
        assert list(mgr._unbound_stores["req-cut"]) == [10, 11]
        assert _finished_results(mgr) == [JobResult(10, False)]

        mgr._reap_unbound_stores()
        assert "req-cut" not in mgr._unbound_stores
        assert _finished_results(mgr) == [
            JobResult(10, False),
            JobResult(11, False),
        ]
        assert "req-cut" in mgr._failed_req_ids
        assert any("failing 2 job(s)" in warning for warning in warnings)

    def test_submit_store_parks_unbound_batch(self):
        """submit_store on an unbound id appends a batch with a fresh
        submitted_at stamp so the unbound-store sweep can age it out."""
        mgr = _make_manager()
        job = _job_metadata(
            job_id=1, kv_params=_remote_decoder_kv_params(kv_request_id="req-1")
        )
        before = time.monotonic()
        mgr.submit_store(job)
        after = time.monotonic()
        batches = mgr._unbound_stores["req-1"]
        assert len(batches) == 1
        assert before <= batches[1].submitted_at <= after


# ---------------------------------------------------------------------------
# has_pending_work
# ---------------------------------------------------------------------------


class TestHasPendingWork:
    """has_pending_work() must always return True so the engine keeps
    ticking the offload pipeline — that's the only thread driving
    _control.poll() (incoming peer connects) and session.poll()
    (incoming fetch messages on existing sessions)."""

    def test_returns_true_unconditionally_to_keep_engine_ticking(self):
        """Even with no sessions and no jobs, has_pending_work() returns
        True so the engine keeps calling get_finished_jobs(), which is
        what drives _control.poll() for inbound peer connects."""
        mgr = _make_manager()
        assert mgr.has_pending_work() is True

    def test_returns_true_even_when_sessions_present(self):
        """The result is the same regardless of session state — there is
        no 'idle' branch."""
        mgr = _make_manager()
        mgr._sessions["peer:1"] = _FakeSession(peer_id="peer:1")  # type: ignore[assignment]
        assert mgr.has_pending_work() is True


# ---------------------------------------------------------------------------
# Shutdown drain
# ---------------------------------------------------------------------------


class _ShutdownFakeData:
    """Fake DataTransport that records cancel/poll/close calls and
    drives the wait-cancel loop with a scriptable `still` queue."""

    def __init__(self, still_queue: list[list[int]] | None = None) -> None:
        # Each list in still_queue is the set of ids the next
        # cancel(mode="wait") should report as still inflight. The last
        # entry repeats once exhausted.
        self._still_queue = list(still_queue) if still_queue else [[]]
        self.cancel_calls: list[tuple[list[int], str]] = []
        self.poll_calls: int = 0
        self.close_calls: int = 0

    def cancel(self, transfer_ids, mode: str = "immediate") -> list[int]:
        ids = list(transfer_ids)
        self.cancel_calls.append((ids, mode))
        if mode == "wait":
            if len(self._still_queue) > 1:
                return list(self._still_queue.pop(0))
            return list(self._still_queue[0])
        return []

    def poll(self, peer_id=None):
        self.poll_calls += 1

        class _Empty:
            done: list[int] = []
            failed: list[int] = []

        return _Empty()

    def close(self) -> None:
        self.close_calls += 1


class _ShutdownFakeControl:
    def __init__(self) -> None:
        self.close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


class TestShutdownDrain:
    """shutdown() drains with wait-mode or retains resources and raises."""

    def _prep(
        self,
        still_queue: list[list[int]] | None = None,
        inflight_ids: list[int] | None = None,
    ) -> tuple[P2PSecondaryTierManager, _ShutdownFakeData, _ShutdownFakeControl]:
        mgr = _make_manager()
        data = _ShutdownFakeData(still_queue=still_queue)
        control = _ShutdownFakeControl()
        mgr._data = data  # type: ignore[assignment]
        mgr._control = control  # type: ignore[assignment]
        if inflight_ids:
            session = _FakeSession(peer_id="peer:1", connected=True)
            session._server._inflight = {tid: object() for tid in inflight_ids}
            mgr._sessions["peer:1"] = session  # type: ignore[assignment]
        return mgr, data, control

    def test_shutdown_drains_inflight_via_wait_cancel(self):
        # First cancel(wait) returns the input still inflight; second returns [].
        mgr, data, control = self._prep(
            still_queue=[[42, 43], []],
            inflight_ids=[42, 43],
        )
        mgr.shutdown()

        wait_calls = [c for c in data.cancel_calls if c[1] == "wait"]
        immediate_calls = [c for c in data.cancel_calls if c[1] == "immediate"]
        assert len(wait_calls) >= 1
        assert wait_calls[0][0] == [42, 43]
        assert immediate_calls == []
        # poll() was driven between cancel attempts.
        assert data.poll_calls >= 1
        # _data and _control were closed exactly once each, after the drain.
        assert data.close_calls == 1
        assert control.close_calls == 1

    def test_shutdown_reconciles_lost_return_before_cancel_snapshot(self):
        mgr, data, _ = self._prep()
        session = _FakeSession(peer_id="peer:1", connected=True)
        recovered_owner = object()
        session._server.recover_on_reconcile[77] = recovered_owner
        mgr._sessions["peer:1"] = session  # type: ignore[assignment]

        mgr._drain_inflight_for_shutdown()

        assert session._server.reconcile_calls == 1
        assert session._server._inflight[77] is recovered_owner
        assert data.cancel_calls == [([77], "wait")]

    def test_shutdown_recovery_failure_retains_owner_before_snapshot(self):
        mgr, data, _ = self._prep()
        session = _FakeSession(peer_id="peer:1", connected=True)
        recovery_error = _ConstructorFailure("ambiguous recovery")
        session._server.reconcile_error = recovery_error
        mgr._sessions["peer:1"] = session  # type: ignore[assignment]

        with pytest.raises(_ConstructorFailure) as raised:
            mgr._drain_inflight_for_shutdown()

        assert raised.value is recovery_error
        assert session._server.reconcile_calls == 1
        assert data.cancel_calls == []

    def test_shutdown_closes_control_and_preserves_first_drain_failure(self):
        manager, data, control = self._prep()
        first = _ConstructorFailure("drain")
        control_error = _ConstructorFailure("control")
        manager._drain_inflight_for_shutdown = MagicMock(side_effect=first)
        control.close = MagicMock(side_effect=control_error)

        with pytest.raises(_ConstructorFailure) as raised:
            manager.shutdown()

        assert raised.value is first
        control.close.assert_called_once_with()
        assert data.close_calls == 0

    def test_shutdown_timeout_raises_and_retains_resources(self, monkeypatch):
        # Drain never completes — wait-cancel keeps returning the inflight set.
        # Use a synthetic clock so the test does not depend on real wallclock
        # being able to advance in <50ms on a loaded CI node:
        #   call 1 (deadline calc): 100.0  -> deadline = 100.05
        #   call 2 (loop predicate): 100.0 -> enters loop, one wait-cancel
        #   call 3 (loop predicate): 100.06 -> exits and fails closed
        monkeypatch.setattr(manager_module, "_SHUTDOWN_DRAIN_TIMEOUT_S", 0.05)
        monkeypatch.setattr(manager_module, "_DRAIN_SLEEP_S", 0.0)
        # The final two values drive the successful retry below.
        times = iter([100.0, 100.0, 100.06, 101.0, 101.0])
        # Patch via a fake module on `manager_module.time` so we do not mutate
        # the global `time` module — other code in the process (e.g. the
        # buildkite test collector's pytest_runtest_logreport hook) calls
        # time.monotonic() before monkeypatch teardown.
        fake_time = SimpleNamespace(monotonic=lambda: next(times), sleep=time.sleep)
        monkeypatch.setattr(manager_module, "time", fake_time)

        mgr, data, control = self._prep(
            still_queue=[[42]],
            inflight_ids=[42],
        )
        with pytest.raises(RuntimeError, match="resources retained"):
            mgr.shutdown()

        wait_calls = [c for c in data.cancel_calls if c[1] == "wait"]
        immediate_calls = [c for c in data.cancel_calls if c[1] == "immediate"]
        assert len(wait_calls) == 1
        assert wait_calls[0][0] == [42]
        assert immediate_calls == []
        assert "peer:1" in mgr._sessions
        assert 42 in mgr._sessions["peer:1"]._server._inflight
        assert data.close_calls == 0
        assert control.close_calls == 1

        # A later shutdown may retry after the transport proves quiescence.
        data._still_queue = [[]]
        mgr.shutdown()
        assert mgr._sessions == {}
        assert data.close_calls == 1
        assert control.close_calls == 2

    def test_shutdown_no_inflight_skips_drain(self):
        mgr, data, control = self._prep()
        mgr.shutdown()

        assert data.cancel_calls == []
        assert data.poll_calls == 0
        assert data.close_calls == 1
        assert control.close_calls == 1


# ---------------------------------------------------------------------------
# Bidirectional regression test — both managers act as client AND server
# toward each other on the same peer_id. This is the case the unification
# is meant to fix.
# ---------------------------------------------------------------------------


class _LoopbackControl:
    """In-memory control transport that pairs two managers head-to-head.

    Each manager hands its outbound message buffer to the other's inbound
    queue on poll(). connect() returns a connection whose send() writes
    into the peer's inbound side; recv() reads what the peer's connect-or-
    poll path delivered for us.
    """

    def __init__(self, local_id: str) -> None:
        self._local_id = local_id
        self._peer: _LoopbackControl | None = None
        self._inbound_outbound: dict[str, _LoopbackConnection] = {}
        # Pending inbound for a peer that has not yet been registered.
        self._pending: list[tuple[str, dict]] = []

    def pair(self, peer: _LoopbackControl) -> None:
        self._peer = peer
        peer._peer = self

    def connect(self, peer_id: str):
        if peer_id in self._inbound_outbound:
            raise AssertionError(f"already connected to {peer_id}")
        conn = _LoopbackConnection(self, peer_id)
        self._inbound_outbound[peer_id] = conn
        return conn

    def poll(self):
        # Drain whatever the peer has sent toward us.
        new = []
        if self._peer is not None:
            for pid, msg in self._peer._drain_outbound_to(self._local_id):
                conn = self._inbound_outbound.get(pid)
                if conn is None:
                    conn = _LoopbackConnection(self, pid)
                    self._inbound_outbound[pid] = conn
                    new.append(conn)
                conn._inbox.append(msg)
        return new

    def _drain_outbound_to(self, peer_local_id: str):
        # Peer is calling our poll → return all messages we've sent toward
        # peer_local_id (which is the peer's own local_id).
        out: list[tuple[str, dict]] = []
        for pid, conn in list(self._inbound_outbound.items()):
            # Each outgoing message goes to peer_local_id and is tagged
            # with the sender's local_id (i.e., self._local_id).
            for msg in conn._outbox:
                out.append((self._local_id, msg))
            conn._outbox.clear()
        return out

    def close(self):
        for conn in self._inbound_outbound.values():
            conn.close()
        self._inbound_outbound.clear()


class _LoopbackConnection:
    def __init__(self, transport: _LoopbackControl, peer_id: str) -> None:
        self._transport = transport
        self.peer_id = peer_id
        self._inbox: list[dict] = []
        self._outbox: list[dict] = []
        self._closed = False

    @property
    def alive(self) -> bool:
        return not self._closed

    def send(self, msg: dict) -> None:
        if self._closed:
            raise RuntimeError("send on closed conn")
        self._outbox.append(msg)

    def recv(self) -> list[dict]:
        msgs = self._inbox
        self._inbox = []
        return msgs

    def mark_dead(self) -> None:
        self._closed = True

    def close(self) -> None:
        self._closed = True


class _FakeData:
    """Minimal NIXL fake that lets matched transfers complete on the next poll."""

    def __init__(self, local_id: str) -> None:
        self._local_id = local_id
        self.block_len = 4096
        self.base_addr = 0x1000
        self.num_blocks = 16
        self.config_fingerprint = ""
        self._remote_peers: dict[str, dict] = {}
        self._removed_peers: list[str] = []
        self._inflight_done: list[int] = []
        self._next_id = 0

    def get_agent_metadata(self) -> bytes:
        return f"meta-{self._local_id}".encode()

    def add_remote_peer(
        self, peer_id, agent_metadata, base_addr, num_blocks, block_len
    ) -> None:
        self._remote_peers[peer_id] = {
            "agent_metadata": agent_metadata,
            "base_addr": base_addr,
            "num_blocks": num_blocks,
            "block_len": block_len,
        }

    def remove_remote_peer(self, peer_id: str) -> None:
        self._removed_peers.append(peer_id)
        self._remote_peers.pop(peer_id, None)

    def write_blocks(self, peer_id, local_idxs, remote_idxs):
        if peer_id not in self._remote_peers:
            return None
        tid = self._next_id
        self._next_id += 1
        self._inflight_done.append(tid)
        return tid

    def poll(self, peer_id=None):
        from vllm.v1.kv_offload.tiering.p2p.data.base import PollResult

        done = self._inflight_done[:]
        self._inflight_done.clear()
        return PollResult(done=done, failed=[])

    def ack_completions(self, peer_id, transfer_ids) -> None:
        del peer_id, transfer_ids

    def cancel(self, transfer_ids) -> None:
        pass

    def close(self) -> None:
        pass


def _build_paired_managers() -> tuple[P2PSecondaryTierManager, P2PSecondaryTierManager]:
    """Two managers each acting as both client and server toward the other.

    Wires _LoopbackControl pair + per-side _FakeData so transfers complete
    on the next poll. The test drives polling by calling get_finished_jobs(),
    which invokes _poll_once synchronously on the calling thread.
    """
    mgr_a = _make_manager()
    mgr_b = _make_manager()
    mgr_a._local_id = "A:1"
    mgr_b._local_id = "B:2"

    ctrl_a = _LoopbackControl(mgr_a._local_id)
    ctrl_b = _LoopbackControl(mgr_b._local_id)
    ctrl_a.pair(ctrl_b)

    mgr_a._control = ctrl_a  # type: ignore[assignment]
    mgr_b._control = ctrl_b  # type: ignore[assignment]
    mgr_a._data = _FakeData(mgr_a._local_id)  # type: ignore[assignment]
    mgr_b._data = _FakeData(mgr_b._local_id)  # type: ignore[assignment]

    return mgr_a, mgr_b


class TestBidirectionalManager:
    """Two managers each load FROM and serve TO the other over a single peer."""

    def test_both_loads_succeed(self):
        mgr_a, mgr_b = _build_paired_managers()

        a_loads_kv = "req-AtoB-load"  # A loads, B serves
        b_loads_kv = "req-BtoA-load"  # B loads, A serves

        a_decoder_params = {
            "remote_prefiller": {
                "kv_request_id": a_loads_kv,
                "remote_host": "B",
                "remote_port": 2,
            },
        }
        b_decoder_params = {
            "remote_prefiller": {
                "kv_request_id": b_loads_kv,
                "remote_host": "A",
                "remote_port": 1,
            },
        }
        a_prefiller_params = {"remote_decoder": {"kv_request_id": b_loads_kv}}
        b_prefiller_params = {"remote_decoder": {"kv_request_id": a_loads_kv}}

        # 1. Both sides open client-role sessions toward the peer.
        mgr_a.on_new_request(_req_context(a_decoder_params))
        mgr_b.on_new_request(_req_context(b_decoder_params))

        # 2. Both sides store the blocks the peer will fetch.
        mgr_a.submit_store(
            _job_metadata(
                job_id=100,
                keys=[b"a-block"],
                block_ids=[0],
                kv_params=a_prefiller_params,
            )
        )
        mgr_b.submit_store(
            _job_metadata(
                job_id=200,
                keys=[b"b-block"],
                block_ids=[0],
                kv_params=b_prefiller_params,
            )
        )

        # 3. Both sides submit loads.
        mgr_a.submit_load(
            _job_metadata(
                job_id=101,
                keys=[b"b-block"],
                block_ids=[0],
                kv_params=a_decoder_params,
            )
        )
        mgr_b.submit_load(
            _job_metadata(
                job_id=201,
                keys=[b"a-block"],
                block_ids=[0],
                kv_params=b_decoder_params,
            )
        )

        # 4. Drive several poll iterations on each side. Each
        # get_finished_jobs() call invokes _poll_once synchronously.
        all_a: list[JobResult] = []
        all_b: list[JobResult] = []
        for _ in range(8):
            all_a.extend(list(mgr_a.get_finished_jobs()))
            all_b.extend(list(mgr_b.get_finished_jobs()))

        session_a = mgr_a._sessions["B:2"]
        session_b = mgr_b._sessions["A:1"]
        assert session_a.ready and session_b.ready
        assert session_a._remote_epoch == session_b._local_epoch
        assert session_b._remote_epoch == session_a._local_epoch

        # Both load jobs and both store jobs must complete successfully.
        a_ok = {r.job_id for r in all_a if r.success}
        b_ok = {r.job_id for r in all_b if r.success}
        # A: load job 101 + store job 100
        assert 101 in a_ok, f"A loads succeeded: {a_ok}"
        assert 100 in a_ok, f"A stores succeeded: {a_ok}"
        # B: load job 201 + store job 200
        assert 201 in b_ok, f"B loads succeeded: {b_ok}"
        assert 200 in b_ok, f"B stores succeeded: {b_ok}"


# ---------------------------------------------------------------------------
# _accept_new_peers — duplicate connection rejection
# ---------------------------------------------------------------------------


class _RecordingConn:
    """Inbound connection stub that records close()/peer_id only."""

    def __init__(self, peer_id: str) -> None:
        self.peer_id = peer_id
        self.close_calls: int = 0

    def close(self) -> None:
        self.close_calls += 1


class TestAcceptNewPeers:
    """A second inbound from an already-connected peer is rejected and the
    new conn is closed; the existing session is left untouched."""

    def test_duplicate_connection_is_closed_and_existing_session_untouched(self):
        mgr = _make_manager()
        peer_id = "10.0.0.1:8000"
        existing = _FakeSession(peer_id=peer_id, connected=True)
        mgr._sessions[peer_id] = existing  # type: ignore[assignment]

        new_conn = _RecordingConn(peer_id)
        mgr._accept_new_peers([new_conn])

        # Manager swallowed the ValueError and closed the duplicate conn.
        assert new_conn.close_calls == 1
        # Existing session was NOT re-attached.
        assert existing.attached == []
        # Session map unchanged.
        assert mgr._sessions[peer_id] is existing

    def test_creates_session_for_new_peer(self):
        """An inbound conn from a peer with no existing session creates
        a fresh connected session and registers it under conn.peer_id.
        The prefiller has no pre-created pending session anymore — the
        first signal of a peer's existence is its inbound connection."""

        class FakeData:
            block_len = 4096
            base_addr = 0x1000
            num_blocks = 16
            config_fingerprint = ""

            def get_agent_metadata(self):
                return b"meta"

            def add_remote_peer(self, *args, **kwargs):
                pass

        mgr = _make_manager()
        mgr._data = FakeData()  # type: ignore[assignment]
        peer_id = "10.0.0.1:8000"

        # Real ControlConnection-shaped fake: send/close/peer_id only.
        sent: list[dict] = []

        class _Conn:
            def __init__(self, pid: str) -> None:
                self.peer_id = pid
                self.alive = True

            def send(self, msg: dict) -> None:
                sent.append(msg)

            def close(self) -> None:
                self.alive = False

        mgr._accept_new_peers([_Conn(peer_id)])  # type: ignore[arg-type]

        assert peer_id in mgr._sessions
        assert mgr._sessions[peer_id].connected is True
        # Session sent its ConnectMsg on the new connection.
        assert any(m for m in sent)


# ---------------------------------------------------------------------------
# _poll_once orchestration
# ---------------------------------------------------------------------------


class TestPollOnce:
    """_poll_once reaps before admission, then polls and reaps again."""

    def test_orchestrates_accept_poll_and_reap(self):
        mgr = _make_manager()

        # Alive session whose poll() returns one load + one store.
        peer_alive = "10.0.0.2:9000"
        alive = _FakeSession(
            peer_id=peer_alive,
            alive=True,
            connected=True,
            loads=[LoadResult(job_id=11, kv_request_id="req-11", success=True)],
            stores=[StoreResult(job_id=22, success=True)],
        )
        mgr._sessions[peer_alive] = alive  # type: ignore[assignment]

        # Dead session whose pending close() jobs surface as failures.
        peer_dead = "10.0.0.3:9999"
        dead = _FakeSession(
            peer_id=peer_dead,
            alive=False,
            connected=True,
            close_jobs=[33],
            close_req_ids=["req-33"],
            close_stores=[44],
        )
        mgr._sessions[peer_dead] = dead  # type: ignore[assignment]

        class _Ctrl:
            def poll(self_inner):
                return []

        class _Data:
            def __init__(self):
                self.reap_calls = 0

            def remove_remote_peer(self_inner, pid):
                pass

            def reap_retired_peers(self_inner):
                self_inner.reap_calls += 1

        mgr._control = _Ctrl()  # type: ignore[assignment]
        data = _Data()
        mgr._data = data  # type: ignore[assignment]

        mgr._poll_once()

        # Every session was polled — alive's results landed.
        # Dead session was reaped — its close() failures landed.
        finished = _finished_results(mgr)
        ok = {(r.job_id, r.success) for r in finished}
        assert (11, True) in ok  # alive load result
        assert (22, True) in ok  # alive store result
        assert (33, False) in ok  # dead session's pending load
        assert (44, False) in ok  # dead session's pending store
        assert "req-33" in mgr._failed_req_ids
        assert peer_dead not in mgr._sessions
        assert peer_alive in mgr._sessions
        assert data.reap_calls == 1

    def test_reaps_dead_same_peer_before_accepting_replacement(self, monkeypatch):
        mgr = _make_manager()
        peer_id = "10.0.0.3:9999"
        old = _FakeSession(peer_id=peer_id, alive=False, connected=True)
        mgr._sessions[peer_id] = old  # type: ignore[assignment]
        incoming = _RecordingConn(peer_id)

        class _Ctrl:
            def poll(self):
                return [incoming]

        class _Data:
            block_len = 4096

            def __init__(self):
                self.removed: list[str] = []

            def remove_remote_peer(self, removed_peer_id):
                self.removed.append(removed_peer_id)

            def reap_retired_peers(self):
                pass

        replacements: list[_FakeSession] = []

        def make_session(**kwargs):
            replacement = _FakeSession(
                peer_id=kwargs["peer_id"], alive=True, connected=True
            )
            replacements.append(replacement)
            return replacement

        mgr._control = _Ctrl()  # type: ignore[assignment]
        data = _Data()
        mgr._data = data  # type: ignore[assignment]
        monkeypatch.setattr(manager_module, "P2PSession", make_session)

        mgr._poll_once()

        assert old.close_calls == 1
        assert data.removed == [peer_id]
        assert incoming.close_calls == 0
        assert len(replacements) == 1
        assert mgr._sessions[peer_id] is replacements[0]

    def test_retried_replacement_is_accepted_after_quarantine_drains(self, monkeypatch):
        mgr = _make_manager()
        peer_id = "10.0.0.4:9999"
        old = _FakeSession(
            peer_id=peer_id,
            alive=False,
            connected=True,
            close_complete=False,
        )
        mgr._sessions[peer_id] = old  # type: ignore[assignment]
        attempts = [_RecordingConn(peer_id), _RecordingConn(peer_id)]

        class _Ctrl:
            def poll(self):
                return [attempts.pop(0)]

        class _Data:
            block_len = 4096

            def __init__(self):
                self.removed: list[str] = []

            def remove_remote_peer(self, removed_peer_id):
                self.removed.append(removed_peer_id)

            def reap_retired_peers(self):
                pass

        replacements: list[_FakeSession] = []

        def make_session(**kwargs):
            replacement = _FakeSession(
                peer_id=kwargs["peer_id"], alive=True, connected=True
            )
            replacements.append(replacement)
            return replacement

        mgr._control = _Ctrl()  # type: ignore[assignment]
        data = _Data()
        mgr._data = data  # type: ignore[assignment]
        monkeypatch.setattr(manager_module, "P2PSession", make_session)

        first = attempts[0]
        mgr._poll_once()
        assert first.close_calls == 1
        assert replacements == []
        assert peer_id in mgr._retiring_sessions

        old.close_complete = True
        second = attempts[0]
        mgr._poll_once()

        assert second.close_calls == 0
        assert data.removed == [peer_id]
        assert len(replacements) == 1
        assert mgr._sessions[peer_id] is replacements[0]

    def test_new_fetch_id_binds_and_replays_unbound_batches(self):
        """When session.poll() reports a kv_request_id whose FetchMsg
        arrived this tick, the manager binds it to that session and
        replays every parked submit_store batch via add_stored_blocks."""
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        mgr = _make_manager()
        peer = "10.0.0.1:8000"
        sess = _FakeSession(
            peer_id=peer,
            alive=True,
            connected=True,
            new_fetch_ids=["req-1"],
        )
        mgr._sessions[peer] = sess  # type: ignore[assignment]
        mgr._unbound_stores["req-1"] = {
            5: _UnboundStoreBatch(job_id=5, keys=[b"k1"], block_ids=[0]),
            6: _UnboundStoreBatch(job_id=6, keys=[b"k2"], block_ids=[1]),
        }

        class _Ctrl:
            def poll(self_inner):
                return []

        mgr._control = _Ctrl()  # type: ignore[assignment]
        mgr._poll_once()

        assert mgr._kv_to_session["req-1"] is sess
        assert "req-1" not in mgr._unbound_stores
        replayed = [
            (kv_req_id, list(keys), job_id)
            for kv_req_id, keys, _, job_id in sess.stores_added
        ]
        assert replayed == [("req-1", [b"k1"], 5), ("req-1", [b"k2"], 6)]

    def test_session_result_is_owned_before_pre_or_post_ack_cut(self):
        for cut_after_ack in (False, True):
            primary = _ConstructorFailure("ack interrupted")

            class InterruptedAck(_FakeSession):
                interrupt = True

                def ack_results(
                    self,
                    *args,
                    _cut_after_ack=cut_after_ack,
                    _primary=primary,
                ):
                    if self.interrupt:
                        self.interrupt = False
                        if _cut_after_ack:
                            super().ack_results(*args)
                        raise _primary
                    super().ack_results(*args)

            mgr = _make_manager()
            session = InterruptedAck(loads=[LoadResult(71, "req-71", True)])

            with pytest.raises(_ConstructorFailure) as raised:
                mgr._adopt_session_result(session, session.pending_results())
            assert raised.value is primary
            assert _finished_results(mgr) == [JobResult(71, True)]

            mgr._adopt_session_result(session, session.pending_results())
            assert _finished_results(mgr) == [JobResult(71, True)]
            assert session.pending_results().loads == []

    def test_fetch_replay_does_not_duplicate_session_owned_store(self):
        from vllm.v1.kv_offload.tiering.p2p.manager import _UnboundStoreBatch

        primary = _ConstructorFailure("after session adopted store")

        class InterruptedStore(_FakeSession):
            interrupt = True

            def add_stored_blocks(self, *args):
                super().add_stored_blocks(*args)
                if self.interrupt:
                    self.interrupt = False
                    raise primary

        mgr = _make_manager()
        session = InterruptedStore(new_fetch_ids=["req-cut"])
        batch = _UnboundStoreBatch(job_id=72, keys=[b"k"], block_ids=[0])
        mgr._unbound_stores["req-cut"] = {72: batch}

        with pytest.raises(_ConstructorFailure) as raised:
            mgr._adopt_session_result(session, session.pending_results())
        assert raised.value is primary
        assert list(mgr._unbound_stores["req-cut"]) == [72]
        assert len(session.stores_added) == 1

        mgr._adopt_session_result(session, session.pending_results())
        assert "req-cut" not in mgr._unbound_stores
        assert len(session.stores_added) == 1
        assert session.pending_results().new_fetch_ids == []

    def test_new_fetch_id_with_no_unbound_still_binds(self):
        """A FetchMsg for a kv_request_id with no parked batches still
        records the binding so subsequent submit_stores route fast."""
        mgr = _make_manager()
        peer = "10.0.0.1:8000"
        sess = _FakeSession(
            peer_id=peer,
            alive=True,
            connected=True,
            new_fetch_ids=["req-fast"],
        )
        mgr._sessions[peer] = sess  # type: ignore[assignment]

        class _Ctrl:
            def poll(self_inner):
                return []

        mgr._control = _Ctrl()  # type: ignore[assignment]
        mgr._poll_once()

        assert mgr._kv_to_session["req-fast"] is sess
        assert sess.stores_added == []

    def test_failed_load_records_kv_request_id(self):
        """A LoadResult(success=False) from session.poll() must add its
        kv_request_id to _failed_req_ids so future lookups return MISS."""
        mgr = _make_manager()
        peer = "10.0.0.1:8000"
        sess = _FakeSession(
            peer_id=peer,
            alive=True,
            connected=True,
            loads=[LoadResult(job_id=5, kv_request_id="req-5", success=False)],
        )
        mgr._sessions[peer] = sess  # type: ignore[assignment]

        class _Ctrl:
            def poll(self_inner):
                return []

        mgr._control = _Ctrl()  # type: ignore[assignment]

        mgr._poll_once()

        assert _finished_results(mgr) == [JobResult(job_id=5, success=False)]
        assert "req-5" in mgr._failed_req_ids


# ---------------------------------------------------------------------------
# drain_jobs
# ---------------------------------------------------------------------------


class _DrainCtrl:
    """Trivial control fake whose poll() returns an empty list."""

    def poll(self):
        return []


class TestDrainJobs:
    def test_returns_immediately_when_quiescent(self):
        """No sessions and no inflight: drain_jobs returns without sleeping."""
        mgr = _make_manager()
        mgr._control = _DrainCtrl()  # type: ignore[assignment]

        sleeps: list[float] = []
        # If drain_jobs sleeps when nothing is pending, that's a regression.
        import vllm.v1.kv_offload.tiering.p2p.manager as m

        original_sleep = m.time.sleep
        m.time.sleep = lambda s: sleeps.append(s)  # type: ignore[assignment]
        try:
            mgr.drain_jobs()
        finally:
            m.time.sleep = original_sleep  # type: ignore[assignment]

        assert sleeps == []

    def test_returns_when_session_has_no_inflight_or_inbound(self):
        """A session with empty _inbound and _inflight does not block drain."""
        mgr = _make_manager()
        mgr._control = _DrainCtrl()  # type: ignore[assignment]
        mgr._sessions["peer:1"] = _FakeSession(peer_id="peer:1")  # type: ignore[assignment]
        # Should return on the first iteration.
        mgr.drain_jobs()

    def test_retiring_session_destination_participates_in_drain(self, monkeypatch):
        from vllm.v1.kv_offload.tiering.p2p.manager import _RetiringSession

        mgr = _make_manager()
        session = _FakeSession(peer_id="peer:retiring", close_complete=False)
        session._client._inbound[1] = object()
        mgr._retiring_sessions[session.peer_id] = _RetiringSession(session=session)
        polls = 0

        def poll_once():
            nonlocal polls
            polls += 1
            if polls == 2:
                session._client._inbound.clear()
                mgr._retiring_sessions.clear()

        mgr._poll_once = poll_once  # type: ignore[method-assign]
        monkeypatch.setattr(manager_module.time, "sleep", lambda _: None)

        mgr.drain_jobs()

        assert polls == 2

    def test_logs_warning_after_5s_then_completes(self, monkeypatch):
        """A session that stays inflight past 5s triggers the warning, and
        once it clears the loop returns."""
        mgr = _make_manager()
        mgr._control = _DrainCtrl()  # type: ignore[assignment]
        sess = _FakeSession(peer_id="peer:1")
        sess._server._inflight = {1: object()}  # non-empty
        mgr._sessions["peer:1"] = sess  # type: ignore[assignment]

        # Synthetic monotonic clock: 100.0 for the start stamp, then 106.0
        # for the first elapsed-check (past the 5s warning threshold), then
        # steady at 106.0 for any later checks.
        clock = iter([100.0, 106.0])

        def fake_monotonic() -> float:
            try:
                return next(clock)
            except StopIteration:
                return 106.0

        monkeypatch.setattr(manager_module, "_DRAIN_SLEEP_S", 0.0)

        # Spy on the warning logger directly — vllm's logger does not
        # propagate to root, so caplog can't see it.
        warnings: list[str] = []

        def record_warning(msg, *args, **_kwargs):
            warnings.append(msg % args if args else msg)

        monkeypatch.setattr(manager_module.logger, "warning", record_warning)

        # Clear inflight after the first sleep so drain can exit on the
        # next iteration's `pending` check.
        n_sleeps = 0

        def clearing_sleep(_s):
            nonlocal n_sleeps
            n_sleeps += 1
            sess._server._inflight = {}

        # Patch via a fake module on `manager_module.time` so we do not mutate
        # the global `time` module — other code in the process (e.g. the
        # buildkite test collector's pytest_runtest_logreport hook) calls
        # time.monotonic() before monkeypatch teardown.
        fake_time = SimpleNamespace(monotonic=fake_monotonic, sleep=clearing_sleep)
        monkeypatch.setattr(manager_module, "time", fake_time)

        mgr.drain_jobs()

        assert any("still draining after 5s" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# on_schedule_end
# ---------------------------------------------------------------------------


class TestOnScheduleEnd:
    def test_is_noop(self):
        """on_schedule_end is a documented no-op; just confirm it doesn't
        raise and doesn't mutate state."""
        mgr = _make_manager()
        before_sessions = dict(mgr._sessions)
        before_jobs = dict(mgr._finished_jobs)
        assert (
            mgr.on_schedule_end(
                ScheduleEndContext(new_req_ids=[], preempted_req_ids=())
            )
            is None
        )
        assert mgr._sessions == before_sessions
        assert mgr._finished_jobs == before_jobs


# ---------------------------------------------------------------------------
# Connection death mid-transfer (real P2PSession via paired managers)
# ---------------------------------------------------------------------------


class TestConnectionDeathMidTransfer:
    """When a peer's control connection dies while a load is in flight,
    the load and CPU destination remain quarantine-owned because control
    death is not remote-DMA quiescence. The
    prefiller-side store no longer travels through the session at store
    time (it's parked in _unbound_stores keyed by kv_request_id), so its
    cleanup on connection death is via on_request_finished or the
    unbound-store timeout — covered separately below."""

    def test_dead_connection_with_pending_load_quarantines_destination(self):
        mgr_a, mgr_b = _build_paired_managers()

        a_decoder_params = {
            "remote_prefiller": {
                "kv_request_id": "req-load",
                "remote_host": "B",
                "remote_port": 2,
            },
        }
        a_prefiller_params = {"remote_decoder": {"kv_request_id": "req-store"}}

        # Open the outbound session A->B and submit one load + one store.
        mgr_a.on_new_request(_req_context(a_decoder_params))
        mgr_a.submit_store(
            _job_metadata(
                job_id=900,
                keys=[b"a-block"],
                block_ids=[0],
                kv_params=a_prefiller_params,
            )
        )
        mgr_a.submit_load(
            _job_metadata(
                job_id=901,
                keys=[b"b-block"],
                block_ids=[0],
                kv_params=a_decoder_params,
            )
        )

        # Drain anything the loopback can deliver synchronously, but stop
        # before the remote side has had time to complete the transfers.
        list(mgr_a.get_finished_jobs())

        # Sanity: store 900 is parked in unbound_stores, not in any
        # session — the producer no longer learns the peer at store time.
        assert "req-store" in mgr_a._unbound_stores

        # Kill the control connection out from under the session.
        peer_id = "B:2"
        sess = mgr_a._sessions[peer_id]
        assert sess._conn is not None
        sess._conn.mark_dead()

        # ZMQ death does not prove the peer's NIXL writer quiescent. Reap
        # therefore retains the load's session and CPU destination without a
        # normal failed completion. Store 900 is independently unbound.
        results: list[JobResult] = []
        for _ in range(3):
            results.extend(list(mgr_a.get_finished_jobs()))

        outcomes = {(r.job_id, r.success) for r in results}
        assert (901, False) not in outcomes
        assert peer_id in mgr_a._retiring_sessions
        assert mgr_a._retiring_sessions[peer_id].session is sess
        assert sess.close_complete is False
        assert sess._client.has_active_loads is True
        assert peer_id not in mgr_a._data._removed_peers
        assert (900, False) not in outcomes, f"store should still be parked: {outcomes}"
        assert "req-load" in mgr_a._failed_req_ids
        # Session left live routing but remains quarantine-owned.
        assert peer_id not in mgr_a._sessions
        # Store batch is still parked.
        assert "req-store" in mgr_a._unbound_stores

        # The engine signals the producer's request is done. That is a
        # no-op for the parked batch — `on_request_finished` does not
        # evict unbound stores; only `_reap_unbound_stores` does, after
        # the unbound-store timeout. Job 900 stays unfinished here.
        mgr_a.on_request_finished(_req_context(a_prefiller_params))
        finishes = {(r.job_id, r.success) for r in _finished_results(mgr_a)}
        assert (900, False) not in finishes
        assert (900, True) not in finishes
        assert "req-store" in mgr_a._unbound_stores


# ---------------------------------------------------------------------------
# Tests for host/port resolution in __init__ (env-var defaults)
# ---------------------------------------------------------------------------


class TestBindHostPortDefaults:
    """host/port fall back to VLLM_P2P_SIDE_CHANNEL_* when not in config."""

    @staticmethod
    def _construct(monkeypatch, dp_index=0, **kwargs) -> P2PSecondaryTierManager:
        """Build a manager with the transports/file-mapper stubbed out.

        The host is used verbatim (no resolution). The transport constructor
        args are recorded on ``mgr._test_calls`` so tests can assert the ZMQ
        identity (``host:port``) stays decoupled from the NIXL agent name
        (a uuid).
        """
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        monkeypatch.setattr(
            manager_module,
            "FileMapper",
            SimpleNamespace(
                from_offloading_spec=lambda **_: SimpleNamespace(
                    get_run_config=lambda: {}
                )
            ),
        )
        calls: dict = {}
        monkeypatch.setattr(
            manager_module,
            "NixlTransport",
            lambda agent_name, *a, **k: (
                calls.update(nixl_name=agent_name, data_kwargs=k) or SimpleNamespace()
            ),
        )
        monkeypatch.setattr(
            manager_module,
            "TorchTransferTransport",
            lambda agent_name, *a, **k: (
                calls.update(torch_name=agent_name, data_kwargs=k)
                or SimpleNamespace(available=True)
            ),
        )
        monkeypatch.setattr(
            manager_module,
            "ZmqTransport",
            lambda local_id, host, port, *a, **k: (
                calls.update(zmq_id=local_id, zmq_host=host, zmq_port=port)
                or SimpleNamespace()
            ),
        )
        spec = SimpleNamespace(
            blocks_per_chunk=1,
            config=SimpleNamespace(
                parallel=SimpleNamespace(data_parallel_index=dp_index)
            ),
        )
        mgr = P2PSecondaryTierManager(spec, memoryview(b""), **kwargs)
        mgr._test_calls = calls
        return mgr

    def test_defaults_from_env_unset(self, monkeypatch):
        monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_HOST", raising=False)
        monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_PORT", raising=False)
        mgr = self._construct(monkeypatch)
        # localhost default is used verbatim as the dial-back identity.
        assert mgr._local_id == "localhost:5710"

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_HOST", "10.1.2.3")
        monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5799")
        mgr = self._construct(monkeypatch)
        assert mgr._local_id == "10.1.2.3:5799"

    def test_explicit_config_wins(self, monkeypatch):
        monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_HOST", "10.1.2.3")
        monkeypatch.setenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5799")
        mgr = self._construct(monkeypatch, host="192.0.2.5", port=6001)
        assert mgr._local_id == "192.0.2.5:6001"

    def test_dp_index_offsets_default_port(self, monkeypatch):
        monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_HOST", raising=False)
        monkeypatch.delenv("VLLM_P2P_SIDE_CHANNEL_PORT", raising=False)
        mgr = self._construct(monkeypatch, dp_index=2)
        assert mgr._local_id == "localhost:5712"

    def test_dp_index_offsets_explicit_port(self, monkeypatch):
        mgr = self._construct(monkeypatch, dp_index=1, host="192.0.2.5", port=6000)
        assert mgr._local_id == "192.0.2.5:6001"

    @pytest.mark.parametrize(
        "bind_host", ["localhost", "127.0.0.1", "::1", "0.0.0.0", "::", "192.0.2.5"]
    )
    def test_host_used_verbatim(self, monkeypatch, bind_host):
        # The host is never rewritten — loopbacks and wildcards alike become
        # the dial-back identity verbatim (mirrors the NIXL connector).
        mgr = self._construct(monkeypatch, host=bind_host, port=5710)
        assert mgr._local_id == f"{bind_host}:5710"

    def test_nixl_name_decoupled_from_identity(self, monkeypatch):
        # The ZMQ identity is the verbatim host:port; the NIXL agent name is a
        # uuid, distinct from the identity and never used as an address.
        mgr = self._construct(monkeypatch, host="127.0.0.1", port=5710)
        assert mgr._test_calls["zmq_host"] == "127.0.0.1"
        assert mgr._test_calls["zmq_port"] == 5710
        assert mgr._test_calls["zmq_id"] == "127.0.0.1:5710"
        assert mgr._local_id == "127.0.0.1:5710"
        # nixl_name is a valid uuid4 and not the host:port identity.
        nixl_name = mgr._test_calls["nixl_name"]
        assert nixl_name != mgr._local_id
        assert uuid.UUID(nixl_name).version == 4

    def test_same_host_port_gets_distinct_nixl_names(self, monkeypatch):
        # The original collision: two peers sharing a host:port must still get
        # distinct NIXL agent names so add_remote_agent doesn't reject a remote
        # whose name equals the local. The per-process uuid guarantees this.
        mgr_a = self._construct(monkeypatch, host="localhost", port=5710)
        mgr_b = self._construct(monkeypatch, host="localhost", port=5710)
        assert mgr_a._local_id == mgr_b._local_id == "localhost:5710"
        assert mgr_a._nixl_agent_name != mgr_b._nixl_agent_name

    def test_torch_data_transport_is_explicitly_selectable(self, monkeypatch):
        mgr = self._construct(
            monkeypatch,
            data_transport="torch",
            transfer_backend="reference",
            transfer_progress_mode="background",
            transfer_thread_mode="single",
            transfer_options={"provider_option": "value"},
        )

        assert "nixl_name" not in mgr._test_calls
        assert uuid.UUID(mgr._test_calls["torch_name"]).version == 4
        assert mgr._test_calls["data_kwargs"]["backend"] == "reference"
        assert mgr._test_calls["data_kwargs"]["endpoint_id"] == mgr._local_id
        assert mgr._test_calls["data_kwargs"]["progress_mode"] == "background"
        assert mgr._test_calls["data_kwargs"]["thread_mode"] == "single"
        assert mgr._test_calls["data_kwargs"]["options"] == {"provider_option": "value"}

    def test_torch_nixl_inherits_native_ucx_agent_options(self, monkeypatch):
        mgr = self._construct(
            monkeypatch,
            data_transport="torch",
            num_threads=7,
        )

        assert mgr._test_calls["data_kwargs"]["progress_mode"] == "background"
        assert mgr._test_calls["data_kwargs"]["thread_mode"] == "single"
        assert mgr._test_calls["data_kwargs"]["options"] == {
            "backends": ["UCX"],
            "num_threads": 7,
            "capture_telemetry": True,
        }

    def test_explicit_torch_nixl_options_override_inherited_defaults(self, monkeypatch):
        mgr = self._construct(
            monkeypatch,
            data_transport="torch",
            backends=["UCX"],
            num_threads=7,
            transfer_options={
                "backends": ["UCX"],
                "num_threads": 2,
                "capture_telemetry": False,
            },
        )

        assert mgr._test_calls["data_kwargs"]["options"] == {
            "backends": ["UCX"],
            "num_threads": 2,
            "capture_telemetry": False,
        }

    def test_unknown_data_transport_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="data_transport"):
            self._construct(monkeypatch, data_transport="unknown")
