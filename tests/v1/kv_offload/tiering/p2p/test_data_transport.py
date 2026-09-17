# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for P2P data-plane transports."""

from __future__ import annotations

import ctypes
from enum import Enum, auto
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vllm.v1.kv_offload.tiering.p2p.data import (
    nixl as nixl_module,
)
from vllm.v1.kv_offload.tiering.p2p.data import (
    torch_transfer as torch_transfer_module,
)
from vllm.v1.kv_offload.tiering.p2p.data.base import PollResult
from vllm.v1.kv_offload.tiering.p2p.data.nixl import NixlTransport
from vllm.v1.kv_offload.tiering.p2p.data.torch_transfer import (
    TorchTransferTransport,
)


class _ItemsForbiddenDict(dict):
    """Prove that peer-scoped polling does not enumerate global ownership."""

    def items(self):
        raise AssertionError("peer-scoped poll scanned the global inflight map")


class _InitializationFailure(BaseException):
    pass


class _NoHashNoEq:
    """Opaque Core distractor whose value operations are forbidden."""

    def __hash__(self):
        raise AssertionError("opaque Core handles must not be hashed")

    def __eq__(self, other):
        del other
        raise AssertionError("opaque Core handles must not be compared")


# ---------------------------------------------------------------------------
# DataTransport base class tests
# ---------------------------------------------------------------------------


class TestDataTransportBase:
    """Tests for the DataTransport abstract base properties."""

    def _make_view(self, num_blocks: int = 8, block_len: int = 1024) -> memoryview:
        """Create a memoryview with the given shape."""
        buf = np.zeros((num_blocks, block_len), dtype=np.uint8)
        return memoryview(buf)

    def test_properties(self):
        """base_addr, num_blocks, block_len are set from memoryview shape."""
        view = self._make_view(num_blocks=4, block_len=2048)

        # Use NixlTransport (concrete) with NIXL mocked away
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            transport = NixlTransport("test:1", view)

        assert transport.num_blocks == 4
        assert transport.block_len == 2048
        assert transport.base_addr == ctypes.addressof(ctypes.c_char.from_buffer(view))

    def test_config_fingerprint_empty_when_no_fields(self):
        """No config fields → empty fingerprint."""
        view = self._make_view()
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            transport = NixlTransport("test:1", view, config_fields=None)
        assert transport.config_fingerprint == ""

    def test_config_fingerprint_deterministic(self):
        """Same config fields → same fingerprint."""
        view = self._make_view()
        fields = {"model": "llama", "dtype": "float16", "blocks_per_chunk": 1}
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            t1 = NixlTransport("test:1", view, config_fields=fields)
            t2 = NixlTransport("test:2", view, config_fields=fields)
        assert t1.config_fingerprint == t2.config_fingerprint
        assert len(t1.config_fingerprint) == 16

    def test_config_fingerprint_differs_for_different_fields(self):
        """Different config fields → different fingerprint."""
        view = self._make_view()
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            t1 = NixlTransport("test:1", view, config_fields={"model": "a"})
            t2 = NixlTransport("test:2", view, config_fields={"model": "b"})
        assert t1.config_fingerprint != t2.config_fingerprint


# ---------------------------------------------------------------------------
# NixlTransport tests (with mocked NIXL agent)
# ---------------------------------------------------------------------------


class TestNixlTransportWithMockedAgent:
    """Tests for NixlTransport logic with a mocked NIXL agent."""

    def _make_transport(self) -> NixlTransport:
        """Create a NixlTransport with mocked NIXL internals."""
        view = memoryview(np.zeros((8, 1024), dtype=np.uint8))

        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            transport = NixlTransport("test:1", view)

        # Manually set up a mock agent after construction
        agent = MagicMock()
        agent.add_remote_agent.return_value = "nixl-peer-name"
        agent.get_xfer_descs.return_value = MagicMock()
        agent.prep_xfer_dlist.return_value = MagicMock()
        agent.make_prepped_xfer.return_value = MagicMock(name="handle")
        agent.transfer.return_value = None
        agent.check_xfer_state.return_value = "PROC"
        agent.get_agent_metadata.return_value = b"test-metadata"

        transport._agent = agent
        transport._local_dlist = MagicMock()
        return transport

    def test_available_false_without_nixl(self):
        """Without NIXL installed, available is False."""
        view = memoryview(np.zeros((4, 512), dtype=np.uint8))
        with patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", None):
            transport = NixlTransport("test:1", view)
        assert transport.available is False

    def test_available_true_with_agent(self):
        transport = self._make_transport()
        assert transport.available is True

    @pytest.mark.parametrize("cleanup_fails", [False, True])
    def test_constructor_baseexception_rolls_back_without_masking(self, cleanup_fails):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        primary = _InitializationFailure("post-registration failure")
        cleanup = _InitializationFailure("cleanup failure")
        events: list[str] = []
        agent = MagicMock()
        registration = object()
        local_dlist = object()
        agent.register_memory.return_value = registration
        agent.get_xfer_descs.return_value = object()
        agent.prep_xfer_dlist.return_value = local_dlist

        def release_dlist(handle):
            assert handle is local_dlist
            events.append("dlist")
            if cleanup_fails:
                raise cleanup

        def deregister(handle):
            assert handle is registration
            events.append("registration")
            if cleanup_fails:
                raise cleanup

        def fail_after_registration(message, *args):
            if "registered %d blocks" in message:
                raise primary

        agent.release_dlist_handle.side_effect = release_dlist
        agent.deregister_memory.side_effect = deregister
        with (
            patch.object(nixl_module, "_NixlAgent", return_value=agent),
            patch.object(
                nixl_module, "_NixlAgentConfig", return_value=SimpleNamespace()
            ),
            patch.object(
                nixl_module.logger, "info", side_effect=fail_after_registration
            ),
            pytest.raises(_InitializationFailure) as raised,
        ):
            NixlTransport("test:1", view)

        assert raised.value is primary
        assert events == ["dlist", "registration"]

    def test_peer_setup_failure_rolls_back_and_retains_failed_cleanup(self):
        transport = self._make_transport()
        primary = _InitializationFailure("descriptor setup")
        cleanup = _InitializationFailure("remote cleanup")
        transport._agent.get_xfer_descs.side_effect = primary
        transport._agent.remove_remote_agent.side_effect = cleanup

        with pytest.raises(_InitializationFailure) as raised:
            transport.add_remote_peer("peer:1", b"metadata", 0, 2, 8)

        assert raised.value is primary
        assert transport._peer_nixl_names == {"peer:1": "nixl-peer-name"}
        assert transport._remote_dlists == {}
        transport._agent.remove_remote_agent.side_effect = None
        transport.remove_remote_peer("peer:1")
        assert transport._peer_nixl_names == {}

    def test_get_agent_metadata(self):
        transport = self._make_transport()
        assert transport.get_agent_metadata() == b"test-metadata"

    def test_write_blocks_returns_none_for_unknown_peer(self):
        """write_blocks returns None if peer not registered."""
        transport = self._make_transport()
        result = transport.write_blocks("unknown:1", [0, 1], [2, 3])
        assert result is None

    def test_write_blocks_returns_transfer_id(self):
        """write_blocks returns an integer transfer_id on success."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0, 1], [2, 3])
        assert tid is not None
        assert isinstance(tid, int)

    def test_write_blocks_increments_transfer_id(self):
        """Each write_blocks call gets a unique transfer_id."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid1 = transport.write_blocks("peer:1", [0], [1])
        tid2 = transport.write_blocks("peer:1", [2], [3])
        assert tid1 != tid2

    def test_poll_empty_when_no_inflight(self):
        """poll returns empty when nothing is inflight."""
        transport = self._make_transport()
        result = transport.poll()
        assert result == PollResult(done=(), failed=())

    def test_poll_returns_done_when_transfer_completes(self):
        """Completed transfer appears in poll().done."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])

        # Simulate completion
        transport._agent.check_xfer_state.return_value = "DONE"
        result = transport.poll()

        assert tid in result.done
        assert result.failed == ()
        # Handle released
        transport._agent.release_xfer_handle.assert_called()

    def test_poll_returns_failed_for_error_state(self):
        """Transfer in error state appears in poll().failed."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])

        transport._agent.check_xfer_state.return_value = "ERR"
        result = transport.poll()

        assert result.done == ()
        assert tid in result.failed

    def test_poll_ignores_in_progress(self):
        """Transfers in PROC/PEND state stay inflight."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        transport.write_blocks("peer:1", [0], [1])

        transport._agent.check_xfer_state.return_value = "PROC"
        result = transport.poll()
        assert result.done == ()
        assert result.failed == ()

        transport._agent.check_xfer_state.return_value = "PEND"
        result = transport.poll()
        assert result.done == ()
        assert result.failed == ()

    def test_poll_peer_id_scopes_to_peer(self):
        """poll(peer_id) drains only that peer's transfers.

        Regression: the transport is shared across peer sessions (e.g. a
        single prefiller serving a DP>1 decoder). An unscoped poll by one
        session used to consume and discard sibling sessions' completions,
        starving them until timeout. poll(peer_id) must leave other peers'
        transfers inflight.
        """
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        transport.add_remote_peer("peer:2", b"meta", 0x2000, 8, 1024)

        tid1 = transport.write_blocks("peer:1", [0], [1])
        tid2 = transport.write_blocks("peer:2", [2], [3])
        transport._agent.check_xfer_state.return_value = "DONE"
        transport._inflight = _ItemsForbiddenDict(transport._inflight)

        # Polling peer:1 must not consume peer:2's completed transfer.
        result = transport.poll(peer_id="peer:1")
        assert tid1 in result.done
        assert tid2 not in result.done
        assert tid1 not in transport._inflight
        assert tid2 in transport._inflight
        assert transport._inflight_by_peer == {"peer:2": {tid2: None}}

        # peer:2 sees its own completion when it polls.
        result2 = transport.poll(peer_id="peer:2")
        assert tid2 in result2.done
        assert tid2 not in transport._inflight
        assert transport._inflight_by_peer == {}

    def test_cancel_removes_inflight(self):
        """cancel removes transfers and releases handles."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid in transport._inflight

        result = transport.cancel([tid])
        assert result == []
        assert tid not in transport._inflight
        assert transport._inflight_by_peer == {}
        transport._agent.release_xfer_handle.assert_called()

    def test_cancel_ignores_unknown_ids(self):
        """cancel with unknown IDs doesn't crash."""
        transport = self._make_transport()
        assert transport.cancel([999, 1000]) == []
        assert transport.cancel([999, 1000], mode="wait") == []

    def test_cancel_wait_release_succeeds(self):
        """wait-mode cancel that succeeds pops the entry and returns []."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid in transport._inflight

        result = transport.cancel([tid], mode="wait")
        assert result == []
        assert tid not in transport._inflight
        transport._agent.release_xfer_handle.assert_called_once()

    def test_cancel_wait_release_raises(self):
        """wait-mode cancel keeps the entry and returns the tid on raise."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.release_xfer_handle.side_effect = RuntimeError(
            "NIXL_ERR_REPOST_ACTIVE"
        )

        result = transport.cancel([tid], mode="wait")
        assert result == [tid]
        assert tid in transport._inflight

    def test_cancel_wait_then_poll_completes(self):
        """A wait-cancel that left a tid pending later completes via poll."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)

        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.release_xfer_handle.side_effect = RuntimeError("busy")
        assert transport.cancel([tid], mode="wait") == [tid]
        assert tid in transport._inflight

        transport._agent.release_xfer_handle.side_effect = None
        transport._agent.check_xfer_state.return_value = "DONE"

        result = transport.poll()
        assert tid in result.done
        assert tid not in transport._inflight

    def test_poll_release_failure_retains_handle_and_exact_outcome(self):
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.check_xfer_state.return_value = "DONE"
        transport._agent.release_xfer_handle.side_effect = RuntimeError("busy")

        assert transport.poll() == PollResult(done=(), failed=())
        assert tid in transport._inflight
        assert transport._terminal_by_peer == {"peer:1": {tid: "done"}}

        transport._agent.release_xfer_handle.side_effect = None
        assert transport.poll() == PollResult(done=[tid], failed=())
        assert transport._inflight == {}
        assert transport._inflight_by_peer == {}
        assert transport._terminal_by_peer == {"peer:1": {tid: "done"}}
        assert transport.poll("peer:1") == PollResult(done=[tid], failed=())
        transport.ack_completions("peer:1", (tid,))
        assert transport._terminal_by_peer == {}

    def test_immediate_cancel_failure_keeps_native_owner_and_blocks_close(self):
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        tid = transport.write_blocks("peer:1", [0], [1])
        transport._agent.release_xfer_handle.side_effect = RuntimeError("active")

        assert transport.cancel([tid], mode="immediate") == []
        assert tid in transport._inflight
        with pytest.raises(RuntimeError, match="active transfer"):
            transport.close()

        transport._agent.release_xfer_handle.side_effect = None
        transport.close()
        assert transport._agent is None

    def test_peer_retirement_repairs_interrupted_native_index_publication(self):
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None

        # Model a BaseException after the primary handle-owner store and before
        # the derived peer-index store.
        transport._inflight_by_peer.clear()
        transport.remove_remote_peer("peer:1")
        assert transport._inflight_by_peer == {"peer:1": {tid: None}}
        assert "peer:1" in transport._retiring_peers
        assert "peer:1" in transport._remote_dlists
        assert transport.peer_retirement_complete("peer:1") is False

        transport._agent.check_xfer_state.return_value = "DONE"
        transport.reap_retired_peers()
        assert tid not in transport._inflight
        assert transport._terminal_by_peer == {"peer:1": {tid: "done"}}
        assert transport.poll("peer:1") == PollResult(done=[tid], failed=())
        assert "peer:1" in transport._remote_dlists
        assert transport.peer_retirement_complete("peer:1") is False

        transport.ack_completions("peer:1", (tid,))
        transport.reap_retired_peers()
        assert "peer:1" not in transport._remote_dlists
        assert transport.peer_retirement_complete("peer:1") is True

    def test_add_and_remove_remote_peer(self):
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        assert "peer:1" in transport._remote_dlists

        transport.remove_remote_peer("peer:1")
        assert "peer:1" not in transport._remote_dlists
        transport._agent.release_dlist_handle.assert_called()
        transport._agent.remove_remote_agent.assert_called()

    def test_close_releases_everything(self):
        """close releases all handles and clears state."""
        transport = self._make_transport()
        transport.add_remote_peer("peer:1", b"meta", 0x1000, 8, 1024)
        transport.write_blocks("peer:1", [0], [1])

        transport.close()
        assert transport._agent is None
        assert transport._inflight == {}
        assert transport._inflight_by_peer == {}
        assert transport._remote_dlists == {}


# ---------------------------------------------------------------------------
# NIXL agent-config selection
# ---------------------------------------------------------------------------


class TestNixlAgentConfigSelection:
    """Tests that backends/num_threads pick the right nixl_agent_config call.

    Mirrors the conditional in
    vllm/distributed/kv_transfer/kv_connector/v1/nixl/base_worker.py:325-329.
    """

    def _make_view(self) -> memoryview:
        return memoryview(np.zeros((4, 512), dtype=np.uint8))

    def test_non_ucx_backends_passes_backends_kwarg(self):
        """When any non-UCX backend is requested, pass backends + telemetry."""
        agent_cls = MagicMock()
        config_fn = MagicMock(return_value=MagicMock(name="cfg"))
        with (
            patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", agent_cls),
            patch(
                "vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgentConfig", config_fn
            ),
        ):
            NixlTransport("test:1", self._make_view(), backends=["MOONCAKE"])

        config_fn.assert_called_once_with(backends=["MOONCAKE"], capture_telemetry=True)
        # num_threads must NOT be passed on the non-UCX branch.
        assert "num_threads" not in config_fn.call_args.kwargs

    def test_ucx_only_passes_num_threads(self):
        """UCX-only configuration passes num_threads + telemetry, no backends."""
        agent_cls = MagicMock()
        config_fn = MagicMock(return_value=MagicMock(name="cfg"))
        with (
            patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", agent_cls),
            patch(
                "vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgentConfig", config_fn
            ),
        ):
            NixlTransport("test:1", self._make_view(), num_threads=8)

        config_fn.assert_called_once_with(num_threads=8, capture_telemetry=True)
        assert "backends" not in config_fn.call_args.kwargs

    def test_default_backends_is_ucx_only(self):
        """No backends arg → defaults to UCX-only branch."""
        agent_cls = MagicMock()
        config_fn = MagicMock(return_value=MagicMock(name="cfg"))
        with (
            patch("vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgent", agent_cls),
            patch(
                "vllm.v1.kv_offload.tiering.p2p.data.nixl._NixlAgentConfig", config_fn
            ),
        ):
            NixlTransport("test:1", self._make_view())

        # Default num_threads=4, no backends kwarg.
        config_fn.assert_called_once_with(num_threads=4, capture_telemetry=True)


# ---------------------------------------------------------------------------
# PyTorch endpoint-transfer adapter tests
# ---------------------------------------------------------------------------


class _WorkState(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    SUCCEEDED = COMPLETED
    FAILED = auto()
    CANCELLED = auto()


class _TransferOp(Enum):
    WRITE = "write"


class _ProgressMode(Enum):
    MANUAL = "manual"
    BACKGROUND = "background"


class _ThreadMode(Enum):
    SINGLE = "single"
    SERIALIZED = "serialized"


class _TransferError(Exception):
    pass


class _FakeResource:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakeWork(_FakeResource):
    def __init__(self) -> None:
        super().__init__()
        self._state = _WorkState.PENDING
        self.state_reads = 0
        self.cancel_completes = True
        self.cancel_calls = 0
        self._error: Exception | None = None
        self.test_calls = 0
        self.error_reads = 0
        self.exception_calls = 0
        self.state_errors: list[BaseException] = []

    @property
    def state(self) -> _WorkState:
        self.state_reads += 1
        if self.state_errors:
            raise self.state_errors.pop(0)
        return self._state

    @state.setter
    def state(self, value: _WorkState) -> None:
        self._state = value

    @property
    def error(self) -> Exception | None:
        self.error_reads += 1
        return self._error

    @error.setter
    def error(self, value: Exception | None) -> None:
        self._error = value

    def test(self) -> bool:
        self.test_calls += 1
        return self._state is not _WorkState.PENDING

    def exception(self) -> Exception | None:
        self.exception_calls += 1
        return self._error

    def cancel(self) -> bool:
        self.cancel_calls += 1
        if self.cancel_completes:
            self.state = _WorkState.CANCELLED
        return self.cancel_completes


class _FakeRegistration(_FakeResource):
    def __init__(self, buffer: memoryview, name: str) -> None:
        super().__init__()
        self.buffer = buffer
        self.name = name
        self.region_calls: list[tuple[int, int, int | None, int | None]] = []

    def region(
        self,
        offset: int,
        nbytes: int,
        *,
        stride: int | None = None,
        count: int | None = None,
    ) -> tuple[str, int, int]:
        self.region_calls.append((offset, nbytes, stride, count))
        return ("local", offset, nbytes)


class _FakePeer(_FakeResource):
    def __init__(self, metadata: bytes) -> None:
        super().__init__()
        self.metadata = metadata
        self.endpoint_id: str | None = None
        self.expected_endpoint_id: str | None = None
        self.region_calls: list[tuple[str, int, int, int | None, int | None]] = []

    def region(
        self,
        registration_name: str,
        offset: int,
        nbytes: int,
        *,
        stride: int | None = None,
        count: int | None = None,
    ) -> tuple[str, int, int]:
        self.region_calls.append((registration_name, offset, nbytes, stride, count))
        return ("remote", offset, nbytes)


class _FakePlan(_FakeResource):
    def __init__(self, endpoint: _FakeEndpoint) -> None:
        super().__init__()
        self.endpoint = endpoint
        self.submit_indices_calls: list[dict] = []

    def submit_indices(
        self,
        op: str,
        *,
        local_indices: np.ndarray,
        remote_indices: np.ndarray,
    ) -> _FakeWork:
        call = {
            "plan": self,
            "op": op,
            "local_indices": local_indices,
            "remote_indices": remote_indices,
        }
        self.submit_indices_calls.append(call)
        work = _FakeWork()
        self.endpoint.works.append(work)
        return work


class _FakeBackend:
    def register(self, memory: object, *, adopt_handle: object | None = None):
        del memory, adopt_handle


def _fake_register_backend(
    name: str,
    factory: object,
    *,
    replace: bool = False,
    factory_api_version: int = 2,
) -> None:
    del name, factory, replace, factory_api_version


class _FakeBackendFactoryV2:
    pass


class _FakeEndpoint(_FakeResource):
    instances: list[_FakeEndpoint] = []

    def __init__(
        self,
        name: str,
        *,
        endpoint_id: str | None,
        backend: str,
        progress_mode: _ProgressMode,
        thread_mode: _ThreadMode,
        options: dict,
    ) -> None:
        super().__init__()
        self.name = name
        self.endpoint_id = endpoint_id
        self.backend = backend
        self.progress_mode = progress_mode
        self.thread_mode = thread_mode
        self.options = options
        self.capabilities = SimpleNamespace(write=True)
        self.registration: _FakeRegistration | None = None
        self.registrations: list[object] = []
        self.peers: list[_FakePeer] = []
        self.plans: list[_FakePlan] = []
        self.prepare_calls: list[dict] = []
        self.write_calls: list[dict] = []
        self.progress_calls = 0
        self.works: list[_FakeWork] = []
        self.__class__.instances.append(self)

    @property
    def open_registrations(self) -> tuple[object, ...]:
        return tuple(
            registration
            for registration in self.registrations
            if not getattr(registration, "closed", False)
        )

    @property
    def open_peers(self) -> tuple[object, ...]:
        return tuple(peer for peer in self.peers if not getattr(peer, "closed", False))

    @property
    def open_plans(self) -> tuple[object, ...]:
        return tuple(plan for plan in self.plans if not getattr(plan, "closed", False))

    @property
    def open_works(self) -> tuple[object, ...]:
        return tuple(work for work in self.works if not getattr(work, "closed", False))

    def register(self, buffer: memoryview, *, name: str) -> _FakeRegistration:
        self.registration = _FakeRegistration(buffer, name)
        self.registrations.append(self.registration)
        return self.registration

    def export_metadata(self, registrations: list[object]) -> bytes:
        assert registrations == [self.registration]
        return b"torch-transfer-metadata"

    def import_peer(
        self,
        metadata: bytes,
        *,
        expected_endpoint_id: str | None = None,
    ) -> _FakePeer:
        peer = _FakePeer(metadata)
        peer.endpoint_id = expected_endpoint_id
        peer.expected_endpoint_id = expected_endpoint_id
        self.peers.append(peer)
        return peer

    def prepare(self, **kwargs) -> _FakePlan:
        self.prepare_calls.append(kwargs)
        plan = _FakePlan(self)
        self.plans.append(plan)
        return plan

    def write(self, plan: object, **kwargs) -> _FakeWork:
        self.write_calls.append({"plan": plan, **kwargs})
        work = _FakeWork()
        self.works.append(work)
        return work

    def progress(self) -> None:
        self.progress_calls += 1


class TestTorchTransferTransport:
    def _make_transport(
        self,
        num_blocks: int = 4,
        block_len: int = 16,
        progress_mode: str = "background",
        work_state_table: object | None = _WorkState,
    ) -> tuple[TorchTransferTransport, _FakeEndpoint]:
        _FakeEndpoint.instances.clear()
        view = memoryview(np.zeros((num_blocks, block_len), dtype=np.uint8))
        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=2,
            Backend=_FakeBackend,
            BackendFactoryV2=_FakeBackendFactoryV2,
            Endpoint=_FakeEndpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            register_backend=_fake_register_backend,
        )
        if work_state_table is not None:
            api.WorkState = work_state_table
        with (
            patch(
                "vllm.v1.kv_offload.tiering.p2p.data.torch_transfer._load_transfer_api",
                return_value=api,
            ),
            patch(
                "vllm.v1.kv_offload.tiering.p2p.data.torch_transfer._register_nixl_backend"
            ) as register_nixl,
        ):
            transport = TorchTransferTransport(
                "test-endpoint", view, progress_mode=progress_mode
            )
        register_nixl.assert_called_once_with()
        return transport, _FakeEndpoint.instances[-1]

    def test_unavailable_without_experimental_torch_api(self):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        with patch(
            "vllm.v1.kv_offload.tiering.p2p.data.torch_transfer._load_transfer_api",
            return_value=None,
        ):
            transport = TorchTransferTransport("test-endpoint", view)

        assert transport.available is False

    def test_constructor_baseexception_closes_resources_without_masking(self):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        primary = _InitializationFailure("region construction")
        cleanup = _InitializationFailure("cleanup failure")
        events: list[str] = []
        holder: dict[str, object] = {}
        cleanup_fails = True

        class FailingEndpoint(_FakeEndpoint):
            def register(self, buffer: memoryview, *, name: str):
                registration = super().register(buffer, name=name)
                holder["endpoint"] = self
                holder["registration"] = registration

                def fail_region(*args, **kwargs):
                    raise primary

                def close_registration():
                    events.append("registration")
                    if cleanup_fails:
                        raise cleanup
                    registration.closed = True

                def close_endpoint():
                    events.append("endpoint")
                    if cleanup_fails:
                        raise cleanup
                    self.closed = True

                registration.region = fail_region  # type: ignore[method-assign]
                registration.close = close_registration  # type: ignore[method-assign]
                self.close = close_endpoint  # type: ignore[method-assign]
                return registration

        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=2,
            Backend=_FakeBackend,
            BackendFactoryV2=_FakeBackendFactoryV2,
            Endpoint=FailingEndpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            WorkState=_WorkState,
            register_backend=_fake_register_backend,
        )
        with (
            patch.object(torch_transfer_module, "_load_transfer_api", return_value=api),
            patch.object(torch_transfer_module, "_register_nixl_backend"),
            patch.object(
                torch_transfer_module.logger,
                "warning",
                side_effect=_InitializationFailure("logging failure"),
            ),
            pytest.raises(_InitializationFailure) as raised,
        ):
            TorchTransferTransport("test-endpoint", view)

        assert raised.value is primary
        assert events == ["registration"]
        recovery = primary.recovery_transport
        assert recovery.construction_cleanup_complete is False
        assert recovery._registration is holder["registration"]
        assert recovery._endpoint is holder["endpoint"]

        cleanup_fails = False
        recovery.close()
        assert events == ["registration", "registration", "endpoint"]
        assert recovery.construction_cleanup_complete is True
        assert recovery._registration is None
        assert recovery._endpoint is None

    def test_registration_recovery_uses_identity_with_opaque_baseline(self):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        primary = _InitializationFailure("after registration publication")
        baseline = _NoHashNoEq()
        holder: dict[str, object] = {}

        class InterruptedEndpoint(_FakeEndpoint):
            def __init__(self, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.registrations.append(baseline)

            def register(self, buffer: memoryview, *, name: str):
                registration = super().register(buffer, name=name)
                holder["endpoint"] = self
                holder["registration"] = registration
                raise primary

        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=2,
            Backend=_FakeBackend,
            BackendFactoryV2=_FakeBackendFactoryV2,
            Endpoint=InterruptedEndpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            WorkState=_WorkState,
            register_backend=_fake_register_backend,
        )
        with (
            patch.object(torch_transfer_module, "_load_transfer_api", return_value=api),
            patch.object(torch_transfer_module, "_register_nixl_backend"),
            pytest.raises(_InitializationFailure) as raised,
        ):
            TorchTransferTransport("test-endpoint", view)

        assert raised.value is primary
        assert holder["registration"].closed is True
        assert holder["endpoint"].closed is True

    def test_endpoint_constructor_recovery_requires_all_exact_identity_fields(self):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        primary = _InitializationFailure("after endpoint publication")
        holder: dict[str, _FakeEndpoint] = {}

        class InterruptedEndpoint(_FakeEndpoint):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                holder["endpoint"] = self
                primary.recovery_endpoint = self
                raise primary

        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=2,
            Backend=_FakeBackend,
            BackendFactoryV2=_FakeBackendFactoryV2,
            Endpoint=InterruptedEndpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            WorkState=_WorkState,
            register_backend=_fake_register_backend,
        )
        with (
            patch.object(torch_transfer_module, "_load_transfer_api", return_value=api),
            patch.object(torch_transfer_module, "_register_nixl_backend"),
            pytest.raises(_InitializationFailure) as raised,
        ):
            TorchTransferTransport("test-endpoint", view)

        assert raised.value is primary
        assert holder["endpoint"].closed is True

    def test_foreign_recovery_endpoint_with_same_id_is_never_mutated(self):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        primary = _InitializationFailure("foreign recovery")

        class ForeignEndpoint:
            endpoint_id = "test-endpoint"
            name = "test-endpoint"
            backend = "nixl"
            progress_mode = _ProgressMode.BACKGROUND
            thread_mode = _ThreadMode.SERIALIZED

            def __init__(self):
                self.close_calls = 0

            def close(self):
                self.close_calls += 1

        foreign = ForeignEndpoint()
        primary.recovery_endpoint = foreign

        def fail_endpoint(*args, **kwargs):
            raise primary

        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=2,
            Backend=_FakeBackend,
            BackendFactoryV2=_FakeBackendFactoryV2,
            Endpoint=fail_endpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            WorkState=_WorkState,
            register_backend=_fake_register_backend,
        )
        with (
            patch.object(torch_transfer_module, "_load_transfer_api", return_value=api),
            patch.object(torch_transfer_module, "_register_nixl_backend"),
            pytest.raises(_InitializationFailure) as raised,
        ):
            TorchTransferTransport("test-endpoint", view)

        assert raised.value is primary
        assert foreign.close_calls == 0
        assert not hasattr(primary, "recovery_transport")

    def test_core_typed_same_id_mismatched_name_is_never_mutated(self):
        """Concrete Core type and endpoint ID alone do not prove ownership."""
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        primary = _InitializationFailure("foreign Core recovery")
        holder: dict[str, _FakeEndpoint] = {}

        class InterruptedEndpoint(_FakeEndpoint):
            def __init__(self, *args, **kwargs):
                candidate = object.__new__(type(self))
                _FakeEndpoint.__init__(candidate, *args, **kwargs)
                candidate.name = "foreign-name"
                holder["candidate"] = candidate
                primary.recovery_endpoint = candidate
                raise primary

        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=2,
            Backend=_FakeBackend,
            BackendFactoryV2=_FakeBackendFactoryV2,
            Endpoint=InterruptedEndpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            WorkState=_WorkState,
            register_backend=_fake_register_backend,
        )
        with (
            patch.object(torch_transfer_module, "_load_transfer_api", return_value=api),
            patch.object(torch_transfer_module, "_register_nixl_backend"),
            pytest.raises(_InitializationFailure) as raised,
        ):
            TorchTransferTransport("test-endpoint", view)

        candidate = holder["candidate"]
        assert raised.value is primary
        assert candidate.endpoint_id == "test-endpoint"
        assert candidate.closed is False
        assert not hasattr(primary, "recovery_transport")

    def test_constructor_close_lost_return_retains_exact_owner_for_retry(self):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        primary = _InitializationFailure("region construction")
        cleanup = _InitializationFailure("close lost return")
        close_calls = 0

        class FailingEndpoint(_FakeEndpoint):
            def register(self, buffer: memoryview, *, name: str):
                registration = super().register(buffer, name=name)

                def fail_region(*args, **kwargs):
                    raise primary

                def close_registration():
                    nonlocal close_calls
                    close_calls += 1
                    registration.closed = True
                    if close_calls == 1:
                        raise cleanup

                registration.region = fail_region  # type: ignore[method-assign]
                registration.close = close_registration  # type: ignore[method-assign]
                return registration

        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=2,
            Backend=_FakeBackend,
            BackendFactoryV2=_FakeBackendFactoryV2,
            Endpoint=FailingEndpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            WorkState=_WorkState,
            register_backend=_fake_register_backend,
        )
        with (
            patch.object(torch_transfer_module, "_load_transfer_api", return_value=api),
            patch.object(torch_transfer_module, "_register_nixl_backend"),
            pytest.raises(_InitializationFailure) as raised,
        ):
            TorchTransferTransport("test-endpoint", view)

        recovery = raised.value.recovery_transport
        assert recovery._registration.closed is True
        assert recovery._endpoint.closed is False
        recovery.close()
        assert close_calls == 2
        assert recovery.construction_cleanup_complete is True
        assert recovery._endpoint is None

    @pytest.mark.parametrize("core_version", [None, 1, True, "2"])
    def test_rejects_unadvertised_core_factory_version(self, core_version):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=core_version,
            Backend=_FakeBackend,
            BackendFactoryV2=_FakeBackendFactoryV2,
            Endpoint=_FakeEndpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            WorkState=_WorkState,
            register_backend=_fake_register_backend,
        )
        with (
            patch.object(torch_transfer_module, "_load_transfer_api", return_value=api),
            patch.object(
                torch_transfer_module, "_register_nixl_backend"
            ) as register_nixl,
            pytest.raises(RuntimeError, match="construction/factory v2") as raised,
        ):
            TorchTransferTransport("test-endpoint", view)

        assert f"BACKEND_FACTORY_API_VERSION={core_version!r}" in str(raised.value)
        register_nixl.assert_not_called()

    def test_accepts_newer_core_with_backward_compatible_v2_factory(self):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=3,
            Backend=_FakeBackend,
            BackendFactoryV2=_FakeBackendFactoryV2,
            Endpoint=_FakeEndpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            WorkState=_WorkState,
            register_backend=_fake_register_backend,
        )
        with (
            patch.object(torch_transfer_module, "_load_transfer_api", return_value=api),
            patch.object(torch_transfer_module, "_register_nixl_backend"),
        ):
            transport = TorchTransferTransport("test-endpoint", view)
        assert transport.available is True
        transport.close()

    def test_rejects_core_missing_factory_v2_type(self):
        view = memoryview(np.zeros((2, 8), dtype=np.uint8))
        api = SimpleNamespace(
            BACKEND_FACTORY_API_VERSION=2,
            Backend=_FakeBackend,
            Endpoint=_FakeEndpoint,
            ProgressMode=_ProgressMode,
            ThreadMode=_ThreadMode,
            TransferOp=_TransferOp,
            TransferError=_TransferError,
            WorkState=_WorkState,
            register_backend=_fake_register_backend,
        )
        with (
            patch.object(torch_transfer_module, "_load_transfer_api", return_value=api),
            patch.object(
                torch_transfer_module, "_register_nixl_backend"
            ) as register_nixl,
            pytest.raises(RuntimeError, match="BackendFactoryV2=missing"),
        ):
            TorchTransferTransport("test-endpoint", view)
        register_nixl.assert_not_called()

    @pytest.mark.parametrize("provider_version", [None, 1, True, "2", 3])
    def test_rejects_unadvertised_provider_factory_version(
        self, monkeypatch, provider_version
    ):
        provider = SimpleNamespace(
            TORCH_TRANSFER_FACTORY_API_VERSION=provider_version,
            register_torch_backend=MagicMock(),
        )
        monkeypatch.setattr(
            torch_transfer_module.importlib,
            "import_module",
            lambda name: provider,
        )

        with pytest.raises(RuntimeError, match="supported factory API version 2"):
            torch_transfer_module._register_nixl_backend()
        provider.register_torch_backend.assert_not_called()

    def test_registers_named_blocks_and_exports_metadata(self):
        transport, endpoint = self._make_transport()

        assert transport.available is True
        assert endpoint.backend == "nixl"
        assert endpoint.endpoint_id == "test-endpoint"
        assert endpoint.progress_mode is _ProgressMode.BACKGROUND
        assert endpoint.thread_mode is _ThreadMode.SERIALIZED
        assert endpoint.options == {}
        assert endpoint.registration is not None
        assert endpoint.registration.name == "kv_blocks"
        assert endpoint.registration.region_calls == [(0, 16, 16, 4)]
        assert transport.get_agent_metadata() == b"torch-transfer-metadata"

    def test_peer_metadata_builds_reusable_indexed_plan(self):
        transport, endpoint = self._make_transport(num_blocks=3)

        transport.add_remote_peer("peer:1", b"peer-metadata", 0xBAD, 2, 16)

        peer = endpoint.peers[-1]
        assert peer.metadata == b"peer-metadata"
        assert peer.expected_endpoint_id == "peer:1"
        assert peer.region_calls == [("kv_blocks", 0, 16, 16, 2)]
        prepare = endpoint.prepare_calls[-1]
        assert prepare["indexed"] is True
        assert len(prepare["local"]) == 1
        assert len(prepare["remote"]) == 1

    def test_core_metadata_failures_become_peer_protocol_errors(self):
        transport, endpoint = self._make_transport()
        endpoint.import_peer = MagicMock(side_effect=_TransferError("bad envelope"))

        with pytest.raises(ValueError, match="invalid endpoint-transfer metadata"):
            transport.add_remote_peer("peer:1", b"malformed", 0, 4, 16)

        assert "peer:1" not in transport._remote_peers

    def test_peer_plan_recovery_uses_identity_with_opaque_baselines(self):
        transport, endpoint = self._make_transport()
        baseline_peer = _NoHashNoEq()
        baseline_plan = _NoHashNoEq()
        endpoint.peers.append(baseline_peer)  # type: ignore[arg-type]
        endpoint.plans.append(baseline_plan)  # type: ignore[arg-type]
        primary = _InitializationFailure("after plan publication")
        original_prepare = endpoint.prepare

        def prepare_then_interrupt(**kwargs):
            original_prepare(**kwargs)
            raise primary

        endpoint.prepare = prepare_then_interrupt
        with pytest.raises(_InitializationFailure) as raised:
            transport.add_remote_peer("peer:1", b"metadata", 0, 4, 16)

        assert raised.value is primary
        assert transport._peer_setup is None
        assert "peer:1" not in transport._remote_peers
        assert endpoint.peers[-1].closed is True
        assert endpoint.plans[-1].closed is True

        endpoint.prepare = original_prepare
        transport.add_remote_peer("peer:1", b"metadata", 0, 4, 16)
        assert "peer:1" in transport._remote_peers

    def test_peer_route_publication_cut_rolls_back_before_retry(self):
        transport, endpoint = self._make_transport()

        class InterruptingOwner(dict):
            interrupt = True

            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                if self.interrupt:
                    self.interrupt = False
                    raise _InitializationFailure("after adapter route publication")

        transport._remote_peers = InterruptingOwner()
        with pytest.raises(_InitializationFailure, match="route publication"):
            transport.add_remote_peer("peer:1", b"metadata", 0, 4, 16)

        assert transport._peer_setup is None
        assert transport._remote_peers == {}
        assert endpoint.peers[-1].closed is True
        assert endpoint.plans[-1].closed is True
        transport.add_remote_peer("peer:1", b"metadata", 0, 4, 16)
        assert "peer:1" in transport._remote_peers

    def test_write_and_peer_scoped_poll(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        transport.add_remote_peer("peer:2", b"two", 0, 4, 16)

        tid1 = transport.write_blocks("peer:1", [0, 2], [1, 3])
        tid2 = transport.write_blocks("peer:2", [1], [0])
        assert tid1 is not None and tid2 is not None
        transport._inflight = _ItemsForbiddenDict(transport._inflight)
        assert endpoint.write_calls == []
        plan_call = endpoint.plans[0].submit_indices_calls[0]
        assert plan_call["op"] is _TransferOp.WRITE
        assert plan_call["local_indices"].tolist() == [0, 2]
        assert plan_call["remote_indices"].tolist() == [1, 3]
        assert plan_call["local_indices"].dtype == np.int32
        assert plan_call["remote_indices"].dtype == np.int32
        assert plan_call["local_indices"].flags.c_contiguous
        assert plan_call["remote_indices"].flags.c_contiguous

        endpoint.works[0].state = _WorkState.SUCCEEDED
        endpoint.works[1].state = _WorkState.FAILED
        assert transport.poll(peer_id="peer:1") == PollResult(done=[tid1], failed=())
        assert endpoint.works[0].closed is True
        assert endpoint.works[0].state_reads == 1
        assert endpoint.works[0].test_calls == 0
        assert endpoint.works[0].error_reads == 0
        assert endpoint.works[0].exception_calls == 0
        assert endpoint.works[1].closed is False
        assert transport.poll(peer_id="peer:2") == PollResult(done=(), failed=[tid2])
        assert endpoint.works[1].closed is True
        assert endpoint.works[1].state_reads == 1
        assert endpoint.works[1].test_calls == 0
        assert endpoint.works[1].error_reads == 0
        assert endpoint.works[1].exception_calls == 0
        assert endpoint.progress_calls == 0
        assert transport._inflight_by_peer == {}

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            (_WorkState.PENDING, None),
            (_WorkState.RUNNING, None),
            (_WorkState.COMPLETED, "done"),
            (_WorkState.FAILED, "failed"),
            (_WorkState.CANCELLED, "failed"),
        ],
    )
    def test_current_core_poll_reads_only_authoritative_state(self, state, expected):
        transport, endpoint = self._make_transport()
        work = endpoint.write(object())
        work.state = state
        work.test = MagicMock(side_effect=AssertionError("test() was queried"))

        assert transport._test_work(work) == expected
        assert work.state_reads == 1
        assert work.test.call_count == 0
        assert work.error_reads == 0
        assert work.exception_calls == 0

    def test_transient_state_error_retains_work_then_recovers_completion(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        work.state = _WorkState.SUCCEEDED
        work.state_errors.append(RuntimeError("temporary observation failure"))

        assert transport.poll() == PollResult(done=(), failed=())
        assert tid in transport._inflight
        assert work.closed is False
        assert transport.poll() == PollResult(done=[tid], failed=())
        assert tid not in transport._inflight
        assert work.closed is True
        assert work.state_reads == 2

    @pytest.mark.parametrize(
        "work_state_table",
        [
            None,
            SimpleNamespace(
                PENDING=_WorkState.PENDING,
                RUNNING=_WorkState.RUNNING,
                COMPLETED=_WorkState.COMPLETED,
                FAILED=_WorkState.FAILED,
            ),
        ],
    )
    def test_poll_preserves_legacy_fallback_without_complete_work_state_table(
        self, work_state_table
    ):
        transport, endpoint = self._make_transport(work_state_table=work_state_table)
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        work.state = _WorkState.SUCCEEDED
        work.error = RuntimeError("legacy terminal failure")

        assert transport._work_state_outcomes is None
        assert transport.poll() == PollResult(done=(), failed=[tid])
        # The legacy terminal error is already authoritative; compatibility
        # code need not read a second state surface after observing it.
        assert work.state_reads == 0
        assert work.test_calls == 1
        assert work.error_reads == 1
        assert work.exception_calls == 0

    def test_poll_reuses_empty_result_and_retains_terminal_close_failure(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        work.state = _WorkState.SUCCEEDED
        close = MagicMock(side_effect=RuntimeError("retry cleanup"))
        work.close = close

        first = transport.poll()
        second = transport.poll()
        assert first is second
        assert first == PollResult(done=(), failed=())
        assert tid in transport._inflight
        assert transport._inflight_by_peer == {"peer:1": {tid: None}}
        assert close.call_count == 2
        assert work.state_reads == 1

        close.side_effect = None
        assert transport.poll() == PollResult(done=[tid], failed=())
        assert tid not in transport._inflight
        assert transport._inflight_by_peer == {}
        assert close.call_count == 3
        assert work.state_reads == 1

    def test_poll_retries_close_committed_before_bookkeeping(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        work.state = _WorkState.SUCCEEDED
        close_calls = 0

        def interrupted_close():
            nonlocal close_calls
            close_calls += 1
            work.closed = True
            if close_calls == 1:
                raise _InitializationFailure("after Work close commit")

        work.close = interrupted_close  # type: ignore[method-assign]
        with pytest.raises(_InitializationFailure, match="after Work close commit"):
            transport.poll()

        assert tid in transport._inflight
        assert transport._terminal_by_peer == {"peer:1": {tid: "done"}}
        assert transport.poll() == PollResult(done=[tid], failed=())
        assert close_calls == 2
        assert work.state_reads == 1
        assert transport._inflight_by_peer == {}
        assert transport._terminal_by_peer == {"peer:1": {tid: "done"}}
        assert transport.poll("peer:1") == PollResult(done=[tid], failed=())
        transport.ack_completions("peer:1", (tid,))
        assert transport._terminal_by_peer == {}

    def test_cancel_clears_terminal_journal_after_poll_close_failure(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        work.state = _WorkState.SUCCEEDED
        work.close = MagicMock(side_effect=RuntimeError("retry cleanup"))

        assert transport.poll() == PollResult(done=(), failed=())
        assert transport._terminal_by_peer == {"peer:1": {tid: "done"}}

        work.close.side_effect = None
        assert transport.cancel([tid], mode="wait") == []
        assert transport._inflight == {}
        assert transport._inflight_by_peer == {}
        assert transport._terminal_by_peer == {}
        transport.close()

    def test_manual_mode_advances_endpoint_once_per_nonempty_poll(self):
        transport, endpoint = self._make_transport(progress_mode="manual")
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None

        endpoint.works[-1].state = _WorkState.SUCCEEDED
        assert transport.poll() == PollResult(done=[tid], failed=())
        assert endpoint.progress_calls == 1

    def test_wait_cancel_retains_pending_work_then_close_is_idempotent(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        work.cancel_completes = False

        assert transport.cancel([tid], mode="wait") == [tid]
        assert work.closed is False

        work.state = _WorkState.CANCELLED
        assert transport.poll() == PollResult(done=(), failed=[tid])
        transport.ack_completions("peer:1", (tid,))
        transport.close()
        transport.close()

        assert work.closed is True
        assert endpoint.closed is True
        assert endpoint.registration is not None
        assert endpoint.registration.closed is True
        assert all(peer.closed for peer in endpoint.peers)
        assert all(plan.closed for plan in endpoint.plans)

    def test_immediate_cancel_retains_active_work_and_prevents_unsafe_close(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        work.cancel_completes = False

        assert transport.cancel([tid], mode="immediate") == []
        assert tid in transport._inflight
        with pytest.raises(RuntimeError, match="active transfer"):
            transport.close()

        work.state = _WorkState.CANCELLED
        assert transport.poll().failed == [tid]
        transport.ack_completions("peer:1", (tid,))
        transport.close()

    @pytest.mark.parametrize("outcome", [_WorkState.SUCCEEDED, _WorkState.FAILED])
    @pytest.mark.parametrize("retirement_path", ["reap", "poll", "remove_after_poll"])
    def test_peer_retirement_preserves_completions_until_ack(
        self, outcome, retirement_path
    ):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        peer, plan = endpoint.peers[-1], endpoint.plans[-1]
        if retirement_path != "remove_after_poll":
            transport.remove_remote_peer("peer:1")
        work.state = outcome
        if retirement_path == "reap":
            transport.reap_retired_peers()
        else:
            transport.poll("peer:1")
        if retirement_path == "remove_after_poll":
            transport.remove_remote_peer("peer:1")

        expected = (
            PollResult(done=[tid], failed=())
            if outcome is _WorkState.SUCCEEDED
            else PollResult(done=(), failed=[tid])
        )
        assert work.closed
        assert not peer.closed and not plan.closed
        assert transport.peer_retirement_complete("peer:1") is False
        assert transport.poll("peer:1") == expected
        assert transport.poll("peer:1") == expected
        transport.ack_completions("peer:1", (tid,))
        transport.reap_retired_peers()
        assert peer.closed and plan.closed
        assert transport.peer_retirement_complete("peer:1") is True
        transport.close()

    def test_unrelated_cancel_keeps_unacknowledged_retiring_peer(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        transport.remove_remote_peer("peer:1")
        work = endpoint.works[-1]
        work.state = _WorkState.SUCCEEDED
        work.close = MagicMock(side_effect=RuntimeError("retry cleanup"))
        assert transport.poll("peer:1") == PollResult(done=(), failed=())
        work.close.side_effect = None
        assert transport.poll("peer:1") == PollResult(done=[tid], failed=())

        assert transport.cancel((), mode="wait") == []
        assert not endpoint.peers[-1].closed and not endpoint.plans[-1].closed
        assert transport.poll("peer:1") == PollResult(done=[tid], failed=())
        transport.ack_completions("peer:1", (tid,))
        transport.reap_retired_peers()
        assert transport.peer_retirement_complete("peer:1") is True
        transport.close()

    def test_dead_peer_teardown_is_deferred_until_work_is_terminal(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        peer = endpoint.peers[-1]
        plan = endpoint.plans[-1]

        transport.remove_remote_peer("peer:1")
        assert "peer:1" in transport._retiring_peers
        assert transport.peer_retirement_complete("peer:1") is False
        assert transport._inflight_by_peer == {"peer:1": {tid: None}}
        assert not work.closed and not peer.closed and not plan.closed
        assert transport.write_blocks("peer:1", [0], [1]) is None

        work.state = _WorkState.SUCCEEDED
        transport.reap_retired_peers()
        assert work.closed and not peer.closed and not plan.closed
        assert transport.poll("peer:1") == PollResult(done=[tid], failed=())
        transport.ack_completions("peer:1", (tid,))
        transport.reap_retired_peers()
        assert work.closed and peer.closed and plan.closed
        assert "peer:1" not in transport._remote_peers
        assert "peer:1" not in transport._retiring_peers
        assert transport._inflight_by_peer == {}
        assert transport.peer_retirement_complete("peer:1") is True
        transport.close()

    def test_peer_retirement_repairs_interrupted_work_index_publication(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        tid = transport.write_blocks("peer:1", [0], [1])
        assert tid is not None
        work = endpoint.works[-1]
        peer = endpoint.peers[-1]
        plan = endpoint.plans[-1]

        # Model a BaseException after the strong Work owner was stored but
        # before publication in the derived per-peer index.
        transport._inflight_by_peer.clear()
        transport.remove_remote_peer("peer:1")
        assert transport._inflight_by_peer == {"peer:1": {tid: None}}
        assert "peer:1" in transport._retiring_peers
        assert not work.closed and not peer.closed and not plan.closed
        assert transport.peer_retirement_complete("peer:1") is False

        work.state = _WorkState.SUCCEEDED
        transport.reap_retired_peers()
        assert work.closed and not peer.closed and not plan.closed
        assert transport.poll("peer:1") == PollResult(done=[tid], failed=())
        transport.ack_completions("peer:1", (tid,))
        transport.reap_retired_peers()
        assert work.closed and peer.closed and plan.closed
        assert tid not in transport._inflight
        assert transport.peer_retirement_complete("peer:1") is True
        transport.close()

    def test_peer_close_failure_quarantines_plan_until_retry(self):
        transport, endpoint = self._make_transport()
        transport.add_remote_peer("peer:1", b"one", 0, 4, 16)
        plan = endpoint.plans[-1]
        peer = endpoint.peers[-1]
        close_calls = 0

        def close_plan():
            nonlocal close_calls
            close_calls += 1
            if close_calls == 1:
                raise RuntimeError("partially committed close")
            plan.closed = True

        plan.close = close_plan
        with pytest.raises(RuntimeError, match="partially committed"):
            transport.remove_remote_peer("peer:1")

        assert "peer:1" in transport._remote_peers
        assert "peer:1" in transport._retiring_peers
        assert transport.write_blocks("peer:1", [0], [1]) is None
        assert peer.closed is False

        transport.remove_remote_peer("peer:1")
        assert close_calls == 2
        assert plan.closed is True
        assert peer.closed is True
        assert "peer:1" not in transport._remote_peers
        assert "peer:1" not in transport._retiring_peers
