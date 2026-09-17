# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch endpoint-transfer data plane for P2P KV tiering."""

from __future__ import annotations

import importlib
import itertools
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Literal, NamedTuple

import numpy as np

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.p2p.data.base import (
    CancelMode,
    DataTransport,
    PollResult,
)

logger = init_logger(__name__)

_EMPTY_POLL_RESULT: PollResult = PollResult(done=(), failed=())
_REGISTRATION_NAME = "kv_blocks"
_PENDING_STATES = frozenset({"CREATED", "PENDING", "RUNNING", "SUBMITTED"})
_DONE_STATES = frozenset({"COMPLETE", "COMPLETED", "DONE", "SUCCEEDED"})
_FAILED_STATES = frozenset({"CANCELLED", "FAILED", "TIMED_OUT"})
_UNKNOWN_WORK_OUTCOME = object()
_FACTORY_API_VERSION = 2


class _RemotePeer(NamedTuple):
    peer: object
    plan: object


class _Inflight(NamedTuple):
    peer_id: str
    work: object
    recovery_token: object | None


@dataclass
class _PeerSetup:
    """Cold-path receipt for Core peer/plan construction."""

    peer_id: str
    # Core resources are opaque and need not be hashable or implement sane
    # equality. Construction recovery is deliberately identity-only.
    baseline_peers: tuple[object, ...]
    baseline_plans: tuple[object, ...]
    peer: object | None = None
    plan: object | None = None


_TerminalOutcome = Literal["done", "failed"]


def _load_transfer_api() -> Any | None:
    """Load the experimental API without making it a vLLM dependency."""
    try:
        module = importlib.import_module("torch.distributed._transfer")
    except ImportError:
        return None
    return (
        module
        if hasattr(module, "Endpoint") and hasattr(module, "TransferOp")
        else None
    )


def _register_nixl_backend() -> None:
    """Register the optional provider only for an explicitly selected path."""
    try:
        module = importlib.import_module("nixl.torch_transfer")
    except ImportError as exc:
        raise ImportError(
            "The PyTorch NIXL transfer backend requires nixl.torch_transfer"
        ) from exc
    provider_version = getattr(module, "TORCH_TRANSFER_FACTORY_API_VERSION", None)
    if type(provider_version) is not int or provider_version != _FACTORY_API_VERSION:
        raise RuntimeError(
            "nixl.torch_transfer does not advertise the supported factory "
            f"API version {_FACTORY_API_VERSION}"
        )
    register = getattr(module, "register_torch_backend", None)
    if not callable(register):
        raise ImportError(
            "nixl.torch_transfer does not expose register_torch_backend()"
        )
    register()


def _require_adoptive_nixl_spi(transfer_api: Any) -> None:
    """Reject Core unless it explicitly advertises the v2 factory contract."""
    core_version = getattr(transfer_api, "BACKEND_FACTORY_API_VERSION", None)
    factory_v2 = getattr(transfer_api, "BackendFactoryV2", None)
    if (
        type(core_version) is not int
        or core_version < _FACTORY_API_VERSION
        or factory_v2 is None
    ):
        raise RuntimeError(
            "the installed torch.distributed._transfer Core does not "
            "advertise the required construction/factory v2 SPI "
            f"(BACKEND_FACTORY_API_VERSION={core_version!r}, "
            f"BackendFactoryV2={'present' if factory_v2 is not None else 'missing'}); "
            "refusing outcome-ambiguous v1 fallback"
        )


class TorchTransferTransport(DataTransport):
    """Adapts ``torch.distributed._transfer`` to tiering's block protocol.

    The adapter is the only vLLM module coupled to the experimental PyTorch
    contract. The native ``NixlTransport`` remains the default data plane. This
    tiering path registers its DRAM mapping, so CUDA producer ordering and a
    device completion event are intentionally outside this adapter's contract.
    """

    def __init__(
        self,
        endpoint_name: str,
        view: memoryview,
        config_fields: dict | None = None,
        endpoint_id: str | None = None,
        backend: str = "nixl",
        progress_mode: str = "background",
        thread_mode: str = "serialized",
        options: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(view, config_fields=config_fields)
        self._endpoint_name = endpoint_name
        # Construction recovery needs an exact identity. vLLM production
        # supplies host:port; direct users get the already-unique agent name.
        self._endpoint_id = endpoint_id if endpoint_id is not None else endpoint_name
        self._backend = backend
        self._progress_mode = progress_mode
        self._manual_progress = progress_mode == "manual"
        self._thread_mode = thread_mode
        self._options = dict(options or {})
        self._endpoint: Any = None
        self._registration: Any = None
        self._local_regions: tuple[object, ...] = ()
        self._remote_peers: dict[str, _RemotePeer] = {}
        self._peer_setup: _PeerSetup | None = None
        self._retiring_peers: set[str] = set()
        self._inflight: dict[int, _Inflight] = {}
        # Ordered transfer IDs only; ``_inflight`` remains the sole Work owner.
        # The manager polls once per peer, so this avoids re-scanning sibling
        # sessions while preserving submission order within each peer.
        self._inflight_by_peer: dict[str, dict[int, None]] = {}
        # Exact outcomes remain until the owning session durably adopts and
        # acknowledges them.  Entries whose IDs remain in ``_inflight`` are
        # terminal Works whose close still needs to be retried.
        self._terminal_by_peer: dict[str, dict[int, _TerminalOutcome]] = {}
        self._next_id = itertools.count()
        self._write_op: object | None = None
        self._transfer_error_types: tuple[type[BaseException], ...] = ()
        self._work_state_outcomes: (
            dict[object, Literal["done", "failed"] | None] | None
        ) = None
        # Failed construction can outlive ``__init__`` through
        # exception.recovery_transport. Exact Core owners stay here until a
        # retrying child-first close commits.
        self._construction_failed = False
        self._construction_cleanup_errors: tuple[str, ...] = ()

        transfer_api = _load_transfer_api()
        if transfer_api is not None:
            self._init(transfer_api, view)

    @property
    def available(self) -> bool:
        return self._endpoint is not None and not self._construction_failed

    @property
    def construction_cleanup_complete(self) -> bool:
        """Whether a failed constructor retains no Core cleanup owner."""
        return not self._construction_failed

    def _init(self, transfer_api: Any, view: memoryview) -> None:
        self._write_op = transfer_api.TransferOp.WRITE
        self._work_state_outcomes = self._complete_work_state_outcomes(transfer_api)
        transfer_error = getattr(transfer_api, "TransferError", None)
        if isinstance(transfer_error, type) and issubclass(
            transfer_error, BaseException
        ):
            self._transfer_error_types = (transfer_error,)
        if self._backend == "nixl":
            _require_adoptive_nixl_spi(transfer_api)
            _register_nixl_backend()
        progress_mode = transfer_api.ProgressMode(self._progress_mode)
        thread_mode = transfer_api.ThreadMode(self._thread_mode)
        self._manual_progress = progress_mode is transfer_api.ProgressMode.MANUAL
        try:
            self._endpoint = transfer_api.Endpoint(
                self._endpoint_name,
                endpoint_id=self._endpoint_id,
                backend=self._backend,
                progress_mode=progress_mode,
                thread_mode=thread_mode,
                options=self._options,
            )
            if not bool(getattr(self._endpoint.capabilities, "write", False)):
                raise NotImplementedError(
                    "The selected PyTorch transfer backend does not support WRITE"
                )
            baseline_registrations = tuple(self._endpoint.open_registrations)
            try:
                self._registration = self._endpoint.register(
                    view, name=_REGISTRATION_NAME
                )
            except BaseException:
                # Core publishes a Registration before returning it.  Recover a
                # result lost between CALL and STORE_ATTR and leave endpoint
                # rollback to close the exact child.
                candidates = tuple(
                    registration
                    for registration in self._endpoint.open_registrations
                    if not any(
                        registration is original for original in baseline_registrations
                    )
                )
                if len(candidates) > 1:
                    raise RuntimeError(
                        "multiple Core registrations appeared during one setup"
                    ) from None
                if candidates:
                    self._registration = candidates[0]
                raise
            self._local_regions = (
                self._registration.region(
                    0,
                    self._block_len,
                    stride=self._block_len,
                    count=self._num_blocks,
                ),
            )
        except BaseException as exc:
            if self._endpoint is None:
                try:
                    recovery_endpoint = getattr(exc, "recovery_endpoint", None)
                except BaseException as lookup_exc:
                    recovery_endpoint = None
                    with suppress(BaseException):
                        exc.add_note(
                            "reading recovery_endpoint also raised: "
                            f"{type(lookup_exc).__qualname__}"
                        )
                if recovery_endpoint is not None:
                    try:
                        trusted = (
                            type(recovery_endpoint) is transfer_api.Endpoint
                            and getattr(recovery_endpoint, "endpoint_id", None)
                            == self._endpoint_id
                            and getattr(recovery_endpoint, "name", None)
                            == self._endpoint_name
                            and getattr(recovery_endpoint, "backend", None)
                            == self._backend
                            and getattr(recovery_endpoint, "progress_mode", None)
                            is progress_mode
                            and getattr(recovery_endpoint, "thread_mode", None)
                            is thread_mode
                        )
                    except BaseException as match_exc:
                        trusted = False
                        with suppress(BaseException):
                            exc.add_note(
                                "validating recovery_endpoint also raised: "
                                f"{type(match_exc).__qualname__}"
                            )
                    if trusted:
                        self._endpoint = recovery_endpoint
                    else:
                        # A real Core Endpoint may belong to another live
                        # caller. No exact identity proof means no mutation.
                        with suppress(BaseException):
                            exc.add_note(
                                "ignored a non-matching recovery_endpoint during "
                                "vLLM PyTorch transport construction"
                            )
            self._local_regions = ()
            self._construction_failed = True
            try:
                self._close_construction_resources()
            except BaseException as cleanup_exc:
                self._construction_cleanup_errors += (
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                )
                try:
                    exc.recovery_transport = self  # type: ignore[attr-defined]
                except BaseException:
                    recovery_error = RuntimeError(
                        "Torch transfer construction cleanup is incomplete; "
                        "retry exception.recovery_transport.close()"
                    )
                    recovery_error.recovery_transport = self  # type: ignore[attr-defined]
                    raise recovery_error from exc
            raise

    def get_agent_metadata(self) -> bytes:
        assert self._endpoint is not None
        assert self._registration is not None
        metadata = self._endpoint.export_metadata([self._registration])
        if isinstance(metadata, bytes):
            return metadata
        to_bytes = getattr(metadata, "to_bytes", None)
        if callable(to_bytes):
            return to_bytes()
        try:
            return bytes(metadata)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Endpoint.export_metadata() must return bytes-compatible metadata"
            ) from exc

    def add_remote_peer(
        self,
        peer_id: str,
        agent_metadata: bytes,
        base_addr: int,
        num_blocks: int,
        block_len: int,
    ) -> None:
        """Import a peer and prepare reusable indexed block descriptors."""
        assert self._endpoint is not None
        self._reconcile_peer_setup()
        if peer_id in self._remote_peers:
            raise ValueError(f"peer {peer_id!r} is already registered")
        if block_len != self._block_len:
            raise ValueError(
                f"block_len mismatch: remote={block_len}, local={self._block_len}"
            )
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        # PyTorch metadata owns the remote descriptor. base_addr stays in the
        # existing handshake for native NIXL compatibility but is not trusted.
        del base_addr
        setup = _PeerSetup(
            peer_id=peer_id,
            baseline_peers=tuple(self._endpoint.open_peers),
            baseline_plans=tuple(self._endpoint.open_plans),
        )
        self._peer_setup = setup
        try:
            peer = self._endpoint.import_peer(
                agent_metadata,
                expected_endpoint_id=peer_id,
            )
            setup.peer = peer
            remote_regions = (
                peer.region(
                    _REGISTRATION_NAME,
                    0,
                    block_len,
                    stride=block_len,
                    count=num_blocks,
                ),
            )
            plan = self._endpoint.prepare(
                local=self._local_regions,
                remote=remote_regions,
                indexed=True,
            )
            setup.plan = plan
            self._remote_peers[peer_id] = _RemotePeer(peer, plan)
            self._peer_setup = None
        except BaseException as exc:
            try:
                self._reconcile_peer_setup()
            except BaseException as cleanup_exc:
                with suppress(BaseException):
                    logger.warning(
                        "TorchTransferTransport %s: peer setup cleanup for %s "
                        "remains pending after %s",
                        self._endpoint_name,
                        peer_id,
                        cleanup_exc,
                    )
            if self._transfer_error_types and isinstance(
                exc, self._transfer_error_types
            ):
                raise ValueError(
                    f"invalid endpoint-transfer metadata for peer {peer_id!r}"
                ) from exc
            raise

    def remove_remote_peer(self, peer_id: str) -> None:
        self._reconcile_peer_setup()
        self._repair_peer_index(peer_id)
        if self._inflight_by_peer.get(peer_id) or self._terminal_by_peer.get(peer_id):
            self._retiring_peers.add(peer_id)
            return
        self._close_remote_peer(peer_id)

    def reap_retired_peers(self) -> None:
        """Drain retired work without discarding unacknowledged outcomes."""

        for peer_id in tuple(self._retiring_peers):
            self._repair_peer_index(peer_id)
            self.poll(peer_id)
            if self._inflight_by_peer.get(peer_id) or self._terminal_by_peer.get(
                peer_id
            ):
                continue
            self._close_remote_peer(peer_id)

    def peer_retirement_complete(self, peer_id: str) -> bool:
        self._repair_peer_index(peer_id)
        return (
            peer_id not in self._retiring_peers
            and peer_id not in self._remote_peers
            and not self._inflight_by_peer.get(peer_id)
            and not self._terminal_by_peer.get(peer_id)
        )

    def _close_remote_peer(self, peer_id: str) -> None:
        entry = self._remote_peers.get(peer_id)
        if entry is None:
            self._retiring_peers.discard(peer_id)
            return
        # Quarantine before native teardown starts. A partially committed
        # close is retryable but must never leave this Plan submit-visible.
        self._retiring_peers.add(peer_id)
        entry.plan.close()  # type: ignore[attr-defined]
        entry.peer.close()  # type: ignore[attr-defined]
        del self._remote_peers[peer_id]
        self._retiring_peers.discard(peer_id)

    def write_blocks(
        self,
        peer_id: str,
        local_idxs: list[int],
        remote_idxs: list[int],
    ) -> int | None:
        return self._write_blocks(
            peer_id,
            local_idxs,
            remote_idxs,
            recovery_token=None,
        )

    def write_blocks_owned(
        self,
        peer_id: str,
        local_idxs: list[int],
        remote_idxs: list[int],
        *,
        recovery_token: object,
    ) -> int | None:
        if recovery_token is None:
            raise ValueError("recovery_token must be a unique non-None object")
        return self._write_blocks(
            peer_id,
            local_idxs,
            remote_idxs,
            recovery_token=recovery_token,
        )

    def _write_blocks(
        self,
        peer_id: str,
        local_idxs: list[int],
        remote_idxs: list[int],
        *,
        recovery_token: object | None,
    ) -> int | None:
        """Submit an indexed, asynchronous WRITE to a registered peer."""
        entry = self._remote_peers.get(peer_id)
        if entry is None or peer_id in self._retiring_peers:
            logger.warning(
                "TorchTransferTransport %s: no transfer plan for peer=%s",
                self._endpoint_name,
                peer_id,
            )
            return None
        if len(local_idxs) != len(remote_idxs):
            raise ValueError("local_idxs and remote_idxs must have equal length")
        assert self._endpoint is not None
        assert self._write_op is not None
        # Core snapshots and validates these packed arrays inside one submission
        # transaction; no transient public IndexSelection is allocated.
        if recovery_token is None:
            # Keep the legacy API free of an extra successful-path keyword.
            work = entry.plan.submit_indices(
                self._write_op,
                local_indices=np.asarray(local_idxs, dtype=np.int32),
                remote_indices=np.asarray(remote_idxs, dtype=np.int32),
            )
        else:
            work = entry.plan.submit_indices(
                self._write_op,
                local_indices=np.asarray(local_idxs, dtype=np.int32),
                remote_indices=np.asarray(remote_idxs, dtype=np.int32),
                recovery_token=recovery_token,
            )
        transfer_id = next(self._next_id)
        self._record_inflight(
            transfer_id,
            _Inflight(peer_id, work, recovery_token),
        )
        return transfer_id

    def recover_transfer_id(
        self,
        peer_id: str,
        recovery_token: object,
    ) -> int | None:
        if recovery_token is None:
            raise ValueError("recovery_token must be a unique non-None object")
        transport_matches = [
            (transfer_id, entry)
            for transfer_id, entry in self._inflight.items()
            if entry.recovery_token is recovery_token
        ]
        if len(transport_matches) > 1:
            raise RuntimeError("multiple Core transfers share one recovery token")
        assert self._endpoint is not None
        work_matches = [
            work
            for work in self._endpoint.open_works
            if getattr(work, "recovery_token", None) is recovery_token
        ]
        if len(work_matches) > 1:
            raise RuntimeError("multiple open Works share one recovery token")
        if transport_matches:
            transfer_id, entry = transport_matches[0]
            if entry.peer_id != peer_id:
                raise RuntimeError("recovery token belongs to a different peer")
            if work_matches and work_matches[0] is not entry.work:
                raise RuntimeError("transport and Core recovery owners disagree")
            self._repair_peer_index(peer_id)
            return transfer_id
        if not work_matches:
            return None
        if peer_id not in self._remote_peers:
            raise RuntimeError("recovered Work belongs to an unknown peer")
        transfer_id = next(self._next_id)
        self._record_inflight(
            transfer_id,
            _Inflight(peer_id, work_matches[0], recovery_token),
        )
        return transfer_id

    def poll(self, peer_id: str | None = None) -> PollResult:
        if self._terminal_by_peer:
            retry_result = self._retry_terminal_poll(peer_id)
            if retry_result is not None:
                return retry_result
        if not self._inflight:
            return _EMPTY_POLL_RESULT
        assert self._endpoint is not None
        if self._manual_progress:
            try:
                self._endpoint.progress()
            except Exception as exc:
                logger.warning(
                    "TorchTransferTransport %s: progress failed: %s",
                    self._endpoint_name,
                    exc,
                )

        terminal: list[tuple[int, _Inflight, Literal["done", "failed"]]] | None = None
        candidate_ids: Iterable[int] = (
            self._inflight
            if peer_id is None
            else self._inflight_by_peer.get(peer_id, ())
        )
        for transfer_id in candidate_ids:
            entry = self._inflight[transfer_id]
            outcome = self._test_work(entry.work)
            if outcome is None:
                continue
            if terminal is None:
                terminal = []
            terminal.append((transfer_id, entry, outcome))

        done: list[int] | None = None
        failed: list[int] | None = None
        for transfer_id, entry, outcome in terminal or ():
            self._terminal_by_peer.setdefault(entry.peer_id, {})[transfer_id] = outcome
            try:
                entry.work.close()  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning(
                    "TorchTransferTransport %s: terminal work close failed "
                    "for transfer_id=%d: %s",
                    self._endpoint_name,
                    transfer_id,
                    exc,
                )
                continue
            self._pop_inflight(transfer_id)
            if outcome == "done":
                if done is None:
                    done = []
                done.append(transfer_id)
            else:
                if failed is None:
                    failed = []
                failed.append(transfer_id)
        for retiring_peer in tuple(self._retiring_peers):
            if not self._inflight_by_peer.get(
                retiring_peer
            ) and not self._terminal_by_peer.get(retiring_peer):
                self._close_remote_peer(retiring_peer)
        if done is None and failed is None:
            return _EMPTY_POLL_RESULT
        return PollResult(
            done=done if done is not None else (),
            failed=failed if failed is not None else (),
        )

    def ack_completions(
        self,
        peer_id: str,
        transfer_ids: Iterable[int],
    ) -> None:
        """Forget only outcomes already adopted by ``peer_id``'s session."""
        journal = self._terminal_by_peer.get(peer_id)
        for transfer_id in transfer_ids:
            if type(transfer_id) is not int:
                raise TypeError("transfer_id must be an exact int")
            if journal is not None and transfer_id in journal:
                if transfer_id in self._inflight:
                    raise RuntimeError(
                        "cannot acknowledge a terminal transfer before Work cleanup"
                    )
                self._clear_terminal(peer_id, transfer_id)
                journal = self._terminal_by_peer.get(peer_id)
                continue
            if any(
                transfer_id in other
                for owner_id, other in self._terminal_by_peer.items()
                if owner_id != peer_id
            ):
                raise RuntimeError("terminal transfer belongs to a different peer")

    def cancel(
        self,
        transfer_ids: Iterable[int],
        mode: CancelMode = "immediate",
    ) -> list[int]:
        if self._endpoint is not None and self._manual_progress:
            with _LogExceptions(self._endpoint_name, "progress during cancel"):
                self._endpoint.progress()
        still_inflight: list[int] = []
        for transfer_id in transfer_ids:
            entry = self._inflight.get(transfer_id)
            if entry is None:
                self._discard_terminal_id(transfer_id)
                continue
            if bool(getattr(entry.work, "closed", False)):
                # Core rejects state/cancel after close. Its close retry
                # finishes any interrupted Core-only graph-unlink epilogue.
                entry.work.close()  # type: ignore[attr-defined]
                self._pop_inflight(transfer_id)
                self._clear_terminal(entry.peer_id, transfer_id)
                continue
            self._cancel_work(entry.work)
            if self._test_work(entry.work) is None:
                still_inflight.append(transfer_id)
                continue
            entry.work.close()  # type: ignore[attr-defined]
            self._pop_inflight(transfer_id)
            self._clear_terminal(entry.peer_id, transfer_id)
        if mode == "immediate" and still_inflight:
            logger.warning(
                "TorchTransferTransport %s: cancellation left %d work item(s) "
                "active; retaining their resources",
                self._endpoint_name,
                len(still_inflight),
            )
            return []
        for retiring_peer in tuple(self._retiring_peers):
            if not self._inflight_by_peer.get(
                retiring_peer
            ) and not self._terminal_by_peer.get(retiring_peer):
                self._close_remote_peer(retiring_peer)
        return still_inflight

    def close(self) -> None:
        if self._construction_failed:
            self._close_construction_resources()
            return
        if self._endpoint is None:
            return
        self._reconcile_peer_setup()
        self.cancel(tuple(self._inflight), mode="immediate")
        if self._inflight:
            raise RuntimeError(
                f"cannot close endpoint with {len(self._inflight)} active transfer(s)"
            )
        if self._inflight_by_peer:
            raise RuntimeError("inflight peer index is inconsistent")
        if self._terminal_by_peer:
            raise RuntimeError("terminal Work journal is inconsistent")
        for peer_id in tuple(self._remote_peers):
            self.remove_remote_peer(peer_id)
        self._local_regions = ()
        if self._registration is not None:
            self._registration.close()
            self._registration = None
        self._endpoint.close()
        self._endpoint = None

    def _close_construction_resources(self) -> None:
        """Retry exact child-first cleanup after a failed constructor."""
        if self._registration is not None:
            self._registration.close()
            self._registration = None
        if self._endpoint is not None:
            self._endpoint.close()
            self._endpoint = None
        self._construction_failed = False

    def _reconcile_peer_setup(self) -> None:
        """Recover and roll back an interrupted Core child construction."""
        setup = self._peer_setup
        if setup is None:
            return
        endpoint = self._endpoint
        if endpoint is None:
            raise RuntimeError("peer setup receipt outlived its Core endpoint")

        if setup.peer is None:
            candidates = tuple(
                peer
                for peer in endpoint.open_peers
                if not any(peer is original for original in setup.baseline_peers)
                and getattr(peer, "endpoint_id", None) == setup.peer_id
            )
            if len(candidates) > 1:
                raise RuntimeError("multiple Core peers match one setup receipt")
            if candidates:
                setup.peer = candidates[0]
        if setup.plan is None:
            candidates = tuple(
                plan
                for plan in endpoint.open_plans
                if not any(plan is original for original in setup.baseline_plans)
            )
            if len(candidates) > 1:
                raise RuntimeError("multiple Core plans match one setup receipt")
            if candidates:
                setup.plan = candidates[0]

        committed = self._remote_peers.get(setup.peer_id)
        if committed is not None:
            if committed.peer is not setup.peer or committed.plan is not setup.plan:
                raise RuntimeError(
                    "Core peer setup receipt conflicts with adapter owner"
                )
            # A live receipt proves add_remote_peer did not return normally.
            # Remove the prematurely published route before child teardown so
            # callers never observe a usable peer after setup reported failure.
            self._remote_peers.pop(setup.peer_id, None)

        # Quarantine before fallible child-first teardown.  A cleanup cut leaves
        # this receipt and peer ID non-submit-visible until the next retry.
        self._retiring_peers.add(setup.peer_id)
        if setup.plan is not None:
            setup.plan.close()  # type: ignore[attr-defined]
            setup.plan = None
        if setup.peer is not None:
            setup.peer.close()  # type: ignore[attr-defined]
            setup.peer = None
        self._peer_setup = None
        self._retiring_peers.discard(setup.peer_id)

    def _record_inflight(self, transfer_id: int, entry: _Inflight) -> None:
        self._inflight[transfer_id] = entry
        self._inflight_by_peer.setdefault(entry.peer_id, {})[transfer_id] = None

    def _repair_peer_index(self, peer_id: str) -> None:
        """Rebuild missing derived entries before peer-lifetime decisions.

        Primary Work ownership is published first. This cold-path scan closes
        the asynchronous-exception window before retirement observes the
        derived index, while keeping successful submission and polling intact.
        """

        peer_transfers = self._inflight_by_peer.get(peer_id)
        for transfer_id, entry in self._inflight.items():
            if entry.peer_id != peer_id or (
                peer_transfers is not None and transfer_id in peer_transfers
            ):
                continue
            if peer_transfers is None:
                peer_transfers = self._inflight_by_peer.setdefault(peer_id, {})
            peer_transfers[transfer_id] = None

    def _pop_inflight(self, transfer_id: int) -> _Inflight:
        entry = self._inflight.get(transfer_id)
        if entry is None:
            raise KeyError(transfer_id)
        self._discard_peer_index(entry.peer_id, transfer_id)
        committed = self._inflight.pop(transfer_id, None)
        if committed is None:
            raise KeyError(transfer_id)
        return committed

    def _discard_peer_index(self, peer_id: str, transfer_id: int) -> None:
        peer_transfers = self._inflight_by_peer.get(peer_id)
        if peer_transfers is None:
            return
        peer_transfers.pop(transfer_id, None)
        if not peer_transfers:
            self._inflight_by_peer.pop(peer_id, None)

    def _clear_terminal(self, peer_id: str, transfer_id: int) -> None:
        journal = self._terminal_by_peer.get(peer_id)
        if journal is None:
            return
        journal.pop(transfer_id, None)
        if not journal:
            self._terminal_by_peer.pop(peer_id, None)

    def _discard_terminal_id(self, transfer_id: int) -> None:
        for peer_id, journal in tuple(self._terminal_by_peer.items()):
            if transfer_id in journal:
                self._discard_peer_index(peer_id, transfer_id)
                self._clear_terminal(peer_id, transfer_id)
                return

    def _retry_terminal_poll(self, peer_id: str | None) -> PollResult | None:
        """Finish Work cleanup without querying state after close."""

        if peer_id is None:
            peer_ids = tuple(self._terminal_by_peer)
        elif self._terminal_by_peer.get(peer_id):
            peer_ids = (peer_id,)
        else:
            return None
        if not peer_ids:
            return None

        for owner_id in peer_ids:
            for transfer_id in tuple(self._terminal_by_peer.get(owner_id, {})):
                entry = self._inflight.get(transfer_id)
                if entry is not None:
                    try:
                        entry.work.close()  # type: ignore[attr-defined]
                    except Exception as exc:
                        logger.warning(
                            "TorchTransferTransport %s: terminal Work close retry "
                            "failed for transfer_id=%d: %s",
                            self._endpoint_name,
                            transfer_id,
                            exc,
                        )
                        continue
                    self._pop_inflight(transfer_id)
                else:
                    self._discard_peer_index(owner_id, transfer_id)
        return self._peek_terminal(peer_id)

    def _peek_terminal(self, peer_id: str | None) -> PollResult:
        """Return cleaned terminal outcomes without transferring ownership."""
        peer_ids = tuple(self._terminal_by_peer) if peer_id is None else (peer_id,)
        done: list[int] | None = None
        failed: list[int] | None = None
        for owner_id in peer_ids:
            for transfer_id, outcome in self._terminal_by_peer.get(
                owner_id, {}
            ).items():
                if transfer_id in self._inflight:
                    continue
                if outcome == "done":
                    if done is None:
                        done = []
                    done.append(transfer_id)
                else:
                    if failed is None:
                        failed = []
                    failed.append(transfer_id)
        if done is None and failed is None:
            return _EMPTY_POLL_RESULT
        return PollResult(done=done or (), failed=failed or ())

    def _test_work(self, work: object) -> Literal["done", "failed"] | None:
        outcomes = self._work_state_outcomes
        if outcomes is None:
            return self._test_work_compat(work)

        try:
            # Current Core's state property performs the provider refresh and
            # is authoritative for success/failure. Do not also read test(),
            # error, or exception: each would refresh the same Work again.
            state = work.state  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning(
                "TorchTransferTransport %s: work.state failed: %s",
                self._endpoint_name,
                exc,
            )
            # Observation failure does not prove terminal transport failure.
            # Retain Work and all dependent resources for a later recovery.
            return None
        try:
            outcome = outcomes.get(state, _UNKNOWN_WORK_OUTCOME)
        except TypeError:
            outcome = _UNKNOWN_WORK_OUTCOME
        if outcome is _UNKNOWN_WORK_OUTCOME:
            raise ValueError(f"unknown PyTorch transfer work state: {state!r}")
        return outcome  # type: ignore[return-value]

    def _test_work_compat(self, work: object) -> Literal["done", "failed"] | None:
        """Classify older experimental Work surfaces without an enum table."""

        try:
            result = work.test()  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning(
                "TorchTransferTransport %s: work.test failed: %s",
                self._endpoint_name,
                exc,
            )
            return "failed"
        if isinstance(result, bool):
            if not result:
                return None
            error = getattr(work, "error", None)
            if callable(error):
                error = error()
            if error is not None:
                return "failed"
            exception = getattr(work, "exception", None)
            if callable(exception) and exception() is not None:
                return "failed"
            state = self._state_name(getattr(work, "state", None))
            if state in _FAILED_STATES:
                return "failed"
            return "done"

        state = self._state_name(result)
        if state is None:
            raise TypeError("Work.test() must return bool or WorkState")
        if state in _PENDING_STATES:
            return None
        if state in _DONE_STATES:
            return "done"
        if state in _FAILED_STATES:
            return "failed"
        raise ValueError(f"unknown PyTorch transfer work state: {result!r}")

    @staticmethod
    def _complete_work_state_outcomes(
        transfer_api: Any,
    ) -> dict[object, Literal["done", "failed"] | None] | None:
        work_state = getattr(transfer_api, "WorkState", None)
        try:
            outcomes: dict[object, Literal["done", "failed"] | None] = {
                work_state.PENDING: None,
                work_state.RUNNING: None,
                work_state.COMPLETED: "done",
                work_state.FAILED: "failed",
                work_state.CANCELLED: "failed",
            }
        except (AttributeError, TypeError):
            return None
        # Aliased or otherwise incomplete tables cannot safely classify the
        # full current Core contract; use the compatibility path instead.
        return outcomes if len(outcomes) == 5 else None

    @staticmethod
    def _state_name(state: object) -> str | None:
        state = getattr(state, "name", state)
        return state.upper() if isinstance(state, str) else None

    def _cancel_work(self, work: object) -> None:
        with _LogExceptions(self._endpoint_name, "work cancellation"):
            work.cancel()  # type: ignore[attr-defined]

    def _close_resource(self, resource: object | None) -> None:
        if resource is None:
            return
        with _LogExceptions(self._endpoint_name, "resource close"):
            resource.close()  # type: ignore[attr-defined]


class _LogExceptions:
    """Log cleanup failures without masking the original transport outcome."""

    def __init__(self, endpoint_name: str, action: str) -> None:
        self._endpoint_name = endpoint_name
        self._action = action

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc is None:
            return False
        with suppress(BaseException):
            logger.warning(
                "TorchTransferTransport %s: %s failed: %s",
                self._endpoint_name,
                self._action,
                exc,
            )
        return True
