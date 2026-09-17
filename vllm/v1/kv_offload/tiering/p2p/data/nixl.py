# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
NixlTransport: Data-plane transport for RDMA-based KV block transfers via NIXL.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from contextlib import suppress
from typing import Any, Literal, NamedTuple

import numpy as np

from vllm.distributed.nixl_utils import NixlWrapper as _NixlAgent
from vllm.distributed.nixl_utils import nixl_agent_config as _NixlAgentConfig
from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.p2p.data.base import (
    CancelMode,
    DataTransport,
    PollResult,
)

logger = init_logger(__name__)

# Shared sentinel returned by poll() in the steady state (no inflight, or
# no transfer changed state since the last poll). Tuples make it immutable;
# callers only iterate / membership-test / equality-check.
_EMPTY_POLL_RESULT: PollResult = PollResult(done=(), failed=())


class _Inflight(NamedTuple):
    """A submitted-but-not-yet-drained transfer.

    ``peer_id`` lets poll() scope to a single owning session, since the
    transport is shared across all peer sessions of the engine.
    """

    peer_id: str
    handle: object
    recovery_token: object | None


_TerminalOutcome = Literal["done", "failed"]


class NixlTransport(DataTransport):
    """Manages a NIXL agent, memory registration, and block transfers.

    Wraps the NIXL C library behind a Python interface so the rest of the
    P2P tier code never touches NIXL types directly. Tracks inflight
    handles internally and returns completed/failed tags on poll.
    """

    def __init__(
        self,
        agent_name: str,
        view: memoryview,
        config_fields: dict | None = None,
        backends: list[str] | None = None,
        num_threads: int = 4,
    ) -> None:
        super().__init__(view, config_fields=config_fields)
        self._agent_name = agent_name
        self._backends = list(backends) if backends else ["UCX"]
        self._num_threads = num_threads
        self._agent: Any = None
        self._reg: Any = None
        self._local_dlist: Any = None
        self._remote_dlists: dict[str, object] = {}
        self._peer_nixl_names: dict[str, str] = {}
        # transfer_id → _Inflight(peer_id, handle).
        self._inflight: dict[int, _Inflight] = {}
        # Ordered IDs let a session poll only its peer. The flat map above is
        # still the sole owner used by cancellation and shutdown.
        self._inflight_by_peer: dict[str, dict[int, None]] = {}
        # Exact outcomes remain until the owning session durably adopts and
        # acknowledges them.  Entries whose IDs remain in ``_inflight`` are
        # terminal handles whose native cleanup still needs to be retried.
        self._terminal_by_peer: dict[str, dict[int, _TerminalOutcome]] = {}
        self._retiring_peers: set[str] = set()
        self._next_id = itertools.count()

        try:
            self._init(view)
        except BaseException:
            self._rollback_initialization()
            raise

    @property
    def available(self) -> bool:
        return self._agent is not None

    def _init(self, view: memoryview) -> None:
        if _NixlAgent is None:
            return

        non_ucx_backends = [b for b in self._backends if b != "UCX"]
        if non_ucx_backends:
            cfg = _NixlAgentConfig(backends=self._backends, capture_telemetry=True)
            logger.info(
                "NixlTransport %s: NIXL backends=%s",
                self._agent_name,
                self._backends,
            )
        else:
            cfg = _NixlAgentConfig(
                num_threads=self._num_threads, capture_telemetry=True
            )
            logger.info(
                "NixlTransport %s: NIXL backends=[UCX] num_threads=%d",
                self._agent_name,
                self._num_threads,
            )
        self._agent = _NixlAgent(self._agent_name, cfg)

        total_size = self._num_blocks * self._block_len
        reg_descs = [(self._base_addr, total_size, 0, "")]
        self._reg = self._agent.register_memory(reg_descs, mem_type="DRAM")

        block_tuples = [
            (self._base_addr + i * self._block_len, self._block_len, 0)
            for i in range(self._num_blocks)
        ]
        xfer_dlist = self._agent.get_xfer_descs(block_tuples, mem_type="DRAM")
        self._local_dlist = self._agent.prep_xfer_dlist("NIXL_INIT_AGENT", xfer_dlist)
        logger.info(
            "NixlTransport %s: registered %d blocks", self._agent_name, self._num_blocks
        )

    def get_agent_metadata(self) -> bytes:
        assert self._agent is not None
        return self._agent.get_agent_metadata()

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_remote_peer(
        self,
        peer_id: str,
        agent_metadata: bytes,
        base_addr: int,
        num_blocks: int,
        block_len: int,
    ) -> None:
        if peer_id in self._peer_nixl_names or peer_id in self._remote_dlists:
            raise ValueError(f"peer {peer_id!r} is already registered")
        nixl_name = self._agent.add_remote_agent(agent_metadata)
        # Publish native-agent ownership before the next fallible operation so
        # failed setup stays discoverable by close() if rollback itself fails.
        self._peer_nixl_names[peer_id] = nixl_name
        try:
            block_descs = [
                (base_addr + i * block_len, block_len, 0) for i in range(num_blocks)
            ]
            xfer_dlist = self._agent.get_xfer_descs(block_descs, mem_type="DRAM")
            remote_dlist = self._agent.prep_xfer_dlist(nixl_name, xfer_dlist)
            self._remote_dlists[peer_id] = remote_dlist
        except BaseException:
            try:
                self.remove_remote_peer(peer_id)
            except BaseException as cleanup_exc:
                self._log_cleanup_failure("remote-peer rollback", cleanup_exc)
            raise

    def remove_remote_peer(self, peer_id: str) -> None:
        self._repair_peer_index(peer_id)
        if self._inflight_by_peer.get(peer_id) or self._terminal_by_peer.get(peer_id):
            self._retiring_peers.add(peer_id)
            return
        self._close_remote_peer(peer_id)

    def reap_retired_peers(self) -> None:
        """Advance native work before releasing its peer descriptors."""

        for peer_id in tuple(self._retiring_peers):
            self._repair_peer_index(peer_id)
            # Also covers a transport-owned submission whose return was
            # interrupted before ServerRole could record its transfer ID.
            self.poll(peer_id)
            if self._inflight_by_peer.get(peer_id) or self._terminal_by_peer.get(
                peer_id
            ):
                continue
            self._close_remote_peer(peer_id)

    def _close_remote_peer(self, peer_id: str) -> None:
        nixl_name = self._peer_nixl_names.get(peer_id)
        dlist = self._remote_dlists.get(peer_id)
        if self._agent is None:
            self._remote_dlists.pop(peer_id, None)
            self._peer_nixl_names.pop(peer_id, None)
            self._retiring_peers.discard(peer_id)
            return
        if nixl_name is None and dlist is None:
            self._retiring_peers.discard(peer_id)
            return
        self._retiring_peers.add(peer_id)
        if dlist is not None:
            self._agent.release_dlist_handle(dlist)
            self._remote_dlists.pop(peer_id, None)
        if nixl_name:
            # NixlWrapper treats a repeated native invalidation as cleanup
            # success, making retry after an interrupted dict-pop idempotent.
            self._agent.remove_remote_agent(nixl_name)
            self._peer_nixl_names.pop(peer_id, None)
        self._retiring_peers.discard(peer_id)

    def peer_retirement_complete(self, peer_id: str) -> bool:
        self._repair_peer_index(peer_id)
        return (
            peer_id not in self._peer_nixl_names
            and peer_id not in self._remote_dlists
            and not self._inflight_by_peer.get(peer_id)
            and not self._terminal_by_peer.get(peer_id)
            and peer_id not in self._retiring_peers
        )

    # ------------------------------------------------------------------
    # Transfer submission and polling
    # ------------------------------------------------------------------

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
        native_agent = getattr(self._agent, "agent", None)
        if not callable(getattr(native_agent, "makeXferReqOwned", None)):
            raise RuntimeError(
                "owned transfer-return recovery requires NIXL makeXferReqOwned"
            )
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
        """Submit a WRITE transfer to *peer_id*.

        Returns a transfer ID, or None if the peer is not registered.
        The ID is returned via poll() when the transfer completes or fails.
        """
        remote_dlist = self._remote_dlists.get(peer_id)
        if remote_dlist is None or peer_id in self._retiring_peers:
            logger.warning(
                "NixlTransport %s: write_blocks NO REMOTE DLIST for peer=%s "
                "(known peers=%s)",
                self._agent_name,
                peer_id,
                list(self._remote_dlists.keys()),
            )
            return None
        logger.debug(
            "NixlTransport %s: write_blocks NIXL.transfer peer=%s blocks=%d",
            self._agent_name,
            peer_id,
            len(local_idxs),
        )
        handle = self._agent.make_prepped_xfer(
            "WRITE",
            self._local_dlist,
            np.asarray(local_idxs, dtype=np.int32),
            remote_dlist,
            np.asarray(remote_idxs, dtype=np.int32),
        )
        transfer_id = next(self._next_id)
        # Once post enters native code its outcome is ambiguous. Publish the
        # strong handle owner first so peer retirement can always find it.
        self._record_inflight(
            transfer_id,
            _Inflight(peer_id, handle, recovery_token),
        )
        self._agent.transfer(handle)
        return transfer_id

    def recover_transfer_id(
        self,
        peer_id: str,
        recovery_token: object,
    ) -> int | None:
        if recovery_token is None:
            raise ValueError("recovery_token must be a unique non-None object")
        matches = [
            (transfer_id, entry)
            for transfer_id, entry in self._inflight.items()
            if entry.recovery_token is recovery_token
        ]
        if len(matches) > 1:
            raise RuntimeError("multiple native transfers share one recovery token")
        if not matches:
            # transfer() is entered only after primary-map publication. An
            # unrecorded make_prepped_xfer result is still unposted and owned
            # by the binding's retry-safe RAII wrapper.
            return None
        transfer_id, entry = matches[0]
        if entry.peer_id != peer_id:
            raise RuntimeError("recovery token belongs to a different peer")
        self._repair_peer_index(peer_id)
        return transfer_id

    def poll(self, peer_id: str | None = None) -> PollResult:
        """Poll inflight transfers.

        When *peer_id* is given, only transfers submitted for that peer_id are
        checked and drained — the transport is shared across peer sessions, so
        an unscoped poll by one session would consume and discard siblings'
        completions. *peer_id* None polls every peer (shutdown drain only).

        Returns PollResult(done=..., failed=...) with transfer IDs.
        Completed handles are released automatically.
        """
        if self._terminal_by_peer:
            retry_result = self._retry_terminal_poll(peer_id)
            if retry_result is not None:
                return retry_result
        if not self._inflight:
            return _EMPTY_POLL_RESULT

        done_ids: list[int] | None = None
        failed_ids: list[int] | None = None

        candidate_ids: Iterable[int] = (
            self._inflight
            if peer_id is None
            else self._inflight_by_peer.get(peer_id, ())
        )
        for transfer_id in candidate_ids:
            entry = self._inflight[transfer_id]
            try:
                state = self._agent.check_xfer_state(entry.handle)
            except Exception as exc:
                logger.warning(
                    "NixlTransport %s: check_xfer_state failed for transfer_id=%d: %s",
                    self._agent_name,
                    transfer_id,
                    exc,
                )
                continue
            if state == "DONE":
                if done_ids is None:
                    done_ids = []
                done_ids.append(transfer_id)
            elif state not in ("PROC", "PEND"):
                if failed_ids is None:
                    failed_ids = []
                failed_ids.append(transfer_id)

        if done_ids is None and failed_ids is None:
            return _EMPTY_POLL_RESULT

        for outcome, ids in (("done", done_ids), ("failed", failed_ids)):
            for tid in ids or ():
                entry = self._inflight[tid]
                self._terminal_by_peer.setdefault(entry.peer_id, {})[tid] = outcome
                try:
                    self._agent.release_xfer_handle(entry.handle)
                except Exception as exc:
                    logger.warning(
                        "NixlTransport %s: release_xfer_handle failed for "
                        "transfer_id=%d: %s",
                        self._agent_name,
                        tid,
                        exc,
                    )
                    continue
                self._pop_inflight(tid)

        return self._peek_terminal(peer_id)

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
                        "cannot acknowledge a terminal transfer before native cleanup"
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
        """Cancel inflight transfers by their IDs.

        See ``DataTransport.cancel`` for the contract. In "wait" mode,
        transfers whose ``release_xfer_handle`` raises (NIXL could not
        complete the abort because the backend is still draining) stay
        in ``self._inflight`` so a later ``poll()`` will observe them.
        """
        still_inflight: list[int] = []
        for tid in transfer_ids:
            entry = self._inflight.get(tid)
            if entry is None:
                self._discard_terminal_id(tid)
                continue
            try:
                self._agent.release_xfer_handle(entry.handle)
            except Exception as exc:
                logger.debug(
                    "NixlTransport %s: cancel pending for transfer_id=%d: %s",
                    self._agent_name,
                    tid,
                    exc,
                )
                still_inflight.append(tid)
                continue
            self._pop_inflight(tid)
            self._clear_terminal(entry.peer_id, tid)
        if mode == "immediate":
            if still_inflight:
                logger.warning(
                    "NixlTransport %s: cancellation left %d handle(s) active; "
                    "retaining their resources",
                    self._agent_name,
                    len(still_inflight),
                )
            return []
        return still_inflight

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._agent is None:
            return
        self._retry_terminal_poll(None)
        self.cancel(tuple(self._inflight), mode="wait")
        if self._inflight:
            raise RuntimeError(
                f"cannot close NIXL agent with {len(self._inflight)} active transfer(s)"
            )
        if self._inflight_by_peer or self._terminal_by_peer:
            raise RuntimeError("NIXL inflight bookkeeping is inconsistent")
        for peer_id in set(self._remote_dlists) | set(self._peer_nixl_names):
            self._close_remote_peer(peer_id)
        if self._local_dlist is not None:
            self._agent.release_dlist_handle(self._local_dlist)
            self._local_dlist = None
        if self._reg is not None:
            self._agent.deregister_memory(self._reg)
            self._reg = None
        self._agent = None

    def _record_inflight(self, transfer_id: int, entry: _Inflight) -> None:
        self._inflight[transfer_id] = entry
        self._inflight_by_peer.setdefault(entry.peer_id, {})[transfer_id] = None

    def _repair_peer_index(self, peer_id: str) -> None:
        """Recover cold-path visibility after interrupted index publication.

        ``_inflight`` is the strong owner and is deliberately published before
        the derived peer index. Peer teardown calls this helper before trusting
        the index, so an asynchronous exception between those two stores can
        retain work safely without adding a check to live submit or poll paths.
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
        # Repair the derived index first; the primary owner is the commit.
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
        """Retry native release without losing an observed exact outcome."""

        if peer_id is None:
            peer_ids = tuple(self._terminal_by_peer)
        elif self._terminal_by_peer.get(peer_id):
            peer_ids = (peer_id,)
        else:
            return None
        if not peer_ids:
            return None

        for owner_id in peer_ids:
            for tid in tuple(self._terminal_by_peer.get(owner_id, {})):
                entry = self._inflight.get(tid)
                if entry is not None:
                    try:
                        self._agent.release_xfer_handle(entry.handle)
                    except Exception as exc:
                        logger.warning(
                            "NixlTransport %s: terminal release retry failed for "
                            "transfer_id=%d: %s",
                            self._agent_name,
                            tid,
                            exc,
                        )
                        continue
                    self._pop_inflight(tid)
                else:
                    self._discard_peer_index(owner_id, tid)
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rollback_initialization(self) -> None:
        """Release every acquired constructor resource without masking failure."""
        agent = self._agent
        local_dlist = self._local_dlist
        reg = self._reg
        self._local_dlist = None
        self._reg = None
        self._agent = None
        if agent is None:
            return
        if local_dlist is not None:
            try:
                agent.release_dlist_handle(local_dlist)
            except BaseException as exc:
                self._log_cleanup_failure("local descriptor rollback", exc)
        if reg is not None:
            try:
                agent.deregister_memory(reg)
            except BaseException as exc:
                self._log_cleanup_failure("memory-registration rollback", exc)

    def _log_cleanup_failure(self, action: str, exc: BaseException) -> None:
        with suppress(BaseException):
            logger.warning(
                "NixlTransport %s: %s failed: %s", self._agent_name, action, exc
            )
