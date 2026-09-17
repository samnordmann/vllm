# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
P2P KV cache sharing protocol constants and documentation.

Protocol Overview
=================

Two session types communicate over a bidirectional message channel:
- P2PClientSession (requests blocks from a server)
- P2PServerSession (serves blocks to a client)

Connection Lifecycle
--------------------

1. Each side opens a connection and sends ConnectMsg with its identity,
   random session epoch, target epoch (initially the all-zero discovery
   sentinel), wire version, RDMA metadata, memory layout, and fingerprint.
2. Each side validates the peer's ConnectMsg, replies with ConnectAckMsg, and
   sends a Connect targeting the candidate's source epoch.
3. A side registers peer metadata and becomes ready only after a validated
   Connect targeting its current local epoch and an Ack targeting that same
   local epoch name one remote epoch. It then flushes queued messages.
4. Until ready, a side periodically retries its immutable discovery Connect.
   A ready side challenges a different epoch but withholds its Ack until the
   old data-plane owner is retired; two-frame successor proof makes the old
   control session dead so manager-owned quiescence/removal runs first.
5. Either side may send DisconnectMsg to gracefully close.

Block Transfer Flow (happy path)
---------------------------------

1. Client sends FetchMsg with a kv_request_id and lists of
   block keys + remote indexes where it wants the data written.
   A request may run several lookup→fetch rounds; symmetric-P2P
   messages carry a session-global ROUND_SEQ operation token so each
   round's supply, demand, and completion stay isolated even if a
   kv_request_id is later reused. The terminal empty FetchMsg is the
   server-side "request finished" signal for the id: parked
   LookupMsg batches are popped and ``cb.finish_request`` fires on
   each.
2. Server matches requested blocks against locally stored blocks:
   - Blocks already available are transferred immediately via RDMA.
   - Blocks not yet available are recorded as "demanded" and
     transferred when the server later stores them.
3. When all blocks for a kv_request_id are transferred, the server
   sends TransferDoneMsg (success=True) to the client.
4. Client reports the load job as complete.

Abort Flow (timeout path)
--------------------------

1. If the client times out waiting for TransferDoneMsg, it sends
   AbortFetchMsg to cancel the request.
2. Server cancels inflight transfers for that kv_request_id and
   replies with AbortAckMsg.
3. Client receives AbortAckMsg and reports the load job as failed.
4. If AbortAckMsg itself times out, the client retains the destination and
   retries the same abort identity; a deadline alone is not quiescence proof.

Message Format
--------------

All messages are dicts serialized with msgpack. Every message has a TYPE_KEY
key identifying its type. Handshake messages carry an exact supported major
and minor version. Every state-mutating message carries the exact source and
target session epochs negotiated by the two-sided handshake.

Security
--------

Session epochs provide reconnect freshness, not authentication. The ZMQ
control plane is trusted and unauthenticated. Sessions validate incoming
messages before role/data mutation; malformed current-channel messages fail
the session, while well-formed frames from a retired epoch are dropped.

The config_fingerprint field in ConnectMsg ensures peers have
compatible model configurations (model, dtype, block sizes). Mismatches
are rejected during the handshake.
"""

TYPE_KEY = "type"

WIRE_MAJOR_KEY = "wire_major"
WIRE_MINOR_KEY = "wire_minor"
SOURCE_EPOCH_KEY = "source_epoch"
TARGET_EPOCH_KEY = "target_epoch"

WIRE_PROTOCOL_MAJOR = 1
WIRE_PROTOCOL_MINOR = 0
SESSION_EPOCH_NBYTES = 16
UNSPECIFIED_EPOCH = bytes(SESSION_EPOCH_NBYTES)

MAX_WIRE_LIST_ITEMS = 65_536
MAX_WIRE_KEY_BYTES = 4_096
MAX_WIRE_STRING_CHARS = 4_096
MAX_AGENT_METADATA_BYTES = 64 * 1024 * 1024
MAX_BLOCK_INDEX = (1 << 31) - 1
_MAX_UINT64 = (1 << 64) - 1

# msgpack's positive-integer wire domain is uint64. Operation identities never
# wrap: once this value has been issued, the session fails closed and must be
# replaced before another lookup/fetch generation can begin.
MAX_ROUND_SEQ = (1 << 64) - 1


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _require(msg: dict, key: str, typ: type, *, name: str = "") -> None:
    """Require an exact wire type (not a subclass such as bool for int)."""
    val = msg.get(key)
    if type(val) is not typ:
        label = name or key
        raise ValueError(f"{label}: expected {typ.__name__}, got {type(val).__name__}")


def _require_pos_int(
    msg: dict,
    key: str,
    *,
    maximum: int = _MAX_UINT64,
    name: str = "",
) -> None:
    """Require an exact positive integer within the declared wire bound."""
    val = msg.get(key)
    if type(val) is not int or not 0 < val <= maximum:
        label = name or key
        raise ValueError(f"{label}: expected positive int <= {maximum}, got {val!r}")


def _require_non_neg_int(
    msg: dict,
    key: str,
    *,
    maximum: int = _MAX_UINT64,
    name: str = "",
) -> None:
    """Require an exact non-negative integer within the wire bound."""
    val = msg.get(key)
    if type(val) is not int or not 0 <= val <= maximum:
        label = name or key
        raise ValueError(
            f"{label}: expected non-negative int <= {maximum}, got {val!r}"
        )


def _require_round_seq(msg: dict, key: str) -> None:
    """Require an exact uint64 session-global operation identity."""
    val = msg.get(key)
    if type(val) is not int or not 0 <= val <= MAX_ROUND_SEQ:
        raise ValueError(f"{key}: expected uint64 operation token, got {val!r}")


def _require_list(msg: dict, key: str, *, name: str = "") -> None:
    """Require an exact, bounded list."""
    val = msg.get(key)
    if type(val) is not list:
        label = name or key
        raise ValueError(f"{label}: expected list, got {type(val).__name__}")
    if len(val) > MAX_WIRE_LIST_ITEMS:
        label = name or key
        raise ValueError(f"{label}: length {len(val)} exceeds {MAX_WIRE_LIST_ITEMS}")


def _require_bounded_str(msg: dict, key: str) -> None:
    _require(msg, key, str)
    value = msg[key]
    if not value or len(value) > MAX_WIRE_STRING_CHARS:
        raise ValueError(
            f"{key}: expected 1..{MAX_WIRE_STRING_CHARS} characters, "
            f"got length {len(value)}"
        )


def _require_bounded_str_allow_empty(msg: dict, key: str) -> None:
    _require(msg, key, str)
    value = msg[key]
    if len(value) > MAX_WIRE_STRING_CHARS:
        raise ValueError(f"{key}: length {len(value)} exceeds {MAX_WIRE_STRING_CHARS}")


def _require_epoch(msg: dict, key: str) -> bytes:
    _require(msg, key, bytes)
    epoch = msg[key]
    if len(epoch) != SESSION_EPOCH_NBYTES:
        raise ValueError(
            f"{key}: expected {SESSION_EPOCH_NBYTES} bytes, got {len(epoch)}"
        )
    return epoch


def _require_nonzero_epoch(msg: dict, key: str) -> bytes:
    epoch = _require_epoch(msg, key)
    if epoch == UNSPECIFIED_EPOCH:
        raise ValueError(f"{key}: all-zero epoch is reserved for discovery targets")
    return epoch


def validate_wire_version(msg: dict) -> None:
    """Require this protocol's exact major/minor pair, excluding booleans."""
    _require_non_neg_int(msg, WIRE_MAJOR_KEY)
    _require_non_neg_int(msg, WIRE_MINOR_KEY)
    version = (msg[WIRE_MAJOR_KEY], msg[WIRE_MINOR_KEY])
    supported = (WIRE_PROTOCOL_MAJOR, WIRE_PROTOCOL_MINOR)
    if version != supported:
        raise ValueError(
            f"unsupported wire version {version[0]}.{version[1]}; "
            f"expected {supported[0]}.{supported[1]}"
        )


def validate_channel(msg: dict) -> None:
    """Validate a state frame's source/target session epochs."""
    _require_nonzero_epoch(msg, SOURCE_EPOCH_KEY)
    _require_nonzero_epoch(msg, TARGET_EPOCH_KEY)


def _validate_keys(keys: list) -> None:
    for key in keys:
        if type(key) is not bytes or len(key) > MAX_WIRE_KEY_BYTES:
            raise ValueError(
                "keys: expected bytes entries no larger than "
                f"{MAX_WIRE_KEY_BYTES}; got {type(key).__name__} "
                f"of length {len(key) if isinstance(key, bytes) else 'n/a'}"
            )


# ---------------------------------------------------------------------------
# Message classes
# ---------------------------------------------------------------------------


class ConnectMsg:
    """Client → Server: initial handshake request.

    Fields:
        PEER_ID: Local peer identity string.
        AGENT_METADATA: RDMA agent metadata (opaque bytes).
        BASE_ADDR: Base memory address of the KV block region.
        NUM_BLOCKS: Number of blocks in the KV block region.
        BLOCK_LEN: Size in bytes of each block (must match between peers).
        CONFIG_FINGERPRINT: SHA-256 prefix of the model configuration.
            Peers with different fingerprints are incompatible.
        HASH_SEED: The peer's effective prefix-cache hash seed (PYTHONHASHSEED
            if set, otherwise the built-in default). Block hashes chain from a
            seed derived from it, so peers with different values compute
            different hashes for identical content and must not exchange blocks.
        SOURCE_EPOCH: Random epoch owned by this session incarnation.
        TARGET_EPOCH: The peer epoch this Connect answers, or the all-zero
            discovery sentinel until one is known.
    """

    TYPE = "connect"
    PEER_ID = "peer_id"
    AGENT_METADATA = "agent_metadata"
    BASE_ADDR = "base_addr"
    NUM_BLOCKS = "num_blocks"
    BLOCK_LEN = "block_len"
    CONFIG_FINGERPRINT = "config_fingerprint"
    HASH_SEED = "hash_seed"
    WIRE_MAJOR = WIRE_MAJOR_KEY
    WIRE_MINOR = WIRE_MINOR_KEY
    SOURCE_EPOCH = SOURCE_EPOCH_KEY
    TARGET_EPOCH = TARGET_EPOCH_KEY

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        validate_wire_version(msg)
        _require_nonzero_epoch(msg, ConnectMsg.SOURCE_EPOCH)
        _require_epoch(msg, ConnectMsg.TARGET_EPOCH)
        _require_bounded_str(msg, ConnectMsg.PEER_ID)
        _require(msg, ConnectMsg.AGENT_METADATA, bytes)
        if len(msg[ConnectMsg.AGENT_METADATA]) > MAX_AGENT_METADATA_BYTES:
            raise ValueError(
                f"agent_metadata: length exceeds {MAX_AGENT_METADATA_BYTES} bytes"
            )
        _require_non_neg_int(msg, ConnectMsg.BASE_ADDR)
        _require_pos_int(msg, ConnectMsg.NUM_BLOCKS, maximum=MAX_BLOCK_INDEX + 1)
        _require_pos_int(msg, ConnectMsg.BLOCK_LEN)
        _require_bounded_str_allow_empty(msg, ConnectMsg.CONFIG_FINGERPRINT)
        _require_bounded_str(msg, ConnectMsg.HASH_SEED)
        base_addr = msg[ConnectMsg.BASE_ADDR]
        span = msg[ConnectMsg.NUM_BLOCKS] * msg[ConnectMsg.BLOCK_LEN]
        if span > _MAX_UINT64 - base_addr:
            raise ValueError("remote memory span exceeds uint64 address space")


class ConnectAckMsg:
    """Server → Client: handshake acknowledgement.

    Fields:
        PEER_ID: Server's peer identity string.
    """

    TYPE = "connect_ack"
    PEER_ID = "peer_id"
    WIRE_MAJOR = WIRE_MAJOR_KEY
    WIRE_MINOR = WIRE_MINOR_KEY
    SOURCE_EPOCH = SOURCE_EPOCH_KEY
    TARGET_EPOCH = TARGET_EPOCH_KEY

    @staticmethod
    def validate(msg: dict) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        validate_wire_version(msg)
        _require_nonzero_epoch(msg, ConnectAckMsg.SOURCE_EPOCH)
        _require_nonzero_epoch(msg, ConnectAckMsg.TARGET_EPOCH)
        _require_bounded_str(msg, ConnectAckMsg.PEER_ID)


class DisconnectMsg:
    """Either → Either: graceful connection close.

    No additional fields beyond TYPE_KEY.
    """

    TYPE = "disconnect"
    SOURCE_EPOCH = SOURCE_EPOCH_KEY
    TARGET_EPOCH = TARGET_EPOCH_KEY

    @staticmethod
    def validate(msg: dict, *, channel_validated: bool = False) -> None:
        if not channel_validated:
            validate_channel(msg)


class FetchMsg:
    """Client → Server: request blocks for one lookup round.

    A non-empty fetch closes only its round. The terminal empty FetchMsg
    (``KEYS`` and ``BLOCK_INDEXES`` both empty) is the "request
    finished" signal: the server pops parked LookupMsg batches, calls
    ``cb.finish_request`` on each, and drains any leftover supply.

    Fields:
        KV_REQUEST_ID: Identifies this block transfer request.
        KEYS: List of block keys (OffloadKey bytes). May be empty.
        BLOCK_INDEXES: List of remote block indexes (same length as KEYS).
        ROUND_SEQ: Session-global uint64 operation token for the lookup/fetch
            generation this message closes. It is never reused in a session.
    """

    TYPE = "fetch"
    KV_REQUEST_ID = "kv_request_id"
    KEYS = "keys"
    BLOCK_INDEXES = "block_indexes"
    ROUND_SEQ = "round_seq"
    SOURCE_EPOCH = SOURCE_EPOCH_KEY
    TARGET_EPOCH = TARGET_EPOCH_KEY

    @staticmethod
    def validate(msg: dict, *, channel_validated: bool = False) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        if not channel_validated:
            validate_channel(msg)
        _require_bounded_str(msg, FetchMsg.KV_REQUEST_ID)
        _require_round_seq(msg, FetchMsg.ROUND_SEQ)
        _require_list(msg, FetchMsg.KEYS)
        _require_list(msg, FetchMsg.BLOCK_INDEXES)
        keys = msg[FetchMsg.KEYS]
        indexes = msg[FetchMsg.BLOCK_INDEXES]
        _validate_keys(keys)
        if len(keys) != len(indexes):
            raise ValueError(
                f"keys/block_indexes length mismatch: {len(keys)} vs {len(indexes)}"
            )
        for idx in indexes:
            if type(idx) is not int or not 0 <= idx <= MAX_BLOCK_INDEX:
                raise ValueError(f"block_indexes: invalid index {idx!r}")


class LookupMsg:
    """Client → Server: probe which block keys the peer holds.

    Sent on the consumer side under symmetric P2P (do_p2p_fetch=true)
    after the consumer has aggregated per-block lookups across a
    scheduler step. The producer replies with one or more LookupRespMsg
    covering the requested keys.

    Fields:
        KV_REQUEST_ID: Identifies this lookup transaction.
        KEYS: List of block keys (OffloadKey bytes) to probe.
        ROUND_SEQ: Session-global uint64 operation token for this lookup/fetch
            generation; pinned supply is parked under it for the fetch.
    """

    TYPE = "lookup"
    KV_REQUEST_ID = "kv_request_id"
    KEYS = "keys"
    ROUND_SEQ = "round_seq"
    SOURCE_EPOCH = SOURCE_EPOCH_KEY
    TARGET_EPOCH = TARGET_EPOCH_KEY

    @staticmethod
    def validate(msg: dict, *, channel_validated: bool = False) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        if not channel_validated:
            validate_channel(msg)
        _require_bounded_str(msg, LookupMsg.KV_REQUEST_ID)
        _require_list(msg, LookupMsg.KEYS)
        _validate_keys(msg[LookupMsg.KEYS])
        _require_round_seq(msg, LookupMsg.ROUND_SEQ)


class LookupRespMsg:
    """Server → Client: per-key hit/miss answer for a prior LookupMsg.

    Carries two parallel arrays of equal length so each (key,
    hit) pair is self-describing. The producer is free to split or
    coalesce responses across multiple LookupMsgs for the same
    KV_REQUEST_ID — the consumer matches each pair back to its
    pending entry by (KV_REQUEST_ID, key).

    Fields:
        KV_REQUEST_ID: The lookup transaction this responds to.
        KEYS: List of block keys answered by this message.
        HITS: Parallel list of bools — True if the producer holds the
            corresponding block, False otherwise.
        ROUND_SEQ: Echoes the LookupMsg operation token. A missing or stale
            token is rejected/ignored rather than resolving a newer probe.
    """

    TYPE = "lookup_resp"
    KV_REQUEST_ID = "kv_request_id"
    KEYS = "keys"
    HITS = "hits"
    ROUND_SEQ = "round_seq"
    SOURCE_EPOCH = SOURCE_EPOCH_KEY
    TARGET_EPOCH = TARGET_EPOCH_KEY

    @staticmethod
    def validate(msg: dict, *, channel_validated: bool = False) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        if not channel_validated:
            validate_channel(msg)
        _require_bounded_str(msg, LookupRespMsg.KV_REQUEST_ID)
        _require_round_seq(msg, LookupRespMsg.ROUND_SEQ)
        _require_list(msg, LookupRespMsg.KEYS)
        _require_list(msg, LookupRespMsg.HITS)
        keys = msg[LookupRespMsg.KEYS]
        hits = msg[LookupRespMsg.HITS]
        _validate_keys(keys)
        if len(keys) != len(hits):
            raise ValueError(f"keys/hits length mismatch: {len(keys)} vs {len(hits)}")
        for hit in hits:
            if type(hit) is not bool:
                raise ValueError(f"hits: invalid value {hit!r}")


class TransferDoneMsg:
    """Server → Client: all blocks transferred for a request.

    Fields:
        KV_REQUEST_ID: The request that completed.
        SUCCESS: Whether the transfer completed successfully.
        ROUND_SEQ: The fetch round that completed. Several loads can be
            in flight per id (the scheduler submits loads incrementally),
            so completions are matched by round.
    """

    TYPE = "transfer_done"
    KV_REQUEST_ID = "kv_request_id"
    SUCCESS = "success"
    ROUND_SEQ = "round_seq"
    SOURCE_EPOCH = SOURCE_EPOCH_KEY
    TARGET_EPOCH = TARGET_EPOCH_KEY

    @staticmethod
    def validate(msg: dict, *, channel_validated: bool = False) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        if not channel_validated:
            validate_channel(msg)
        _require_bounded_str(msg, TransferDoneMsg.KV_REQUEST_ID)
        _require(msg, TransferDoneMsg.SUCCESS, bool)
        _require_round_seq(msg, TransferDoneMsg.ROUND_SEQ)


class AbortFetchMsg:
    """Client → Server: cancel a pending request.

    Fields:
        KV_REQUEST_ID: The request to cancel.
        ROUND_SEQ: The fetch round to cancel.
    """

    TYPE = "abort_fetch"
    KV_REQUEST_ID = "kv_request_id"
    ROUND_SEQ = "round_seq"
    SOURCE_EPOCH = SOURCE_EPOCH_KEY
    TARGET_EPOCH = TARGET_EPOCH_KEY

    @staticmethod
    def validate(msg: dict, *, channel_validated: bool = False) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        if not channel_validated:
            validate_channel(msg)
        _require_bounded_str(msg, AbortFetchMsg.KV_REQUEST_ID)
        _require_round_seq(msg, AbortFetchMsg.ROUND_SEQ)


class AbortAckMsg:
    """Server → Client: acknowledge cancellation.

    Fields:
        KV_REQUEST_ID: The request that was cancelled.
        ROUND_SEQ: The round that was cancelled; echoes AbortFetchMsg.
    """

    TYPE = "abort_ack"
    KV_REQUEST_ID = "kv_request_id"
    ROUND_SEQ = "round_seq"
    SOURCE_EPOCH = SOURCE_EPOCH_KEY
    TARGET_EPOCH = TARGET_EPOCH_KEY

    @staticmethod
    def validate(msg: dict, *, channel_validated: bool = False) -> None:
        """Raise ValueError if any field has an invalid type or value."""
        if not channel_validated:
            validate_channel(msg)
        _require_bounded_str(msg, AbortAckMsg.KV_REQUEST_ID)
        _require_round_seq(msg, AbortAckMsg.ROUND_SEQ)
