# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused top-k finalization and one-sided reduce-scatter for latent MoE."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.models.kimi_k3.nvidia.ops.cute_dsl.latent_moe_tail.primitives import (
    NUM_LAMPORT_BUFFERS,
    PACKED_BYTES,
    VEC_BF16,
    bf16x8_to_packed_u32x4,
    fragment_is_dirty,
    load_global_u32x4,
    load_volatile_u32,
    packed_u32x4_to_bf16x8,
    red_async_release_gpu_add_u32,
    sanitize_negative_zero,
    store_global_u32x4,
    store_lamport_sentinel_128,
    to_cute,
    to_cute_dynamic_m,
)

_DEFAULT_TOKEN_CTAS = 16
_SUPPORTED_WORLD_SIZES = (2, 4, 8, 16)


class FusedTopKReduceScatterKernel:
    """Finalize local expert rows and reduce-scatter token slices."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        hidden_dim: int,
        top_k: int,
        max_local_tokens: int,
        token_ctas: int,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.max_local_tokens = max_local_tokens
        self.token_ctas = min(token_ctas, max_local_tokens)
        self.threads = hidden_dim // VEC_BF16

    @cute.jit
    def __call__(
        self,
        gemm2_permuted: cute.Tensor,
        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
        output: cute.Tensor,
        workspace: cute.Tensor,
        flags: cute.Tensor,
        peer_ptrs: cute.Tensor,
        global_tokens: Int32,
        local_tokens: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            gemm2_permuted,
            expert_weights,
            expanded_idx_to_permuted_idx,
            output,
            workspace,
            flags,
            peer_ptrs,
            global_tokens,
            local_tokens,
        ).launch(
            grid=(self.token_ctas, self.world_size, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        gemm2_permuted: cute.Tensor,
        expert_weights: cute.Tensor,
        expanded_idx_to_permuted_idx: cute.Tensor,
        output: cute.Tensor,
        workspace: cute.Tensor,
        flags: cute.Tensor,
        peer_ptrs: cute.Tensor,
        global_tokens: Int32,
        local_tokens: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token_cta, destination, _ = cute.arch.block_idx()

        current_index = cute.arch.load((flags.iterator + 0).llvm_ptr, Uint32)
        dirty_index = cute.arch.load((flags.iterator + 1).llvm_ptr, Uint32)
        bytes_per_buffer = cute.arch.load((flags.iterator + 2).llvm_ptr, Uint32)
        dirty_num_stages = cute.arch.load((flags.iterator + 3).llvm_ptr, Uint32)
        bytes_to_clear = cute.arch.load((flags.iterator + 4).llvm_ptr, Uint32)
        current_elements = Int64(current_index) * (Int64(bytes_per_buffer) // Int64(2))
        dirty_elements = Int64(dirty_index) * (Int64(bytes_per_buffer) // Int64(2))

        global_tid = (
            Int64(token_cta) * self.world_size + Int64(destination)
        ) * self.threads + Int64(tidx)
        total_threads = Int64(self.token_ctas * self.world_size * self.threads)
        clear_fragments = (Int64(bytes_to_clear) + PACKED_BYTES - 1) // PACKED_BYTES
        clear_idx = global_tid
        if dirty_num_stages > Uint32(0):
            while clear_idx < clear_fragments:
                clear_ptr = cute.make_ptr(
                    BFloat16,
                    (
                        workspace.iterator + dirty_elements + clear_idx * VEC_BF16
                    ).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                store_lamport_sentinel_128(clear_ptr)
                clear_idx = clear_idx + total_threads

        token = token_cta
        while token < local_tokens:
            global_token = destination * local_tokens + token
            local_accum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            for element in cutlass.range_constexpr(VEC_BF16):
                local_accum[element] = Float32(0.0)

            for slot in cutlass.range_constexpr(self.top_k):
                permuted_idx = expanded_idx_to_permuted_idx[global_token, slot]
                if permuted_idx >= Int32(0):
                    permuted_element = (
                        Int64(permuted_idx) * self.hidden_dim + Int64(tidx) * VEC_BF16
                    )
                    permuted_ptr = cute.make_ptr(
                        BFloat16,
                        (gemm2_permuted.iterator + permuted_element).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    )
                    values = packed_u32x4_to_bf16x8(
                        load_global_u32x4(permuted_ptr, volatile=False)
                    ).to(Float32)
                    weight = expert_weights[global_token, slot].to(Float32)
                    for element in cutlass.range_constexpr(VEC_BF16):
                        local_accum[element] = (
                            local_accum[element] + values[element] * weight
                        )

            local_packed = sanitize_negative_zero(
                bf16x8_to_packed_u32x4(local_accum.load().to(BFloat16))
            )
            peer_base = cute.arch.load(
                (peer_ptrs.iterator + destination).llvm_ptr, Int64
            )
            destination_element = current_elements + (
                (Int64(token) * self.world_size + self.rank) * self.hidden_dim
                + Int64(tidx) * VEC_BF16
            )
            store_global_u32x4(
                peer_base + destination_element * 2,
                local_packed,
                volatile=False,
            )

            if destination == self.rank:
                rank_words = cute.make_rmem_tensor(
                    cute.make_layout((self.world_size, 4), stride=(4, 1)),
                    Uint32,
                )
                valid = False
                while not valid:
                    valid = True
                    for source_rank in cutlass.range_constexpr(self.world_size):
                        remote_element = current_elements + (
                            (Int64(token) * self.world_size + source_rank)
                            * self.hidden_dim
                            + Int64(tidx) * VEC_BF16
                        )
                        remote_ptr = cute.make_ptr(
                            BFloat16,
                            (workspace.iterator + remote_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                        remote = load_global_u32x4(remote_ptr, volatile=True)
                        for word in cutlass.range_constexpr(4):
                            rank_words[source_rank, word] = remote[word]
                        valid = valid & (not fragment_is_dirty(remote))

                reduced = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
                for element in cutlass.range_constexpr(VEC_BF16):
                    reduced[element] = Float32(0.0)
                for source_rank in cutlass.range_constexpr(self.world_size):
                    values = packed_u32x4_to_bf16x8(
                        rank_words[source_rank, None].load()
                    ).to(Float32)
                    for element in cutlass.range_constexpr(VEC_BF16):
                        reduced[element] = reduced[element] + values[element]

                output_element = Int64(token) * self.hidden_dim + Int64(tidx) * VEC_BF16
                store_global_u32x4(
                    Int64((output.iterator + output_element).toint()),
                    bf16x8_to_packed_u32x4(reduced.load().to(BFloat16)),
                    volatile=False,
                )

            if tidx == 0:
                red_async_release_gpu_add_u32(flags.iterator + 8, Uint32(1))
            token = token + self.token_ctas

        if token_cta == 0 and destination == 0 and tidx == 0:
            access_counter = flags.iterator + 8
            arrived = load_volatile_u32(access_counter)
            target = Uint32(global_tokens)
            while arrived < target:
                arrived = load_volatile_u32(access_counter)
            next_index = (current_index + Uint32(1)) % Uint32(NUM_LAMPORT_BUFFERS)
            actual_bytes = Uint32(local_tokens) * Uint32(
                self.world_size * self.hidden_dim * 2
            )
            cute.arch.store((flags.iterator + 0).llvm_ptr, next_index)
            cute.arch.store((flags.iterator + 1).llvm_ptr, current_index)
            cute.arch.store((flags.iterator + 2).llvm_ptr, bytes_per_buffer)
            cute.arch.store((flags.iterator + 3).llvm_ptr, Uint32(1))
            cute.arch.store((flags.iterator + 4).llvm_ptr, actual_bytes)
            for index in cutlass.range_constexpr(5, 8):
                cute.arch.store((flags.iterator + index).llvm_ptr, Uint32(0))
            cute.arch.store(access_counter.llvm_ptr, Uint32(0))


_COMPILED: dict[tuple[object, ...], Any] = {}


def _compile_key(
    *,
    rank: int,
    world_size: int,
    hidden_dim: int,
    top_k: int,
    max_local_tokens: int,
    token_ctas: int,
) -> tuple[object, ...]:
    return (
        torch.accelerator.current_device_index(),
        rank,
        world_size,
        hidden_dim,
        top_k,
        max_local_tokens,
        token_ctas,
    )


def _runtime_args(
    gemm2_permuted: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    flags: torch.Tensor,
    peer_ptrs: torch.Tensor,
) -> tuple[object, ...]:
    return (
        to_cute_dynamic_m(gemm2_permuted, mode=0, assumed_align=16),
        to_cute_dynamic_m(expert_weights, mode=0, assumed_align=16),
        to_cute_dynamic_m(expanded_idx_to_permuted_idx, mode=0, assumed_align=16),
        to_cute_dynamic_m(output, mode=0, assumed_align=16),
        to_cute(workspace, 16),
        to_cute(flags, 16),
        to_cute(peer_ptrs, 16),
        Int32(expert_weights.shape[0]),
        Int32(output.shape[0]),
        cuda.CUstream(torch.cuda.current_stream(output.device).cuda_stream),
    )


def _compile(
    *,
    rank: int,
    world_size: int,
    hidden_dim: int,
    top_k: int,
    max_local_tokens: int,
    token_ctas: int,
    output: torch.Tensor,
    workspace: torch.Tensor,
    flags: torch.Tensor,
    peer_ptrs: torch.Tensor,
) -> None:
    key = _compile_key(
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        top_k=top_k,
        max_local_tokens=max_local_tokens,
        token_ctas=token_ctas,
    )
    if key in _COMPILED:
        return
    device = output.device
    max_global_tokens = max_local_tokens * world_size
    gemm2 = torch.empty(
        (max_global_tokens * top_k, hidden_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    weights = torch.empty(
        (max_global_tokens, top_k), dtype=torch.bfloat16, device=device
    )
    expanded_idx = torch.empty(
        (max_global_tokens, top_k), dtype=torch.int32, device=device
    )
    kernel = FusedTopKReduceScatterKernel(
        rank=rank,
        world_size=world_size,
        hidden_dim=hidden_dim,
        top_k=top_k,
        max_local_tokens=max_local_tokens,
        token_ctas=token_ctas,
    )
    _COMPILED[key] = cute.compile(
        kernel,
        *_runtime_args(
            gemm2,
            weights,
            expanded_idx,
            output,
            workspace,
            flags,
            peer_ptrs,
        ),
    )


@dataclass(frozen=True)
class DeferredTopKReduceScatterContract:
    group_id: int
    rank: int
    world_size: int
    device: torch.device
    hidden_dim: int
    top_k: int
    max_local_tokens: int
    token_ctas: int


class DeferredTopKReduceScatter:
    """Cached symmetric workspace and launcher for the fused EP tail."""

    _instances: ClassVar[
        dict[DeferredTopKReduceScatterContract, DeferredTopKReduceScatter]
    ] = {}

    @classmethod
    def initialize(
        cls,
        *,
        group: dist.ProcessGroup,
        hidden_dim: int,
        top_k: int,
        max_local_tokens: int,
        token_ctas: int = _DEFAULT_TOKEN_CTAS,
    ) -> DeferredTopKReduceScatter:
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        device = torch.device("cuda", torch.accelerator.current_device_index())
        contract = DeferredTopKReduceScatterContract(
            group_id=id(group),
            rank=rank,
            world_size=world_size,
            device=device,
            hidden_dim=hidden_dim,
            top_k=top_k,
            max_local_tokens=max_local_tokens,
            token_ctas=min(token_ctas, max_local_tokens),
        )
        op = cls._instances.get(contract)
        if op is None:
            op = cls(contract, group)
            cls._instances[contract] = op
        return op

    def __init__(
        self,
        contract: DeferredTopKReduceScatterContract,
        group: dist.ProcessGroup,
    ) -> None:
        if contract.world_size not in _SUPPORTED_WORLD_SIZES:
            raise ValueError(
                "Deferred top-k reduce-scatter supports world sizes "
                f"{_SUPPORTED_WORLD_SIZES}, got {contract.world_size}."
            )
        if contract.device.type != "cuda":
            raise ValueError("Deferred top-k reduce-scatter requires CUDA.")
        if torch.cuda.get_device_capability(contract.device)[0] != 10:
            raise ValueError("Deferred top-k reduce-scatter requires SM100.")
        if contract.hidden_dim % VEC_BF16:
            raise ValueError("hidden_dim must be divisible by eight.")
        threads = contract.hidden_dim // VEC_BF16
        if not 32 <= threads <= 1024:
            raise ValueError("hidden_dim requires between 32 and 1024 threads.")
        if contract.top_k <= 0 or contract.max_local_tokens <= 0:
            raise ValueError("top_k and max_local_tokens must be positive.")

        self.contract = contract
        max_local_tokens = contract.max_local_tokens
        world_size = contract.world_size
        hidden_dim = contract.hidden_dim
        device = contract.device

        self._workspace = symm_mem.empty(
            (
                NUM_LAMPORT_BUFFERS,
                max_local_tokens,
                world_size,
                hidden_dim,
            ),
            dtype=torch.bfloat16,
            device=device,
        )
        self._symm_mem = symm_mem.rendezvous(self._workspace, group)
        self._workspace.view(torch.int32).fill_(-0x80000000)
        bytes_per_buffer = max_local_tokens * world_size * hidden_dim * 2
        self._flags = torch.tensor(
            [0, 2, bytes_per_buffer, 0, 0, 0, 0, 0, 0],
            dtype=torch.uint32,
            device=device,
        )
        peer_ptrs = [
            self._symm_mem.get_buffer(
                peer,
                self._workspace.shape,
                self._workspace.dtype,
            ).data_ptr()
            for peer in range(world_size)
        ]
        if any(pointer == 0 for pointer in peer_ptrs):
            raise RuntimeError("Symmetric peer mapping is unavailable.")
        self._peer_ptrs = torch.tensor(peer_ptrs, dtype=torch.int64, device=device)
        compile_output = torch.empty(
            (max_local_tokens, hidden_dim), dtype=torch.bfloat16, device=device
        )

        torch.accelerator.synchronize(device)
        dist.barrier(group=group, device_ids=[device.index])
        for owner in range(world_size):
            if contract.rank == owner:
                _compile(
                    rank=contract.rank,
                    world_size=world_size,
                    hidden_dim=hidden_dim,
                    top_k=contract.top_k,
                    max_local_tokens=max_local_tokens,
                    token_ctas=contract.token_ctas,
                    output=compile_output,
                    workspace=self._workspace,
                    flags=self._flags,
                    peer_ptrs=self._peer_ptrs,
                )
            dist.barrier(group=group, device_ids=[device.index])

    def __call__(
        self,
        source: UnfinalizedMoEOutput,
        output: torch.Tensor,
    ) -> None:
        contract = self.contract
        global_tokens = source.expert_weights.shape[0]
        local_tokens = output.shape[0]
        if global_tokens != local_tokens * contract.world_size:
            raise ValueError(
                "Deferred top-k reduce-scatter requires a uniform token split."
            )
        if not 1 <= local_tokens <= contract.max_local_tokens:
            raise ValueError(
                f"local token count must be in [1, {contract.max_local_tokens}]."
            )

        expected = (
            (
                source.gemm2_permuted,
                torch.bfloat16,
                contract.hidden_dim,
                "gemm2_permuted",
            ),
            (source.expert_weights, torch.bfloat16, contract.top_k, "expert_weights"),
            (
                source.expanded_idx_to_permuted_idx,
                torch.int32,
                contract.top_k,
                "expanded_idx_to_permuted_idx",
            ),
            (output, torch.bfloat16, contract.hidden_dim, "output"),
        )
        for tensor, dtype, width, name in expected:
            if (
                tensor.ndim != 2
                or tensor.shape[1] != width
                or tensor.dtype != dtype
                or tensor.device != contract.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be contiguous CUDA {dtype} with width {width}."
                )

        key = _compile_key(
            rank=contract.rank,
            world_size=contract.world_size,
            hidden_dim=contract.hidden_dim,
            top_k=contract.top_k,
            max_local_tokens=contract.max_local_tokens,
            token_ctas=contract.token_ctas,
        )
        with torch.accelerator.device_index(contract.device.index):
            _COMPILED[key](
                *_runtime_args(
                    source.gemm2_permuted,
                    source.expert_weights,
                    source.expanded_idx_to_permuted_idx,
                    output,
                    self._workspace,
                    self._flags,
                    self._peer_ptrs,
                )
            )
