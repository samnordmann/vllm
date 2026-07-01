# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_ep_group
from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave


def _view_r128c4_scale_payload(
    payload: torch.Tensor, logical_rows: int, columns: int
) -> torch.Tensor:
    """Expose a flat R128C4 buffer with the shape expected by MoE plumbing.

    The physical row count is padded to 128. The expert kernel consumes only the
    pointer and derives its logical row count from the activation tensor.
    """
    if columns <= 0 or columns % 4 != 0:
        raise ValueError(
            f"R128C4 scale columns must be positive and divisible by 4: {columns}"
        )
    padded_rows = (logical_rows + 127) // 128 * 128
    expected_numel = padded_rows * columns
    if payload.numel() != expected_numel:
        raise ValueError(
            "R128C4 scale payload size mismatch: "
            f"got {payload.numel()}, expected {expected_numel} "
            f"for {logical_rows=} and {columns=}"
        )
    return payload.view(padded_rows, columns)


def get_local_sizes():
    dp_metadata = get_forward_context().dp_metadata
    assert dp_metadata is not None
    return dp_metadata.get_chunk_sizes_across_dp_rank()


class FlashInferNVLinkOneSidedPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    """FlashInfer implementation using the Moe AlltoAll kernel."""

    all2all_manager: All2AllManagerBase

    def __init__(
        self,
        max_num_tokens: int,
        top_k: int,
        num_experts: int,
        hidden_size: int,
        num_dispatchers: int = 1,
        dispatch_dtype_bytes_per_elem: int = 0,
        dispatch_scale_bytes_per_token: int = 0,
        dispatch_scale_r128c4: bool = False,
    ):
        super().__init__()
        self.max_num_tokens = max_num_tokens
        self.top_k = top_k
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.num_dispatchers_ = num_dispatchers
        self.scale_elems_per_token = dispatch_scale_bytes_per_token
        self.dispatch_scale_r128c4 = dispatch_scale_r128c4
        if dispatch_scale_r128c4 and dispatch_scale_bytes_per_token % 4 != 0:
            raise ValueError(
                "direct R128C4 scale dispatch requires a multiple of 4 columns"
            )

        device_communicator = get_ep_group().device_communicator
        assert device_communicator is not None
        all2all_manager = device_communicator.all2all_manager
        assert all2all_manager is not None
        self.all2all_manager = all2all_manager
        self.all2all_manager.initialize(  # type: ignore[attr-defined]
            max_num_tokens=self.max_num_tokens,
            top_k=self.top_k,
            num_experts=self.num_experts,
            hidden_size=self.hidden_size,
            dispatch_dtype_bytes_per_elem=dispatch_dtype_bytes_per_elem,
            dispatch_scale_bytes_per_token=dispatch_scale_bytes_per_token,
            dispatch_scale_r128c4=dispatch_scale_r128c4,
        )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def output_is_reduced(self) -> bool:
        return True

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1.mul_(topk_weights.to(a1.dtype))

        global_num_tokens_cpu = get_local_sizes()
        self.runtime_max_tokens_per_rank = (
            max(global_num_tokens_cpu)
            if global_num_tokens_cpu is not None
            else a1.shape[0]
        )

        if defer_input_quant:
            a1q, a1q_scale = a1, None
        else:
            a1q, a1q_scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_gscale,
                quant_config.quant_dtype,
                quant_config.per_act_token_quant,
                quant_config.block_shape,
                is_scale_swizzled=False,  # delay swizzle to after comm
                mx_alignment=quant_config.mx_alignment,
            )

        payloads = []
        payloads.append(a1q)
        if a1q_scale is not None:
            payloads.append(a1q_scale)
        topk_ids_payload_index = len(payloads)
        payloads.append(topk_ids)
        payloads.append(topk_weights)

        output_payload_layouts = None
        if a1q_scale is not None and self.dispatch_scale_r128c4:
            from flashinfer.comm.trtllm_moe_alltoall import MoeA2APayloadLayout

            output_payload_layouts = [
                MoeA2APayloadLayout.LINEAR,
                MoeA2APayloadLayout.R128C4,
                MoeA2APayloadLayout.LINEAR,
                MoeA2APayloadLayout.LINEAR,
            ]

        assert self.all2all_manager.moe_alltoall is not None  # type: ignore[attr-defined]
        recv_payloads = self.all2all_manager.moe_alltoall.dispatch(  # type: ignore[attr-defined]
            token_selected_experts=topk_ids,
            input_payloads=payloads,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
            invalid_token_expert_id=-1,  # Follow TRTLLM Pattern
            expert_id_payload_index=topk_ids_payload_index,
            output_payload_layouts=output_payload_layouts,
        )
        if a1q_scale is not None:
            a1q_recv, a1q_scale_recv, topk_ids_recv, topk_weights_recv = recv_payloads
            assert self.scale_elems_per_token > 0
            if self.dispatch_scale_r128c4:
                a1q_scale_recv = _view_r128c4_scale_payload(
                    a1q_scale_recv.view(torch.uint8),
                    self.num_dispatchers_ * self.runtime_max_tokens_per_rank,
                    self.scale_elems_per_token,
                )
            else:
                if (
                    quant_config.quant_dtype == "nvfp4"
                    and quant_config.is_scale_swizzled
                ):
                    a1q_scale_recv = a1q_scale_recv.view(-1, a1q_scale_recv.shape[-1])
                    a1q_scale_recv = a1q_scale_recv.view(torch.uint8)
                    a1q_scale_recv = nvfp4_block_scale_interleave(a1q_scale_recv)
                a1q_scale_recv = a1q_scale_recv.view(-1, self.scale_elems_per_token)
        else:
            a1q_recv, topk_ids_recv, topk_weights_recv = recv_payloads
            a1q_scale_recv = None
        a1q_recv = a1q_recv.view(-1, a1q_recv.shape[-1])
        topk_ids_recv = topk_ids_recv.view(-1, topk_ids_recv.shape[-1])
        topk_weights_recv = topk_weights_recv.view(-1, topk_weights_recv.shape[-1])

        return a1q_recv, a1q_scale_recv, None, topk_ids_recv, topk_weights_recv

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        assert self.all2all_manager.moe_alltoall is not None  # type: ignore[attr-defined]

        ep_size = self.all2all_manager.world_size
        hidden_size = fused_expert_output.shape[-1]
        fused_expert_output = fused_expert_output.view(
            ep_size, self.runtime_max_tokens_per_rank, hidden_size
        )

        self.all2all_manager.moe_alltoall.combine(  # type: ignore[attr-defined]
            payload=fused_expert_output,
            runtime_max_tokens_per_rank=self.runtime_max_tokens_per_rank,
            output=output,
        )
