# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.utils.flashinfer import nvfp4_block_scale_interleave

logger = init_logger(__name__)


def _quantize_and_setup_dispatch(
    a1: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    defer_input_quant: bool = False,
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
    # Defer input quantization to the MoE kernel.
    if defer_input_quant:
        a1q = a1
        a1q_scale = None
    else:
        input_sf = (
            quant_config.a1_gscale
            if quant_config.use_nvfp4_w4a4
            else quant_config.a1_scale
        )

        # NOTE: swizzling pads the scales to multiple of 128
        # which makes the scales tensor different shape than
        # the hidden states, breaking the A2A kernel. So, we
        # delay the swizzling until after the A2A.
        a1q, a1q_scale = moe_kernel_quantize_input(
            a1,
            input_sf,
            quant_dtype=quant_config.quant_dtype,
            per_act_token_quant=quant_config.per_act_token_quant,
            block_shape=quant_config.block_shape,
            is_scale_swizzled=False,
            mx_alignment=quant_config.mx_alignment,
        )

    # Skip gathering scales if we have static quantization
    # (the scale is a scalar, replicated on all ranks) or
    # if quantization is deferred.
    skip_gather_scales = a1q_scale is None or a1q_scale.ndim == 0
    scales = None if skip_gather_scales else [a1q_scale]

    return a1q, scales, a1q_scale


def _unwrap_scale_and_prepare_for_moe(
    scales: list[torch.Tensor] | None,
    quant_config: FusedMoEQuantConfig,
) -> torch.Tensor:
    assert scales is not None and len(scales) == 1
    a1q_scale = scales[0]
    # Apply swizzling after a2a if the MoE kernel needs it.
    if quant_config.quant_dtype == "nvfp4" and quant_config.is_scale_swizzled:
        assert a1q_scale is not None
        if a1q_scale.element_size() == 1:
            a1q_scale = a1q_scale.view(torch.uint8)
        a1q_scale = nvfp4_block_scale_interleave(a1q_scale)

    return a1q_scale


class MoEPrepareAndFinalizeNaiveDPEPModular(mk.FusedMoEPrepareAndFinalizeModular):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the topk weights and ids.
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers
        self._deferred_reduce_scatter = None
        self._defer_this_call = False
        self._moe_config = None
        # Set by FusedMoEWithLoRA.set_mapping() when LoRA is active. When
        # present, prepare() dispatches the per-token LoRA mapping alongside
        # hidden_states and writes the gathered result back to the context so
        # experts can use the per-rank-local mapping.
        self._lora_context = None

    def post_init_setup(self, fused_experts: mk.FusedMoEExperts) -> None:
        self._moe_config = fused_experts.moe_config
        if not envs.VLLM_EXPERIMENTAL_NEMOTRON_DEFERRED_EP_FINALIZE:
            return

        from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
            TrtLlmNvFp4ExpertsModular,
        )

        config = fused_experts.moe_config
        parallel = config.moe_parallel_config
        if not (
            isinstance(fused_experts, TrtLlmNvFp4ExpertsModular)
            and config.in_dtype == torch.bfloat16
            and config.hidden_dim_unpadded == 2048
            and config.experts_per_token == 22
            and config.num_experts == 512
            and parallel.tp_size == 1
            and parallel.dp_size > 1
            and parallel.ep_size == parallel.dp_size
            and parallel.pcp_size == 1
            and parallel.use_ep
            and not parallel.is_sequence_parallel
            and not parallel.enable_eplb
        ):
            logger.warning_once(
                "Experimental Nemotron deferred EP finalization was requested "
                "for an unsupported MoE configuration; using the standard path."
            )
            return

        from vllm.model_executor.layers.fused_moe.cute_dsl import (
            DeferredTopKReduceScatter,
        )

        max_local_tokens = (
            envs.VLLM_EXPERIMENTAL_NEMOTRON_DEFERRED_EP_FINALIZE_MAX_TOKENS
        )
        group = get_dp_group().device_group
        self._deferred_reduce_scatter = DeferredTopKReduceScatter.initialize(
            group=group,
            hidden_dim=config.hidden_dim_unpadded,
            top_k=config.experts_per_token,
            max_local_tokens=max_local_tokens,
        )
        config.defer_moe_finalize_max_num_tokens = max_local_tokens * parallel.dp_size
        logger.info_once(
            "Experimental fused NVFP4 top-k finalization and EP reduce-scatter "
            "is available for uniform batches up to %d tokens/rank.",
            max_local_tokens,
        )

    def _should_defer_finalize(self, local_tokens: int) -> bool:
        op = self._deferred_reduce_scatter
        if (
            op is None
            or local_tokens <= 0
            or local_tokens > op.contract.max_local_tokens
            or not is_forward_context_available()
        ):
            return False
        metadata = get_forward_context().dp_metadata
        if metadata is None:
            return False
        sizes = metadata.get_chunk_sizes_across_dp_rank()
        return (
            sizes is not None
            and len(sizes) == op.contract.world_size
            and all(size == local_tokens for size in sizes)
        )

    def set_lora_context(self, ctx) -> None:
        self._lora_context = ctx

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

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
        """Quantize and Dispatch Topk Weights and Topk Ids."""

        self._defer_this_call = self._should_defer_finalize(a1.shape[0])
        if self._moe_config is not None:
            self._moe_config.defer_moe_finalize = self._defer_this_call

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        # When LoRA is active, dispatch the per-token LoRA id along with
        # hidden_states so every rank receives the correct mapping for the
        # tokens it ends up processing. The punica_wrapper stores indices as
        # int64 but the moe_lora_align_block_size kernel expects int32, so
        # pull the pre-cast view from token_mapping_meta.
        lora_ctx = self._lora_context
        local_token_lora_mapping = None
        if lora_ctx is not None:
            local_token_lora_mapping = (
                lora_ctx.punica_wrapper.token_mapping_meta.token_lora_mapping[
                    : a1.shape[0]
                ]
            )

        extra_tensors: list[torch.Tensor] | None = None
        if scales is not None:
            extra_tensors = list(scales)
        if local_token_lora_mapping is not None:
            if extra_tensors is None:
                extra_tensors = []
            extra_tensors.append(local_token_lora_mapping)

        res = get_ep_group().dispatch(
            a1q,
            topk_weights,
            topk_ids,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

        if extra_tensors is None:
            assert len(res) == 3
            a1q, topk_weights, topk_ids = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 4
            a1q, topk_weights, topk_ids, gathered_extras = res
            gathered_extras = list(gathered_extras)
            if local_token_lora_mapping is not None:
                dispatched_lora_mapping = gathered_extras.pop()
                assert lora_ctx is not None
                lora_ctx.local_token_lora_mapping = dispatched_lora_mapping
            if scales is not None:
                a1q_scale = _unwrap_scale_and_prepare_for_moe(
                    gathered_extras, quant_config
                )
            else:
                a1q_scale = a1q_scale_orig

        return a1q, a1q_scale, None, topk_ids, topk_weights

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor | UnfinalizedMoEOutput,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        if isinstance(fused_expert_output, UnfinalizedMoEOutput):
            if not self._defer_this_call or self._deferred_reduce_scatter is None:
                raise RuntimeError(
                    "Received a deferred MoE output without an active fused "
                    "reduce-scatter consumer."
                )
            self._deferred_reduce_scatter(fused_expert_output, output)
            return

        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()

        out = weight_and_reduce_impl.apply(
            output=None,
            fused_expert_output=fused_expert_output,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        output.copy_(
            get_ep_group().combine(out, is_sequence_parallel=self.is_sequence_parallel)
        )


class MoEPrepareAndFinalizeNaiveDPEPMonolithic(mk.FusedMoEPrepareAndFinalizeMonolithic):
    """
    Naive Prepare/Finalize for Dp/Ep case for Modular Kernels.

    Uses Torch AR/RS or AR for dispatch/combine operations, applied
    to the router logits (the MoE kernel runs the router internally).
    """

    def __init__(
        self,
        is_sequence_parallel: bool = False,
        num_dispatchers: int = 1,
    ) -> None:
        super().__init__()
        self.is_sequence_parallel = is_sequence_parallel
        self._num_dispatchers = num_dispatchers

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return None

    def num_dispatchers(self) -> int:
        return self._num_dispatchers

    def output_is_reduced(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        router_logits: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareMonolithicResultType:
        """Quantize and Dispatch Router Logits."""

        a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
            a1, quant_config, defer_input_quant
        )

        res = get_ep_group().dispatch_router_logits(
            a1q,
            router_logits,
            is_sequence_parallel=self.is_sequence_parallel,
            extra_tensors=scales,
        )

        if scales is None:
            assert len(res) == 2
            a1q, router_logits = res
            a1q_scale = a1q_scale_orig
        else:
            assert len(res) == 3
            a1q, router_logits, scales = res
            a1q_scale = _unwrap_scale_and_prepare_for_moe(scales, quant_config)

        return a1q, a1q_scale, router_logits

    def finalize(
        self,
        fused_expert_output: torch.Tensor,
    ) -> torch.Tensor:
        out = get_ep_group().combine(
            fused_expert_output, is_sequence_parallel=self.is_sequence_parallel
        )
        return out


def make_moe_prepare_and_finalize_naive_dp_ep(
    use_monolithic: bool,
    is_sequence_parallel: bool = False,
    num_dispatchers: int = 1,
) -> MoEPrepareAndFinalizeNaiveDPEPModular | MoEPrepareAndFinalizeNaiveDPEPMonolithic:
    return (
        MoEPrepareAndFinalizeNaiveDPEPMonolithic(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
        if use_monolithic
        else MoEPrepareAndFinalizeNaiveDPEPModular(
            is_sequence_parallel=is_sequence_parallel,
            num_dispatchers=num_dispatchers,
        )
    )
