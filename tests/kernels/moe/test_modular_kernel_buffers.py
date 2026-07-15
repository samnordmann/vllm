# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk

pytestmark = pytest.mark.skip_global_cleanup


class _FakeWorkspaceManager:
    def __init__(self):
        self.requests = None

    def get_simultaneous(self, *requests):
        self.requests = requests
        return [torch.empty(shape, dtype=dtype) for shape, dtype in requests]


class _FakeExperts:
    a2_scale = None

    def __init__(self):
        self.apply_kwargs = None

    def workspace_dtype(self, out_dtype):
        return out_dtype

    def workspace_shapes(self, M, *args):
        return (M, 2), (M, 3), (M, 4)

    def moe_problem_size(self, *args):
        return 2, 5, 4, 3, 1

    def apply(self, **kwargs):
        self.apply_kwargs = kwargs


class _DefaultPrepareFinalize:
    fused_expert_output_buffer = (
        mk.FusedMoEPrepareAndFinalize.fused_expert_output_buffer
    )


def _make_impl(prepare_finalize):
    impl = object.__new__(mk.FusedMoEKernelModularImpl)
    impl.prepare_finalize = prepare_finalize
    impl.fused_experts = _FakeExperts()
    return impl


def _allocate(impl):
    return impl._allocate_buffers(
        out_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        M_chunk=2,
        M_full=5,
        N=4,
        K=3,
        top_k=1,
        global_num_experts=2,
        local_num_experts=2,
        expert_tokens_meta=None,
        activation=mk.MoEActivation.SILU,
    )


def test_default_output_buffer_uses_workspace_allocation(monkeypatch):
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(mk, "current_workspace_manager", lambda: workspace_manager)
    impl = _make_impl(_DefaultPrepareFinalize())

    workspace13, workspace2, fused_out = _allocate(impl)

    assert workspace_manager.requests == (
        ((20,), torch.bfloat16),
        ((2, 3), torch.bfloat16),
    )
    assert workspace13.shape == (2, 2)
    assert workspace2.shape == (2, 3)
    assert fused_out.shape == (5, 4)
    assert fused_out.untyped_storage().data_ptr() == (
        workspace13.untyped_storage().data_ptr()
    )


def test_external_output_buffer_is_passed_to_experts(monkeypatch):
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(mk, "current_workspace_manager", lambda: workspace_manager)
    external_output = torch.empty((5, 4), dtype=torch.bfloat16)
    impl = _make_impl(
        SimpleNamespace(fused_expert_output_buffer=lambda *args: external_output)
    )

    fused_out = impl._fused_experts(
        in_dtype=torch.bfloat16,
        a1q=torch.empty((5, 3), dtype=torch.bfloat16),
        a1q_scale=None,
        w1=torch.empty((2, 4, 3), dtype=torch.bfloat16),
        w2=torch.empty((2, 3, 4), dtype=torch.bfloat16),
        topk_weights=torch.empty((5, 1), dtype=torch.float32),
        topk_ids=torch.empty((5, 1), dtype=torch.int32),
        activation=mk.MoEActivation.SILU,
        global_num_experts=2,
        local_num_experts=2,
        expert_map=None,
        apply_router_weight_on_input=False,
        expert_tokens_meta=None,
    )

    assert workspace_manager.requests == (
        ((10,), torch.bfloat16),
        ((5, 3), torch.bfloat16),
    )
    assert fused_out is external_output
    assert impl.fused_experts.apply_kwargs["output"] is external_output


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device", "contiguity"])
def test_invalid_external_output_buffer_is_rejected(monkeypatch, mismatch):
    workspace_manager = _FakeWorkspaceManager()
    monkeypatch.setattr(mk, "current_workspace_manager", lambda: workspace_manager)

    if mismatch == "shape":
        external_output = torch.empty((4, 4), dtype=torch.bfloat16)
    elif mismatch == "dtype":
        external_output = torch.empty((5, 4), dtype=torch.float32)
    elif mismatch == "device":
        external_output = torch.empty((5, 4), dtype=torch.bfloat16, device="meta")
    else:
        external_output = torch.empty((4, 5), dtype=torch.bfloat16).T

    impl = _make_impl(
        SimpleNamespace(fused_expert_output_buffer=lambda *args: external_output)
    )
    with pytest.raises(AssertionError):
        _allocate(impl)
