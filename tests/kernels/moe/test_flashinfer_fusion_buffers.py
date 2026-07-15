# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    flashinfer_nvlink_one_sided as fi_one_sided,
)

pytestmark = pytest.mark.skip_global_cleanup


class _FakeWorkspaceManager:
    def __init__(self):
        self.requests = None

    def get_simultaneous(self, *requests):
        self.requests = requests
        return [torch.empty(shape, dtype=dtype) for shape, dtype in requests]


class _FakeExperts:
    def workspace_dtype(self, out_dtype):
        return out_dtype

    def workspace_shapes(self, m, *args):
        return (m, 2), (m, 3), (m, 4)


class _DefaultPrepareFinalize:
    fused_expert_output_buffer = (
        mk.FusedMoEPrepareAndFinalize.fused_expert_output_buffer
    )


def _allocate(monkeypatch, prepare_finalize):
    manager = _FakeWorkspaceManager()
    monkeypatch.setattr(mk, "current_workspace_manager", lambda: manager)
    impl = object.__new__(mk.FusedMoEKernelModularImpl)
    impl.prepare_finalize = prepare_finalize
    impl.fused_experts = _FakeExperts()
    outputs = impl._allocate_buffers(
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
    return manager, outputs


def test_default_output_reuses_workspace(monkeypatch):
    manager, (workspace13, workspace2, fused_out) = _allocate(
        monkeypatch, _DefaultPrepareFinalize()
    )

    assert manager.requests == (
        ((20,), torch.bfloat16),
        ((2, 3), torch.bfloat16),
    )
    assert workspace13.shape == (2, 2)
    assert workspace2.shape == (2, 3)
    assert fused_out.shape == (5, 4)
    assert fused_out.untyped_storage().data_ptr() == (
        workspace13.untyped_storage().data_ptr()
    )


def test_external_output_bypasses_workspace_storage(monkeypatch):
    external = torch.empty((5, 4), dtype=torch.bfloat16)
    manager, (_, _, fused_out) = _allocate(
        monkeypatch,
        SimpleNamespace(fused_expert_output_buffer=lambda *args: external),
    )

    assert manager.requests == (
        ((4,), torch.bfloat16),
        ((2, 3), torch.bfloat16),
    )
    assert fused_out is external


@pytest.mark.parametrize("mismatch", ["shape", "dtype", "device", "contiguity"])
def test_invalid_external_output_is_rejected(monkeypatch, mismatch):
    if mismatch == "shape":
        external = torch.empty((4, 4), dtype=torch.bfloat16)
    elif mismatch == "dtype":
        external = torch.empty((5, 4), dtype=torch.float32)
    elif mismatch == "device":
        external = torch.empty((5, 4), dtype=torch.bfloat16, device="meta")
    else:
        external = torch.empty((4, 5), dtype=torch.bfloat16).T

    with pytest.raises(AssertionError):
        _allocate(
            monkeypatch,
            SimpleNamespace(fused_expert_output_buffer=lambda *args: external),
        )


def test_output_multiplier_is_explicitly_gated(monkeypatch):
    prepare_finalize = object.__new__(
        fi_one_sided.FlashInferNVLinkOneSidedPrepareAndFinalize
    )

    monkeypatch.setenv("VLLM_FLASHINFER_FUSED_OUTPUT_SCALE", "0")
    assert not prepare_finalize.try_fuse_output_multiplier(5.0)
    monkeypatch.setenv("VLLM_FLASHINFER_FUSED_OUTPUT_SCALE", "1")
    assert prepare_finalize.try_fuse_output_multiplier(5.0)
    assert prepare_finalize.output_multiplier == 5.0
