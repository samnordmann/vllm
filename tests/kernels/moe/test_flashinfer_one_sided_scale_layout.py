# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.layers.fused_moe.all2all_utils import (
    _use_direct_nvfp4_scale_layout,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    flashinfer_nvlink_one_sided,
)


@pytest.mark.parametrize(
    "override",
    [
        {"quant_dtype": None},
        {"is_scale_swizzled": False},
        {"hidden_size": 7168},
        {"top_k": 8},
        {"num_experts": 256},
        {"ep_size": 8},
    ],
)
def test_direct_nvfp4_scale_layout_target_gate(override):
    target = {
        "quant_dtype": "nvfp4",
        "is_scale_swizzled": True,
        "hidden_size": 8192,
        "top_k": 22,
        "num_experts": 512,
        "ep_size": 4,
    }
    assert _use_direct_nvfp4_scale_layout(**target)
    assert not _use_direct_nvfp4_scale_layout(**(target | override))


@pytest.mark.parametrize(
    ("tokens_per_rank", "padded_rows"),
    [(1, 128), (31, 128), (32, 128), (33, 256), (64, 256), (65, 384)],
)
def test_view_r128c4_scale_payload(tokens_per_rank, padded_rows):
    columns = 8192 // 16
    physical = torch.empty(padded_rows * columns, dtype=torch.uint8)
    result = flashinfer_nvlink_one_sided._view_r128c4_scale_payload(
        physical, logical_rows=4 * tokens_per_rank, columns=columns
    )
    assert result.shape == (padded_rows, columns)
    assert result.data_ptr() == physical.data_ptr()


def test_view_r128c4_scale_payload_rejects_wrong_size():
    with pytest.raises(ValueError, match="size mismatch"):
        flashinfer_nvlink_one_sided._view_r128c4_scale_payload(
            torch.empty(127 * 512, dtype=torch.uint8),
            logical_rows=4,
            columns=512,
        )
