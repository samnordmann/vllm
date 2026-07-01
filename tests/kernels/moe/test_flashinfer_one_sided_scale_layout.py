# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.model_executor.layers.fused_moe.all2all_utils import (
    _use_direct_nvfp4_scale_layout,
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
