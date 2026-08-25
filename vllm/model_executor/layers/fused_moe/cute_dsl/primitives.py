# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CuTe DSL primitives used by the deferred-finalize collective."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Int64, Uint16, Uint32
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

VEC_BF16 = 8
PACKED_BYTES = 16
NUM_LAMPORT_BUFFERS = 3
NEG_ZERO_F32_BITS = 0x80000000
NEG_ZERO_BF16_BITS = 0x8000


class CUDAGraphCompatibleWrapper:
    """DLPack view that does not synchronize with the producer stream."""

    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor

    def __dlpack__(self, stream=None):
        return self.tensor.__dlpack__(stream=-1)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


def to_cute(tensor: torch.Tensor, assumed_align: int = 16) -> cute.Tensor:
    return from_dlpack(
        CUDAGraphCompatibleWrapper(tensor.detach()), assumed_align=assumed_align
    )


def to_cute_dynamic_m(
    tensor: torch.Tensor,
    *,
    mode: int,
    assumed_align: int = 16,
) -> cute.Tensor:
    return to_cute(tensor, assumed_align).mark_compact_shape_dynamic(
        mode=mode,
        stride_order=tensor.dim_order(),
    )


@dsl_user_op
def load_global_u32x4(
    pointer: cute.Pointer,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
):
    address = pointer.toint(loc=loc, ip=ip)
    opcode = "ld.volatile.global.v4.u32" if volatile else "ld.global.v4.u32"
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        f"{opcode} {{$0, $1, $2, $3}}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=volatile,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed = vector.from_elements(
        ir.VectorType.get([4], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(4)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@dsl_user_op
def store_global_u32x4(
    address: Int64,
    packed,
    *,
    volatile: cutlass.Constexpr[bool] = False,
    loc=None,
    ip=None,
) -> None:
    words = [packed[i].ir_value(loc=loc, ip=ip) for i in range(4)]
    opcode = "st.volatile.global.v4.u32" if volatile else "st.global.v4.u32"
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), *words],
        f"{opcode} [$0], {{$1, $2, $3, $4}};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def store_lamport_sentinel_128(pointer: cute.Pointer, *, loc=None, ip=None) -> None:
    address = pointer.toint(loc=loc, ip=ip)
    value = Uint32(NEG_ZERO_F32_BITS).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [address.ir_value(loc=loc, ip=ip), value, value, value, value],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_async_release_gpu_add_u32(
    pointer: cute.Pointer, value: Uint32, *, loc=None, ip=None
) -> None:
    address = pointer.toint(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [
            address.ir_value(loc=loc, ip=ip),
            value.ir_value(loc=loc, ip=ip),
        ],
        "red.async.release.global.gpu.add.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_volatile_u32(pointer: cute.Pointer, *, loc=None, ip=None) -> Uint32:
    address = pointer.toint(loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.volatile.global.u32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def packed_u32x4_to_bf16x8(packed, *, loc=None, ip=None):
    target_type = ir.VectorType.get([VEC_BF16], BFloat16.mlir_type, loc=loc)
    values = llvm.bitcast(
        target_type,
        packed.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(values, VEC_BF16, BFloat16)


@dsl_user_op
def bf16x8_to_packed_u32x4(values, *, loc=None, ip=None):
    target_type = ir.VectorType.get([4], T.i32(), loc=loc)
    packed = llvm.bitcast(
        target_type,
        values.ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(packed, 4, Uint32)


@cute.jit
def sanitize_negative_zero_u32(word):
    low = Uint16(word & Uint32(0xFFFF))
    high = Uint16(word >> Uint32(16))
    if low == Uint16(NEG_ZERO_BF16_BITS):
        word = word & Uint32(0xFFFF0000)
    if high == Uint16(NEG_ZERO_BF16_BITS):
        word = word & Uint32(0x0000FFFF)
    return word


@cute.jit
def sanitize_negative_zero(packed):
    result = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
    for index in cutlass.range_constexpr(4):
        result[index] = sanitize_negative_zero_u32(packed[index])
    return result.load()


@cute.jit
def fragment_is_dirty(packed):
    dirty = packed[0] == Uint32(NEG_ZERO_F32_BITS)
    for index in cutlass.range_constexpr(1, 4):
        dirty = dirty | (packed[index] == Uint32(NEG_ZERO_F32_BITS))
    return dirty
