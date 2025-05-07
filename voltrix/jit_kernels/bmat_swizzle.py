import functools
import os
import torch
from typing import Tuple

from .tuner import jit_tuner


includes = ('"voltrix/bmat_kernels.cuh"',)
template = """
voltrix::hmat_packed_swizzle_cuda(num_row_windows, pointer1, hspa, hspa_packed);
"""

def hmat_packed_swizzle_kernel(
    block_partition: torch.Tensor,
    pointer1: torch.Tensor,
    hspa: torch.Tensor,
    hspa_packed: torch.Tensor,
):
    assert block_partition.is_cuda and block_partition.dtype == torch.int32
    assert pointer1.is_cuda and pointer1.dtype == torch.int32
    assert hspa.is_cuda and hspa.dtype == torch.float
    assert hspa_packed.is_cuda and hspa_packed.dtype == torch.uint32

    num_row_windows = block_partition.shape[0]

    args = (
        num_row_windows,
        pointer1,
        hspa,
        hspa_packed,
    )
    runtime = jit_tuner.compile_and_tune(
        name="hmat_packed_swizzle_kernel",
        keys={},
        space=tuple(),
        includes=includes,
        arg_defs=(
            ("num_row_windows", int),
            ("pointer1", torch.int),
            ("hspa", torch.float),
            ("hspa_packed", torch.uint32),
        ),
        template=template,
        args=args,
    )

    runtime(*args)
