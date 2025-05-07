import warnings
import torch
from typing import Tuple

from .tuner import jit_tuner
from ..jit.compiler import hash_to_hex


includes = ('"voltrix/spmm_kernels.cuh"',)
template = """
voltrix::voltrix_spmm_forward_cuda(
    blk_offsets, hspa_packed, hind,
    num_nodes, num_edges, embedding_dim, input, output, {model}, stream);
"""


def feature_hash(feature: torch.Tensor) -> int:
    """
    Hash a feature tensor to an integer.

    # For a simple and native implementation, we just use the tensor memory address.
    # This is not a suitable hash function, but the weight feature tensor is static and
    # will not change during the lifetime of the program.
    # NOTE:
    # We assume that the feature tensor's memory shape and layout will not change.
    """
    # if tensor's hash_tag attr is set, use it
    if hasattr(feature, "hash_tag") and isinstance(feature.hash_tag, str):
        return hash_to_hex(feature.hash_tag)

    warnings.warn(
        "The feature tensor(i.e. `hspa_packed`)'s hash_tag attr is not set. "
        "Voltrix will use the '0' as the key value for profiling, "
        "which may lead to performance degradation of different cases."
    )
    return hash_to_hex("0")


def spmm_kernel(
    blk_offsets: torch.Tensor,  # pointer1
    hspa_packed: torch.Tensor,
    hind: torch.Tensor,
    num_nodes: int,
    num_edges: int,
    embedding_dim: int,
    input: torch.Tensor,
    output: torch.Tensor,
):

    assert blk_offsets.is_cuda and blk_offsets.dtype == torch.int32
    assert hspa_packed.is_cuda and hspa_packed.dtype == torch.uint32
    assert hind.is_cuda and hind.dtype == torch.int32
    assert input.is_cuda and input.dtype == torch.float
    assert output.is_cuda and output.dtype == torch.float

    args = (
        blk_offsets,
        hspa_packed,
        hind,
        num_nodes,
        num_edges,
        embedding_dim,
        input,
        output,
        torch.cuda.current_stream(),
    )
    runtime = jit_tuner.compile_and_tune(
        name="spmm_kernel",
        keys={
            "feature_hash": feature_hash(hspa_packed),
        },
        space=(
            {"model": 0},
            {"model": 1},
            {"model": 2},
        ),
        includes=includes,
        arg_defs=(
            ("blk_offsets", blk_offsets.dtype),
            ("hspa_packed", hspa_packed.dtype),
            ("hind", hind.dtype),
            ("num_nodes", int),
            ("num_edges", int),
            ("embedding_dim", int),
            ("input", input.dtype),
            ("output", output.dtype),
            ("stream", torch.cuda.Stream),
        ),
        template=template,
        args=args,
        kernel_tag="spmm",
    )

    runtime(*args)
