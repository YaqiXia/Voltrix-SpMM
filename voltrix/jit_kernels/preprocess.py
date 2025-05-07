import torch
from typing import Tuple, List

from .tuner import jit_tuner


includes = ('"voltrix/bmat_kernels.cuh"',)
template = """
voltrix::preprocess(
    edge_list,
    node_pointer,
    num_nodes,
    BLK_H,
    BLK_W,
    block_partition,
    edge_to_column,
    edge_to_row,
    pointer1
);
"""


def preprocess_kernel(
    edge_list: torch.Tensor,
    node_pointer: torch.Tensor,
    block_partition: torch.Tensor,
    edge_to_column: torch.Tensor,
    edge_to_row: torch.Tensor,
    pointer1: torch.Tensor,
):

    assert edge_list.is_cpu and edge_list.dtype == torch.int32
    assert node_pointer.is_cpu and node_pointer.dtype == torch.int32
    assert block_partition.is_cpu and block_partition.dtype == torch.int32
    assert edge_to_column.is_cpu and edge_to_column.dtype == torch.int32
    assert edge_to_row.is_cpu and edge_to_row.dtype == torch.int32
    assert pointer1.is_cpu and pointer1.dtype == torch.int32

    num_nodes = node_pointer.shape[0] - 1

    args = (
        edge_list,
        node_pointer,
        num_nodes,
        block_partition,
        edge_to_column,
        edge_to_row,
        pointer1,
    )


    runtime = jit_tuner.compile_and_tune(
        name="preprocess_kernel",
        keys={},
        space=tuple(),
        includes=includes,
        arg_defs=(
            ("edge_list", torch.int),
            ("node_pointer", torch.int),
            ("num_nodes", int),
            ("block_partition", torch.int),
            ("edge_to_column", torch.int),
            ("edge_to_row", torch.int),
            ("pointer1", torch.int),
        ),
        template=template,
        args=args,
    )

    runtime(*args)



