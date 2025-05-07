import torch
from typing import Tuple

from .tuner import jit_tuner


includes = ('"voltrix/bmat_kernels.cuh"',)
template = """
voltrix::hmat_cuda(node_pointer, edge_list, block_partition, edge_to_column, edge_to_row, pointer1, num_row_windows, num_nodes, num_edges, hspa, hind);
"""


def hmat_gen_kernel(
    node_pointer: torch.Tensor,
    edge_list: torch.Tensor,
    block_partition: torch.Tensor,
    edge_to_column: torch.Tensor,
    edge_to_row: torch.Tensor,
    pointer1: torch.Tensor,
    hspa: torch.Tensor,
    hind: torch.Tensor,
):

    assert node_pointer.is_cuda and node_pointer.dtype == torch.int32
    assert edge_list.is_cuda and edge_list.dtype == torch.int32
    assert block_partition.is_cuda and block_partition.dtype == torch.int32
    assert edge_to_column.is_cuda and edge_to_column.dtype == torch.int32
    assert edge_to_row.is_cuda and edge_to_row.dtype == torch.int32
    assert pointer1.is_cuda and pointer1.dtype == torch.int32
    assert hspa.is_cuda and hspa.dtype == torch.float
    assert hind.is_cuda and hind.dtype == torch.int32

    num_row_windows = block_partition.shape[0]
    num_nodes = node_pointer.shape[0] - 1
    num_edges = edge_list.shape[0]

    args = (
        node_pointer,
        edge_list,
        block_partition,
        edge_to_column,
        edge_to_row,
        pointer1,
        num_row_windows,
        num_nodes,
        num_edges,
        hspa,
        hind,
    )

    runtime = jit_tuner.compile_and_tune(
        name="hmat_gen_kernel",
        keys={},
        space=tuple(),
        includes=includes,
        arg_defs=(
            ("node_pointer", torch.int),
            ("edge_list", torch.int),
            ("block_partition", torch.int),
            ("edge_to_column", torch.int),
            ("edge_to_row", torch.int),
            ("pointer1", torch.int),
            ("num_row_windows", int),
            ("num_nodes", int),
            ("num_edges", int),
            ("hspa", torch.float),
            ("hind", torch.int),
        ),
        template=template,
        args=args,
    )

    runtime(*args)
