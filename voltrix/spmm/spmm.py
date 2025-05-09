import math

import torch

from ..jit_kernels import (
    preprocess_kernel,
    hmat_gen_kernel,
    spmm_kernel,
    hmat_packed_swizzle_kernel,
)

BLK_H = 16
BLK_W = 8


def csr_preprocess(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    num_nodes: int,
):
    assert indptr.is_cpu and indptr.dtype == torch.int32
    assert indices.is_cpu and indices.dtype == torch.int32

    num_edges = indices.numel()
    num_row_windows = math.ceil(num_nodes / BLK_H)

    # allocate preprocess buffers
    edge_to_column = torch.zeros(num_edges, dtype=torch.int32)
    edge_to_row = torch.zeros(num_edges, dtype=torch.int32)
    block_partition = torch.zeros(num_row_windows, dtype=torch.int32)
    pointer1 = torch.zeros(block_partition.numel() + 1, dtype=torch.int32)

    # preprocess
    preprocess_kernel(
        edge_list=indices,
        node_pointer=indptr,
        block_partition=block_partition,
        edge_to_column=edge_to_column,
        edge_to_row=edge_to_row,
        pointer1=pointer1,
    )

    # allocate bmat buffers
    total_blocks = int(pointer1[-1].cpu().item())
    hspa = torch.zeros(
        (
            total_blocks * BLK_H * BLK_W,
        ),
        dtype=torch.float32,
    ).cuda()
    hind = torch.zeros(
        (
            total_blocks * BLK_W,
        ),
        dtype=torch.int32,
    ).cuda()
    hspa_packed = torch.zeros((hspa.numel() // 32,), dtype=torch.uint32).cuda()

    # generate bmat
    indptr = indptr.cuda()
    indices = indices.cuda()

    edge_to_column = edge_to_column.cuda()
    edge_to_row = edge_to_row.cuda()
    block_partition = block_partition.cuda()
    pointer1 = pointer1.cuda()

    hmat_gen_kernel(
        node_pointer=indptr,
        edge_list=indices,
        block_partition=block_partition,
        edge_to_column=edge_to_column,
        edge_to_row=edge_to_row,
        pointer1=pointer1,
        hspa=hspa,
        hind=hind,
    )
    hmat_packed_swizzle_kernel(
        block_partition=block_partition,
        pointer1=pointer1,
        hspa=hspa,
        hspa_packed=hspa_packed,
    )

    return (
        pointer1,  # blk_offsets
        hspa_packed,
        hind,
    )


def spmm(
    blk_offsets: torch.Tensor,  # pointer1
    hspa_packed: torch.Tensor,
    hind: torch.Tensor,
    num_nodes: int,
    num_edges: int,
    feat: torch.Tensor,
):
    num_feats = feat.shape[1]
    output = torch.empty((num_nodes, num_feats), dtype=torch.float32, device=feat.device)

    spmm_kernel(
        blk_offsets,
        hspa_packed,
        hind,
        num_nodes=num_nodes,
        num_edges=num_edges,
        embedding_dim=num_feats,
        input=feat,
        output=output,
    )

    return output
