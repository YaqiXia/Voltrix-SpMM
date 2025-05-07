import argparse
import math

import numpy as np
import scipy.sparse as sp
import torch

import voltrix
from voltrix.utils import calc_diff, GPU_bench


BLK_H = voltrix.BLK_H
BLK_W = voltrix.BLK_W


def gen_data(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    num_nodes = args.num_nodes
    num_feats = args.num_feats

    A = sp.random(num_nodes, num_nodes, density=args.density, format="csr")

    sparse = torch.sparse_csr_tensor(
        torch.tensor(A.indptr, dtype=torch.int32),
        torch.tensor(A.indices, dtype=torch.int32),
        values=torch.ones(A.nnz, dtype=torch.float32),
        size=(num_nodes, num_nodes),
    )

    indices = torch.tensor(A.indices, dtype=torch.int32)
    indptr = torch.tensor(A.indptr, dtype=torch.int32)
    feat = torch.randn(num_nodes, num_feats, dtype=torch.float32)
    num_edges = indices.shape[0]
    args.num_edges = num_edges

    print(f"indptr: {indptr.shape}")
    print(f"indices: {indices.shape}")
    print(f"feat: {feat.shape}")
    print(f"num_edges: {num_edges}")

    return indptr, indices, sparse, feat

def voltrix_spmm(
    indptr, indices, feat, args
): 
    num_row_windows = math.ceil(args.num_nodes / BLK_H)
    num_edges = args.num_edges

    # allocate preprocess buffers
    edge_to_column = torch.zeros(num_edges, dtype=torch.int32)
    edge_to_row = torch.zeros(num_edges, dtype=torch.int32)
    block_partition = torch.zeros(num_row_windows, dtype=torch.int32)
    pointer1 = torch.zeros(block_partition.numel() + 1, dtype=torch.int32)

    # preprocess
    voltrix.preprocess_kernel(
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
            total_blocks * BLK_H,
            BLK_W,
        ),
        dtype=torch.float32,
    ).cuda()
    hind = torch.zeros(
        (
            total_blocks * BLK_H,
            BLK_W,
        ),
        dtype=torch.int32,
    ).cuda()
    hspa_packed = torch.zeros((hspa.numel() // 32,), dtype=torch.uint32).cuda()

    # generate bmat
    indptr = indptr.cuda()
    indices = indices.cuda()
    feat = feat.cuda()

    edge_to_column = edge_to_column.cuda()
    edge_to_row = edge_to_row.cuda()
    block_partition = block_partition.cuda()
    pointer1 = pointer1.cuda()

    voltrix.hmat_gen_kernel(
        node_pointer=indptr,
        edge_list=indices,
        block_partition=block_partition,
        edge_to_column=edge_to_column,
        edge_to_row=edge_to_row,
        pointer1=pointer1,
        hspa=hspa,
        hind=hind,
    )
    voltrix.hmat_packed_swizzle_kernel(
        block_partition=block_partition,
        pointer1=pointer1,
        hspa=hspa,
        hspa_packed=hspa_packed,
    )

    hspa_packed.hash_tag = f"test_{args.seed}_{args.num_nodes}_{args.density}"


    # perform SpMM
    output = torch.zeros((args.num_nodes, args.num_feats), dtype=torch.float32).cuda()
    def func():
        voltrix.spmm_kernel(
            pointer1,
            hspa_packed,
            hind,
            num_nodes=args.num_nodes,
            num_edges=num_edges,
            embedding_dim=args.num_feats,
            input=feat,
            output=output,
        )

        return output
    
    output = func()
    time = GPU_bench(func, iters=10, warmup=10)

    return output, time
    
    

def cusparse_spmm(sparse, feat, args):
    sparse = sparse.cuda()
    feat = feat.cuda()

    def func():
        return sparse @ feat

    output = func()
    time = GPU_bench(func, iters=10, warmup=10)

    return output, time


def main(args):
    indptr, indices, sparse, feat = gen_data(args)

    voltrix_out, voltrix_time = voltrix_spmm(indptr, indices, feat, args)
    cusparse_out, cusparse_time = cusparse_spmm(sparse, feat, args)

    print(f"difference rate: {calc_diff(voltrix_out, cusparse_out) * 100:.3f}%")
    print(f"voltrix time: {voltrix_time:.3f} ms")
    print(f"cusparse time: {cusparse_time:.3f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate data with specified seed.")
    parser.add_argument("--seed", type=int, default=20, help="Random seed value")
    parser.add_argument("--num_nodes", type=int, default=8192, help="Number of nodes")
    parser.add_argument(
        "--density", type=float, default=0.01, help="Density of the matrix"
    )
    parser.add_argument("--num_feats", type=int, default=512, help="Feature dimension")
    args = parser.parse_args()

    main(args)
