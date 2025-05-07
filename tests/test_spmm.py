import argparse

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

    blk_offsets, hspa_packed, hind = voltrix.csr_preprocess(
        indptr,
        indices,
        args.num_nodes,
    )

    # add feature hash tag
    hspa_packed.hash_tag = f"test_{args.seed}_{args.num_nodes}_{args.density}"

    feat = feat.cuda()

    def func():
        return voltrix.spmm(
            blk_offsets,
            hspa_packed,
            hind,
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            feat=feat,
        )
    
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