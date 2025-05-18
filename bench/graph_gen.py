import math
import numpy as np
import scipy.sparse as sp
import argparse
import sys
import os
import torch
import os.path as osp


if __name__ == "__main__":
    assert (
        os.getenv("TCGNN_PATH") is not None
    ), "Please set TCGNN_PATH environment variable"
    sys.path.append(os.getenv("TCGNN_PATH"))
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
    from dataset import *

    # import TCGNN
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate data with specified seed.")
    parser.add_argument("--seed", type=int, default=20, help="Random seed value")
    parser.add_argument("--num_feats", type=int, default=1024, help="Feature dimension")
    parser.add_argument(
        "--base_folder",
        type=str,
        default=f"{os.getenv('DATASET_PATH')}",
        help="Base datasets folder",
    )
    parser.add_argument("--data_name", type=str, default="ddi", help="Data name")
    parser.add_argument(
        "--only_dense", action="store_true", help="Only generate dense data"
    )
    parser.add_argument("--reorder", action="store_true", help="Use acc reorder")
    args = parser.parse_args()
    num_feats = args.num_feats

    assert (".npz" not in args.data_name) and (
        ".mtx" not in args.data_name
    ), "Please do not contain suffix .npz or .mtx in the data name"

    path = osp.join(
        args.base_folder,
        (args.data_name + (".reorder" if args.reorder else "") + ".npz"),
    )

    dataset = TCGNN_dataset(path, num_feats, 16, load_from_txt=False)
    column_index = dataset.column_index
    row_pointers = dataset.row_pointers

    num_nodes = dataset.num_nodes
    num_edges = dataset.num_edges

    print("Indices:", column_index)
    print("Indptr:", row_pointers)

    # Save CSR format as CSV files (row pointers, column indices, non-zero values)
    # np.savetxt("data.csv", A.data, delimiter=",") # This line was commented out in the original
    if not args.only_dense:
        np.savetxt("indices.csv", column_index, delimiter=",", fmt="%d")
        np.savetxt("indptr.csv", row_pointers, delimiter=",", fmt="%d")

    # print("CSR Matrix A (dense format):")
    # print(A.toarray())  # Convert CSR matrix to dense matrix and print

    B = np.random.rand(num_nodes, num_feats)
    # B = dataset.x

    # Print the shape and some data of matrix B
    # print("Matrix B shape:", B.shape)
    # print("Matrix B (first 2 rows):")
    # print(B[:2])

    # Save matrix B as a CSV file, keeping 3 decimal places
    def save_to_file(data, filename):
        with open(filename, "wb") as out_file:
            out_file.write(data.astype(np.float32).tobytes())

    # np.savetxt("feat.csv", B, delimiter=",", fmt="%.3f") # This line was commented out in the original
    save_to_file(B, "feat.csv")

    BLK_H = 16

    num_thd = int(114)
    # num_thd = math.ceil(num_nodes / BLK_H) * num_feats // 512 # This line was commented out in the original
    assert num_feats % BLK_H == 0, "num_feats should be multiple of BLK_H"
    # assert num_nodes % BLK_H == 0, "num_nodes should be multiple of BLK_H" # This line was commented out
    if num_nodes % BLK_H != 0:
        num_nodes = num_nodes + BLK_H - num_nodes % BLK_H
    base_size = num_nodes * num_feats // BLK_H**2 // num_thd
    rem_size = num_nodes * num_feats // BLK_H**2 % num_thd

    block_offsets = np.empty(num_thd + 1, dtype=np.int32)
    block_offsets.fill(base_size)
    block_offsets[0] = 0
    block_offsets[-1] = base_size + rem_size

    block_offsets = np.cumsum(block_offsets)

    if not args.only_dense:
        np.savetxt("block_offsets.csv", block_offsets, delimiter=",", fmt="%d")

    print("Block offsets:", block_offsets)

    indices = torch.tensor(column_index).cuda().int()
    offsets = torch.tensor(row_pointers).cuda().int()

    N = offsets.numel() - 1
    csr = (
        torch.sparse_csr_tensor(
            offsets, indices, values=torch.ones_like(indices).float(), size=(N, N)
        )
        .cuda()
        .float()
    )
    weight = torch.tensor(B).cuda().float()

    o = csr @ weight
    o_base = o.detach().cpu().numpy()

    save_to_file(o_base, "output_base.csv")

    import numpy as np
    from scipy.sparse import csr_matrix, coo_matrix
    from scipy.io import mmwrite, mmread
    import subprocess
    import os

    # Assume the number of rows in the matrix can be inferred from indptr
    num_rows = len(row_pointers) - 1

    data = np.ones_like(column_index, dtype=float)
    # Construct CSR matrix
    A_csr = csr_matrix(
        (data, column_index, row_pointers), shape=(num_rows, column_index.max() + 1)
    )

    # Convert to COO format
    A_coo = A_csr.tocoo()

    # Save as a .mtx file in Matrix Market format
    mmwrite("data.mtx", A_coo)
    print(f"Save to data.mtx")

    print("Data generated successfully.")