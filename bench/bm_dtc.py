import sys
import numpy as np
import os.path as osp
import argparse
import torch


BLK_H = 16
BLK_W = 8
import DTCSpMM

# ExecutionPlan = {
#   # reorderd
#   "YeastH.reorder": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"], 496: [False, "float", "nonsplit"]},
#   "OVCAR-8H.reorder": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"], 496: [False, "float", "nonsplit"]},
#   "Yeast.reorder": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"], 496: [False, "float", "nonsplit"]},
#   "DD.reorder": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float4", "split"], 496: [False, "float4", "split"]},
#   "web-BerkStan.reorder": {128: [False, "float2", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "nonsplit"], 496: [False, "float", "nonsplit"]},
#   "reddit.reorder": {128: [True, "float4", "split"], 256: [True, "float4", "split"], 512: [True, "float4", "split"], 496: [True, "float4", "split"]},
#   "ddi.reorder": {128: [True, "float", "nonsplit"], 256: [True, "float", "nonsplit"], 512: [True, "float4", "split"], 496: [True, "float4", "split"]},
#   "protein.reorder": {128: [False, "float4", "split"], 256: [False, "float4", "split"], 512: [False, "float4", "split"], 496: [True, "float4", "split"]},

#   # origin
#   "YeastH": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"]},
#   "OVCAR-8H": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"]},
#   "Yeast": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float", "nonsplit"]},
#   "DD": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float4", "split"]},
#   "web-BerkStan": {128: [False, "float2", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "nonsplit"]},
#   "reddit": {128: [False, "float4", "split"], 256: [False, "float4", "split"], 512: [False, "float4", "split"]},
#   "ddi": {128: [False, "float", "nonsplit"], 256: [False, "float", "nonsplit"], 512: [False, "float4", "nonsplit"]},
#   "protein": {128: [False, "float4", "split"], 256: [False, "float4", "split"], 512: [False, "float4", "split"]},
#   # more
#   "citeseer": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
#   "cora": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"], 1024: [False, "float4", "split"], 2048: [False, "float4", "split"], 4096: [False, "float4", "split"], 8192: [False, "float4", "split"]},
#   "pubmed": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
#   "ppi": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
#   "PROTEINS_full": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
#   "OVCAR-8H": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
#   "amazon0505": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
#   "artist": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
#   "com-amazon": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
#   "soc-BlogCatalog": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
#   "amazon0601": {128: [False, "float4", "nonsplit"], 256: [False, "float4", "nonsplit"], 512: [False, "float4", "split"]},
# }

## Load matrix from files.
# dataset = args.dataset
# Set your own path to the dataset.


# 1. 读取 CSR 矩阵的三个部分（data, indices, indptr）并重建稀疏矩阵 A
indices = np.loadtxt("indices.csv", delimiter=",", dtype=int)
indptr = np.loadtxt("indptr.csv", delimiter=",", dtype=int)


def read_from_file(filename, dtype):
    return np.fromfile(filename, dtype=dtype)


feat = read_from_file("feat.csv", np.float32).reshape(indptr.shape[0] - 1, -1)
data = np.ones_like(indices)

##--------------------------Test for preprocess------------##
indices_tensor = torch.from_numpy(indices).int()
indptr_tensor = torch.from_numpy(indptr).int()
feat = torch.from_numpy(feat).float()
num_nodes = indptr_tensor.shape[0] - 1
num_edges = indices_tensor.shape[0]


# path = '/data/xyq/TCGNN-Pytorch/ncu_cuda_test/spmm_test'
# matrix = Custom_dataset(path, None, None)
num_rows = num_nodes
num_nnz = num_edges
print("NUM_ROW, NNZ: ", num_rows, " ", num_nnz)
column_index = indices_tensor
row_pointers = indptr_tensor
# Process data.
num_row_windows = (num_rows + BLK_H - 1) // BLK_H
edgeToColumn = torch.zeros(num_nnz, dtype=torch.int)
edgeToRow = torch.zeros(num_nnz, dtype=torch.int)
blockPartition = torch.zeros(num_row_windows, dtype=torch.int)
column_index_ori = column_index.cuda()
row_pointers_ori = row_pointers.cuda()

blockPartition_cuda = blockPartition.cuda()
edgeToColumn_cuda = edgeToColumn.cuda()
edgeToRow_cuda = edgeToRow.cuda()


# Optimize CPU.
# RowWindowOffset, TCblockRowid,\
#       TCblocktileId, TCblockoffset, SparseAToXindex,\
#         block_count = DTCSpMM.preprocess(column_index, row_pointers, num_rows, BLK_H, BLK_W, blockPartition, edgeToColumn, edgeToRow)

# Optimize GPU.
(
    RowWindowOffset,
    TCblockRowid,
    TCblocktileId,
    TCblockoffset,
    SparseAToXindex,
    block_count,
) = DTCSpMM.preprocess_gpu(
    column_index_ori,
    row_pointers_ori,
    num_rows,
    BLK_H,
    BLK_W,
    blockPartition_cuda,
    edgeToColumn_cuda,
    edgeToRow_cuda,
)
print(blockPartition_cuda.sum())
# DTCSpMM.preprocess_gpu(column_index_ori, row_pointers_ori, num_rows, 8, BLK_W, blockPartition_cuda, edgeToColumn_cuda, edgeToRow_cuda)
# print(blockPartition_cuda.sum())
# exit()
# breakpoint()
iters = 1000

# import TES
# Pointer1, Pointer2, local_map, global_map = TES.preprocess_hcsr(column_index, row_pointers, num_rows,  \
#             BLK_H,	BLK_W, blockPartition, edgeToColumn, edgeToRow)
# Pointer1 = Pointer1.contiguous().cuda()
# total_blocks = Pointer1[-1].cpu().item()
# hspa = torch.zeros([total_blocks * BLK_W * BLK_H], dtype=torch.float32).cuda()
# hind = torch.zeros([total_blocks * BLK_W], dtype=torch.int32).cuda()
# blockPartition_ori  = blockPartition.cuda()
# edgeToColumn_ori  = edgeToColumn.cuda()
# edgeToRow_ori  = edgeToRow.cuda()
# TES.hmat_gen(row_pointers_ori, column_index_ori, blockPartition_ori, edgeToColumn_ori, edgeToRow_ori, Pointer1, hspa, hind)
# Run tests.
weight = torch.tensor(
    read_from_file("feat.csv", np.float32).reshape(indptr.shape[0] - 1, -1)
).cuda()

feat_size = weight.shape[1]
X = weight
print("feat_size =", feat_size)

parser = argparse.ArgumentParser(description="DTCSpMM test")
parser.add_argument("--use_balance", action="store_true", help="Use balance mode")
args = parser.parse_args()

balance_choice = args.use_balance
exeplan = "float4_split"

torch.cuda.synchronize()
if not balance_choice:
    X_out = DTCSpMM.run_DTCSpMM(
        X,
        RowWindowOffset,
        TCblocktileId,
        TCblockoffset,
        SparseAToXindex,
        num_rows,
        num_nnz,
        exeplan,
    )[0]
else:
    X_out = DTCSpMM.run_DTCSpMM_balance(
        X,
        TCblockRowid,  # 每个TC blocks的 行 idx
        TCblocktileId,  # TC blocks中非零元素的位置展开后并在一起
        TCblockoffset,  # 指导 TCblocktileId 中每个block的偏移量
        SparseAToXindex,  # TC blocks数量 * 8
        num_rows,
        exeplan,
    )[0] / 1000


o = X_out

o_base = read_from_file("output_base.csv", np.float32).reshape(*list(o.shape))
o = o.detach().cpu().numpy()
print(np.allclose(o, o_base, atol=1e-1))


try:
    with open("DTCSpMM_exe_time_and_throughput.csv", "r") as f:
        print(f"[DTC-SPMM] Elapsed time: {f.readline().split(",")[1]} ms")

    with open("DTCSpMM_exe_time_and_throughput.csv", "w") as f:
        pass
except Exception as e:
    print(f"An error occurred: {e}")