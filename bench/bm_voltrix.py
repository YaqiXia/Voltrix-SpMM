import numpy as np
import torch

import voltrix
from voltrix.utils import calc_diff, GPU_bench


def read_from_file(filename, dtype):
    return np.fromfile(filename, dtype=dtype)


indices = torch.tensor(np.loadtxt("indices.csv", delimiter=",", dtype=np.int32), dtype=torch.int32)
indptr = torch.tensor(np.loadtxt("indptr.csv", delimiter=",", dtype=np.int32), dtype=torch.int32)
N = indptr.numel() - 1
weight = torch.tensor(read_from_file("feat.csv", np.float32)).cuda().view(N, -1)

blk_ofs, hspa_packed, hind = voltrix.csr_preprocess(
    indptr,
    indices,
    N,
)

def spmm():
    return voltrix.spmm(
        blk_ofs, hspa_packed, hind, num_nodes=N, num_edges=indices.numel(), feat=weight
    )


o = spmm()
o = o.detach().cpu()
o_base = torch.tensor(
    read_from_file("output_base.csv", np.float32).reshape(*list(o.shape))
)
print(f"difference rate: {calc_diff(o, o_base) * 100:.3f}%")

time = GPU_bench(spmm, iters=10, warmup=10, kernel_name="spmm")
print(f"[Voltrix] time: {time:.4f} ms")
