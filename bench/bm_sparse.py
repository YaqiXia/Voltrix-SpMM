import numpy as np
import torch

# np.savetxt("indices.csv", A.indices, delimiter=",", fmt="%d")
# np.savetxt("indptr.csv", A.indptr, delimiter=",", fmt="%d")
indices = np.loadtxt("indices.csv", delimiter=",", dtype=np.int32)
offsets = np.loadtxt("indptr.csv", delimiter=",", dtype=np.int32)


import scipy.sparse as sp

indices = torch.tensor(indices, dtype=torch.int32).cuda()
offsets = torch.tensor(offsets, dtype=torch.int32).cuda()

N = offsets.numel() - 1
csr = torch.sparse_csr_tensor(offsets, indices, values=torch.ones_like(indices).float(), size=(N, N)).cuda()
print(indices.numel())

def read_from_file(filename, dtype):
    return np.fromfile(filename, dtype=dtype)
weight = torch.tensor(read_from_file("feat.csv", np.float32)).cuda().view(N, -1)

def f():
    return csr @ weight


iters = 100

for _ in range(10):
    o = f()

torch.cuda.synchronize()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)


start_event.record()
for _ in range(iters):
    o = f()
end_event.record()
torch.cuda.synchronize()


o = f()


o_base = read_from_file("output_base.csv", np.float32).reshape(*list(o.shape))
o = o.detach().cpu().numpy()
print(np.allclose(o, o_base, atol=1e-1))

print(f"[cuSPARSE] Elapsed time: {start_event.elapsed_time(end_event) / iters:.4f} ms")
