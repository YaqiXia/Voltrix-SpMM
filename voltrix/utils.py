from logging import warning
import os
import sys
from typing import List
import torch


def nan_nums(x: torch.Tensor):
    return torch.isnan(x).sum().item()


def inf_nums(x: torch.Tensor):
    return torch.isinf(x).sum().item()


def check_nan_inf(x: torch.Tensor):
    if nan_nums(x) > 0 or inf_nums(x) > 0:
        warning(f"with {nan_nums(x)} nans and {inf_nums(x)} infs")


def relative_error(value: torch.Tensor, real: torch.Tensor, exclude_zeros=True):
    value = value.double().flatten()
    real = real.double().flatten()

    if not exclude_zeros:
        epsilon = 1e-9
        return ((value - real).abs() / (real.abs() + epsilon)).mean().item()
    else:
        mask = (real.abs() == 0) | real.isinf() | value.isinf()
        num = mask.sum().item()

        value = value[~mask]
        real = real[~mask]

        return ((value - real).abs() / (real.abs())).mean().item()


def calc_diff(x, y, dtype=torch.float):
    x, y = x.to(dtype), y.to(dtype)
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def round_quant(x: torch.Tensor, lfp_t=torch.float8_e4m3fn):
    return x.to(lfp_t)


def round_dequant(x: torch.Tensor, hfp_t=torch.float16):
    return x.to(hfp_t)


def per_tensor_quant(
    x: torch.Tensor, lfp_t=torch.float8_e4m3fn, scale_fp_t=torch.float32
):
    # 448 for E4M3 constexpr max value
    scale = x.abs().max() / 448
    return (x / scale).to(lfp_t), scale.to(scale_fp_t)


def per_tensor_dequant(x: torch.Tensor, scale: torch.Tensor, hfp_t=torch.float16):
    return x.to(hfp_t) * scale


def block_quant(x: torch.Tensor, blk_shape, lfp_t=torch.float8_e4m3fn):
    assert (
        x.dim() == 2 and x.size(0) % blk_shape[0] == 0 and x.size(1) % blk_shape[1] == 0
    )

    m = x.size(0)
    n = x.size(1)
    bm = blk_shape[0]
    bn = blk_shape[1]
    qm = x.size(0) // bm
    qn = x.size(1) // bn
    x = (
        x.view(qm, bm, qn, bn).permute(0, 2, 1, 3).contiguous().view(qm, qn, bm * bn)
    )  # qm, qn, bm * bn
    scales = x.abs().max(dim=2, keepdim=True).values / 448  # qm, qn, 1
    scales[scales == 0] = 1.0

    x = (
        (x / scales)
        .clamp(-448, 448)
        .view(qm, qn, bm, bn)
        .permute(0, 2, 1, 3)
        .contiguous()
        .view(m, n)
        .to(lfp_t)
    )
    scales = scales.view(qm, qn).to(torch.float32)
    return x, scales


def block_quantv1(x: torch.Tensor, blk_shape, lfp_t=torch.float8_e4m3fn):
    return x.to(lfp_t)


def block_dequant(
    x: torch.Tensor, scales: torch.Tensor, blk_shape, hfp_t=torch.float16
):
    m, n = x.size()

    bm, bn = blk_shape
    qm, qn = scales.size()

    assert m == qm * bm and n == qn * bn

    scales = scales.view(qm, qn, 1)
    x = (
        x.to(hfp_t).view(qm, bm, qn, bn).permute(0, 2, 1, 3).reshape(qm, qn, bm * bn)
        * scales
    )
    x = x.view(qm, qn, bm, bn).permute(0, 2, 1, 3).reshape(m, n)
    return x


def gaussian(size, mean=0, std=1, dtype=torch.float16, device=torch.device("cuda:0")):
    return torch.normal(mean, std, size=size, device=device, dtype=dtype)


def guassion_with_channel_outlier(
    size,
    dim: int,
    outlier_ratio=0.01,
    g_mean=0,
    g_std=1,
    o_mean=0,
    o_std=100,
    dtype=torch.bfloat16,
    device=torch.device("cuda:0"),
):
    x = gaussian(size, g_mean, g_std, dtype=dtype, device=device)

    num_channels = int(x.size(dim) * outlier_ratio)
    outlier_indices = torch.randperm(x.size(dim), device=device)[:num_channels]

    outlier_shape = [x.size(i) if i != dim else num_channels for i in range(x.ndim)]
    outliers = gaussian(outlier_shape, o_mean, o_std, dtype=dtype, device=device)

    x.index_copy_(dim, outlier_indices, outliers)

    return x


class DurationTimer:
    """
    Example:
        With DurationTimer() as t:
            ...
        duration = t.get_duration()
    """

    def __init__(self, cond=True, device=None, is_sync=True):
        self.cond = cond
        self.device = device if device is not None else torch.cuda.current_device()
        # self.device = device
        self.is_sync = is_sync

        self.duration = None

    def __enter__(self):
        if self.cond:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()  # type: ignore
        return self

    def __exit__(self, *args):
        if self.cond:
            self.end.record()  # type: ignore
            if self.sync:
                torch.cuda.synchronize(self.device)
                self.duration: float = self.start.elapsed_time(self.end)

    def sync(self):
        if self.cond:
            torch.cuda.synchronize(self.device)
            self.duration = self.start.elapsed_time(self.end)
        return self

    def get_duration(self):  # return in ms
        if self.duration is not None:
            return self.duration
        else:
            return self.start.elapsed_time(self.end)


class empty_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class suppress_stdout_stderr:
    def __enter__(self):
        self.outnull_file = open(os.devnull, "w")
        self.errnull_file = open(os.devnull, "w")

        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)

        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)

        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)

        self.outnull_file.close()
        self.errnull_file.close()


def bench_kineto(
    fn,
    kernel_names,
    num_tests: int = 30,
    suppress_kineto_output: bool = False,
    trace_path: str = None,
    barrier_comm_profiling: bool = False,
    flush_l2: bool = False,
):
    # Conflict with Nsight Systems
    using_nsys = os.environ.get("DG_NSYS_PROFILING", False)

    # For some auto-tuning kernels with prints
    fn()

    # Profile
    suppress = (
        suppress_stdout_stderr
        if suppress_kineto_output and not using_nsys
        else empty_suppress
    )
    with suppress():
        schedule = (
            torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=1)
            if not using_nsys
            else None
        )
        profiler = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA], schedule=schedule
            )
            if not using_nsys
            else empty_suppress()
        )
        with profiler:
            for i in range(2):
                # NOTES: use a large kernel and a barrier to eliminate the unbalanced CPU launch overhead
                if barrier_comm_profiling:
                    import torch.distributed as dist

                    lhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    rhs = torch.randn((8192, 8192), dtype=torch.float, device="cuda")
                    lhs @ rhs
                    dist.all_reduce(torch.ones(1, dtype=torch.float, device="cuda"))
                for _ in range(num_tests):
                    if flush_l2:
                        torch.empty(
                            int(256e6 // 4), dtype=torch.int, device="cuda"
                        ).zero_()
                    fn()

                if not using_nsys:
                    profiler.step()

    # Return 1 if using Nsight Systems
    if using_nsys:
        return 1

    # Parse the profiling table
    assert isinstance(kernel_names, str) or isinstance(kernel_names, tuple)
    is_tupled = isinstance(kernel_names, tuple)
    prof_lines = (
        profiler.key_averages()
        .table(sort_by="cuda_time_total", max_name_column_width=160)
        .split("\n")
    )
    kernel_names = (kernel_names,) if isinstance(kernel_names, str) else kernel_names
    assert all([isinstance(name, str) for name in kernel_names])
    for name in kernel_names:
        assert (
            sum([name in line for line in prof_lines]) == 1
        ), f"Errors of the kernel {name} in the profiling table"

    # Save chrome traces
    if trace_path is not None:
        profiler.export_chrome_trace(trace_path)

    # Return average kernel times
    units = {"ms": 1e3, "us": 1e6}
    kernel_times = []
    for name in kernel_names:
        for line in prof_lines:
            if name in line:
                time_str = line.split()[-2]
                for unit, scale in units.items():
                    if unit in time_str:
                        kernel_times.append(float(time_str.replace(unit, "")) / scale)
                        break
                break
    return tuple(kernel_times) if is_tupled else kernel_times[0]


def GPU_bench(func, iters=100, warmup=30, kernel_name=None) -> float:
    """
    Benchmark a function on GPU.
    if kernel_name is given, use kineto to profile only one matched kernel.
    return in ms
    """
    if kernel_name is None:
        for _ in range(warmup):
            func()

        with DurationTimer() as t:
            for _ in range(iters):
                func()
        return t.get_duration() / iters
    else:
        if kernel_name == "fp8_gemm":
            print("HHHHHHHHHHHHHH")
        return (
            bench_kineto(
                func,
                kernel_name,
                num_tests=iters,
                suppress_kineto_output=True,
                flush_l2=True,
            )
            * 1e3
        )


def CPU_bench(func, iters=100, warmup=30):
    for _ in range(warmup):
        func()

    import time

    start = time.time()
    for _ in range(iters):
        func()
    end = time.time()
    # return in ms
    return (end - start) * 1000 / iters


def profiler_enter():
    from torch.cuda import cudart, check_error

    check_error(cudart().cudaProfilerStart())


def profiler_exit():
    from torch.cuda import cudart, check_error

    check_error(cudart().cudaProfilerStop())


class Profiler:
    def __enter__(self):
        profiler_enter()

    def __exit__(self, *args):
        profiler_exit()


class WithNVTX:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        from torch.cuda import nvtx

        nvtx.range_push(self.name)
        return self

    def __exit__(self, *args):
        from torch.cuda import nvtx

        nvtx.range_pop()
        return self


if __name__ == "__main__":
    x = guassion_with_channel_outlier(
        [10, 10], 1, 0.2, g_mean=0, g_std=0, o_mean=100, o_std=0
    )
    print(x)
