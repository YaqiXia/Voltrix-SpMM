import multiprocessing as mp
import time
import copy
import os
import sys
import torch
from typing import Any, Dict

from ..jit import build, cpp_format, generate, Runtime
from ..project import (
    DEBUG_FLAG,
    PRINT_AUTOTUNE_FLAG,
)


def suppress_output(func, *args, **kwargs):
    with open(os.devnull, "w") as fnull:
        old_stdout = os.dup(sys.stdout.fileno())
        old_stderr = os.dup(sys.stderr.fileno())
        os.dup2(fnull.fileno(), sys.stdout.fileno())
        os.dup2(fnull.fileno(), sys.stderr.fileno())
        try:
            return func(*args, **kwargs) 
        finally:
            os.dup2(old_stdout, sys.stdout.fileno())
            os.dup2(old_stderr, sys.stderr.fileno())


def parallel_build(name, arg_defs, code, tuned_keys):
    try:
        if os.getenv(DEBUG_FLAG, None):
            runtime = build(name, arg_defs, code)
        else:
            runtime = suppress_output(build, name, arg_defs, code)
    except Exception as e:
        import traceback
        traceback.print_exc()
        runtime = None
    return runtime, tuned_keys


class JITTuner:
    def __init__(self) -> None:
        self.tuned = {}

    def compile_and_tune(
        self,
        name: str,
        keys: Dict[str, Any],
        space: tuple,
        includes: tuple,
        arg_defs: tuple,
        template: str,
        args: tuple,
        kernel_tag=None,  # optional, kernel name for ncu profiling. If not given, we will use cuda event to capture range time.
    ) -> Runtime:
        # NOTES: we always assume the space and template will not change
        # We also assume the GPU device will not be changed
        # NOTES: the function must have no accumulated side effects
        keys = {k: keys[k] for k in sorted(keys.keys())}
        signature = (name, f"{keys}")
        if signature in self.tuned:
            if os.getenv(DEBUG_FLAG, None):
                print(f"Using cached JIT kernel {name} with keys {keys}")
            return self.tuned[signature]

        if os.getenv(DEBUG_FLAG, None):
            print(f"Auto-tuning JIT kernel {name} with keys {keys}")

        assert signature not in self.tuned
        assert args is not None
        space = (dict(),) if len(space) == 0 else space

        # kernels = []
        # for tuned_keys in space:
        #     assert isinstance(tuned_keys, dict)
        #     full_keys = copy.deepcopy(keys)
        #     full_keys.update(tuned_keys)
        #     code = generate(includes, arg_defs, cpp_format(template, full_keys))

        #     # Illegal build must raise errors
        #     kernels.append((build(name, arg_defs, code), tuned_keys))

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = []
            for tuned_keys in space:
                assert isinstance(tuned_keys, dict)
                full_keys = copy.deepcopy(keys)
                full_keys.update(tuned_keys)
                code = generate(includes, arg_defs, cpp_format(template, full_keys))

                result = pool.apply_async(
                    parallel_build, (name, arg_defs, code, tuned_keys)
                )
                results.append(result)

            # kernels = [res.get() for res in results]
            kernels = []
            for res in results:
                kernel = res.get()
                if kernel[0] is not None:
                    kernels.append(kernel)

        best_runtime, best_time, best_keys = None, None, None
        times = []
        for runtime, tuned_keys in kernels:
            if len(space) > 1:
                # Check kernel validity
                return_code = runtime(*args)
                if return_code != 0:
                    # Pass illegal kernels, e.g. insufficient shared memory capacity
                    if os.getenv(DEBUG_FLAG, None):
                        print(
                            f"Illegal JIT kernel {name} with keys {keys} and tuned keys {tuned_keys}: error code {return_code}"
                        )
                    continue

                # Measure performance with L2 flush and a large GEMM kernel before to reduce overhead between kernels
                """
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                try:
                    torch.empty(int(256e6 // 4), dtype=torch.int, device='cuda').zero_()
                    # _ = torch.randn((4096, 4096), dtype=torch.float, device='cuda') @ torch.randn((4096, 4096), dtype=torch.float, device='cuda')
                    start_event.record()
                    for i in range(8):
                        assert runtime(*args) == 0
                    end_event.record()
                    end_event.synchronize()
                except:
                    continue
                elapsed_time = start_event.elapsed_time(end_event)
                """

                from ..utils import GPU_bench

                # _ = torch.randn((4096, 4096), dtype=torch.float, device='cuda') @ torch.randn((4096, 4096), dtype=torch.float, device='cuda')
                def func():
                    return runtime(*args)

                elapsed_time = GPU_bench(func, iters=8, kernel_name=kernel_tag)
                times.append(elapsed_time)
            else:
                elapsed_time = 0

            # Compare if better
            if best_time is None or elapsed_time < best_time:
                best_runtime, best_time, best_keys = runtime, elapsed_time, tuned_keys
            if os.getenv(DEBUG_FLAG, None):
                print(
                    f"Tuned JIT kernel {name} with keys {keys} and tuned keys {tuned_keys} has time {elapsed_time}"
                )
        assert (
            best_runtime is not None
        ), f"Failed to tune JIT kernel {name} with keys {keys}"

        # Cache the best runtime and return
        if os.getenv(DEBUG_FLAG, None) or os.getenv(PRINT_AUTOTUNE_FLAG, None):
            print(
                f"JIT kernel {name}[in {len(kernels)}/{len(results)}] with keys {keys} has tuned keys {best_keys} and time {best_time:.4f}ms"
            )
            # print(f"times: {times}")

        self.tuned[signature] = best_runtime
        return best_runtime


jit_tuner = JITTuner()
