

import os
import subprocess
import sys

import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Launch Voltrix-SPMM kernel.")
parser.add_argument("--rode_home", type=str, help="Path to RoDe")
parser.add_argument("--feat_dim", type=int, choices=[256, 512, 1024], help="Feature dimension")
parser.add_argument("--data", type=str, default="data.mtx", help="data file")
args = parser.parse_args()

rode_home = args.rode_home
feat_dim = args.feat_dim
exe_file = os.path.join(rode_home, "build", "eval", f"eval_spmm_f32_n{feat_dim}")

if not os.path.exists(exe_file):
    raise FileNotFoundError(f"Command {exe_file} not found. Please build RoDe first.")

cmd = f"{exe_file} {args.data}"
result = subprocess.run(
    f"{cmd}",
    shell=True,
    text=True,
    capture_output=True,
    stdin=subprocess.DEVNULL,
    preexec_fn=os.setsid,
)

output = result.stdout
output = output.split(", ")
rode_time = float(output[-2])
sputnik_time = float(output[1])

print(f"[RoDe] Elapsed time: {rode_time / 10:.4f} ms")
print(f"[Sputnik] Elapsed time: {sputnik_time / 10:.4f} ms")


