import subprocess
import os
import argparse
from typing import Dict, List


from numpy import isin
import torch


base_mtd_cmds = {
    "Voltrix": "python bm_voltrix.py",
    "DTC-SPMM": "python bm_dtc.py",
    "TC-GNN": "./tcgnn",
    "GE-SPMM": "./gespmm",
    "cuSPARSE": "python bm_sparse.py",
    "RoDe": "python bm_rode.py",
    "Sputnik": "python bm_sputnik.py",
}

feature_dims = [256, 512, 1024]
do_not_reorder = ["TC-GNN", "GE-SPMM", "cuSPARSE", "RoDe", "Sputnik"]
only_do_reorder = ["DTC-SPMM", "Voltrix"]

time_pattern = {
    "Voltrix": "[Voltrix] time: ",
    "DTC-SPMM": "[DTC-SPMM] Elapsed time: ",
    "TC-GNN": "[TC-GNN] Kernel time: ",
    "GE-SPMM": "[GE-SPMM] Embedding time: ",
    "cuSPARSE": "[cuSPARSE] Elapsed time: ",
    "RoDe": "[RoDe] Elapsed time: ",
    "Sputnik": "[Sputnik] Elapsed time: ",
}



def append_mtd_cmds(method, **kwargs):
    assert method in base_mtd_cmds.keys()

    if method in ["RoDe", "Sputnik"]:
        rode_home = os.getenv("RODE_HOME")
        feat_dim = kwargs["feat_dim"]

        assert isinstance(rode_home, str), "RODE_HOME environment variable not set."
        assert feat_dim in feature_dims

        return f"--rode_home={rode_home} --feat_dim={feat_dim}"
    
    return ""




def get_npz_files(directory):
    files = [
        os.path.splitext(f)[0]
        for f in os.listdir(directory)
        if f.endswith(".npz") and os.path.isfile(os.path.join(directory, f))
    ]
    return files


def bench(args):
    dataset_folder = args.datasets_folder
    proj_folder = args.proj_folder
    npz_files = list(
        filter(lambda f: "reorder" not in f, get_npz_files(args.datasets_folder))
    )
    output_file = args.output_file

    if not args.append and os.path.exists(output_file):
        os.remove(output_file)

    with open(output_file, "a") as f:
        f.write("Method,Dataset,FeatDim,Reorder,Time (ms)\n")

    print("Generating profiling plan...")
    do_not_reorder_methods = list(
        filter(lambda x: x in do_not_reorder, base_mtd_cmds.keys())
    )
    reorder_methods = list(filter(lambda x: x in only_do_reorder, base_mtd_cmds.keys()))
    print("Methods that do not require reordering: ", do_not_reorder_methods)
    print("Methods that require reordering: ", reorder_methods)

    print(f"Collecting total graph dataset: {npz_files}")
    for npz_file in npz_files:
        for feat_dim in feature_dims:
            print(
                "====================================================================="
            )
            print(f"Generating graph data for {npz_file} with {feat_dim} features...")
            assert subprocess.run(
                f"python graph_gen.py --base_folder={dataset_folder} --data_name={npz_file} --num_feat={feat_dim}",
                shell=True,
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            ).returncode == 0, "Error in generating graph data."

            for method in do_not_reorder_methods:
                print(f"Profiling {method} kernel...")
                result = subprocess.run(
                    f"{base_mtd_cmds[method]} {append_mtd_cmds(method, feat_dim=feat_dim)}",
                    shell=True,
                    text=True,
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                )
                time = "NAN"
                if time_pattern[method] in result.stdout:
                    time = result.stdout.split(time_pattern[method])[1].split(" ms")[0]

                with open(output_file, "a") as f:
                    f.write(f"{method},{npz_file},{feat_dim},N,{time}\n")

                print(f"{method} kernel time: {time} ms")

            # do reorder
            print(f"Reordering graph data for {npz_file} with {feat_dim} features...")
            assert subprocess.run(
                f"python graph_gen.py --base_folder={dataset_folder} --data_name={npz_file} --num_feat={feat_dim} --reorder",
                shell=True,
                text=True,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            ).returncode == 0, "Error in reordering graph data."

            for method in reorder_methods:
                print(f"Profiling {method} kernel...")
                result = subprocess.run(
                    f"{base_mtd_cmds[method]} {append_mtd_cmds(method, feat_dim=feat_dim)}",
                    shell=True,
                    text=True,
                    capture_output=True,
                    stdin=subprocess.DEVNULL,
                    preexec_fn=os.setsid,
                )

                time = "NAN"
                if time_pattern[method] in result.stdout:
                    time = result.stdout.split(time_pattern[method])[1].split(" ms")[0]

                with open(output_file, "a") as f:
                    f.write(f"{method},{npz_file},{feat_dim},Y,{time}\n")

                print(f"{method} kernel time: {time} ms")


if __name__ == "__main__":
    x = torch.randn((1,)).cuda()  # for occupy the GPU

    parser = argparse.ArgumentParser(description="Benchmark all methods.")

    parser.add_argument(
        "--datasets_folder",
        type=str,
        default=os.getenv("DATASET_PATH"),
    )
    parser.add_argument("--output_file", type=str, default="results.csv")
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help="Append to the output file.",
    )
    parser.add_argument("--proj_folder", type=str, default="./", help="Project folder.")

    args = parser.parse_args()

    bench(args)