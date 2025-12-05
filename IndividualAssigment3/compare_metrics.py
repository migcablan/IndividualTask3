import time
import csv
import multiprocessing as mp

from sequential import generate_matrix, matmul_basic
from parallel import matmul_parallel
import numpy as np


def benchmark_all(size, n_procs):
    A = generate_matrix(size, size)
    B = generate_matrix(size, size)

    # --- Basic ---
    start = time.perf_counter()
    C_basic = matmul_basic(A, B)
    end = time.perf_counter()
    t_basic = end - start

    # --- Parallel ---
    start = time.perf_counter()
    C_par = matmul_parallel(A, B, n_procs=n_procs)
    end = time.perf_counter()
    t_par = end - start

    # --- NumPy vectorized ---
    A_np = np.array(A, dtype=float)
    B_np = np.array(B, dtype=float)
    start = time.perf_counter()
    C_vec = A_np @ B_np
    end = time.perf_counter()
    t_vec = end - start

    return t_basic, t_par, t_vec


if __name__ == "__main__":
    cores = mp.cpu_count()
    print(f"Available cores: {cores}")

    SIZES = [32, 64, 128, 256, 512, 1024, 2048]

    with open("mm_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "size",
            "version",
            "processes",
            "time_s",
            "speedup",
            "efficiency",
            "cores_total",
        ])

        for n in SIZES:
            for p in [2, 4, cores]:
                print(f"\nMatrix size: {n}x{n}, parallel processes: {p}")
                t_basic, t_par, t_vec = benchmark_all(n, p)

                speedup_par = t_basic / t_par
                efficiency_par = speedup_par / p
                speedup_vec = t_basic / t_vec

                print(f"Basic time:        {t_basic:.4f} s")
                print(f"Parallel time:     {t_par:.4f} s")
                print(f"NumPy time:        {t_vec:.4f} s")
                print(f"Parallel speedup:  {speedup_par:.2f}x")
                print(f"Parallel efficiency: {efficiency_par:.2f}")
                print(f"NumPy speedup:     {speedup_vec:.2f}x")
                print(f"Resources: processes={p}, total_cores={cores}")

                # Basic
                writer.writerow([
                    n,
                    "basic",
                    1,
                    t_basic,
                    1.0,
                    1.0,
                    cores,
                ])

                # Parallel
                writer.writerow([
                    n,
                    "parallel",
                    p,
                    t_par,
                    speedup_par,
                    efficiency_par,
                    cores,
                ])

                # NumPy (vectorized)
                writer.writerow([
                    n,
                    "numpy",
                    1,
                    t_vec,
                    speedup_vec,
                    "",
                    cores,
                ])
