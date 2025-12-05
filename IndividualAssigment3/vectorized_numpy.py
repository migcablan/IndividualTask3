import numpy as np
import time
import multiprocessing as mp


def benchmark_numpy(size):
    A = np.random.rand(size, size)
    B = np.random.rand(size, size)
    start = time.perf_counter()
    C = A @ B
    end = time.perf_counter()
    elapsed = end - start
    return elapsed


if __name__ == "__main__":
    cores = mp.cpu_count()
    print(f"Available cores for BLAS/NumPy: {cores}")
    SIZES = [32, 64, 128, 256, 512, 1024, 2048]
    for n in SIZES:
        t = benchmark_numpy(n)
        print(f"NumPy vectorized {n}x{n}: {t:.4f} s")
