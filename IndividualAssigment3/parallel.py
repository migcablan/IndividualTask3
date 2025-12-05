import random
import time
import multiprocessing as mp


def generate_matrix(n, m):
    return [[random.random() for _ in range(m)] for _ in range(n)]


def matmul_worker(args):
    A_chunk, B = args
    m = len(A_chunk[0])
    p = len(B[0])
    C_chunk = []
    for i in range(len(A_chunk)):
        row = [0.0] * p
        for k in range(m):
            aik = A_chunk[i][k]
            for j in range(p):
                row[j] += aik * B[k][j]
        C_chunk.append(row)
    return C_chunk


def matmul_parallel(A, B, n_procs=None):
    if n_procs is None:
        n_procs = mp.cpu_count()
    n = len(A)
    chunk_size = (n + n_procs - 1) // n_procs
    chunks = []
    for i in range(0, n, chunk_size):
        chunks.append(A[i:i + chunk_size])

    with mp.Pool(processes=n_procs) as pool:
        C_chunks = pool.map(matmul_worker, [(chunk, B) for chunk in chunks])

    C = []
    for chunk in C_chunks:
        C.extend(chunk)
    return C


def benchmark_parallel(size, n_procs=None):
    A = generate_matrix(size, size)
    B = generate_matrix(size, size)
    start = time.perf_counter()
    C = matmul_parallel(A, B, n_procs=n_procs)
    end = time.perf_counter()
    elapsed = end - start
    return elapsed


if __name__ == "__main__":
    SIZES = [32, 64, 128, 256, 512, 1024, 2048]
    for n in SIZES:
        for p in [2, 4, mp.cpu_count()]:
            t = benchmark_parallel(n, n_procs=p)
            print(f"Parallel Python {n}x{n} with {p} processes: {t:.4f} s")
