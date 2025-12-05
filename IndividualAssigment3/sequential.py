import random
import time


def generate_matrix(n, m):
    return [[random.random() for _ in range(m)] for _ in range(n)]


def matmul_basic(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(m):
            aik = A[i][k]
            for j in range(p):
                C[i][j] += aik * B[k][j]
    return C


def benchmark_basic(size):
    A = generate_matrix(size, size)
    B = generate_matrix(size, size)
    start = time.perf_counter()
    C = matmul_basic(A, B)
    end = time.perf_counter()
    elapsed = end - start
    return elapsed


if __name__ == "__main__":
    SIZES = [32, 64, 128, 256, 512, 1024, 2048]
    for n in SIZES:
        t = benchmark_basic(n)
        print(f"Basic Python {n}x{n}: {t:.4f} s")
