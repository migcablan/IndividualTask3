import os
import psutil
import multiprocessing as mp


def get_num_cores():
    return mp.cpu_count()


def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_cpu_percent(interval=0.5):
    return psutil.cpu_percent(interval=interval)
