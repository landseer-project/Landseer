# smart_gpu_allocator.py
import time
import threading
import pynvml

class GPUAllocator:
    def __init__(self, max_utilization=60, max_memory_mb=8000, cooldown_sec=10, verbose=False):
        self.max_util = max_utilization
        self.max_mem = max_memory_mb
        self.cooldown = cooldown_sec
        self.verbose = verbose

        self.lock = threading.Lock()
        pynvml.nvmlInit()
        self.num_gpus = pynvml.nvmlDeviceGetCount()

        self.last_assigned = [0.0] * self.num_gpus
        self.active = [False] * self.num_gpus
        self.next_idx = 0

    def _gpu_ok(self, idx):
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_util = util.gpu
        mem_used = mem.used / (1024 ** 2)  # MiB

        if self.verbose:
            print(f"[GPU {idx}] Util: {gpu_util}%, Mem: {mem_used:.0f} MiB, Active: {self.active[idx]}")

        return gpu_util < self.max_util and mem_used < self.max_mem

    def allocate_gpu(self):
        while True:
            with self.lock:
                start_idx = self.next_idx
                for _ in range(self.num_gpus):
                    idx = self.next_idx
                    now = time.time()

                    cooldown_ok = (now - self.last_assigned[idx]) > self.cooldown
                    util_ok = self._gpu_ok(idx)
                    idle = not self.active[idx]

                    if cooldown_ok and util_ok and idle:
                        self.last_assigned[idx] = now
                        self.active[idx] = True
                        self.next_idx = (idx + 1) % self.num_gpus
                        if self.verbose:
                            print(f"[GPU Allocator] Allocated GPU {idx}")
                        return idx

                    self.next_idx = (self.next_idx + 1) % self.num_gpus

            time.sleep(1)

    def release_gpu(self, idx):
        with self.lock:
            self.active[idx] = False
            if self.verbose:
                print(f"[GPU Allocator] Released GPU {idx}")

    def __del__(self):
        pynvml.nvmlShutdown()
