import threading

class GPUAllocator:
    def __init__(self, max_gpus=4):
        self.max_gpus = max_gpus
        self.gpu_usage = {gpu_id: 0 for gpu_id in range(max_gpus)}
        self.lock = threading.Lock()

    def allocate_gpu(self):
        with self.lock:
            # Find the GPU with the lowest usage
            best_gpu = min(self.gpu_usage, key=self.gpu_usage.get)
            self.gpu_usage[best_gpu] += 1
            return best_gpu

    def release_gpu(self, gpu_id):
        with self.lock:
            self.gpu_usage[gpu_id] = max(0, self.gpu_usage[gpu_id] - 1)