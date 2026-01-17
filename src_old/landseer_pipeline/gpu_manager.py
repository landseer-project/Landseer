import pynvml
import time
from typing import Optional, List
import logging

class GPUManager:
    def __init__(self, max_temp: float = 60.0, cooldown_time: int = 300):
        self.max_temp = max_temp
        self.cooldown_time = cooldown_time
        self.gpu_states = {}
        self._init_nvml()
        
    def _init_nvml(self):
        try:
            pynvml.nvmlInit()
            self.device_count = pynvml.nvmlDeviceGetCount()
            for i in range(self.device_count):
                self.gpu_states[i] = {
                    'in_use': False,
                    'last_used': 0,
                    'temperature': 0
                }
        except Exception as e:
            logging.error(f"Failed to initialize NVML: {e}")
            raise

    def get_available_gpu(self) -> Optional[int]:
        """Get the least utilized GPU that is below temperature threshold."""
        current_time = time.time()
        
        for i in range(self.device_count):
            if self.gpu_states[i]['in_use']:
                continue
                
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            self.gpu_states[i]['temperature'] = temp
            
            if temp < self.max_temp:
                # Check if GPU has cooled down
                if current_time - self.gpu_states[i]['last_used'] > self.cooldown_time:
                    self.gpu_states[i]['in_use'] = True
                    self.gpu_states[i]['last_used'] = current_time
                    return i
                    
        return None

    def release_gpu(self, gpu_id: int):
        """Release a GPU after use."""
        if 0 <= gpu_id < self.device_count:
            self.gpu_states[gpu_id]['in_use'] = False
            self.gpu_states[gpu_id]['last_used'] = time.time()

    def get_gpu_stats(self) -> List[dict]:
        """Get current stats for all GPUs."""
        stats = []
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            stats.append({
                'id': i,
                'temperature': temp,
                'memory_used': memory.used,
                'memory_total': memory.total,
                'gpu_utilization': utilization.gpu,
                'in_use': self.gpu_states[i]['in_use']
            })
        return stats

    def __del__(self):
        try:
            pynvml.nvmlShutdown()
        except:
            pass 