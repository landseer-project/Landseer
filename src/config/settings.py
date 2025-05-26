import os
import torch
from dataclasses import dataclass, field

@dataclass
class Settings:
    data_dir: str = "./data"
    logs_dir: str = "./logs" 
    output_dir: str = "./output"
    use_gpu: bool = True

    def __post_init__(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        if not os.path.exists(self.logs_dir):
            os.makedirs(self.logs_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if self.use_gpu and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"