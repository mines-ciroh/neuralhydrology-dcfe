from pathlib import Path
import torch
from neuralhydrology.nh_run import start_run

# by default we assume that you have at least one CUDA-capable NVIDIA GPU
if torch.cuda.is_available():
    print("GPU")
    start_run(config_file=Path("/home/adam/dev/repos/neuralhydrology-dcfe/examples/08-SHM/Exp1_SHM_CAMELS_HPC.yml"))