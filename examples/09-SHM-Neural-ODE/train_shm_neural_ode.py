from pathlib import Path
import matplotlib.pyplot as plt
from neuralhydrology.nh_run import start_run, eval_run
import pickle

# Start run begins a training run on the input configuration.
path_to_config = "/home/adam/dev/repos/neuralhydrology-dcfe/examples/09-SHM-Neural-ODE/2basin-test-dynamic.yml"
start_run(config_file= Path(path_to_config), gpu=-1) # -1 for now (cpu)