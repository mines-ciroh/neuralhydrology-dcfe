from pathlib import Path

from neuralhydrology.nh_run import start_run


def main():
    # Start run begins a training run on the input configuration.
    path_to_config = "/home/adam/dev/repos/neuralhydrology-dcfe/examples/09-SHM-Neural-ODE/2basin-test-dynamic.yml"
    # path_to_config = "/Users/danielmckenzie/Documents/Active_Research/Ziyu/neuralhydrology-dcfe/examples/09-SHM-Neural-ODE/2basin-test-dynamic.yml"
    start_run(config_file=Path(path_to_config), gpu=-1)  # -1 for now (cpu)


if __name__ == "__main__":
    main()
