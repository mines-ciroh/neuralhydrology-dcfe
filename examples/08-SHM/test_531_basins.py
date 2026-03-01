from pathlib import Path
from neuralhydrology.evaluation.evaluate import start_evaluation
from neuralhydrology.nh_run import eval_run

if __name__ == "__main__":
    run_dir = Path("/home/adam/dev/repos/neuralhydrology-dcfe/runs/EXP1_SHM_CAMELS_noSnow_HPC_Dynamic_2302_193830")
    eval_run(run_dir=run_dir, period="test")