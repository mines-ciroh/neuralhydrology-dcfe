from pathlib import Path
import matplotlib.pyplot as plt
from neuralhydrology.nh_run import start_run, eval_run
import pickle

test = False
plot = True
run_dir_dynamic = "/home/adam/dev/repos/neuralhydrology-dcfe/runs/example_shm_neural_ode_dynamic_2603_081358" # Modify this to your specific run path!
if test:
    eval_run(run_dir=Path(run_dir_dynamic), period="test", gpu=-1)
if plot:
    basin = '02177000' # Modify to '02349900' if you'd like to see the results for that basin.
    with open(Path(run_dir_dynamic) / "test" / "model_epoch002" / "test_results.p", "rb") as fp:
        dynamic_shm_predictions = pickle.load(fp)
    with open(Path(run_dir_dynamic) / "test" / "model_epoch002" / "test_all_output.p", "rb") as fp:
        dynamic_shm_all_output = pickle.load(fp)
        
    fig, ax = plt.subplots(figsize=(16,8), nrows=1, ncols=1)
    dates = dynamic_shm_predictions[basin]["1D"]["xr"]["datetime"]
    observed = dynamic_shm_predictions[basin]["1D"]["xr"]["QObs(mm/d)_obs"]
    dynamic_nse = dynamic_shm_predictions[basin]["1D"]["NSE"]
    ax.plot(dates,
            observed,
            label="Observed",
            linewidth=2)

    ax.plot(dates,
            dynamic_shm_predictions[basin]["1D"]["xr"]["QObs(mm/d)_sim"],
            label=f"Dynamic (Neural ODE), NSE: {dynamic_nse:.4f}",
            linestyle="--",
            alpha=0.75)

    ax.set_title(f"Basin {basin}: Dynamic (Neural ODE) vs. Observed, 2 Epochs")
    ax.set_ylabel("Q [mm/d]")
    ax.set_xlabel("Year")
    plt.legend()
    plt.show()
    plt.grid()