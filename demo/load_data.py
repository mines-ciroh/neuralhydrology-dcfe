"""
This module loads the demo data. There are hardcoded paths and constants. If you want to add new data, you will need to modify this file.
"""
from pathlib import Path
import pickle
import pandas as pd

SHM_path = Path("demo/demo_data/SHM/")
LSTM_path = Path("demo/demo_data/LSTM/")
dCFE_path = Path("demo/demo_data/dCFE/")

def load_SHM_data():
    num_epochs = 10
    
    run_dir_dynamic = SHM_path / "demo_SHM_dynamic"
    run_dir_oracle = SHM_path / "demo_SHM_oracle_average"
    run_dir_operational = SHM_path / "demo_SHM_operational_average"
    
    # TODO: think we should consolidate predictions and all_output into a single structure
    dynamic_shm_data = {}
    oracle_shm_data = {}
    operational_shm_data = {}
    
    for epoch in range(1, num_epochs + 1):
        dynamic_shm_data["epoch" + str(epoch)] = {}
        oracle_shm_data["epoch" + str(epoch)] = {}
        operational_shm_data["epoch" + str(epoch)] = {}

        # Load Predictions
        with open(Path(run_dir_dynamic) / "test" / f"model_epoch{epoch:03d}" / "test_results.p", "rb") as fp:
            dynamic_shm_data["epoch" + str(epoch)]["predictions"] = pickle.load(fp)
        with open(Path(run_dir_oracle) / "test" / f"model_epoch{epoch:03d}" / "test_results.p", "rb") as fp:
            oracle_shm_data["epoch" + str(epoch)]["predictions"] = pickle.load(fp)
        with open(Path(run_dir_operational) / "test" / f"model_epoch{epoch:03d}" / "test_results.p", "rb") as fp:
            operational_shm_data["epoch" + str(epoch)]["predictions"] = pickle.load(fp)

        # Load Internal States and Parameters
        with open(Path(run_dir_dynamic) / "test" / f"model_epoch{epoch:03d}" / "test_all_output.p", "rb") as fp:
            dynamic_shm_data["epoch" + str(epoch)]["all_output"] = pickle.load(fp)
        with open(Path(run_dir_oracle) / "test" / f"model_epoch{epoch:03d}" / "test_all_output.p", "rb") as fp:
            oracle_shm_data["epoch" + str(epoch)]["all_output"] = pickle.load(fp)
        with open(Path(run_dir_operational) / "test" / f"model_epoch{epoch:03d}" / "test_all_output.p", "rb") as fp:
            operational_shm_data["epoch" + str(epoch)]["all_output"] = pickle.load(fp)
    
    df = pd.DataFrame({
        "SHM_Dynamic": dynamic_shm_data,
        "SHM_Oracle": oracle_shm_data,
        "SHM_Operational": operational_shm_data,
    })
    
    return df

def load_LSTM_data():
    return

def load_dCFE_data():
    return

if __name__ == "__main__":
    load_SHM_data()