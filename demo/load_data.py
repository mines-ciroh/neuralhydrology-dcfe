"""
This module loads the demo data. There are hardcoded paths and constants.
If you want to add new data, you will need to modify this file.
"""
from pathlib import Path
import pickle
import pandas as pd

shm_path = Path("demo/demo_data/shm/")
lstm_path = Path("demo/demo_data/lstm/")
dcfe_path = Path("demo/demo_data/dcfe/")

def load_shm_data():
    """Loads the SHM predictions, internal states, and parameters for GUI

    Returns:
        pd.DataFrame: Dataframe containing predictions and all_output
        for each hybrid model variant (Dynamic, Oracle, Operational)
        
        Each variant contains num_epochs epochs, which are the model epochs
        that were saved during training.
        
        Each epoch contains:
            - predictions: a dictionary containing the SHM predictions for each basin.
            - all_output: a dictionary containing the internal states and parameters for each basin.
    """
    num_epochs = 10

    run_dir_dynamic = shm_path / "demo_shm_dynamic"
    run_dir_oracle = shm_path / "demo_shm_oracle_average"
    run_dir_operational = shm_path / "demo_shm_operational_average"
    
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
        "shm_dynamic": dynamic_shm_data,
        "shm_oracle": oracle_shm_data,
        "shm_operational": operational_shm_data,
    })
    
    return df

def load_lstm_data():
    """Load LSTM predictions for GUI

    Returns:
        dict: Dictionary containing LSTM predictions for each epoch
    """
    num_epochs = 10
    run_dir_lstm = lstm_path
  
    lstm_data = {}
    for epoch in range(1, num_epochs + 1):
        lstm_data["epoch" + str(epoch)] = {}
     
        with open(Path(run_dir_lstm) / "test" / f"model_epoch{epoch:03d}" / "test_results.p", "rb") as fp:
            lstm_data["epoch" + str(epoch)] = pickle.load(fp)
            
    return lstm_data # LSTM has a different data structure than the hybrid models.

def load_dcfe_data():
    """Loads the dCFE predictions, internal states, and parameters for the GUI

    Returns:
        pd.DataFrame: Dataframe containing predictions and all_output
        for each hybrid model variant (Dynamic, Oracle, Operational)
        
        Each variant contains num_epochs epochs, which are the model epochs
        that were saved during training.
        
        Each epoch contains:
            - predictions: a dictionary containing the SHM predictions for each basin.
            - all_output: a dictionary containing the internal states and parameters for each basin.
    """
    num_epochs = 10
    run_dir_dynamic = dcfe_path / "demo_dcfe_dynamic"
    run_dir_oracle = dcfe_path / "demo_dcfe_oracle_average"
    run_dir_operational = dcfe_path / "demo_dcfe_operational_average"
    
    dynamic_dCFE_data = {}
    oracle_dCFE_data = {}
    operational_dCFE_data = {}
    
    for epoch in range(1, num_epochs + 1):
        dynamic_dCFE_data["epoch" + str(epoch)] = {}
        oracle_dCFE_data["epoch" + str(epoch)] = {}
        operational_dCFE_data["epoch" + str(epoch)] = {}

        # Load Predictions
        with open(Path(run_dir_dynamic) / "test" / f"model_epoch{epoch:03d}" / "test_results.p", "rb") as fp:
            dynamic_dCFE_data["epoch" + str(epoch)]["predictions"] = pickle.load(fp)
        with open(Path(run_dir_oracle) / "test" / f"model_epoch{epoch:03d}" / "test_results.p", "rb") as fp:
            oracle_dCFE_data["epoch" + str(epoch)]["predictions"] = pickle.load(fp)
        with open(Path(run_dir_operational) / "test" / f"model_epoch{epoch:03d}" / "test_results.p", "rb") as fp:
            operational_dCFE_data["epoch" + str(epoch)]["predictions"] = pickle.load(fp)

        # Load Internal States and Parameters
        with open(Path(run_dir_dynamic) / "test" / f"model_epoch{epoch:03d}" / "test_all_output.p", "rb") as fp:
            dynamic_dCFE_data["epoch" + str(epoch)]["all_output"] = pickle.load(fp)
        with open(Path(run_dir_oracle) / "test" / f"model_epoch{epoch:03d}" / "test_all_output.p", "rb") as fp:
            oracle_dCFE_data["epoch" + str(epoch)]["all_output"] = pickle.load(fp)
        with open(Path(run_dir_operational) / "test" / f"model_epoch{epoch:03d}" / "test_all_output.p", "rb") as fp:
            operational_dCFE_data["epoch" + str(epoch)]["all_output"] = pickle.load(fp)
    
    df = pd.DataFrame({
        "dcfe_dynamic": dynamic_dCFE_data,
        "dcfe_oracle": oracle_dCFE_data,
        "dcfe_operational": operational_dCFE_data,
    })
    return df

if __name__ == "__main__":
    # Debugging data loaders
    shm_data = load_shm_data()
    lstm_data = load_lstm_data()
    dcfe_data = load_dcfe_data()