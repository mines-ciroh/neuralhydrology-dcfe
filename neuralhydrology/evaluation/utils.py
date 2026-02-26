import itertools
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch
import pandas as pd
from ruamel.yaml import YAML


def load_basin_id_encoding(run_dir: Path) -> Dict[str, int]:
    id_to_int_file = run_dir / "train_data" / "id_to_int.yml"
    if id_to_int_file.is_file():
        with id_to_int_file.open("r") as fp:
            yaml = YAML(typ="safe")
            id_to_int = yaml.load(fp)
        return id_to_int

    else:
        id_to_int_file = run_dir / "train_data" / "id_to_int.p"
        if id_to_int_file.is_file():
            with id_to_int_file.open("rb") as fp:
                id_to_int = pickle.load(fp)
            return id_to_int
        else:
            raise FileNotFoundError(f"No id-to-int file found in {id_to_int_file.parent}. "
                                    "Looked for (new) yaml file or (old) pickle file")



def metrics_to_dataframe(results: dict, metrics: Iterable[str], targets: Iterable[str]) -> pd.DataFrame:
    """Extract all metric values from result dictionary and convert to pandas.DataFrame

    Parameters
    ----------
    results : dict
        Dictionary, containing the results of the model evaluation as returned by the `Tester.evaluate()`.
    metrics : Iterable[str]
        Iterable of metric names (without frequency suffix).
    targets : Iterable[str]
        Iterable of target variable names.

    Returns
    -------
    A basin indexed DataFrame with one column per metric. In case of multi-frequency runs, the metric names contain
    the corresponding frequency as a suffix.
    """
    metrics_dict = defaultdict(dict)
    for basin, basin_data in results.items():
        for freq, freq_results in basin_data.items():
            for target, metric in itertools.product(targets, metrics):
                metric_key = metric
                if len(targets) > 1:
                    metric_key = f"{target}_{metric}"
                if len(basin_data) > 1:
                    # For multi-frequency runs, metrics include a frequency suffix.
                    metric_key = f"{metric_key}_{freq}"
                if metric_key in freq_results.keys():
                    metrics_dict[basin][metric_key] = freq_results[metric_key]
                else:
                    # in case the current period has no valid samples, the result dict has no metric-key
                    metrics_dict[basin][metric_key] = np.nan

    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.index.name = "basin"

    return df

def tensor_dict_to_numpy(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively convert torch tensors in a dict to CPU numpy arrays.

    Parameters
    ----------
    data : Dict[str, Any]
        Dictionary that may contain torch tensors and nested dictionaries.

    Returns
    -------
    Dict[str, Any]
        Dictionary with torch tensors converted to numpy arrays.
    """
    converted = {}
    for key, value in data.items():
        if isinstance(value, dict):
            converted[key] = tensor_dict_to_numpy(value)
        elif torch.is_tensor(value):
            converted[key] = [value.detach().cpu().numpy()]
        else:
            converted[key] = value
    return converted

