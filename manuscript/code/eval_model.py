from pathlib import Path
from neuralhydrology.evaluation import metrics, get_tester
import pandas as pd
from neuralhydrology.utils.config import Config

run_dir = Path("/home/ziyu/neuralhydrology-dcfe/runs/full_basin_snow_dcfe_0304_175915")
run_config = Config(run_dir/"config.yml")

# create a tester instance and start evaluation
tester = get_tester(cfg=Config(run_dir / "config.yml"), run_dir=run_dir, period="test", init_model=True)
results = tester.evaluate(save_results=True, metrics=run_config.metrics)