from pathlib import Path
from neuralhydrology.evaluation.tester import RegressionTester
from neuralhydrology.utils.config import Config

run_dir = Path("/home/adam/dev/repos/neuralhydrology-dcfe/examples/08-SHM/runs/test_SHM_2402_103807")

tester = RegressionTester(run_dir=run_dir, cfg=Config(run_dir / "config.yml"))

tester.evaluate(save_all_output=True)