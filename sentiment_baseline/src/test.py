from dataset import test_loader
from train import trainer
import runtime_config
import json, os

cfg = runtime_config.CFG
output_dir = cfg['dirs']['output']

test_results = trainer.evaluate(eval_dataset=test_loader.dataset, metric_key_prefix="test")
print(f"Evaluation Results: {test_results}")

with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
    json.dump(test_results, f, indent=2)