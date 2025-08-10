# utils.py
# Helper functions (visualization, metrics, logger, etc.)

import matplotlib.pyplot as plt
import json
import os
import yaml

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def plot_top5(preds, labels, categories):
    # Visualize top-5 predictions for selected samples
    pass

def log_metrics(metrics, log_file="results/training.log", json_file="results/metrics.json"):
    # Print and save metrics
    for k, v in metrics.items():
        print(f"{k}: {v}")
    with open(log_file, "a") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    # Save to JSON
    if os.path.exists(json_file):
        with open(json_file, "r") as jf:
            data = json.load(jf)
    else:
        data = {}
    data.update(metrics)
    with open(json_file, "w") as jf:
        json.dump(data, jf, indent=2)
