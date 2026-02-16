import yaml, os

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)