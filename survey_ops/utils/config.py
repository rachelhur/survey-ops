import json
import os

from pathlib import Path

def dict_to_nested(data):
    """Converts {'model.lr': 0.1} to {'model': {'lr': 0.1}}"""
    nested = {}
    for key, value in data.items():
        keys = key.split('.')
        d = nested
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return nested

def save_config(args=None, config_dict=None, outdir=None):
    """Saves the experiment arguments as a nested JSON."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Convert argparse Namespace to nested dict
    if args is not None:
        config_dict = dict_to_nested(vars(args))
    
    with open(out_path / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)