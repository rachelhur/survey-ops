import json

from importlib import resources
import json

def load_internal_mapping(filename):
    # This looks inside survey_ops/core/mappings/ even if the package is installed
    pkg_path = resources.files("data.lookups").joinpath(filename)
    with pkg_path.open("r") as f:
        return json.load(f)

class Config:
    def __init__(self, config_path, config_dict=None):
        self.config_path = config_path
        if dict is not None:
            with open(config_path, 'w') as f:
                json.dump(config_dict, f)

        with open(config_path, 'r') as f:
            self._data = json.load(f)

    def get(self, path=None, default=None):
        if path is None:
            return self._data

        node = self._data
        for key in path.split("."):
            if not isinstance(node, dict):
                return default
            node = node.get(key, default)
            if node is default:
                return default
        return node
    
    def set(self, path, value):
        keys = path.split(".")
        node = self._data

        # walk to parent
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]

        node[keys[-1]] = value

    def save(self, path=None, indent=2):
        path = path or self.config_path
        with open(path, "w") as f:
            json.dump(self._data, f, indent=indent)

import json
import os

class Config:
    def __init__(self, config_path, data):
        self.config_path = config_path
        self._data = {}
        if os.path.exists(config_path):
            with open(config_path) as f:
                self._data = json.load(f)

    def get(self, path=None, default=None):
        if path is None: return self._data
        node = self._data
        for key in path.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node
    
    def set(self, path, value):
        keys = path.split(".")
        node = self._data
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = value

    def merge_dict(self, data):
        """Deep merge a dictionary into existing config."""
        for key, value in data.items():
            if isinstance(value, dict):
                # Recursively merge
                node = self._data.setdefault(key, {})
                self._merge_recursive(node, value)
            else:
                self._data[key] = value

    def _merge_recursive(self, target, source):
        for k, v in source.items():
            if isinstance(v, dict):
                target[k] = self._merge_recursive(target.get(k, {}), v)
            else:
                target[k] = v
        return target

    def save(self, indent=2):
        with open(self.config_path, "w") as f:
            json.dump(self._data, f, indent=indent)


import json
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

def load_global_config(config_path: Path):
    global_cfg = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            global_cfg = json.load(f)
        # global_cfg = Config(config_path)
    else:
        raise NotImplementedError("Will eventually write a function which creates the necessary global_config.json")
    return global_cfg