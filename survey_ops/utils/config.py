import json

from importlib import resources
import json

def load_internal_mapping(filename):
    # This looks inside survey_ops/core/mappings/ even if the package is installed
    pkg_path = resources.files("data.lookups").joinpath(filename)
    with pkg_path.open("r") as f:
        return json.load(f)

class Config:
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path) as f:
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

