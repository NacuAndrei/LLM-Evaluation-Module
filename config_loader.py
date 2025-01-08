from ruamel.yaml import YAML
import os

class ConfigLoader:
    @staticmethod
    def load_config() -> dict:
        yaml_config_file = YAML(typ="safe")
        with open(os.environ["CONFIG_PATH"], "r") as file:
            return yaml_config_file.load(file)
