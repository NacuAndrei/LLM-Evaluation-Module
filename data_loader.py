import json

class DataLoader:
    @staticmethod
    def load_dataset(absolute_path: str):
        with open(absolute_path, "r") as json_file:
            return json.load(json_file)