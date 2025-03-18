import os
import json


def save_metadata_exp(metadata, path, name):
    os.makedirs(path,exist_ok=True)
    metadata_file_name = f"{name}.json"
    metadata_path = os.path.join(path, metadata_file_name)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
