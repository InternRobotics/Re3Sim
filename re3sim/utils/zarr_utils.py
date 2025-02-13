import zarr
import numpy as np


def zarr_to_dict(zarr_file):
    root = zarr.open(zarr_file, mode="r")
    return {
        key: zarr_to_dict(value) if isinstance(value, zarr.Group) else np.array(value)
        for key, value in root.items()
    }


def dict_to_zarr(root, data):
    for key, value in data.items():
        if isinstance(value, dict):
            dict_to_zarr(root.create_group(key), value)
        else:
            root[key] = value
