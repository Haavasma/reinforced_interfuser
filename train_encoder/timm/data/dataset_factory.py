from copy import copy, deepcopy
import os

from .dataset import IterableImageDataset, ImageDataset
from .carla_dataset import CarlaMVDetDataset
from typing import Tuple
from sklearn.model_selection import train_test_split


def _search_split(root, split):
    # look for sub-folder with name of split in root and use that if it exists
    split_name = split.split("[")[0]
    try_root = os.path.join(root, split_name)
    if os.path.exists(try_root):
        return try_root
    if split_name == "validation":
        try_root = os.path.join(root, "val")
        if os.path.exists(try_root):
            return try_root
    return root


def create_dataset(
    name,
    root,
    split="validation",
    search_split=True,
    is_training=False,
    batch_size=None,
    **kwargs
):
    name = name.lower()
    if name.startswith("tfds"):
        ds = IterableImageDataset(
            root,
            parser=name,
            split=split,
            is_training=is_training,
            batch_size=batch_size,
            **kwargs
        )
    else:
        # FIXME support more advance split cfg for ImageFolder/Tar datasets in the future
        kwargs.pop(
            "repeats", 0
        )  # FIXME currently only Iterable dataset support the repeat multiplier
        if search_split and os.path.isdir(root):
            root = _search_split(root, split)
        ds = ImageDataset(root, parser=name, **kwargs)
    return ds


def create_carla_dataset(
    root, eval_size=0.2, **kwargs
) -> Tuple[CarlaMVDetDataset, CarlaMVDetDataset]:
    train_ds = CarlaMVDetDataset(root, "det", **kwargs)
    route_frames = train_ds.route_frames
    eval_ds = deepcopy(train_ds)
    train, eval = train_test_split(route_frames, test_size=eval_size, random_state=42)

    train_ds.route_frames = train
    eval_ds.route_frames = eval

    return train_ds, eval_ds
