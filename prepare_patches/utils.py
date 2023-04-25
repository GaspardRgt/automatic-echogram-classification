import os
from pathlib import Path
import torch

def getPatchMaskPath(patch_data_path: str or PosixPath):
    name = os.path.basename(os.path.normpath(patch_data_path))
    l = name.split("_")
    l[3] = "maskpatch"
    parent = Path(os.path.dirname(patch_data_path))
    grandpa = Path(os.path.dirname(parent))
    son = "_".join(l)
    return (grandpa / "masks") / son

def get_patchXY(patch_path):
    mask_path = getPatchMaskPath(patch_path)
    X = torch.load(patch_path)
    Y = torch.load(mask_path)
    return X, Y
