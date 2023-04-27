
import os
import math
import random
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

from tqdm import tqdm


# Thresholding function
def ApplyThresholds(x: torch.Tensor(), low:int = -80., high: int = None):      # A lot faster using native functions
    """Applies a threshold on a CHW tensor by limiting values to a minimum (low) and a maximum (high). 
    Also assigns NaN values to low. Returns a new tensor with thresholded values.
    
    Args:
        x (torch.Tensor()): a CHW image format tensor.
        low (int)(optional): the minimum value to which any value below will be assigned. Defaults to None.
        high (int)(optional): the maximum value to which any value above will be assigned. Defaults to None.
    """
    low_tensor = torch.full(x.shape, fill_value=low)
    if high:
        high_tensor = torch.full(x.shape, fill_value=high)
        return torch.minimum(torch.maximum(torch.nan_to_num(x, nan=low),
                                           low_tensor),
                             high_tensor)
    else:
        return torch.maximum(torch.nan_to_num(x, nan=low),
                             low_tensor)
    

# A function to remove the signal of fishes (not included in the classification)
def ApplyMask(x: torch.Tensor(), mask: torch.Tensor(), low: torch.float = -80.):
    """Applies a mask to a given tensor x. 
    A mask is a tensor of same shape as x and containing only 1 and torch.nan as values.

    Args:
        x (torch.Tensor()): the input tensor.
        mask (torch.Tensor()): the corresponding mask.
        low (torch.float)(optional): the value to which resulting nan values are converted. Defaults to -80dB.
    """
    assert x.shape == mask.shape, "Data and Mask should have same shape"
    return torch.nan_to_num(torch.add(x, mask), nan=low)


# Retrieving data and returning in the right format
def getData(dir_path: str or PosixPath,
            depthmax: int,
            depthmin: int = 0):
    """Retrieves HDF5 echointegration and classification files from directory.
    Crops the depth of the data.
    Returns data with a few modifications for further use in segmentation.

    Args:
        data_dir (str): the directory in which the HDF5 are located.
        depthmax (int): the maximum depth to be kept when croping data.
        depthmin (int)(optional): the minimum depth to be kept when croping data.
    """
    # Finding the data directory
    os.chdir(dir_path)
    print(f"Retrieving data from: {os.getcwd()}...")

    # Reading files
    for filename in os.listdir():
        if filename == "Echointegration.mat":
            EI = h5py.File(filename)
        elif filename.split("_")[-1] == "6groups.mat":
            CLASSES = h5py.File(filename)
        else: ()
    try:
        EI.keys()
    except:
        raise Exception(f"Issue while retrieving EI file in {dir_path}")
    try:
        CLASSES.keys()
    except:
        raise Exception(f"Issue while retrieving CLASSES file in {dir_path}")
    print(f"Files retrieved: {EI, CLASSES}")

    # Getting matrixes from HDF5 files, converting to tensors and correct format
    # Data

    Sv_surface = torch.from_numpy(np.array(EI.get('Sv_surface'))[:, :, depthmin:depthmax]).permute(0, 2, 1)
    Sv_surface = torch.nan_to_num(Sv_surface, nan=-80.)

    Mask_clean_FISH = torch.from_numpy(np.array(EI.get('Mask_clean_FISH'))[:, :, depthmin:depthmax]).permute(0, 2, 1) * 0 - 200
    Mask_clean_FISH = torch.nan_to_num(Mask_clean_FISH, nan= 0)

    Sv_surface = ApplyMask(Sv_surface, Mask_clean_FISH)
    Sv_surface = ApplyThresholds(Sv_surface, low=-80., high=-50.)

    minSv, maxSv = torch.min(Sv_surface), torch.max(Sv_surface)
    Sv_surface_01 = (Sv_surface - minSv) / (maxSv - minSv)

    # Label
    Ind_best_class = torch.nan_to_num(
        input=torch.from_numpy(np.array(CLASSES.get('ind_Best_class'))[:, depthmin:depthmax]).permute(1, 0),
        nan=-2.) + 2.
    
    Ind_best_class = Ind_best_class.to(torch.long)
    
    EI.close()
    CLASSES.close()

    print(f"\nSv_surface tensor shape: {Sv_surface_01.shape} -> [color_channels (4), height, width] (CHW)")
    print(f"Best_class tensor shape: {Ind_best_class.shape} -> [height, width]")

    return Sv_surface_01, Ind_best_class


def layoutRandomGrid(img: torch.Tensor(), 
                     grid_dx: int = 36, 
                     grid_dy: int = 36,
                     shift: int = 0,
                     offset: tuple = (0, 0, 0, 0)):
    """Creates a grid layout for a CHW tensor, with rectangular patches of given dimensions.
    
    Args:
        x (torch.Tensor()): the CHW image tensor on which we the grid should be laid out.
        patch_height (int)(optional): the height of the patches. Defaults to 36.
        patch_width (int)(optional): the width of the patches. Defaults to 36.
        shit (int)(optional): range of the random shift applied to each center. Defaults to 0.
        offset (tuple)(optional): the distance to keep between the edges of the grid and those of the image. Is read (top, bottom, left, right). Defaults to (0, 0, 0, 0). 
    """
    height, width = len(img[0, :, 0]), len(img[0, 0, :])
    k_vertical = (height - offset[0] - offset[1]) // grid_dy
    k_horizontal = (width - offset[2] - offset[3]) // grid_dx
    
    Layout_Grid = []

    for i in range(k_vertical):
        y = offset[0] + int(grid_dy*(i + 1/2))
        for j in range(k_horizontal):
            x = offset[2] + int(grid_dx*(j + 1/2))
            Layout_Grid.append((x + random.randint(-shift, shift), y + random.randint(-shift, shift)))
    
    return Layout_Grid

# crop and save a copy of the data in dataset directory. For maks, apply a custom torch.transforms to obtain a regular format mask


def cropNcopy(ndim:int, image_tensor:torch.Tensor(), center:tuple, dx:int = 224, dy:int = 224, transform=None):
    x, y = center
    assert ndim == 2 or ndim == 3, "ndmin should be 2 or 3"
    if ndim == 2:
        if transform:
            return transform(image_tensor[y-dy//2:y+dy//2, x-dx//2:x+dx//2])
        else:
            return image_tensor[y-dy//2:y+dy//2, x-dx//2:x+dx//2]
    else:
        if transform:
            return transform(image_tensor[:, y-dy//2:y+dy//2, x-dx//2:x+dx//2])
        else:
            return image_tensor[:, y-dy//2:y+dy//2, x-dx//2:x+dx//2]
    
class Im2Mask:
    def __init__(self, target_channels):
        self.target_channels = target_channels

    def __call__(self, image):      
        l = []
        for i in range(9):
            l.append(torch.where(image==float(i), 1., 0.).unsqueeze(0))
        return torch.cat(l, axis=0)
    

def saveSquaresFromGrid(Centers:np.ndarray,
                    patch_width:int,
                    patch_height:int, 
                    image_tensor:torch.Tensor(),
                    masks:torch.Tensor(),
                    dataset_dir:str or PosixPath,
                    bin_mask:bool = False):
    
    ei_filepath = Path(os.getcwd())
    ei = os.path.basename(ei_filepath)
    leg_filepath = ei_filepath.parent.absolute()
    leg = os.path.basename(leg_filepath)
    save_dir_data = dataset_dir / "echos"
    save_dir_masks = dataset_dir / "masks"

    print(f"Collecting patches from {ei} of {leg}...")
    for i in range(len(Centers)):

            save_path_data = save_dir_data / f"ABRACOS1_{leg}_{ei}_echopatch_{i}.pth"
            save_path_masks = save_dir_masks / f"ABRACOS1_{leg}_{ei}_maskpatch_{i}.pth"

            center = Centers[i]
            # Saving data
            torch.save(cropNcopy(ndim=3, 
                                 image_tensor=image_tensor,
                                 center=center, 
                                 dx=patch_width,
                                 dy=patch_height),
                       save_path_data)
            
            # Saving masks
            if bin_mask:
                im2mask = Im2Mask(target_channels=9)
            else:
                im2mask = None
            torch.save(cropNcopy(ndim=2,
                                image_tensor=masks,
                                center=center,
                                dx=patch_width,
                                dy=patch_height,
                                transform=im2mask), 
                    save_path_masks)
