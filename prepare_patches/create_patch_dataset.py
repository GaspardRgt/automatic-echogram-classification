# Creating a first dataset based only on AB1_LEG1_EI01
import torch
import numpy as np

import os
from pathlib import Path

import argparse

import get_data


parser = argparse.ArgumentParser()

parser.add_argument("--data_path",
                    help="path from which to retrieve data")

parser.add_argument("--dest_path",
                    help="path in which to save formated data")

parser.add_argument("--depthmin",
                    default=0)

parser.add_argument("--depthmax",
                    default=280)

parser.add_argument("--grid_dx",
                    default=254)

parser.add_argument("--grid_dy",
                    default=270)

parser.add_argument("--random_shift",
                    default=15)

parser.add_argument("--grid_offset",
                    default=(0, 0, 0, 0))

parser.add_argument("--patch_height",
                    default=224)

parser.add_argument("--patch_width",
                    default=224)

parser.add_argument("--bin_mask",
                    default=False)

args = parser.parse_args()

# Creating directories
DATA_PATH = Path(args.data_path)
os.chdir(DATA_PATH)

DEST_PATH = Path(args.dest_path)
DATASET_PATH = DEST_PATH / "dataset"

RAW_PATH = DATASET_PATH / "echos"
ANNO_PATH = DATASET_PATH / "masks"

DATASET_PATH.mkdir(parents=True, exist_ok=True)
RAW_PATH.mkdir(parents=True, exist_ok=True)
ANNO_PATH.mkdir(parents=True, exist_ok=True)

for path in os.listdir(DATA_PATH):
    os.chdir(DATA_PATH)
    DATA_SUBDIR_PATH = path

    # Retrieving data
    Sv_surface, Ind_best_class = get_data.getData(dir_path=DATA_SUBDIR_PATH,
                                         depthmin=args.depthmin,
                                         depthmax=args.depthmax)

    # Creating grid layout
    Centers = get_data.layoutRandomGrid(Sv_surface,
                        grid_dx=args.grid_dx,
                        grid_dy=args.grid_dy,
                        shift=args.random_shift,
                        offset=args.grid_offset)

    # Saving patches to data and masks directories
    get_data.saveSquaresFromGrid(Centers=Centers,
                        patch_width=224,
                        patch_height=224,
                        image_tensor=Sv_surface,
                        masks=Ind_best_class,
                        dataset_dir=DATASET_PATH,
                        bin_mask=args.bin_mask)
