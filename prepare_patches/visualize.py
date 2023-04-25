
# Visualizing functions
import os
import random
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import colors
from matplotlib import cm

from prepare_patches.utils import getPatchMaskPath, get_patchXY

def plot_echogram_with_contrast(x, xmin, xmax, height = 280, ratio = 3.5, size=6):
    image_tensor = x[:3, :height, xmin:xmax]

    transform = transforms.ToPILImage()
    img = transform(pic=image_tensor)

    plt.figure(figsize=(size, size))

    # Resizing : we want a particular W/H ratio
    width = xmax - xmin
    r = width / height
    alpha = r / ratio
    img = img.resize(size=(width, int(height*alpha)),
                     resample=0)

    plt.axis("off")
    plt.imshow(img)

    plt.title(f"Echogram with 3 lowest freq | xmin: {xmin} | size: {xmax-xmin} pings x {height//2}m")


def plot_echoclasses(x: np.ndarray, xmin, xmax, height=280, color_display : str = "complete", ratio = 3.5, size=6):
    # Choose cmap depending on color_display
    if color_display == "complete":
        color_list = np.array([(255,255,255), (0,0,0), (130,130,130), (0,0,128), (28,134,238), (255,21,147), (0,204,0), (0,100,0), (255,184,16)]) / 255
    elif color_display == "classes only":
        color_list = np.array([(255,255,255), (130,130,130), (130,130,130), (0,0,128), (28,134,238), (255,21,147), (0,204,0), (0,100,0), (255,184,16)]) / 255
    
    cmap = colors.ListedColormap(color_list)

    numpy_image = x / 8
    img = Image.fromarray(np.uint8(cmap(numpy_image)[:, xmin:xmax, :-1]*255))

    fig = plt.figure(figsize=(size, size))

    bounds = np.arange(10)
    norm = colors.BoundaryNorm(bounds, cmap.N)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Classes", shrink=0.3)

    # Resizing : we want a particular W/H ratio
    width = xmax - xmin
    r = width / height
    alpha = r / ratio
    img = img.resize(size=(width, int(height*alpha)),
                     resample=0)
    plt.axis("off")
    plt.imshow(img)

    plt.title(f"6 classes classification, ind_Best_class | xmin = {xmin} | size = {width} pings x {height//2}m")

def plot_patches(dataset_path, n=3, data_transform=None):
    os.chdir(dataset_path)
    raw_path = dataset_path / "echos"
    random_patch_paths = [(raw_path / path) for path in random.sample(os.listdir(raw_path), k=n)]

    for patch_path in random_patch_paths:
        X, Y = get_patchXY(patch_path)
        if data_transform:
            X, Y = data_transform(X, Y)
        fig, ax = plt.subplots(1, 2)
        # Plot data RGB
        image_tensor = X[:3, :, :]

        transform = transforms.ToPILImage()
        X_img = transform(pic=image_tensor)

        ax[0].imshow(X_img)
        ax[0].set_title(f"Echogram patch (3 lowest freqs)")
        ax[0].axis("off")

        # Plot masks
        color_list = np.array([(255,255,255), (0,0,0), (130,130,130), (0,0,128), (28,134,238), (255,21,147), (0,204,0), (0,100,0), (255,184,16)]) / 255
        cmap = colors.ListedColormap(color_list)

        numpy_image = Y / 8
        Y_img = Image.fromarray(np.uint8(cmap(numpy_image)[:, :, :-1]*255))

        bounds = np.arange(10)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Classes", shrink=0.3)

        ax[1].imshow(Y_img)
        ax[1].set_title("masks")
        ax[1].axis("off")

        fig.suptitle(f"Patch id: {os.path.basename(os.path.normpath(patch_path))}", fontsize=16)


def plot_patchesV2(dataset_path, n=3, data_transform=None):
    os.chdir(dataset_path)
    raw_path = dataset_path / "echos"
    random_patch_paths = [(raw_path / path) for path in random.sample(os.listdir(raw_path), k=n)]

    patch_path = random_patch_paths[0]
    img, mask = get_patchXY(patch_path)
    X, Y = img.unsqueeze(0), mask.unsqueeze(0)

    for patch_path in random_patch_paths[1:]:
        img, mask = get_patchXY(patch_path)
        X, Y = torch.concat((X, img.unsqueeze(0)), dim=0), torch.concat((Y, mask.unsqueeze(0)), dim=0)    

    if data_transform:
        X, Y = data_transform(X, Y)

    for k in range(X.shape[0]):  
        fig, ax = plt.subplots(1, 2)
        # Plot data RGB
        image_tensor = X[k, :3, :, :]

        trans = torchvision.transforms.ToPILImage()
        X_img = trans(image_tensor)

        ax[0].imshow(X_img)
        ax[0].set_title(f"Echogram patch (3 lowest freqs)")
        ax[0].axis("off")

        # Plot masks
        color_list = np.array([(255,255,255), (0,0,0), (130,130,130), (0,0,128), (28,134,238), (255,21,147), (0,204,0), (0,100,0), (255,184,16)]) / 255
        cmap = colors.ListedColormap(color_list)

        numpy_image = Y[k] / 8
        Y_img = Image.fromarray(np.uint8(cmap(numpy_image)[:, :, :-1]*255))

        bounds = np.arange(10)
        norm = colors.BoundaryNorm(bounds, cmap.N)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), label="Classes", shrink=0.3)

        ax[1].imshow(Y_img)
        ax[1].set_title("masks")
        ax[1].axis("off")

        fig.suptitle(f"Patch id: {os.path.basename(os.path.normpath(patch_path))}", fontsize=16)
