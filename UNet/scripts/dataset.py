
# import the necessary packages
import torch
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, imagePaths, maskPaths, transforms):
        # store the image and mask filepaths, and augmentation transforms
        self.imagePaths = imagePaths
        self.maskPaths = maskPaths
        self.transforms = transforms

    def __len__(self):
        # return the number of total samples contained in the dataset
        return len(self.imagePaths)

    def __getitem__(self, idx):
        # grab the image path from the current index
        image = torch.load(self.imagePaths[idx])

        # read the associated mask from disk
        mask = torch.load(self.maskPaths[idx])

        # check to see if we are applying any transformations
        if self.transforms is not None:
            image, mask = self.transforms(image, mask) # we use PyTorch reference segmentation transforms

        # return a tuple of the image and its mask
        return (image, mask)
