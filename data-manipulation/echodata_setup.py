"""
Creates train and test directories from the echointegration data of ABRACOS1.
"""
### I. First Steps
# 1. Retrieve Sv_surface matrixes from each file in the 'ABRACOS1' directory.
# 2. Concatenate theses matrixes to obtain a single matrix.
# 3. Compare shape with concatenated files from the 'Annotations Everton' directory (Sv_surface matrixes should be of same shape)
# 3bis. (opt) Visualize top left corner of both matrix : classes should be coherent with signal
# 4. Make crops to constitute a dataset: for now let's pick 1000 random 128x128x4 crops on the first layer of signal (where all 4 frequencies are available, i.e. 0-140m)

### Now we have X (dataset of 1000 128x128x4 Sv_surface images) and the corresponding Y (labels: 1000 128x128x4 annotated images)
### II. Next steps
# 1. Split (X, Y) into (X_train, Y_train) / (X_test, Y_test). Random split. Ratio 0.8
# 2. Create and fill the train and test directories
# Each should have 6 sub directories named after the classes:
# - 1
# - 2
# - 3
# - 4
# - 5
# - 6

### I. Retrieving data and creating the dataset

# imports
import os
import h5py

import numpy as np
import torch


os.chdir("/home1/datawork/gringuen/Echoclasses/Files/EI_Abracos1/Transects/LEG1")

dir_path = "/home1/datawork/gringuen/Echoclasses/Files/EI_Abracos1/Transects/LEG1"
Sv_tot = torch.

for dirpath, dirnames, filenames in os.walk(dir_path):
    print(f"Retrieving data from {dirpath} directory...")
    for filename in filenames:
        if filename == "Echointegration.mat":
            filepath = dirpath + filename
            EI = h5py.File(filepath,'r')
            Sv_surface_array = np.array(EI.get("Sv_surface")) # /!\ CHW format /!\
            Sv_surface_tensor = torch.from_numpy(Sv_surface_array).permute(2, 1, 0) # Now in HWC -> to be easier for append, but Sv_tot will have to end up in CHW
            Sv_tot = torch.cat((Sv_tot, Sv_surface_tensor), dim=1)
            




