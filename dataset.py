import os
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from augmentations import Transformer, Crop, Cutout, Noise, Normalize, Blur, Flip

class ADNI_Dataset(Dataset):

    def __init__(self, config, labels, aug, *args, **kwargs): # ADNI
        super().__init__(*args, **kwargs)
        
        self.config = config
        self.data_dir = config.data
        self.labels = labels
        self.aug = aug
        self.transforms = Transformer()
        self.transforms.register(Normalize(), probability=1.0)

        if self.aug == True and self.config.mode == 'pretraining':
            if self.config.tf == 'all_tf':
                self.transforms.register(Flip(), probability=0.5)
                self.transforms.register(Blur(sigma=(0.1, 1)), probability=0.5)
                self.transforms.register(Noise(sigma=(0.1, 1)), probability=0.5)
                self.transforms.register(Cutout(patch_size=np.ceil(np.array(self.config.input_size)/4)), probability=0.5)
                self.transforms.register(Crop(np.ceil(0.75*np.array(self.config.input_size)), "random", resize=True), probability=0.5)

            elif self.config.tf == 'cutout':
                self.transforms.register(Cutout(patch_size=np.ceil(np.array(self.config.input_size)/4)), probability=1.0)

            elif self.config.tf == 'crop':
                self.transforms.register(Crop(np.ceil(0.75*np.array(self.config.input_size)), "random", resize=True), probability=1.0)
        
    def collate_fn(self, list_samples):
        list_x = torch.stack([torch.as_tensor(x, dtype=torch.float) for (x, y, z) in list_samples], dim=0)
        list_y = torch.stack([torch.as_tensor(y, dtype=torch.float) for (x, y, z) in list_samples], dim=0)
        list_z = [z for (x, y, z) in list_samples]

        return (list_x, list_y, list_z)

    def __getitem__(self, idx):
        labels = self.labels[self.config.label_name].values[idx]
        file_name = self.labels['File_name'].values[idx]
        path = os.path.join(self.data_dir, file_name)
        img = nib.load(os.path.join(path, 'brain_to_MNI_nonlin.nii.gz'))
        img = np.swapaxes(img.get_data(),1,2)
        img = np.flip(img,1)
        img = np.flip(img,2)
        img = resize(img, (self.config.input_size[1], self.config.input_size[2], self.config.input_size[3]), mode='constant')
        img = torch.from_numpy(img).float().view(self.config.input_size[0], self.config.input_size[1], self.config.input_size[2], self.config.input_size[3])
        img = img.numpy()
        
        np.random.seed()
        x = self.transforms(img)

        return (x, labels, file_name)

    def __len__(self):
        return len(self.labels)