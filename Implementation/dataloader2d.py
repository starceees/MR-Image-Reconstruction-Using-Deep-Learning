import os
from glob import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from preprocess import normalize_slice, random_augmentations

class SlicesDataset(Dataset):
    def __init__(self, image_paths, label_paths=None, is_train=False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.is_train = is_train
        self.slice_index_mapping = []
        
        # Each file is assumed to be a 3D volume; we extract each axial slice.
        for idx, path in enumerate(image_paths):
            img = nib.load(path)
            num_slices = img.shape[2]
            for slice_idx in range(num_slices):
                self.slice_index_mapping.append((idx, slice_idx))
                
    def __len__(self):
        return len(self.slice_index_mapping)
    
    def __getitem__(self, idx):
        file_idx, slice_idx = self.slice_index_mapping[idx]
        # Load the slice from image
        img = nib.load(self.image_paths[file_idx]).get_fdata()[:, :, slice_idx]
        img = normalize_slice(img)
        
        if self.label_paths:
            lbl = nib.load(self.label_paths[file_idx]).get_fdata()[:, :, slice_idx]
            # Threshold the label (assuming binary segmentation)
            lbl = (lbl > 0.5).astype(np.float32)
            if self.is_train:
                img, lbl = random_augmentations(img, lbl)
            # Convert to torch tensors
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            lbl = torch.tensor(lbl, dtype=torch.long)
            return img, lbl
        else:
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0)

def get_loaders(config):
    data_root = config['data_root']
    images = sorted(glob(os.path.join(data_root, "imagesTr", "*.nii.gz")))
    labels = sorted(glob(os.path.join(data_root, "labelsTr", "*.nii.gz")))
    
    # Split the dataset into train, validation, and test sets.
    train_img, valtest_img, train_lbl, valtest_lbl = train_test_split(
        images, labels, test_size=0.3, random_state=config['seed'])
    val_img, test_img, val_lbl, test_lbl = train_test_split(
        valtest_img, valtest_lbl, test_size=0.5, random_state=config['seed'])
    
    train_ds = SlicesDataset(train_img, train_lbl, is_train=True)
    val_ds = SlicesDataset(val_img, val_lbl, is_train=False)
    test_ds = SlicesDataset(test_img, test_lbl, is_train=False)
    
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], shuffle=False)
    
    return {'train': train_loader, 'val': val_loader, 'test': test_loader}
