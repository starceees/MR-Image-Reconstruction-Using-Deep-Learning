import numpy as np
import torch
import torchvision.transforms.functional as TF

def normalize_slice(slice):
    # Normalize the image slice to [0, 1]
    return (slice - slice.min()) / (slice.max() - slice.min() + 1e-8)

def random_augmentations(img, lbl):
    # Random horizontal flip.
    if np.random.rand() > 0.5:
        img = np.fliplr(img).copy()
        lbl = np.fliplr(lbl).copy()
    
    # Random rotation between -15 and 15 degrees.
    angle = np.random.uniform(-15, 15)
    # Unsqueeze to add channel dimension for rotation.
    img_tensor = torch.tensor(img).unsqueeze(0)
    lbl_tensor = torch.tensor(lbl).unsqueeze(0)
    img_tensor = TF.rotate(img_tensor, angle, interpolation=TF.InterpolationMode.BILINEAR)
    lbl_tensor = TF.rotate(lbl_tensor, angle, interpolation=TF.InterpolationMode.NEAREST)
    
    # Random zoom between 0.9 and 1.1.
    zoom = np.random.uniform(0.9, 1.1)
    h, w = img_tensor.shape[-2:]
    new_h = int(h * zoom)
    new_w = int(w * zoom)
    img_tensor = TF.resize(img_tensor, [new_h, new_w], interpolation=TF.InterpolationMode.BILINEAR)
    lbl_tensor = TF.resize(lbl_tensor, [new_h, new_w], interpolation=TF.InterpolationMode.NEAREST)
    
    # Center crop back to original size.
    img_tensor = TF.center_crop(img_tensor, [h, w])
    lbl_tensor = TF.center_crop(lbl_tensor, [h, w])
    
    return img_tensor.squeeze(0).numpy(), lbl_tensor.squeeze(0).numpy()
