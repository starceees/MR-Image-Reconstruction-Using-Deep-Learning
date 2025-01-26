import torch 
import fastmri
import fastmri.data.transforms as T
from fastmri.data.subsample import create_mask_for_mask_type

def complex_to_channels(kspace):
    """
    COnvert complex k-space [H,W,2] -> [2,H,W],
    1.e. seperate real and imaginary 2 channels 
    """
    # kpace data for multi-coil data [H,W,2]
    # k-sapce data for single coil [1,2,H,W] 
    # we need to remove the extra leading dimention and then permute the axes 
    if ksapce.dim() == 4:
        kspace = kspace[0]

    #Now shape is [H,W,2]l; permute to [2,H,W]
    kspace = kspace.permute(2,0,1)