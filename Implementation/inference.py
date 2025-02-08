import os
import yaml
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from glob import glob
from matplotlib.backends.backend_pdf import PdfPages
from torchvision.transforms import functional as TF

from dataloader2d import get_loaders
from unet2dmodel import UNet2D
from metrics import calculate_metrics, aggregate_metrics

# =============================================================================
# 1. Slice-wise Inference using DataLoader
# =============================================================================
def run_inference(model, loader, device):
    """
    Runs inference on a test DataLoader and aggregates metrics.
    """
    model.eval()
    all_metrics = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            metrics = calculate_metrics(outputs, y)
            all_metrics.append(metrics)
    final_metrics = aggregate_metrics(all_metrics)
    print("Slice-wise inference on test DataLoader:")
    print(f"Dice Score: {final_metrics['dice']:.4f}")
    print(f"Jaccard Index: {final_metrics['jaccard']:.4f}")
    return final_metrics

# =============================================================================
# 2. 2D Slice-wise Inference on 3D Volumes
# =============================================================================
def infer_2d_slices(volume_np, model, device):
    """
    Performs inference on each axial slice of a 3D volume using a trained 2D UNet model.
    
    Args:
        volume_np (numpy.ndarray): Input 3D volume with shape (D, H, W).
        model (torch.nn.Module): Trained 2D UNet model.
        device (torch.device): Device on which to run inference.
    
    Returns:
        pred_mask_np (numpy.ndarray): Predicted segmentation mask for the volume (D, H, W).
    """
    D, H, W = volume_np.shape
    pred_mask_np = np.zeros((D, H, W), dtype=np.uint8)

    # Convert the volume to a tensor and normalize it.
    volume_tensor = torch.from_numpy(volume_np).float()
    volume_tensor = (volume_tensor - volume_tensor.min()) / (volume_tensor.max() - volume_tensor.min() + 1e-8)

    model.eval()
    with torch.no_grad():
        for z in range(D):
            # Extract the z-th slice and add batch and channel dimensions: (1, 1, H, W)
            slice_2d = volume_tensor[z, :, :].unsqueeze(0).unsqueeze(0).to(device)
            
            # Resize the slice to the model's expected input size (here assumed to be 256x256)
            target_size = (256, 256)
            slice_2d_resized = TF.resize(slice_2d, target_size)
            
            # Inference on the resized slice
            logits = model(slice_2d_resized)  # (1, num_classes, H_resized, W_resized)
            pred_mask = torch.argmax(logits, dim=1).cpu().numpy().squeeze(0)  # (H_resized, W_resized)
            
            # Resize the predicted mask back to original (H, W) using nearest-neighbor interpolation.
            pred_mask_tensor = torch.from_numpy(pred_mask).unsqueeze(0).unsqueeze(0).float()
            pred_mask_resized = TF.resize(
                pred_mask_tensor,
                (H, W),
                interpolation=TF.InterpolationMode.NEAREST
            ).squeeze().numpy().astype(np.uint8)
            
            pred_mask_np[z, :, :] = pred_mask_resized

    return pred_mask_np

# =============================================================================
# 3. Inference and Visualization Function
# =============================================================================
def visualize_2d_inference(config, model, device):
    """
    Loads test volumes from the imagesTs folder, runs slice-wise inference on each
    volume using a 2D UNet model, and saves a PDF with visualizations showing the
    original center slice and a red-overlay segmentation.
    """
    ROOT_DIR = config['data_root']
    test_folder = os.path.join(ROOT_DIR, "imagesTs")
    test_images = sorted(glob(os.path.join(test_folder, "*.nii.gz")))
    output_pdf = "inference_results_2d.pdf"
    
    with PdfPages(output_pdf) as pdf:
        for idx, img_path in enumerate(test_images):
            base_name = os.path.basename(img_path)
            print(f"Processing {base_name} with slice-wise 2D inference...")
            
            # Load the full 3D volume (assumed shape: D x H x W)
            img_nifti = nib.load(img_path)
            img_np = img_nifti.get_fdata(dtype=np.float32)
            
            # Run inference on each slice of the volume.
            pred_mask_np = infer_2d_slices(img_np, model, device)
            
            # For visualization, use the center axial slice.
            D, H, W = img_np.shape
            z_mid = D // 2
            original_slice = img_np[z_mid, :, :]
            mask_slice = pred_mask_np[z_mid, :, :]
            
            # Create an RGB image from the grayscale original.
            img_rgb = np.stack([original_slice] * 3, axis=-1)
            img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-8)
            
            # Create a red mask overlay.
            mask_rgb = np.zeros_like(img_rgb)
            mask_rgb[:, :, 0] = mask_slice  # Red channel gets the mask.
            mask_overlay = (img_rgb * 0.5) + (mask_rgb * 0.5)
            
            # Plot the original and overlay side by side.
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f"{base_name} - Center Axial Slice (z={z_mid})", fontsize=16)
            axs[0].imshow(original_slice, cmap='gray')
            axs[0].set_title("Original")
            axs[1].imshow(mask_overlay)
            axs[1].set_title("Masked (Red Overlay)")
            for ax in axs:
                ax.set_xticks([])
                ax.set_yticks([])
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            print(f"Finished processing {base_name}")
    
    print(f"Inference results saved to: {output_pdf}")

# =============================================================================
# 4. Main Guard
# =============================================================================
if __name__ == "__main__":
    # Load configuration parameters from YAML.
    with open("parameters.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Get the test DataLoader (for slice-wise metrics) from the dataloader.
    loaders = get_loaders(config)
    test_loader = loaders['test']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize the UNet model.
    model = UNet2D(
        in_channels=1,
        out_channels=config['num_classes'],
        base_filters=config['base_filters'],
        bilinear=config['bilinear']
    )
    model.to(device)
    
    # Load the best model checkpoint.
    checkpoint_path = "checkpoints/best_model.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    
    # 1. Run inference using the DataLoader and print aggregated metrics.
    run_inference(model, test_loader, device)
    
    # 2. Run slice-wise inference on full volumes from imagesTs and visualize the results.
    visualize_2d_inference(config, model, device)
