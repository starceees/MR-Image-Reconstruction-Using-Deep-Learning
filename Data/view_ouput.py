import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Paths to your original MRI (e.g., la_001_0000.nii.gz) and predicted segmentation (la_001.nii.gz).
# Make sure these two images have matching dimensions.
orig_path = "/home/raghuram/ARPL/MR-Image-Reconstruction-Using-Deep-Learning/Task02_Heart/imagesTs/la_001.nii.gz" 
seg_path  = "/home/raghuram/ARPL/MR-Image-Reconstruction-Using-Deep-Learning/Data/segoutput/la_001.nii.gz"   

# Output PDF file
output_pdf = "overlay_results_2.pdf"

# Load images
orig_img = nib.load(orig_path).get_fdata()  # shape (X, Y, Z)
seg_img  = nib.load(seg_path).get_fdata()   # shape (X, Y, Z), typically 0=background, 1=structure

with PdfPages(output_pdf) as pdf:
    # Loop over all slices in the Z dimension
    for z in range(orig_img.shape[2]):
        # Create a figure for each slice
        fig, ax = plt.subplots(figsize=(5, 5))

        # 1) Display the original image in grayscale
        ax.imshow(orig_img[..., z], cmap="gray", origin="lower")

        # 2) Create a masked array for the segmentation so that label=0 is transparent
        mask = np.ma.masked_where(seg_img[..., z] == 0, seg_img[..., z])

        # 3) Overlay the segmentation in red
        ax.imshow(mask, cmap="magma", alpha=1, origin="lower")

        ax.set_title(f"Slice {z}")
        ax.axis("off")

        # Write the figure (this slice) as a new page in the PDF
        pdf.savefig(fig)
        plt.close(fig)

print(f"Overlay PDF saved as {output_pdf}")

