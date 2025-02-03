import os
import random
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_nii_file(file_path):
    """Load a NIfTI file and return the data and affine."""
    img = nib.load(file_path)
    return img.get_fdata(), img.affine

def visualize_and_save_to_pdf(images_dir, labels_dir, output_pdf):
    """Visualize random images with labels and save to a PDF."""
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.nii')]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.nii')]

    with PdfPages(output_pdf) as pdf:
        for img_file in image_files:
            # Load image and corresponding label
            img_path = os.path.join(images_dir, img_file)
            label_path = os.path.join(labels_dir, img_file)  # Assuming same filename for labels

            if not os.path.exists(label_path):
                print(f"Label file not found for {img_file}, skipping.")
                continue

            img_data, _ = load_nii_file(img_path)
            label_data, _ = load_nii_file(label_path)

            # Select a random slice
            slice_idx = random.randint(0, img_data.shape[2] - 1)
            img_slice = img_data[:, :, slice_idx]
            label_slice = label_data[:, :, slice_idx]

            # Plot the image and label
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_slice, cmap='gray')
            axes[0].set_title(f'Image: {img_file}\nSize: {img_data.shape}')
            axes[0].axis('off')

            axes[1].imshow(label_slice, cmap='jet', alpha=0.5)
            axes[1].set_title(f'Label: {img_file}\nSize: {label_data.shape}')
            axes[1].axis('off')

            # Save the figure to the PDF
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Visualization saved to {output_pdf}")

if __name__ == "__main__":
    # Define the paths
    images_dir = '/home/raghuram/ARPL/MR-Image-Reconstruction-Using-Deep-Learning/extracted_data/imagesTr'
    labels_dir = '/home/raghuram/ARPL/MR-Image-Reconstruction-Using-Deep-Learning/extracted_data/labelsTr'
    output_pdf = 'visualization_output.pdf'

    # Run the visualization function
    visualize_and_save_to_pdf(images_dir, labels_dir, output_pdf)