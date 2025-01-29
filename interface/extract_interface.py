import os
import shutil
import gzip
import nibabel as nib

def extract_nii_gz(file_path, target_dir):
    """Extract .nii.gz files to .nii and save them in the target directory."""
    try:
        with gzip.open(file_path, 'rb') as f_in:
            with open(os.path.join(target_  dir, os.path.basename(file_path)[:-3]), 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Extracted: {file_path}")
    except gzip.BadGzipFile:
        print(f"Invalid gzip file, copying as-is: {file_path}")
        shutil.copy(file_path, os.path.join(target_dir, os.path.basename(file_path)))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def extract_images_and_labels(source_dir, target_dir):
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Define the paths for training and testing images and labels
    train_images_dir = os.path.join(source_dir, 'imagesTr')
    train_labels_dir = os.path.join(source_dir, 'labelsTr')
    test_images_dir = os.path.join(source_dir, 'imagesTs')

    # Extract training images
    if os.path.exists(train_images_dir):
        train_images_target = os.path.join(target_dir, 'imagesTr')
        os.makedirs(train_images_target, exist_ok=True)
        for file_name in os.listdir(train_images_dir):
            if file_name.endswith('.nii.gz'):
                extract_nii_gz(os.path.join(train_images_dir, file_name), train_images_target)

    # Extract training labels
    if os.path.exists(train_labels_dir):
        train_labels_target = os.path.join(target_dir, 'labelsTr')
        os.makedirs(train_labels_target, exist_ok=True)
        for file_name in os.listdir(train_labels_dir):
            if file_name.endswith('.nii.gz'):
                extract_nii_gz(os.path.join(train_labels_dir, file_name), train_labels_target)

    # Extract testing images
    if os.path.exists(test_images_dir):
        test_images_target = os.path.join(target_dir, 'imagesTs')
        os.makedirs(test_images_target, exist_ok=True)
        for file_name in os.listdir(test_images_dir):
            if file_name.endswith('.nii.gz'):
                extract_nii_gz(os.path.join(test_images_dir, file_name), test_images_target)

    print(f"Extraction complete. Files extracted to {target_dir}")

if __name__ == "__main__":
    # Define the source and target directories
    source_directory = '/home/raghuram/ARPL/MR-Image-Reconstruction-Using-Deep-Learning/Task02_Heart'
    target_directory = '/home/raghuram/ARPL/MR-Image-Reconstruction-Using-Deep-Learning/extracted_data'

    # Run the extraction function
    extract_images_and_labels(source_directory, target_directory)