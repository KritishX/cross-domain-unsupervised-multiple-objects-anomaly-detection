import os
import shutil
from PIL import Image
import cv2
import numpy as np
import stat

# --- Configuration ---
DATASET_DIR = 'dataset'
REVIEW_DIR = 'review'
MIN_RESOLUTION = (224, 224)
BLUR_THRESHOLD = 30.0  # Lowered threshold
BLANK_THRESHOLD = 10.0
CONVERT_TO_RGB = False
VALID_IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']

def is_image_file(filename):
    """Check if a file has a valid image extension."""
    return any(filename.lower().endswith(ext) for ext in VALID_IMAGE_EXTENSIONS)

def fix_permissions(directory):
    """Recursively adds write permissions for the user to a directory."""
    print(f"Fixing permissions for '{directory}'...")
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            os.chmod(os.path.join(root, name), stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH)
        for name in files:
            os.chmod(os.path.join(root, name), stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
    print("Permissions fixed.")

def clean_dataset(directory):
    """
    Cleans the image dataset based on a set of rules.
    """
    print(f"Starting dataset cleaning in '{directory}'...")

    # Create review directory if it doesn't exist
    if not os.path.exists(REVIEW_DIR):
        os.makedirs(REVIEW_DIR)

    # --- Counters for logging ---
    stats = {
        "processed": 0,
        "deleted_corrupted": 0,
        "deleted_low_res": 0,
        "flagged_blur_blank": 0,
        "converted_to_rgb": 0,
    }

    for subdir, dirs, files in os.walk(directory):
        # 6. Skip Ground-Truth Masks
        if 'ground_truth' in subdir.split(os.path.sep):
            continue

        for filename in files:
            if not is_image_file(filename):
                continue

            stats["processed"] += 1
            filepath = os.path.join(subdir, filename)
            
            # 2. Corruption Check
            try:
                with Image.open(filepath) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print(f"DELETING corrupted image: {filepath} - {e}")
                stats["deleted_corrupted"] += 1
                os.remove(filepath)
                continue

            # Re-open image for other checks
            with Image.open(filepath) as img:
                # 3. Resolution Check
                if img.width < MIN_RESOLUTION[0] or img.height < MIN_RESOLUTION[1]:
                    print(f"DELETING low-res image: {filepath} ({img.size})")
                    stats["deleted_low_res"] += 1
                    os.remove(filepath)
                    continue

                # 4. Color Mode Standardization
                if CONVERT_TO_RGB and img.mode != 'RGB':
                    print(f"CONVERTING non-RGB image to RGB: {filepath} (Mode: {img.mode})")
                    rgb_img = img.convert('RGB')
                    rgb_img.save(filepath)
                    stats["converted_to_rgb"] += 1

            # 5. Blur / Blank Detection (Optional)
            try:
                cv_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if cv_img is None:
                    raise IOError("OpenCV could not read the image.")

                is_blurry = cv2.Laplacian(cv_img, cv2.CV_64F).var() < BLUR_THRESHOLD
                is_blank = np.std(cv_img) < BLANK_THRESHOLD

                if is_blurry or is_blank:
                    review_path = os.path.join(REVIEW_DIR, os.path.basename(subdir) + "_" + filename)
                    print(f"FLAGGING {'blurry' if is_blurry else 'blank'} image: {filepath} -> {review_path}")
                    shutil.move(filepath, review_path)
                    stats["flagged_blur_blank"] += 1
                    continue
            except Exception as e:
                print(f"Could not process with OpenCV, skipping blur/blank check for {filepath} - {e}")


    # 8. Logging / Reporting
    print("\n--- Cleaning Summary ---")
    print(f"Total images processed: {stats['processed']}")
    print(f"Corrupted images deleted: {stats['deleted_corrupted']}")
    print(f"Low-resolution images deleted: {stats['deleted_low_res']}")
    print(f"Images moved to '{REVIEW_DIR}' for review (blur/blank): {stats['flagged_blur_blank']}")
    print(f"Images converted to RGB: {stats['converted_to_rgb']}")
    print("------------------------\n")
    print("10. Ready for Feature Extraction: Dataset contains valid images.")


if __name__ == '__main__':
    if not os.path.isdir(DATASET_DIR):
        print(f"Error: Directory '{DATASET_DIR}' not found.")
        print("Please make sure you have extracted the dataset first.")
    else:
        # 7. Permission Fix
        fix_permissions(DATASET_DIR)
        clean_dataset(DATASET_DIR)