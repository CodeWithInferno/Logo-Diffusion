import os
from PIL import Image
from tqdm import tqdm

def clean_dataset(source_dir, clean_dir):
    """
    Cleans the dataset by verifying each image and copying valid ones to a new directory.

    Args:
        source_dir (str): The path to the directory containing the raw image data.
        clean_dir (str): The path to the directory where clean images will be saved.
    """
    print(f"Starting data cleaning process...")
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {clean_dir}")

    if not os.path.exists(clean_dir):
        os.makedirs(clean_dir)
        print(f"Created clean directory: {clean_dir}")

    valid_images = 0
    corrupted_files = 0
    
    # Find all image files recursively
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} total image files to process.")

    # Use tqdm for a progress bar
    for file_path in tqdm(image_files, desc="Cleaning Images"):
        try:
            # Attempt to open the image to verify it's not corrupt
            with Image.open(file_path) as img:
                img.verify()  # Verify the image integrity

            # If valid, copy it to the clean directory
            # We maintain the class structure (e.g., /Accessories/Nike/image.jpg)
            relative_path = os.path.relpath(file_path, source_dir)
            dest_path = os.path.join(clean_dir, relative_path)
            
            # Create the subdirectory in the destination if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Copy the file
            with open(file_path, 'rb') as f_src, open(dest_path, 'wb') as f_dst:
                f_dst.write(f_src.read())

            valid_images += 1

        except (IOError, SyntaxError) as e:
            # This catches corrupted or invalid image files
            print(f"\nWarning: Corrupted or invalid image file skipped: {file_path} ({e})")
            corrupted_files += 1
        except Exception as e:
            print(f"\nAn unexpected error occurred with file {file_path}: {e}")
            corrupted_files += 1
            
    print("\n--- Data Cleaning Summary ---")
    print(f"Total valid images copied: {valid_images}")
    print(f"Total corrupted/invalid files skipped: {corrupted_files}")
    print(f"Clean data is now available in: {clean_dir}")

if __name__ == "__main__":
    raw_data_directory = "data/logos/datasetcopy/trainandtest"
    cleaned_data_directory = "data/logos/cleaned"
    clean_dataset(raw_data_directory, cleaned_data_directory)
