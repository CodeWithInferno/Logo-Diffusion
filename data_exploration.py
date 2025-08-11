import os
from PIL import Image
import random

def explore_dataset(directory):
    """
    Analyzes the logo dataset by counting files and inspecting random images.

    Args:
        directory (str): The path to the directory containing the logo images.
    """
    print(f"Starting analysis of the dataset in: {directory}")

    try:
        image_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_files.append(os.path.join(root, file))

        file_count = len(image_files)
        print(f"Found {file_count} image files in the directory and its subdirectories.")

        if file_count == 0:
            print("\nWarning: No image files found. Please ensure your dataset is in the correct location.")
            return

        # --- Verification Steps ---
        print("\n--- Verifying a few random images ---")
        num_images_to_check = min(5, file_count)
        random_selection = random.sample(image_files, num_images_to_check)

        for i, file_path in enumerate(random_selection):
            try:
                with Image.open(file_path) as img:
                    print(f"{i+1}. Filename: {os.path.basename(file_path)}")
                    print(f"   - Path: {file_path}")
                    print(f"   - Format: {img.format}")
                    print(f"   - Mode: {img.mode}")
                    print(f"   - Size: {img.size}")

                    # Check for common pitfalls
                    if img.mode not in ['RGB', 'RGBA']:
                        print("   - Warning: Image is not in a standard color mode (RGB/RGBA). This might require conversion.")
                    if max(img.size) < 50:
                        print("   - Warning: Image is very small. This could be problematic for sketch generation.")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory}'.")
        print("Please make sure the path is correct and the data is in the 'data/logos' subdirectory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # The user should place their logo images in a 'data/logos' subdirectory.
    # This script should be run from the root of the project.
    logo_directory = "data/logos/datasetcopy/trainandtest"
    explore_dataset(logo_directory)