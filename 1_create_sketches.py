import os
import cv2
from tqdm import tqdm

def create_sketch(image_path):
    """
    Creates a sketch from an input image using OpenCV.

    Args:
        image_path (str): The path to the input image.

    Returns:
        numpy.ndarray: The sketch image as a NumPy array, or None if an error occurs.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert to grayscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Invert the grayscale image
        invert_img = cv2.bitwise_not(grey_img)
        
        # Apply Gaussian blur
        blur_img = cv2.GaussianBlur(invert_img, (21, 21), 0)
        
        # Invert the blurred image
        invblur_img = cv2.bitwise_not(blur_img)
        
        # Create the sketch using a color dodge blend
        sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)
        
        return sketch_img
    except Exception as e:
        print(f"\nError creating sketch for {image_path}: {e}")
        return None

def process_images_to_sketches(source_dir, sketch_dir):
    """
    Processes all images in a source directory, converts them to sketches,
    and saves them in a new directory.

    Args:
        source_dir (str): The directory containing the clean images.
        sketch_dir (str): The directory where sketch images will be saved.
    """
    print(f"Starting sketch generation...")
    print(f"Source directory: {source_dir}")
    print(f"Destination directory: {sketch_dir}")

    if not os.path.exists(sketch_dir):
        os.makedirs(sketch_dir)
        print(f"Created sketch directory: {sketch_dir}")

    # Find all image files recursively
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} total images to process.")

    # Use tqdm for a progress bar
    for file_path in tqdm(image_files, desc="Creating Sketches"):
        sketch = create_sketch(file_path)
        
        if sketch is not None:
            # Construct the destination path, preserving the subdirectory structure
            relative_path = os.path.relpath(file_path, source_dir)
            dest_path = os.path.join(sketch_dir, relative_path)
            
            # Create the subdirectory in the destination if it doesn't exist
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Save the sketch image
            cv2.imwrite(dest_path, sketch)

    print("\n--- Sketch Generation Summary ---")
    print(f"Successfully generated sketches for all valid images.")
    print(f"Sketches are now available in: {sketch_dir}")

if __name__ == "__main__":
    cleaned_data_directory = "data/logos/cleaned"
    sketches_directory = "data/logos/sketches"
    process_images_to_sketches(cleaned_data_directory, sketches_directory)
