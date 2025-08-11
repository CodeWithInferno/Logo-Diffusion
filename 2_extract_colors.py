# import os
# import json
# import numpy as np
# from PIL import Image
# from sklearn.cluster import KMeans
# from tqdm import tqdm

# def extract_dominant_colors(image_path, n_colors=5):
#     """
#     Extracts the dominant colors from an image using K-Means clustering.

#     Args:
#         image_path (str): The path to the input image.
#         n_colors (int): The number of dominant colors to extract.

#     Returns:
#         list: A list of hex color strings, or None if an error occurs.
#     """
#     try:
#         with Image.open(image_path) as img:
#             # Ensure image is in RGB format
#             img = img.convert('RGB')
            
#             # Resize for faster processing
#             img = img.resize((100, 100))
            
#             # Get pixel data as a NumPy array
#             np_img = np.array(img)
#             pixels = np_img.reshape(-1, 3);
            
#             # Use K-Means to find dominant colors
#             kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
#             kmeans.fit(pixels)
            
#             # Get the RGB values of the cluster centers
#             colors = kmeans.cluster_centers_.astype(int)
            
#             # Convert RGB to hex
#             hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
            
#             return hex_colors
            
#     except Exception as e:
#         print(f"\nError processing {image_path}: {e}")
#         return None

# def process_images_for_colors(source_dir, output_file):
#     """
#     Processes all images in a source directory to extract dominant colors
#     and saves the results to a JSON file.

#     Args:
#         source_dir (str): The directory containing the clean images.
#         output_file (str): The path to the output JSON file.
#     """
#     print(f"Starting color extraction...")
#     print(f"Source directory: {source_dir}")
#     print(f"Output file: {output_file}")

#     color_data = {}

#     # Find all image files recursively
#     image_files = []
#     for root, _, files in os.walk(source_dir):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_files.append(os.path.join(root, file))

#     print(f"Found {len(image_files)} total images to process.")

#     # Use tqdm for a progress bar
#     for file_path in tqdm(image_files, desc="Extracting Colors"):
#         colors = extract_dominant_colors(file_path)
        
#         if colors:
#             # Use relative path as the key
#             relative_path = os.path.relpath(file_path, source_dir)
#             color_data[relative_path] = colors

#     # Save the color data to a JSON file
#     with open(output_file, 'w') as f:
#         json.dump(color_data, f, indent=4)

#     print("\n--- Color Extraction Summary ---")
#     print(f"Successfully extracted color palettes for {len(color_data)} images.")
#     print(f"Color data saved to: {output_file}")

# if __name__ == "__main__":
#     cleaned_data_directory = "data/logos/cleaned"
#     output_json_file = "data/color_palettes.json"
#     process_images_for_colors(cleaned_data_directory, output_json_file)




import os
import json
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
from joblib import Parallel, delayed

# --- This function stays the same ---
def extract_dominant_colors(image_path, n_colors=5):
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB').resize((100, 100))
            np_img = np.array(img)
            pixels = np_img.reshape(-1, 3)
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init='auto', max_iter=100)
            kmeans.fit(pixels)
            colors = kmeans.cluster_centers_.astype(int)
            hex_colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]
            relative_path = os.path.relpath(image_path, start=os.path.dirname(os.path.dirname(image_path)))
            return (relative_path, hex_colors)
    except Exception as e:
        # Return the error to be aware of failures
        return (os.path.relpath(image_path, start=os.path.dirname(os.path.dirname(image_path))), f"Error: {e}")

# --- This main function is now parallelized ---
def process_images_parallel(source_dir, output_file):
    print("Starting PARALLEL color extraction...")
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} total images to process.")

    # n_jobs=-1 tells joblib to use all available CPU cores.
    # This is the key to making it fast!
    results = Parallel(n_jobs=-1)(delayed(extract_dominant_colors)(file) for file in tqdm(image_files, desc="Extracting Colors"))

    # Convert the list of tuples back into a dictionary
    color_data = {path: colors for path, colors in results if isinstance(colors, list)}

    with open(output_file, 'w') as f:
        json.dump(color_data, f, indent=4)

    print("\n--- Color Extraction Summary ---")
    print(f"Successfully processed palettes for {len(color_data)} images.")
    print(f"Color data saved to: {output_file}")


if __name__ == "__main__":
    cleaned_data_directory = "data/logos/cleaned"  # Make sure this is the path to your cleaned images
    output_json_file = "data/color_palettes.json"
    process_images_parallel(cleaned_data_directory, output_json_file)