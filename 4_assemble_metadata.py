import os
import json
import csv
from tqdm import tqdm

def assemble_metadata(cleaned_dir, sketch_dir, color_file, caption_file, output_csv):
    """
    Assembles a master metadata CSV file from the cleaned images, sketches,
    color palettes, and captions.

    Args:
        cleaned_dir (str): Directory of the clean source images.
        sketch_dir (str): Directory of the generated sketch images.
        color_file (str): Path to the color palettes JSON file.
        caption_file (str): Path to the captions JSON file.
        output_csv (str): Path for the output metadata CSV file.
    """
    print("Starting metadata assembly...")

    # --- 1. Load JSON Data ---
    print(f"Loading color data from: {color_file}")
    try:
        with open(color_file, 'r') as f:
            color_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Color file not found at {color_file}. The 'colors' column will be empty.")
        color_data = {}
    
    print(f"Loading caption data from: {caption_file}")
    try:
        with open(caption_file, 'r') as f:
            caption_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Caption file not found at {caption_file}. The 'caption' column will be empty.")
        caption_data = {}

    # --- 2. Find All Source Images ---
    image_files = []
    for root, _, files in os.walk(cleaned_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))
    
    print(f"Found {len(image_files)} source images to process.")

    # --- 3. Create CSV and Write Data ---
    header = ['image_path', 'sketch_path', 'colors', 'caption']
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for img_path in tqdm(image_files, desc="Assembling Metadata"):
            # Use relative paths for consistency
            relative_path = os.path.relpath(img_path, cleaned_dir)
            
            # Construct sketch path
            sketch_path = os.path.join(sketch_dir, relative_path)
            
            # Get colors and caption from the loaded data
            # The .get() method safely returns None if the key doesn't exist
            colors = color_data.get(relative_path)
            caption = caption_data.get(relative_path)
            
            # Format colors as a simple comma-separated string if they exist
            colors_str = ",".join(colors) if colors else ""
            
            # Write the row to the CSV
            writer.writerow([
                os.path.join(cleaned_dir, relative_path),
                sketch_path,
                colors_str,
                caption if caption else ""
            ])

    print("\n--- Metadata Assembly Summary ---")
    print(f"Successfully created metadata file with {len(image_files)} entries.")
    print(f"Metadata saved to: {output_csv}")

if __name__ == "__main__":
    CLEANED_DIRECTORY = "data/logos/cleaned"
    SKETCH_DIRECTORY = "data/logos/sketches"
    COLOR_JSON_FILE = "data/color_palettes.json"
    CAPTION_JSON_FILE = "data/captions.json" # This will be created by the other script
    OUTPUT_CSV_FILE = "data/metadata.csv"
    
    assemble_metadata(
        CLEANED_DIRECTORY,
        SKETCH_DIRECTORY,
        COLOR_JSON_FILE,
        CAPTION_JSON_FILE,
        OUTPUT_CSV_FILE
    )
