# import os
# import json
# import torch
# from PIL import Image
# from tqdm import tqdm
# from transformers import BlipProcessor, BlipForConditionalGeneration

# def generate_caption(image_path, processor, model, device):
#     """
#     Generates a caption for a single image using a BLIP model.

#     Args:
#         image_path (str): The path to the input image.
#         processor: The BLIP processor.
#         model: The BLIP model.
#         device (str): The device to run the model on ('cuda' or 'cpu').

#     Returns:
#         str: The generated caption, or None if an error occurs.
#     """
#     try:
#         with Image.open(image_path) as img:
#             # Ensure image is in RGB format
#             img = img.convert('RGB')
            
#             # Process the image
#             inputs = processor(images=img, return_tensors="pt").to(device)
            
#             # Generate the caption
#             output_ids = model.generate(**inputs, max_length=50)
#             caption = processor.decode(output_ids[0], skip_special_tokens=True)
            
#             return caption.strip()
            
#     except Exception as e:
#         print(f"\nError processing {image_path}: {e}")
#         return None

# def process_images_for_captions(source_dir, output_file):
#     """
#     Generates captions for all images in a source directory and saves them to a JSON file.

#     Args:
#         source_dir (str): The directory containing the clean images.
#         output_file (str): The path to the output JSON file.
#     """
#     print("Starting caption generation...")
    
#     # --- Setup Model ---
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     print("Loading BLIP model... (This may take a few minutes)")
#     try:
#         processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#         model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
#         print("Model loaded successfully.")
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         print("Please ensure you have an internet connection and the 'transformers' library is installed.")
#         return

#     print(f"Source directory: {source_dir}")
#     print(f"Output file: {output_file}")

#     caption_data = {}

#     # Find all image files recursively
#     image_files = []
#     for root, _, files in os.walk(source_dir):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 image_files.append(os.path.join(root, file))

#     print(f"Found {len(image_files)} total images to process.")

#     # Use tqdm for a progress bar
#     for file_path in tqdm(image_files, desc="Generating Captions"):
#         caption = generate_caption(file_path, processor, model, device)
        
#         if caption:
#             # Use relative path as the key
#             relative_path = os.path.relpath(file_path, source_dir)
#             caption_data[relative_path] = caption

#     # Save the caption data to a JSON file
#     with open(output_file, 'w') as f:
#         json.dump(caption_data, f, indent=4)

#     print("\n--- Caption Generation Summary ---")
#     print(f"Successfully generated captions for {len(caption_data)} images.")
#     print(f"Caption data saved to: {output_file}")

# if __name__ == "__main__":
#     cleaned_data_directory = "data/logos/cleaned"
#     output_json_file = "data/captions.json"
#     process_images_for_captions(cleaned_data_directory, output_json_file)












import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# --- Configuration ---
SOURCE_DIR = "data/logos/cleaned"  # The directory containing the clean images.
OUTPUT_FILE = "data/captions.json" # The path to the output JSON file.
BATCH_SIZE = 32                    # How many images to process at once. Adjust based on your GPU VRAM.

def load_progress(output_file):
    """Loads existing captions if the output file exists."""
    if os.path.exists(output_file):
        print(f"Found existing captions file. Loading progress from {output_file}")
        with open(output_file, 'r') as f:
            return json.load(f)
    return {}

def process_in_batches(image_files, processor, model, device):
    """
    Generates captions for a list of image files in batches.
    
    Returns:
        dict: A dictionary of {relative_path: caption}.
    """
    new_captions = {}
    # Wrap the batch loop in tqdm for a progress bar
    for i in tqdm(range(0, len(image_files), BATCH_SIZE), desc="Generating Captions in Batches"):
        batch_paths = image_files[i:i + BATCH_SIZE]
        batch_images = []
        
        # --- Load images for the current batch ---
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                batch_images.append(img)
            except Exception as e:
                print(f"\nSkipping corrupted image {path}: {e}")
        
        if not batch_images:
            continue

        # --- Process the entire batch at once ---
        try:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True, truncation=True).to(device)
            output_ids = model.generate(**inputs, max_length=50)
            captions = processor.batch_decode(output_ids, skip_special_tokens=True)

            # --- Store the results for this batch ---
            for path, caption in zip(batch_paths, captions):
                relative_path = os.path.relpath(path, SOURCE_DIR)
                new_captions[relative_path] = caption.strip()
        except Exception as e:
            print(f"\nError processing batch starting with {batch_paths[0]}: {e}")

    return new_captions


if __name__ == "__main__":
    print("Starting EFFICIENT caption generation...")
    
    # --- Setup Model ---
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    print("Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    print("Model loaded successfully.")

    # --- Load existing progress and determine files to process ---
    existing_captions = load_progress(OUTPUT_FILE)
    all_image_files = []
    for root, _, files in os.walk(SOURCE_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, file)
                relative_path = os.path.relpath(full_path, SOURCE_DIR)
                if relative_path not in existing_captions:
                    all_image_files.append(full_path)

    if not all_image_files:
        print("All images have already been captioned. Nothing to do.")
    else:
        print(f"Found {len(existing_captions)} existing captions.")
        print(f"Processing {len(all_image_files)} new images.")

        # --- Generate new captions ---
        new_captions = process_in_batches(all_image_files, processor, model, device)

        # --- Combine old and new captions and save ---
        existing_captions.update(new_captions)
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(existing_captions, f, indent=4)
        
        print("\n--- Caption Generation Summary ---")
        print(f"Successfully generated {len(new_captions)} new captions.")
        print(f"Total captions now saved in {OUTPUT_FILE}: {len(existing_captions)}")