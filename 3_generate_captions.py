import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

def generate_caption(image_path, processor, model, device):
    """
    Generates a caption for a single image using a BLIP model.

    Args:
        image_path (str): The path to the input image.
        processor: The BLIP processor.
        model: The BLIP model.
        device (str): The device to run the model on ('cuda' or 'cpu').

    Returns:
        str: The generated caption, or None if an error occurs.
    """
    try:
        with Image.open(image_path) as img:
            # Ensure image is in RGB format
            img = img.convert('RGB')
            
            # Process the image
            inputs = processor(images=img, return_tensors="pt").to(device)
            
            # Generate the caption
            output_ids = model.generate(**inputs, max_length=50)
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            
            return caption.strip()
            
    except Exception as e:
        print(f"\nError processing {image_path}: {e}")
        return None

def process_images_for_captions(source_dir, output_file):
    """
    Generates captions for all images in a source directory and saves them to a JSON file.

    Args:
        source_dir (str): The directory containing the clean images.
        output_file (str): The path to the output JSON file.
    """
    print("Starting caption generation...")
    
    # --- Setup Model ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading BLIP model... (This may take a few minutes)")
    try:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and the 'transformers' library is installed.")
        return

    print(f"Source directory: {source_dir}")
    print(f"Output file: {output_file}")

    caption_data = {}

    # Find all image files recursively
    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    print(f"Found {len(image_files)} total images to process.")

    # Use tqdm for a progress bar
    for file_path in tqdm(image_files, desc="Generating Captions"):
        caption = generate_caption(file_path, processor, model, device)
        
        if caption:
            # Use relative path as the key
            relative_path = os.path.relpath(file_path, source_dir)
            caption_data[relative_path] = caption

    # Save the caption data to a JSON file
    with open(output_file, 'w') as f:
        json.dump(caption_data, f, indent=4)

    print("\n--- Caption Generation Summary ---")
    print(f"Successfully generated captions for {len(caption_data)} images.")
    print(f"Caption data saved to: {output_file}")

if __name__ == "__main__":
    cleaned_data_directory = "data/logos/cleaned"
    output_json_file = "data/captions.json"
    process_images_for_captions(cleaned_data_directory, output_json_file)
