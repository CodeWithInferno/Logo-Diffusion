# import os
# import torch
# import pandas as pd
# from PIL import Image
# from torch.utils.data import Dataset
# from torchvision import transforms
# from transformers import CLIPTokenizer, CLIPTextModel

# class LogoDataset(Dataset):
#     """
#     PyTorch Dataset for loading logo images, sketches, and their conditions.
#     """
#     def __init__(self, metadata_path, text_encoder_device='cpu'):
#         """
#         Args:
#             metadata_path (str): Path to the metadata.csv file.
#             text_encoder_device (str): Device to load the text encoder on ('cpu' or 'cuda').
#                                        Kept on CPU by default to save VRAM on the main process.
#         """
#         print("Initializing LogoDataset...")
#         self.metadata = pd.read_csv(metadata_path)
        
#         # --- Initialize Models ---
#         print("Loading CLIP model for text encoding...")
#         self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        
#         # Load the text encoder to the specified device.
#         self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
#         self.text_encoder.to(text_encoder_device)
#         self.text_encoder_device = text_encoder_device

#         print("CLIP model loaded.")

#         # --- Define Image Transformations ---
#         self.image_transforms = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             # *** FIX 1: Correct normalization for 3-channel RGB images ***
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) 
#         ])
        
#         self.sketch_transforms = transforms.Compose([
#             transforms.Resize((512, 512)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5], [0.5]) # This is correct for 1-channel grayscale
#         ])

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx):
#         """
#         Retrieves a single data point from the dataset.
#         """
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
            
#         row = self.metadata.iloc[idx]

#         # --- 1. Load and Transform Images ---
#         try:
#             logo_path = row['image_path']
#             sketch_path = row['sketch_path']
            
#             logo_image = Image.open(logo_path).convert("RGB")
#             sketch_image = Image.open(sketch_path).convert("L") # Ensure grayscale

#             pixel_values = self.image_transforms(logo_image)
#             sketch_condition = self.sketch_transforms(sketch_image)
#         except Exception as e:
#             print(f"Error loading images for index {idx}, path {row.get('image_path', 'N/A')}: {e}")
#             # Return the data from the next valid index
#             return self.__getitem__((idx + 1) % len(self))


#         # --- 2. Process Color and Text Condition ---
#         # *** FIX 2: Combine color and text into a single powerful prompt ***
#         caption = row['caption']
#         colors_str = row['colors']

#         if pd.notna(caption) and isinstance(caption, str):
#             # Prepend color information to the caption
#             if pd.notna(colors_str) and isinstance(colors_str, str):
#                 color_prefix = "Colors: " + ", ".join(colors_str.split(',')[:5]) + ". "
#                 caption = color_prefix + caption
#         else:
#             caption = "a logo" # Provide a default caption if missing

#         # Tokenize and encode the combined caption
#         inputs = self.tokenizer(
#             caption, 
#             padding="max_length", 
#             max_length=self.tokenizer.model_max_length, 
#             truncation=True, 
#             return_tensors="pt"
#         )
#         # The text encoder is already on the correct device from the __init__ method.
#         # We use torch.no_grad() to prevent gradient calculation for this inference step.
#         with torch.no_grad():
#             text_encoder_output = self.text_encoder(inputs.input_ids.to(self.text_encoder_device))
#             text_condition = text_encoder_output.last_hidden_state

#         return {
#             "pixel_values": pixel_values,
#             "sketch_condition": sketch_condition,
#             "text_condition": text_condition.squeeze(0).cpu(), # Move to CPU before returning
#         }

# if __name__ == '__main__':
#     # This is a test block to verify the dataset works as expected.
#     # It will only run if you execute this script directly.
#     print("\n--- Running Dataset Test ---")
    
#     # Create a dummy image file for testing
#     dummy_image = Image.new('RGB', (100, 100), color = 'red')
#     dummy_image_path = 'dummy_image.png'
#     dummy_image.save(dummy_image_path)
    
#     # Create a dummy metadata file for testing
#     dummy_metadata = {
#         'image_path': [dummy_image_path],
#         'sketch_path': [dummy_image_path], # Using the same image for sketch is fine for a shape test
#         'colors': ['#ff0000,#00ff00,#0000ff,#ffff00,#ff00ff'],
#         'caption': ['a sample logo with various colors']
#     }
#     dummy_df = pd.DataFrame(dummy_metadata)
#     dummy_csv_path = 'dummy_metadata.csv'
#     dummy_df.to_csv(dummy_csv_path, index=False)

#     try:
#         # Test with 'cpu' as it's more common for local testing
#         dataset = LogoDataset(metadata_path=dummy_csv_path, text_encoder_device='cpu')
#         print(f"Dataset size: {len(dataset)}")
        
#         sample_item = dataset[0]
        
#         print("\n--- Sample Item Shapes ---")
#         print(f"pixel_values shape: {sample_item['pixel_values'].shape}")
#         print(f"sketch_condition shape: {sample_item['sketch_condition'].shape}")
#         print(f"text_condition shape: {sample_item['text_condition'].shape}")

#         assert sample_item['pixel_values'].shape == (3, 512, 512)
#         assert sample_item['sketch_condition'].shape == (1, 512, 512)
#         assert sample_item['text_condition'].shape == (77, 1024)
#         print("\n✅ Assertions passed! The dataset script appears to be working correctly.")

#     except Exception as e:
#         print(f"\nAn error occurred during the test: {e}")
#     finally:
#         # Clean up the dummy files
#         if os.path.exists(dummy_csv_path):
#             os.remove(dummy_csv_path)
#         if os.path.exists(dummy_image_path):
#             os.remove(dummy_image_path)






# src/dataset.py

import os
import torch
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

class LogoDataset(Dataset):
    def __init__(self, metadata_path, text_encoder_device='cpu'):
        print("Initializing LogoDataset...")
        self.metadata = pd.read_csv(metadata_path)
        
        print("Loading CLIP model for text encoding...")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder.to(text_encoder_device)
        self.text_encoder_device = text_encoder_device
        self.text_encoder.requires_grad_(False)
        print("CLIP model loaded and frozen.")

        self.image_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        self.sketch_transforms = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.metadata.iloc[idx]

        try:
            logo_path = row['image_path']
            sketch_path = row['sketch_path']
            logo_image = Image.open(logo_path).convert("RGB")
            sketch_image = Image.open(sketch_path).convert("L")
            pixel_values = self.image_transforms(logo_image)
            sketch_condition = self.sketch_transforms(sketch_image)
        except Exception:
            return self.__getitem__((idx + 1) % len(self))

        caption = row['caption']
        colors_str = row['colors']

        if pd.notna(caption) and isinstance(caption, str):
            if pd.notna(colors_str) and isinstance(colors_str, str):
                color_prefix = "Logo with colors: " + ", ".join(colors_str.split(',')[:5]) + ". "
                caption = color_prefix + caption
        else:
            caption = "a logo" 

        inputs = self.tokenizer(
            caption, padding="max_length", max_length=self.tokenizer.model_max_length, 
            truncation=True, return_tensors="pt"
        )
        
        with torch.no_grad():
            text_encoder_output = self.text_encoder(inputs.input_ids.to(self.text_encoder_device))
        
        text_condition = text_encoder_output.last_hidden_state.cpu()

        return {
            "pixel_values": pixel_values,
            "sketch_condition": sketch_condition,
            "text_condition": text_condition.squeeze(0),
        }