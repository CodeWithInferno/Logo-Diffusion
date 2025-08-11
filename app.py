import gradio as gr
import torch
from torchvision import transforms
from diffusers import DDPMScheduler
from transformers import CLIPTokenizer
from PIL import Image
import numpy as np
import os
import sys
from safetensors.torch import load_file

# --- Add src to path ---
# This allows importing from the src directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'src'))

from model import FullControlNetModel
from dataset import LogoDataset # We need this for the tokenizer and text encoder setup

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = 'models/checkpoint_step_26000'

# --- Load Models ---
print("Loading models...")
try:
    # We use the dataset class to get the correctly configured tokenizer and text_encoder
    dataset = LogoDataset(metadata_path='data/metadata.csv', text_encoder_device=DEVICE)
    tokenizer = dataset.tokenizer
    text_encoder = dataset.text_encoder

    # Load the full model architecture
    model = FullControlNetModel().to(DEVICE)
    
    # Load the trained ControlNet weights from the specified checkpoint
    model.controlnet.load_state_dict(load_file(os.path.join(CHECKPOINT_PATH, 'diffusion_pytorch_model.safetensors'), device=DEVICE))
    
    vae = model.vae
    unet = model.unet
    controlnet = model.controlnet
    
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    # It's helpful to see the full traceback for debugging
    import traceback
    traceback.print_exc()
    exit()

# --- Noise Scheduler ---
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

# --- Image Transformations ---
sketch_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def generate_logo(prompt: str, sketch_image: Image.Image):
    """
    Generates a logo based on a text prompt and a sketch image.
    """
    if sketch_image is None:
        # Return a placeholder or an error message if no sketch is provided
        return Image.new('RGB', (512, 512), color = 'gray')

    # --- 1. Preprocess Inputs ---
    sketch_image = sketch_image.convert("L")
    sketch_condition = sketch_transforms(sketch_image).unsqueeze(0).to(DEVICE)
    sketch_condition_rgb = sketch_condition.repeat(1, 3, 1, 1)

    inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        text_embeddings = text_encoder(inputs.input_ids.to(DEVICE)).last_hidden_state

    # --- 2. Generation Process ---
    latents = torch.randn((1, unet.in_channels, 64, 64), device=DEVICE)
    noise_scheduler.set_timesteps(50)

    for t in noise_scheduler.timesteps:
        with torch.no_grad():
            down_block_res_samples, mid_block_res_sample = controlnet(
                latents,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=sketch_condition_rgb,
                return_dict=False,
            )
            
            noise_pred = unet(
                latents,
                t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample
            
            latents = noise_scheduler.step(noise_pred, t, latents).prev_sample

    # --- 3. Decode and Post-process ---
    with torch.no_grad():
        latents = 1 / vae.config.scaling_factor * latents
        image = vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image[0] * 255).round().astype("uint8")
        pil_image = Image.fromarray(image)

    return pil_image

# --- Gradio Interface ---
with gr.Blocks() as demo:
    gr.Markdown("# Logo Diffusion")
    gr.Markdown("Generate a logo from a text prompt and a sketch.")
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(label="Prompt", placeholder="e.g., a majestic lion head, minimalist, vector style")
            sketch_input = gr.Image(type="pil", label="Sketch")
            generate_button = gr.Button("Generate")
        with gr.Column():
            output_image = gr.Image(label="Generated Logo")

    generate_button.click(
        fn=generate_logo,
        inputs=[prompt_input, sketch_input],
        outputs=output_image
    )

if __name__ == "__main__":
    # A dummy metadata file is needed for the dataset class to initialize
    if not os.path.exists('data/metadata.csv'):
        os.makedirs('data', exist_ok=True)
        with open('data/metadata.csv', 'w') as f:
            f.write('image_path,sketch_path,colors,caption\n')
            f.write('dummy.png,dummy.png,red,dummy\n')
    if not os.path.exists('dummy.png'):
        Image.new('RGB', (100, 100), color = 'red').save('dummy.png')
        
    demo.launch(share=True)
