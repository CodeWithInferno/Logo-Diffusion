# import os
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from diffusers import DDPMScheduler
# from transformers import get_cosine_schedule_with_warmup
# from tqdm.auto import tqdm

# # Import from our other local scripts
# from dataset import LogoDataset
# from model import ControlNet

# # --- Configuration ---
# # This block creates absolute paths, making the script runnable from any directory
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(SCRIPT_DIR)
# METADATA_PATH = os.path.join(ROOT_DIR, 'data', 'metadata.csv')
# MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, 'models')

# CHECKPOINT_NAME = 'logo_controlnet.pth'
# LEARNING_RATE = 1e-5
# WARMUP_STEPS = 500
# BATCH_SIZE = 8      # Start with 4 for a 24GB card, can increase to 6 or 8 if memory allows
# NUM_EPOCHS = 10     # A starting point, can be increased for better results
# SAVE_STEP_FREQUENCY = 1000 # Save a checkpoint every 1000 steps

# def train():
#     """Main training function."""
#     print("--- Starting Training ---")

#     # --- Setup ---
#     os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"Using device: {device}")

#     print("Initializing Dataset and DataLoader...")
#     dataset = LogoDataset(metadata_path=METADATA_PATH, text_encoder_device=device)
#     dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    
#     print("Initializing Model...")
#     model = ControlNet().to(device)
    
#     print("Initializing Optimizer and Scheduler...")
#     # Important: Only optimize the trainable parameters of our ControlNet
#     trainable_params = list(model.controlnet_down_blocks.parameters()) + \
#                                           list(model.controlnet_mid_block.parameters()) + \
#                      list(model.zero_convs.parameters())
#     optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    
#     num_training_steps = len(dataloader) * NUM_EPOCHS
#     lr_scheduler = get_cosine_schedule_with_warmup(
#         optimizer,
#         num_warmup_steps=WARMUP_STEPS,
#         num_training_steps=num_training_steps
#     )

#     print("Initializing Noise Scheduler...")
#     noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    
#     scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
#     global_step = 0
#     print(f"Starting training for {NUM_EPOCHS} epochs...")

#     # --- Training Loop ---
#     for epoch in range(NUM_EPOCHS):
#         progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
#         for step, batch in enumerate(progress_bar):
#             pixel_values = batch["pixel_values"].to(device)
#             sketch_condition = batch["sketch_condition"].to(device)
#             text_condition = batch["text_condition"].to(device)

#             with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
#                 with torch.no_grad():
#                     latents = model.vae.encode((pixel_values + 1.0) / 2.0).latent_dist.sample() * model.vae.config.scaling_factor
#                     sketch_condition_rgb = sketch_condition.repeat(1,3,1,1)
#                     sketch_latents = model.vae.encode((sketch_condition_rgb + 1.0) / 2.0).latent_dist.sample() * model.vae.config.scaling_factor

#                 noise = torch.randn_like(latents)
#                 timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
#                 noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

#                 noise_pred = model(
#                     noisy_latents=noisy_latents,
#                     timesteps=timesteps,
#                     text_condition=text_condition,
#                     sketch_latents=sketch_latents
#                 )
#                 loss = F.mse_loss(noise_pred.float(), noise.float())

#             optimizer.zero_grad(set_to_none=True)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#             lr_scheduler.step()

#             global_step += 1
#             progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

#             if global_step % SAVE_STEP_FREQUENCY == 0:
#                 save_path = os.path.join(MODEL_OUTPUT_DIR, f"checkpoint_step_{global_step}.pth")
#                 save_weights = {
#                     'controlnet_down_blocks': model.controlnet_down_blocks.state_dict(),
#                     'controlnet_mid_block': model.controlnet_mid_block.state_dict(),
#                     'zero_convs': model.zero_convs.state_dict()
#                 }
#                 torch.save(save_weights, save_path)
#                 print(f"\nCheckpoint saved to {save_path}")

#     print("\n--- Training Complete ---")
#     final_save_path = os.path.join(MODEL_OUTPUT_DIR, CHECKPOINT_NAME)
#     final_weights = {
#         'controlnet_down_blocks': model.controlnet_down_blocks.state_dict(),
#         'controlnet_mid_block': model.controlnet_mid_block.state_dict(),
#         'zero_convs': model.zero_convs.state_dict()
#     }
#     torch.save(final_weights, final_save_path)
#     print(f"Final model saved to {final_save_path}")

# if __name__ == '__main__':
#     # Set the start method to 'spawn' for CUDA compatibility with multiprocessing
#     import torch
#     try:
#         torch.multiprocessing.set_start_method('spawn')
#     except RuntimeError:
#         # The start method can only be set once.
#         pass
#     train()



# src/train.py

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler
from transformers import get_cosine_schedule_with_warmup
from tqdm.auto import tqdm

from dataset import LogoDataset
from model import FullControlNetModel # Make sure you are using the correct model class

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
METADATA_PATH = os.path.join(ROOT_DIR, 'data', 'metadata.csv')
MODEL_OUTPUT_DIR = os.path.join(ROOT_DIR, 'models')

CHECKPOINT_NAME = 'logo_controlnet.pth'
LEARNING_RATE = 1e-5
WARMUP_STEPS = 500
BATCH_SIZE = 4
NUM_EPOCHS = 10
SAVE_STEP_FREQUENCY = 1000

def train():
    print("--- Starting Training ---")

    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Initializing Dataset and DataLoader...")
    dataset = LogoDataset(metadata_path=METADATA_PATH, text_encoder_device=device)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    print("Initializing Model...")
    model = FullControlNetModel().to(device)
    
    print("Initializing Optimizer and Scheduler...")
    optimizer = torch.optim.AdamW(model.controlnet.parameters(), lr=LEARNING_RATE)
    
    num_training_steps = len(dataloader) * NUM_EPOCHS
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps
    )

    print("Initializing Noise Scheduler...")
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")
    
    scaler = torch.amp.GradScaler()
    global_step = 0
    print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        
        for step, batch in enumerate(progress_bar):
            if batch is None: continue
                
            pixel_values = batch["pixel_values"].to(device)
            sketch_condition = batch["sketch_condition"].to(device)
            text_condition = batch["text_condition"].to(device)

            with torch.amp.autocast(device_type="cuda"):
                with torch.no_grad():
                    # Encode the real image to latents
                    latents = model.vae.encode((pixel_values + 1.0) / 2.0).latent_dist.sample() * model.vae.config.scaling_factor
                
                # --- THIS IS THE FIX ---
                # The sketch is NOT encoded. We just expand it to 3 channels.
                sketch_condition_rgb = sketch_condition.repeat(1, 3, 1, 1)

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # The model's forward pass now takes the 3-channel sketch directly
                noise_pred = model(
                    noisy_latents=noisy_latents, 
                    timesteps=timesteps, 
                    text_condition=text_condition, 
                    sketch_condition=sketch_condition_rgb # Pass the 3-channel sketch
                )
                loss = F.mse_loss(noise_pred.float(), noise.float())

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            global_step += 1
            progress_bar.set_postfix(loss=loss.item(), lr=lr_scheduler.get_last_lr()[0])

            if global_step % SAVE_STEP_FREQUENCY == 0:
                save_path = os.path.join(MODEL_OUTPUT_DIR, f"checkpoint_step_{global_step}")
                model.controlnet.save_pretrained(save_path)
                print(f"\nCheckpoint saved to {save_path}")

    print("\n--- Training Complete ---")
    final_save_path = os.path.join(MODEL_OUTPUT_DIR, CHECKPOINT_NAME)
    model.controlnet.save_pretrained(final_save_path)
    print(f"Final model saved to {final_save_path}")

if __name__ == '__main__':
    import torch
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    train()