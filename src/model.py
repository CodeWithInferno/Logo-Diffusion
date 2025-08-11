# import torch
# from torch import nn
# from diffusers import UNet2DConditionModel, AutoencoderKL
# import copy

# class ControlNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         print("Initializing ControlNet model...")

#         self.unet = UNet2DConditionModel.from_pretrained(
#             "runwayml/stable-diffusion-v1-5", subfolder="unet"
#         )
#         self.vae = AutoencoderKL.from_pretrained(
#             "runwayml/stable-diffusion-v1-5", subfolder="vae"
#         )
#         print("Pre-trained models loaded.")

#         print("Freezing original U-Net and VAE weights...")
#         self.unet.requires_grad_(False)
#         self.vae.requires_grad_(False)
        
#         print("Creating trainable copy of U-Net blocks for ControlNet...")
#         self.controlnet_conv_in = copy.deepcopy(self.unet.conv_in)
#         self.controlnet_time_proj = copy.deepcopy(self.unet.time_proj)
#         self.controlnet_time_embedding = copy.deepcopy(self.unet.time_embedding)
#         self.controlnet_down_blocks = copy.deepcopy(self.unet.down_blocks)
#         self.controlnet_mid_block = copy.deepcopy(self.unet.mid_block)

#         print("Creating zero-convolution layers...")
#         self.zero_convs = self._create_zero_convs()
        
#         print("ControlNet initialization complete.")

#     def _create_zero_convs(self):
#         zero_convs = nn.ModuleList()
#         # Input block
#         zero_convs.append(self._create_zero_conv(self.unet.conv_in.out_channels, self.unet.conv_in.out_channels))
#         # Down blocks
#         for down_block in self.unet.down_blocks:
#             for _ in down_block.resnets:
#                 zero_convs.append(self._create_zero_conv(down_block.resnets[0].out_channels, down_block.resnets[0].out_channels))
#             if down_block.downsamplers:
#                 zero_convs.append(self._create_zero_conv(down_block.downsamplers[0].out_channels, down_block.downsamplers[0].out_channels))
#         # Mid block
#         zero_convs.append(self._create_zero_conv(self.unet.mid_block.resnets[-1].out_channels, self.unet.mid_block.resnets[-1].out_channels))
#         return zero_convs

#     def _create_zero_conv(self, in_channels, out_channels):
#         return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

#     def forward(self, noisy_latents, timesteps, text_condition, sketch_latents):
#         # Time embedding
#         t_emb = self.unet.time_proj(timesteps)
#         emb = self.unet.time_embedding(t_emb)

#         # ControlNet forward pass
#         control_hidden_states = self.controlnet_conv_in(sketch_latents)
#         control_down_block_residuals = [control_hidden_states]
#         for down_block in self.controlnet_down_blocks:
#             control_hidden_states, res_samples = down_block(
#                 hidden_states=control_hidden_states, temb=emb, encoder_hidden_states=text_condition
#             )
#             control_down_block_residuals.extend(res_samples)
#         control_mid_block_residual = self.controlnet_mid_block(
#             control_hidden_states, temb=emb, encoder_hidden_states=text_condition
#         )

#         # Apply zero convolutions
#         control_down_block_residuals = [
#             self.zero_convs[i](r) for i, r in enumerate(control_down_block_residuals)
#         ]
#         control_mid_block_residual = self.zero_convs[-1](control_mid_block_residual)

#         # U-Net forward pass with injected control
#         return self.unet(
#             noisy_latents,
#             timesteps,
#             encoder_hidden_states=text_condition,
#             down_block_additional_residuals=control_down_block_residuals,
#             mid_block_additional_residual=control_mid_block_residual,
#         ).sample


# src/model.py

import torch
from torch import nn
from diffusers import UNet2DConditionModel, AutoencoderKL, ControlNetModel

class FullControlNetModel(nn.Module):
    def __init__(self):
        super().__init__()
        print("Initializing Full ControlNet Model...")
        
        self.unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")
        self.vae = AutoencoderKL.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="vae")
        self.controlnet = ControlNetModel.from_unet(self.unet)

        print("Freezing original U-Net and VAE weights...")
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        print("Model initialization complete. The ControlNet part is trainable by default.")

    def forward(self, noisy_latents, timesteps, text_condition, sketch_condition):
        down_block_residuals, mid_block_residual = self.controlnet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_condition,
            controlnet_cond=sketch_condition, # Pass the sketch here
            return_dict=False,
        )

        noise_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=text_condition,
            down_block_additional_residuals=down_block_residuals,
            mid_block_additional_residual=mid_block_residual,
        ).sample
        
        return noise_pred