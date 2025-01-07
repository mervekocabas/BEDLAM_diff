# @GonzaloMartinGarcia
# Training code for 'Fine-Tuning Image-Conditional Diffusion Models is Easier than You Think'.
# This training code is a modified version of the original text-to-image SD training code from the HuggingFace Inc. Team,
# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py.

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
import torch.nn.init as init
import torch.nn.functional as F

from accelerate import Accelerator
from transformers import CLIPTextModel, CLIPTokenizer

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel

class E2E(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        # Load model components
        pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2")
        self.noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-2", subfolder="scheduler", timestep_spacing="trailing")
        #self.noise_scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2", subfolder="scheduler")
        self.tokenizer    = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2", subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="text_encoder")
        self.vae          = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder="vae")
        self.unet         = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet", out_channels=1024, low_cpu_mem_usage=False, ignore_mismatched_sizes=True)
        #self.unet         = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-2", subfolder="unet")

        # Repeat weights
        self.unet.conv_out.weight.data = init.xavier_uniform_(self.unet.conv_out.weight, gain=0.01)
        
        # Explanation: 1280 / 4 = 320 repetitions along the first dimension (channel)
        #self.unet.conv_out.weight.data = pipe.unet.conv_out.weight.data.detach().clone().repeat(320, 1, 1, 1)    
        
        # Divide the original weights by 320
        #scaled_weight = pipe.unet.conv_out.weight.data / 180

        # Repeat the 4-channel weights to create a 320-channel tensor
        #self.unet.conv_out.weight.data = scaled_weight.detach().clone().repeat(180, 1, 1, 1)

        # Repeat biases
        self.unet.conv_out.bias.data = init.zeros_(self.unet.conv_out.bias)
        
        #import ipdb; ipdb.set_trace()
        # Repeat the bias 320 times to match 1280 channels
        #self.unet.conv_out.bias.data = pipe.unet.conv_out.bias.data.detach().clone().repeat(320)
        
        # Divide the original biases by 320
        #scaled_bias = pipe.unet.conv_out.bias.data / 180

        # Repeat the 4-channel biases to create a 320-channel tensor
        #self.unet.conv_out.bias.data = scaled_bias.detach().clone().repeat(180)

        # Freeze or set model components to training mode
        self.vae.requires_grad_(False)
        #self.vae.train()
        self.text_encoder.requires_grad_(False)
        #self.unet.requires_grad_(False)
        self.unet.train()
        #self.unet.enable_xformers_memory_efficient_attention()
        #self.unet.enable_gradient_checkpointing()
        
        # Pre-compute empty text CLIP encoding
        self.empty_token    = self.tokenizer([""], padding="max_length", truncation=True, return_tensors="pt").input_ids
        self.empty_encoding = self.text_encoder(self.empty_token, return_dict=False)[0]
    
    def forward(self, images, categories=None, prompts=None):
        device = images.device
        
        # RGB latent
        rgb_latents = encode_image(self.vae, images)
        #rgb_latents = self.vae.encode(images).latent_dist.mode()
        rgb_latents = (rgb_latents * self.vae.config.scaling_factor).to(device)
        
        # Set timesteps to the first denoising step
        timesteps = torch.ones((rgb_latents.shape[0])) * (self.noise_scheduler.config.num_train_timesteps-1) # 999
        timesteps = (timesteps.long()).to(device)
                    
        # Generate UNet prediction
        encoder_hidden_states = self.empty_encoding.repeat(len(images), 1, 1).to(device)
        model_pred = self.unet(rgb_latents, timesteps, encoder_hidden_states, return_dict=False)[0]
                        
        # Decode latent prediction
        model_pred = model_pred / self.vae.config.scaling_factor
        #out_conv = (nn.Conv2d(4, 720, 3, stride=1, padding=1)).to(device)
        #current_estimate = out_conv(model_pred)
        #return current_estimate
        return model_pred
    
def encode_image(vae, image):
        h = vae.encoder(image)
        moments = vae.quant_conv(h)
        latent, _ = torch.chunk(moments, 2, dim=1)
        return latent
    