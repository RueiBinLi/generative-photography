import os
import torch
import logging
import argparse
import json
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler
from einops import rearrange

from genphoto.pipelines.pipeline_animation import GenPhotoPipeline
from genphoto.models.unet import UNet3DConditionModelCameraCond
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor
from genphoto.utils.util import save_videos_grid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_bokehK_embedding(bokehK_values, target_height, target_width):
    f = bokehK_values.shape[0]
    bokehK_embedding = torch.zeros((f, 3, target_height, target_width), dtype=bokehK_values.dtype)
    
    for i in range(f):
        K_value = bokehK_values[i].item()
        kernel_size = max(K_value, 1)
        sigma = K_value / 3.0

        ax = np.linspace(-(kernel_size / 2), kernel_size / 2, int(np.ceil(kernel_size)))
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
        kernel /= np.sum(kernel)
        scale = kernel[int(np.ceil(kernel_size) / 2), int(np.ceil(kernel_size) / 2)]
        
        bokehK_embedding[i] = scale
    
    return bokehK_embedding



class Camera_Embedding_bokehK(Dataset):
    def __init__(self, bokehK_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.bokehK_values = bokehK_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device  
        self.sample_size = sample_size

    def load(self):

        # if len(self.bokehK_values) != 5:
        #     raise ValueError("Expected 5 bokehK values")

        # Generate prompts for each bokehK value and append bokehK information to caption
        prompts = []
        for bb in self.bokehK_values:
            prompt = f"<bokeh kernel size: {bb.item()}>"
            prompts.append(prompt)
        

        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        

        # Calculate differences between consecutive embeddings (ignoring sequence_length)
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        # Add the difference between the last and the first embedding
        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0) 

        frame = concatenated_differences.size(0)

        # Concatenate differences along the batch dimension (f)
        concatenated_differences = torch.cat(differences, dim=0)

        pad_length = 128 - concatenated_differences.size(1)

        if pad_length > 0:
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))


        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        bokehK_embedding = create_bokehK_embedding(self.bokehK_values, self.sample_size[0], self.sample_size[1]).to(self.device)
        camera_embedding = torch.cat((bokehK_embedding, ccl_embedding), dim=1)
        return camera_embedding
    
def kelvin_to_rgb(kelvin):
    if torch.is_tensor(kelvin):
        kelvin = kelvin.cpu().item()  

    temp = kelvin / 100.0

    if temp <= 66:
        red = 255
        green = 99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0
        if temp <= 19:
            blue = 0
        else:
            blue = 138.5177312231 * np.log(temp - 10) - 305.0447927307

    elif 66 < temp <= 88:
        red = 0.5 * (255 + 329.698727446 * ((temp - 60) ** -0.19332047592))
        green = 0.5 * (288.1221695283 * ((temp - 60) ** -0.1155148492) + 
                       (99.4708025861 * np.log(temp) - 161.1195681661 if temp > 0 else 0))
        blue = 0.5 * (138.5177312231 * np.log(temp - 10) - 305.0447927307 + 255)

    else:
        red = 329.698727446 * ((temp - 60) ** -0.19332047592)
        green = 288.1221695283 * ((temp - 60) ** -0.1155148492)
        blue = 255

    return np.array([red, green, blue], dtype=np.float32) / 255.0


def create_color_temperature_embedding(color_temperature_values, target_height, target_width, min_color_temperature=2000, max_color_temperature=10000):

    f = color_temperature_values.shape[0]
    rgb_factors = []

    # Compute RGB factors based on kelvin_to_rgb function
    for color_temperature in color_temperature_values.squeeze():
        kelvin = min_color_temperature + (color_temperature * (max_color_temperature - min_color_temperature))  # Map normalized color_temperature to actual Kelvin
        rgb = kelvin_to_rgb(kelvin)
        rgb_factors.append(rgb)
    
    # Convert to tensor and expand to target dimensions
    rgb_factors = torch.tensor(rgb_factors).float()  # [f, 3]
    rgb_factors = rgb_factors.unsqueeze(2).unsqueeze(3)  # [f, 3, 1, 1]
    color_temperature_embedding = rgb_factors.expand(f, 3, target_height, target_width)  # [f, 3, target_height, target_width]

    return color_temperature_embedding



class Camera_Embedding_color(Dataset):
    def __init__(self, color_temperature_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.color_temperature_values = color_temperature_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device  
        self.sample_size = sample_size

    def load(self):

        if len(self.color_temperature_values) != 5:
            raise ValueError("Expected 5 color_temperature values")

        # Generate prompts for each color_temperature value and append color_temperature information to caption
        prompts = []
        for ct in self.color_temperature_values:
            prompt = f"<color temperature: {ct.item()}>"
            prompts.append(prompt)
     

        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        

        # Calculate differences between consecutive embeddings (ignoring sequence_length)
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  


        # Add the difference between the last and the first embedding
        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0) 
        frame = concatenated_differences.size(0)
        concatenated_differences = torch.cat(differences, dim=0)

        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))


        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        color_temperature_embedding = create_color_temperature_embedding(self.color_temperature_values, self.sample_size[0], self.sample_size[1]).to(self.device)
        camera_embedding = torch.cat((color_temperature_embedding, ccl_embedding), dim=1)
        return camera_embedding
    
def create_focal_length_embedding(focal_length_values, target_height, target_width, base_focal_length=24.0, sensor_height=24.0, sensor_width=36.0):
    device = 'cpu'
    focal_length_values = focal_length_values.to(device)
    f = focal_length_values.shape[0]  # Number of frames


    # Convert constants to tensors to perform operations with focal_length_values
    sensor_width = torch.tensor(sensor_width, device=device)
    sensor_height = torch.tensor(sensor_height, device=device)
    base_focal_length = torch.tensor(base_focal_length, device=device)

    # Calculate the FOV for the base focal length (min_focal_length)
    base_fov_x = 2.0 * torch.atan(sensor_width * 0.5 / base_focal_length)
    base_fov_y = 2.0 * torch.atan(sensor_height * 0.5 / base_focal_length)

    # Calculate the FOV for each focal length in focal_length_values
    target_fov_x = 2.0 * torch.atan(sensor_width * 0.5 / focal_length_values)
    target_fov_y = 2.0 * torch.atan(sensor_height * 0.5 / focal_length_values)

    # Calculate crop ratio: how much of the image is cropped at the current focal length
    crop_ratio_xs = target_fov_x / base_fov_x  # Crop ratio for horizontal axis
    crop_ratio_ys = target_fov_y / base_fov_y  # Crop ratio for vertical axis

    # Get the center of the image
    center_h, center_w = target_height // 2, target_width // 2

    # Initialize a mask tensor with zeros on CPU
    focal_length_embedding = torch.zeros((f, 3, target_height, target_width), dtype=torch.float32)  # Shape [f, 3, H, W]

    # Fill the center region with 1 based on the calculated crop dimensions
    for i in range(f):
        # Crop dimensions calculated using rounded float values
        crop_h = torch.round(crop_ratio_ys[i] * target_height).int().item()  # Rounded cropped height for the current frame
        crop_w = torch.round(crop_ratio_xs[i] * target_width).int().item()  # Rounded cropped width for the current frame

        # Ensure the cropped dimensions are within valid bounds
        crop_h = max(1, min(target_height, crop_h))
        crop_w = max(1, min(target_width, crop_w))

        # Set the center region of the focal_length embedding to 1 for the current frame
        focal_length_embedding[i, :,
        center_h - crop_h // 2: center_h + crop_h // 2,
        center_w - crop_w // 2: center_w + crop_w // 2] = 1.0

    return focal_length_embedding


class Camera_Embedding_focal(Dataset):
    def __init__(self, focal_length_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.focal_length_values = focal_length_values.to(device)                                                 
        self.tokenizer = tokenizer                 
        self.text_encoder = text_encoder       
        self.device = device               
        self.sample_size = sample_size

    def load(self):

        # if len(self.focal_length_values) != 5:
        #     raise ValueError("Expected 5 focal_length values")

        # Generate prompts for each focal length value and append focal_length information to caption
        prompts = []
        for fl in self.focal_length_values:
            prompt = f"<focal length: {fl.item()}>"
            prompts.append(prompt)
        

        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        

        # Calculate differences between consecutive embeddings (ignoring sequence_length)
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        # Add the difference between the last and the first embedding
        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0) 
        frame = concatenated_differences.size(0)
        concatenated_differences = torch.cat(differences, dim=0)

        pad_length = 128 - concatenated_differences.size(1)
        if pad_length > 0:
        # Pad along the second dimension (77 -> 128), pad only on the right side
            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))


        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        focal_length_embedding = create_focal_length_embedding(self.focal_length_values, self.sample_size[0], self.sample_size[1]).to(self.device)

        camera_embedding = torch.cat((focal_length_embedding, ccl_embedding), dim=1)
        return camera_embedding
    
def create_shutter_speed_embedding(shutter_speed_values, target_height, target_width, base_exposure=0.5):

    f = shutter_speed_values.shape[0]

    # Set a constant full well capacity (fwc)
    fwc = 32000  # Constant value for full well capacity

    # Calculate scale based on EV and sensor full well capacity (fwc)
    scales = (shutter_speed_values / base_exposure) * (fwc / (fwc + 0.0001))

    # Reshape and expand to match image dimensions
    scales = scales.unsqueeze(2).unsqueeze(3).expand(f, 3, target_height, target_width)

    # Use scales to create the final shutter_speed embedding
    shutter_speed_embedding = scales      # Shape [f, 3, H, W]

    return shutter_speed_embedding



class Camera_Embedding_shutter(Dataset):
    def __init__(self, shutter_speed_values, tokenizer, text_encoder, device, sample_size=[256, 384]):
        self.shutter_speed_values = shutter_speed_values.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.device = device  
        self.sample_size = sample_size

    def load(self):

        if len(self.shutter_speed_values) != 5:
            raise ValueError("Expected 5 shutter_speed values")

        # Generate prompts for each shutter_speed value and append shutter_speed information to caption
        prompts = []
        for ss in self.shutter_speed_values:
            prompt = f"<exposure: {ss.item()}>"
            prompts.append(prompt)

        # Tokenize prompts and encode to get embeddings
        with torch.no_grad():
            prompt_ids = self.tokenizer(
                prompts, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).input_ids.to(self.device)

            encoder_hidden_states = self.text_encoder(input_ids=prompt_ids).last_hidden_state  # Shape: (f, sequence_length, hidden_size)
        

        # Calculate differences between consecutive embeddings (ignoring sequence_length)
        differences = []
        for i in range(1, encoder_hidden_states.size(0)):
            diff = encoder_hidden_states[i] - encoder_hidden_states[i - 1]
            diff = diff.unsqueeze(0)
            differences.append(diff)  

        # Add the difference between the last and the first embedding
        final_diff = encoder_hidden_states[-1] - encoder_hidden_states[0]
        final_diff = final_diff.unsqueeze(0)
        differences.append(final_diff)

        # Concatenate differences along the batch dimension (f-1)
        concatenated_differences = torch.cat(differences, dim=0) 
        frame = concatenated_differences.size(0)

        concatenated_differences = torch.cat(differences, dim=0)
        pad_length = 128 - concatenated_differences.size(1)

        if pad_length > 0:

            concatenated_differences_padded = F.pad(concatenated_differences, (0, 0, 0, pad_length))


        ccl_embedding = concatenated_differences_padded.reshape(frame, self.sample_size[0], self.sample_size[1])
        ccl_embedding = ccl_embedding.unsqueeze(1)  
        ccl_embedding = ccl_embedding.expand(-1, 3, -1, -1)
        ccl_embedding = ccl_embedding.to(self.device)
        shutter_speed_embedding = create_shutter_speed_embedding(self.shutter_speed_values, self.sample_size[0], self.sample_size[1]).to(self.device)
        camera_embedding = torch.cat((shutter_speed_embedding, ccl_embedding), dim=1)
        return camera_embedding