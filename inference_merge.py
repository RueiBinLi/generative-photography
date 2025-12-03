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
from inference_settings import Camera_Embedding_bokehK, Camera_Embedding_temp, Camera_Embedding_focal, Camera_Embedding_shutter

import torch as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnsembleCameraEncoder(nn.Module):
    def __init__(self, encoder_dict):
        super().__init__()
        self.encoders = nn.ModuleDict(encoder_dict)

    def forward(self, camera_embedding):
        total_feature = None

        for name, encoder in self.encoders.items():
            if name in camera_embedding:
                input = camera_embedding[name]

                feature = encoder(input)

                if total_feature is None:
                    total_feature = feature
                else:
                    total_feature = total_feature + feature
            else:
                print(f"Warning: {name} embedding not found in input.")
        return total_feature

def load_models(cfg):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    noise_scheduler = DDIMScheduler(**OmegaConf.to_container(cfg.noise_scheduler_kwargs))
    vae = AutoencoderKL.from_pretrained(cfg.pretrained_model_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(cfg.pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(cfg.pretrained_model_path, subfolder="text_encoder").to(device)
    text_encoder.requires_grad_(False)
    unet = UNet3DConditionModelCameraCond.from_pretrained_2d(
        cfg.pretrained_model_path,
        subfolder=cfg.unet_subfolder,
        unet_additional_kwargs=cfg.unet_additional_kwargs
    ).to(device)
    unet.requires_grad_(False)

    camera_encoder_bokehK = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder_bokehK.requires_grad_(False)
    camera_encoder_temp = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder_temp.requires_grad_(False)
    camera_encoder_focal = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder_focal.requires_grad_(False)
    camera_encoder_shutter = CameraCameraEncoder(**cfg.camera_encoder_kwargs).to(device)
    camera_encoder_shutter.requires_grad_(False)
    # camera_adaptor = CameraAdaptor(unet, camera_encoder)
    # camera_adaptor.requires_grad_(False)
    # camera_adaptor.to(device)

    logger.info("Setting the attention processors")
    unet.set_all_attn_processor(
        add_spatial_lora=cfg.lora_ckpt is not None,
        add_motion_lora=cfg.motion_lora_rank > 0,
        lora_kwargs={"lora_rank": cfg.lora_rank, "lora_scale": cfg.lora_scale},
        motion_lora_kwargs={"lora_rank": cfg.motion_lora_rank, "lora_scale": cfg.motion_lora_scale},
        **cfg.attention_processor_kwargs
    )

    if cfg.lora_ckpt is not None:
        print(f"Loading the lora checkpoint from {cfg.lora_ckpt}")
        lora_checkpoints = torch.load(cfg.lora_ckpt, map_location=unet.device)
        if 'lora_state_dict' in lora_checkpoints.keys():
            lora_checkpoints = lora_checkpoints['lora_state_dict']
        _, lora_u = unet.load_state_dict(lora_checkpoints, strict=False)
        assert len(lora_u) == 0
        print(f'Loading done')

    if cfg.motion_module_ckpt is not None:
        print(f"Loading the motion module checkpoint from {cfg.motion_module_ckpt}")
        mm_checkpoints = torch.load(cfg.motion_module_ckpt, map_location=unet.device)
        _, mm_u = unet.load_state_dict(mm_checkpoints, strict=False)
        assert len(mm_u) == 0
        print("Loading done")

    if cfg.camera_adaptor_ckpt_bokehK:
        ckpt = torch.load(cfg.camera_adaptor_ckpt_bokehK, map_location=device)
        camera_encoder_bokehK.load_state_dict(ckpt['camera_encoder_state_dict'], strict=False)

    if cfg.camera_adaptor_ckpt_temp:
        ckpt = torch.load(cfg.camera_adaptor_ckpt_temp, map_location=device)
        camera_encoder_temp.load_state_dict(ckpt['camera_encoder_state_dict'], strict=False)

    if cfg.camera_adaptor_ckpt_focal:
        ckpt = torch.load(cfg.camera_adaptor_ckpt_focal, map_location=device)
        camera_encoder_focal.load_state_dict(ckpt['camera_encoder_state_dict'], strict=False)

    if cfg.camera_adaptor_ckpt_shutter:
        ckpt = torch.load(cfg.camera_adaptor_ckpt_shutter, map_location=device)
        camera_encoder_shutter.load_state_dict(ckpt['camera_encoder_state_dict'], strict=False)

    ensemble_encoder = EnsembleCameraEncoder({
        'bokehK': camera_encoder_bokehK,
        'temp': camera_encoder_temp,
        'shutter': camera_encoder_shutter,
        'focal': camera_encoder_focal,
    })

    pipeline = GenPhotoPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        camera_encoder=ensemble_encoder
    ).to(device)

    pipeline.enable_vae_slicing()

    return pipeline, device


def run_inference(pipeline, tokenizer, text_encoder, base_scene, bokehK_list, color_temperature_list, focal_length_list, shutter_speed_list, filename, output_dir, device, video_length=5, height=256, width=384):
    os.makedirs(output_dir, exist_ok=True)

    bokehK_list_str = bokehK_list
    bokehK_values = json.loads(bokehK_list_str)
    bokehK_values = torch.tensor(bokehK_values).unsqueeze(1)

    camera_embedding_bokehK = Camera_Embedding_bokehK(bokehK_values, tokenizer, text_encoder, device).load()
    camera_embedding_bokehK = rearrange(camera_embedding_bokehK.unsqueeze(0), "b f c h w -> b c f h w")

    color_temperature_list_str = color_temperature_list
    color_temperature_values = json.loads(color_temperature_list_str)
    color_temperature_values = torch.tensor(color_temperature_values).unsqueeze(1)
    
    num_frames = len(color_temperature_values)

    camera_embedding_temp = Camera_Embedding_temp(color_temperature_values, tokenizer, text_encoder, device).load()
    camera_embedding_temp = rearrange(camera_embedding_temp.unsqueeze(0), "b f c h w -> b c f h w")

    focal_length_list_str = focal_length_list
    focal_length_values = json.loads(focal_length_list_str)
    focal_length_values = torch.tensor(focal_length_values).unsqueeze(1)

    num_frames = len(focal_length_values)

    camera_embedding_focal = Camera_Embedding_focal(focal_length_values, tokenizer, text_encoder, device).load()
    camera_embedding_focal = rearrange(camera_embedding_focal.unsqueeze(0), "b f c h w -> b c f h w")

    shutter_speed_list_str = shutter_speed_list
    shutter_speed_values = json.loads(shutter_speed_list_str)
    shutter_speed_values = torch.tensor(shutter_speed_values).unsqueeze(1)

    camera_embedding_shutter = Camera_Embedding_shutter(shutter_speed_values, tokenizer, text_encoder, device).load()
    camera_embedding_shutter = rearrange(camera_embedding_shutter.unsqueeze(0), "b f c h w -> b c f h w")

    embeddings = {
        'bokehK': camera_embedding_bokehK,
        'temp': camera_embedding_temp,
        'focal': camera_embedding_focal,
        'shutter': camera_embedding_shutter,
    }

    with torch.no_grad():
        sample = pipeline(
            prompt=base_scene,
            camera_embedding=embeddings,
            video_length=video_length,
            height=height,
            width=width,
            num_inference_steps=25,
            guidance_scale=8.0
        ).videos[0]

    output_name = filename + '.gif'
    sample_save_path = os.path.join(output_dir, output_name)
    save_videos_grid(sample[None, ...], sample_save_path)
    logger.info(f"Saved generated sample to {sample_save_path}")


def main(config_path, base_scene, bokehK_list, color_temperature_list, focal_length_list, shutter_speed_list, filename):
    torch.manual_seed(42)
    cfg = OmegaConf.load(config_path)
    logger.info("Loading models...")
    pipeline, device = load_models(cfg)
    logger.info("Starting inference...")

    run_inference(pipeline, pipeline.tokenizer, pipeline.text_encoder, base_scene, bokehK_list, color_temperature_list, focal_length_list, shutter_speed_list, filename, cfg.output_dir, device=device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--base_scene", type=str, required=True, help="invariant scene caption as JSON string")
    parser.add_argument("--bokehK_list", type=str, required=False, default='', help="Bokeh K values as JSON string")
    parser.add_argument("--color_temperature_list", type=str, required=False, default='', help="color_temperature values as JSON string")
    parser.add_argument("--focal_length_list", type=str, required=False, default='', help="focal_length values as JSON string")
    parser.add_argument("--shutter_speed_list", type=str, required=False, default='', help="shutter_speed values as JSON string")
    parser.add_argument("--filename", type=str, required=True, help="filename for gif")
    args = parser.parse_args()
    
    if args.bokehK_list == '' and args.color_temperature_list == '' and args.focal_length_list == '' and args.shutter_speed_list == '':
        logger.info("Error: You must input at least one camera setting.")
        print("Error: You must input at least one camera setting.")
        exit(1)
        
    main(args.config, args.base_scene, args.bokehK_list, args.color_temperature_list, args.focal_length_list, args.shutter_speed_list, args.filename)

    # usage example
    # python inference_bokehK.py --config configs/inference_genphoto/adv3_256_384_genphoto_relora_bokehK.yaml --base_scene "A young boy wearing an orange jacket is standing on a crosswalk, waiting to cross the street." --bokehK_list "[2.44, 8.3, 10.1, 17.2, 24.0]"

