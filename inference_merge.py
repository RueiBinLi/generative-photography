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
from genphoto.models.camera_adaptor import CameraCameraEncoder, CameraAdaptor, EnsembleCameraEncoder
from genphoto.utils.util import save_videos_grid
from inference_settings import Camera_Embedding_bokehK, Camera_Embedding_temp, Camera_Embedding_focal, Camera_Embedding_shutter

import torch as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    if getattr(cfg, 'camera_adaptor_ckpt_bokehK', None):
        print(f"Loading BokehK Encoder: {cfg.camera_adaptor_ckpt_bokehK}")
        ckpt = torch.load(cfg.camera_adaptor_ckpt_bokehK, map_location=device)
        camera_encoder_bokehK.load_state_dict(ckpt['camera_encoder_state_dict'], strict=False)
    
    if getattr(cfg, 'camera_adaptor_ckpt_temp', None):
        print(f"Loading Color Temperature Encoder: {cfg.camera_adaptor_ckpt_temp}")
        ckpt = torch.load(cfg.camera_adaptor_ckpt_temp, map_location=device)
        camera_encoder_temp.load_state_dict(ckpt['camera_encoder_state_dict'], strict=False)

    if getattr(cfg, 'camera_adaptor_ckpt_focal', None):
        print(f"Loading Focal Length Encoder: {cfg.camera_adaptor_ckpt_focal}")
        ckpt = torch.load(cfg.camera_adaptor_ckpt_focal, map_location=device)
        camera_encoder_focal.load_state_dict(ckpt['camera_encoder_state_dict'], strict=False)

    if getattr(cfg, 'camera_adaptor_ckpt_shutter', None):
        print(f"Loading Shutter Speed Encoder: {cfg.camera_adaptor_ckpt_shutter}")
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

    logger.info("Loading and Merging LoRAs...")

    active_adapters = []

    def load_lora_safe(ckpt_path, adapter_name):
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"Loading LoRA: {adapter_name} from {ckpt_path}")

            try:
                pipeline.load_lora_weights(ckpt_path, adapter_name=adapter_name)
                active_adapters.append(adapter_name)
            except Exception as e:
                print(f"Failed to load {adapter_name}: {e}")

    load_lora_safe(getattr(cfg, 'camera_adaptor_ckpt_bokehK', None), "bokehK")
    load_lora_safe(getattr(cfg, 'camera_adaptor_ckpt_temp', None), "temp")
    load_lora_safe(getattr(cfg, 'camera_adaptor_ckpt_focal', None), "focal")
    load_lora_safe(getattr(cfg, 'camera_adaptor_ckpt_shutter', None), "shutter")

    if  active_adapters:
        print(f"Activating adapters: {active_adapters}")
        pipeline.set_adapters(active_adapters, adapter_weights=[0.5] * len(active_adapters))
    else:
        print("No LoRAs loaded. Running with base model only.")

    pipeline.enable_vae_slicing()

    return pipeline, device


def run_inference(pipeline, tokenizer, text_encoder, base_scene, bokehK_list, color_temperature_list, focal_length_list, shutter_speed_list, filename, output_dir, device, video_length=5, height=256, width=384):
    os.makedirs(output_dir, exist_ok=True)

    embeddings = {}
    num_frames = video_length # default

    if bokehK_list:
        bokehK_values = json.loads(bokehK_list)
        num_frames = len(bokehK_values)
        bokehK_values = torch.tensor(bokehK_values).unsqueeze(1)
        emb = Camera_Embedding_bokehK(bokehK_values, tokenizer, text_encoder, device).load()
        embeddings['bokehK'] = rearrange(emb.unsqueeze(0), "b f c h w -> b c f h w")

    if color_temperature_list:
        color_temperature_values = json.loads(color_temperature_list)
        num_frames = len(color_temperature_values)
        color_temperature_values = torch.tensor(color_temperature_values).unsqueeze(1)
        emb = Camera_Embedding_temp(color_temperature_values, tokenizer, text_encoder, device).load()
        embeddings['temp'] = rearrange(emb.unsqueeze(0), "b f c h w -> b c f h w")

    if focal_length_list:
        focal_length_values = json.loads(focal_length_list)
        num_frames = len(focal_length_values)
        focal_length_values = torch.tensor(focal_length_values).unsqueeze(1)
        emb = Camera_Embedding_focal(focal_length_values, tokenizer, text_encoder, device).load()
        embeddings['focal'] = rearrange(emb.unsqueeze(0), "b f c h w -> b c f h w")
    
    if shutter_speed_list:
        shtter_speed_values = json.loads(shutter_speed_list)
        num_frames = len(shtter_speed_values)
        shtter_speed_values = torch.tensor(shtter_speed_values).unsqueeze(1)
        emb = Camera_Embedding_shutter(shtter_speed_values, tokenizer, text_encoder, device).load()
        embeddings['shutter'] = rearrange(emb.unsqueeze(0), "b f c h w -> b c f h w")

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
    parser.add_argument("--config", type=str, required=True, help="Path to YAML dfasfd file")
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
    # python inference_merge.py --config configs/inference_genphoto/merge_configs.yaml --base_scene "A young boy wearing an orange jacket is standing on a crosswalk, waiting to cross the street." --bokehK_list "[2.44, 8.3, 10.1, 17.2, 24.0]" --color_temperature_list "[5455.0, 5155.0, 5555.0, 6555.0, 7555.0]"

