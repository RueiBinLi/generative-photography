import torch
import gc
from diffusers import StableDiffusionPipeline

def merge_sequential():
    base_model_id = "/home/sagemaker-user/generative-photography/generative_photography/stable-diffusion-v1-5"
    output_path = "./final_merged_model_sequential"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    lora_queue = [
        {"name": "Bokeh",   "path": "/home/sagemaker-user/generative-photography/generative_photography/weights/checkpoint-bokehK.ckpt", "scale": 0.5},
        {"name": "Focal",   "path": "/home/sagemaker-user/generative-photography/generative_photography/weights/checkpoint-focal_length.ckpt", "scale": 0.5},
        {"name": "Shutter", "path": "/home/sagemaker-user/generative-photography/generative_photography/weights/checkpoint-shutter_speed.ckpt", "scale": 0.5},
        {"name": "Temp",    "path": "/home/sagemaker-user/generative-photography/generative_photography/weights/checkpoint-color_temperature.ckpt", "scale": 0.5},
    ]

    print(f"Loading Base Model: {base_model_id}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float16
    ).to(device)

    print("\nStarting Sequential Merge (One by One)...")

    for i, item in enumerate(lora_queue):
        print(f"[{i+1}/4] Processing {item['name']}...")
        
        try:
            pipe.load_lora_weights(item['path'])
            print(f"   - Loaded weights from {item['path']}")

            pipe.fuse_lora(lora_scale=item['scale'])
            print(f"   - Fused into UNet with scale {item['scale']}")

            pipe.unload_lora_weights()
            print(f"   - Unloaded LoRA memory.")

            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            print(f"   [ERROR] Failed to merge {item['name']}: {e}")
            return

    print("\nAll LoRAs merged successfully!")

    print(f"Saving final model to {output_path}...")
    pipe.save_pretrained(output_path)
    print("Done. You can now load this folder directly in inference.")

if __name__ == "__main__":
    merge_sequential()