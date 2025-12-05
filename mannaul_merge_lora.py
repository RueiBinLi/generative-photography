import torch
import os
import gc
from diffusers import StableDiffusionPipeline

def manual_merge():
    # ================= 設定區域 =================
    base_model_id = "runwayml/stable-diffusion-v1-5"
    output_path = "./final_merged_model_manual"
    device = "cpu" # 使用 CPU 運算以避免 VRAM 不足，合併只需要做一次，慢一點沒關係

    # 定義 4 個 LoRA 的路徑與權重
    lora_configs = [
        {"name": "Bokeh",   "path": "/home/sagemaker-user/generative-photography/generative_photography/weights/checkpoint-bokehK.ckpt", "scale": 0.5},
        {"name": "Focal",   "path": "/home/sagemaker-user/generative-photography/generative_photography/weights/checkpoint-focal_length.ckpt", "scale": 0.5},
        {"name": "Shutter", "path": "/home/sagemaker-user/generative-photography/generative_photography/weights/checkpoint-shutter_speed.ckpt", "scale": 0.5},
        {"name": "Temp",    "path": "/home/sagemaker-user/generative-photography/generative_photography/weights/checkpoint-color_temperature.ckpt", "scale": 0.5},
    ]
    # ==========================================

    print(f"Loading Base Model: {base_model_id}...")
    # 我們只需要 UNet 的 state_dict 來修改
    pipe = StableDiffusionPipeline.from_pretrained(base_model_id, use_safetensors=True)
    unet = pipe.unet
    unet.to(device)
    
    base_state_dict = unet.state_dict()

    print("\nStarting Manual Merge...")

    for item in lora_configs:
        name = item['name']
        path = item['path']
        scale = item['scale']
        
        print(f"[{name}] Reading checkpoint...")
        try:
            ckpt = torch.load(path, map_location=device)
            
            # 關鍵：從 attention_processor_state_dict 提取權重
            if "attention_processor_state_dict" in ckpt:
                lora_dict = ckpt["attention_processor_state_dict"]
            else:
                print(f"  [!] Skipped {name}: 'attention_processor_state_dict' not found.")
                continue

            print(f"  - Merging {len(lora_dict)//2} layers into UNet (scale={scale})...")
            
            merge_count = 0
            
            # 遍歷所有權重，尋找 'down' 層，然後配對 'up' 層
            for key, down_weight in lora_dict.items():
                if "down.weight" in key:
                    # 1. 找出對應的 Up 權重名稱
                    up_key = key.replace("down.weight", "up.weight")
                    
                    if up_key not in lora_dict:
                        continue
                        
                    up_weight = lora_dict[up_key]

                    # 2. 推導 Base Model 的 Key 名稱
                    # 作者的格式: ...attn1.processor.to_q_lora.down.weight
                    # 目標格式:   ...attn1.to_q.weight
                    
                    base_key = key.replace(".processor", "") # 移除 processor
                    base_key = base_key.replace("_lora", "") # 移除 _lora (to_q_lora -> to_q)
                    base_key = base_key.replace(".down.weight", ".weight") # 移除 .down
                    
                    # 特殊處理 to_out (diffusers 的 to_out 通常是 to_out.0.weight)
                    if "to_out" in base_key:
                        base_key = base_key.replace("to_out.weight", "to_out.0.weight")

                    # 3. 執行合併數學
                    if base_key in base_state_dict:
                        # 計算 LoRA 增量: (Up @ Down) * scale
                        # Up: (Out, Rank), Down: (Rank, In) -> Result: (Out, In)
                        delta_w = torch.mm(up_weight, down_weight) * scale
                        
                        # 確保形狀一致 (有些是 1x1 Conv，需要 unsqueeze)
                        if base_state_dict[base_key].shape != delta_w.shape:
                            if len(base_state_dict[base_key].shape) == 4:
                                delta_w = delta_w.unsqueeze(-1).unsqueeze(-1)
                        
                        # 加進去！
                        base_state_dict[base_key] += delta_w
                        merge_count += 1
                    else:
                        # print(f"    [Warning] Base key not found: {base_key}")
                        pass

            print(f"  - Successfully fused {merge_count} layers.")
            
            # 清理記憶體
            del lora_dict
            del ckpt
            gc.collect()

        except Exception as e:
            print(f"  [ERROR] Failed to merge {name}: {e}")
            # 印出詳細錯誤以便除錯
            import traceback
            traceback.print_exc()
            return

    print("\nAll LoRAs merged!")
    
    # 4. 將修改後的權重存回 Pipeline 並存檔
    print(f"Saving final model to {output_path}...")
    pipe.unet.load_state_dict(base_state_dict)
    pipe.save_pretrained(output_path)
    print("Done! You can now use this model in inference.")

if __name__ == "__main__":
    manual_merge()