import os
import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np

# 專案內部引用
from options.options import parse
from archs import create_model

# --- 設定環境 ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 輔助函式 ---
def load_model_wrapper(opt_path, device):
    """
    載入模型與權重的封裝函式
    """
    if not os.path.exists(opt_path):
        raise FileNotFoundError(f"Config file not found: {opt_path}")
        
    opt = parse(opt_path)
    print(f"\n[Loading] {opt['network']['name']} from {opt_path}")
    
    # 建立模型
    model, _, _ = create_model(opt['network'], rank=0)
    model = model.to(device)
    
    # 載入權重
    path_weights = opt['save']['path']
    print(f"   -> Weights: {path_weights}")
    
    if not os.path.exists(path_weights):
        raise FileNotFoundError(f"Weight file not found: {path_weights}")

    checkpoints = torch.load(path_weights, map_location=device, weights_only=False)
    
    # 自動適應不同的權重儲存格式
    if 'params' in checkpoints:
        weights = checkpoints['params']
    elif 'model_state_dict' in checkpoints:
        weights = checkpoints['model_state_dict']
    else:
        weights = checkpoints

    # 移除 DDP 可能產生的 module. 前綴
    new_weights = {}
    for key, value in weights.items():
        if key.startswith('module.'):
            new_weights[key.replace('module.', '')] = value
        else:
            new_weights[key] = value
            
    model.load_state_dict(new_weights, strict=True)
    model.eval()
    
    return model, opt['network']['name']

def pad_tensor(tensor, multiple=8):
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value=0)
    return tensor

def draw_text(image, text, x_pos):
    """
    在圖片上方繪製文字
    """
    draw = ImageDraw.Draw(image)
    try:
        # 嘗試載入系統字體，如果失敗則使用預設
        # Linux 常見字體路徑
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "arial.ttf"
        ]
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, 40)
                break
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()

    # 計算文字大小並置中
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except:
        text_w, text_h = 100, 20 # Fallback
        
    draw.text((x_pos - text_w // 2, 10), text, fill=(0, 0, 0), font=font)

# --- 主程式 ---
def main():
    parser = argparse.ArgumentParser(description="Compare two models inference")
    parser.add_argument('-opt1', type=str, required=True, help='Config file for Model 1 (e.g., Baseline)')
    parser.add_argument('-opt2', type=str, required=True, help='Config file for Model 2 (e.g., Ours)')
    parser.add_argument('-i', '--inp_path', type=str, default='./demo/inputs', help="Input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='./images/comparison_results', help="Output folder")
    args = parser.parse_args()

    # 1. 載入兩個模型
    model1, name1 = load_model_wrapper(args.opt1, device)
    model2, name2 = load_model_wrapper(args.opt2, device)

    # 2. 準備路徑
    os.makedirs(args.out_path, exist_ok=True)
    image_list = sorted([f for f in os.listdir(args.inp_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"\nStart processing {len(image_list)} images...")
    print(f"Output will be saved to: {args.out_path}")

    transform = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    for img_name in tqdm(image_list):
        img_path = os.path.join(args.inp_path, img_name)
        img_pil = Image.open(img_path).convert('RGB')
        
        # --- Resize Logic (防爆顯存) ---
        w, h = img_pil.size
        MAX_SIZE = 1024
        if h > MAX_SIZE or w > MAX_SIZE:
            scale = MAX_SIZE / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            # 確保是 8 的倍數
            new_h = (new_h // 8) * 8
            new_w = (new_w // 8) * 8
            img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_h = (h // 8) * 8
            new_w = (w // 8) * 8
            if new_h != h or new_w != w:
                img_pil = img_pil.resize((new_w, new_h), Image.LANCZOS)
        
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        img_tensor = pad_tensor(img_tensor)

        # --- Inference ---
        with torch.no_grad():
            # Model 1
            out1 = model1(img_tensor, side_loss=False)
            if isinstance(out1, (list, tuple)): out1 = out1[0]
            out1 = torch.clamp(out1, 0, 1)
            
            # Model 2
            out2 = model2(img_tensor, side_loss=False)
            if isinstance(out2, (list, tuple)): out2 = out2[0]
            out2 = torch.clamp(out2, 0, 1)

        # --- Post-process & Stitching ---
        res1 = to_pil(out1.squeeze(0).cpu())
        res2 = to_pil(out2.squeeze(0).cpu())
        
        # 建立大畫布 (Input | Model1 | Model2)
        # 上方保留 60 pixel 寫字
        header_h = 60
        total_w = img_pil.width * 3
        total_h = img_pil.height + header_h
        
        combined = Image.new('RGB', (total_w, total_h), (255, 255, 255))
        
        # 貼上圖片
        combined.paste(img_pil, (0, header_h))
        combined.paste(res1, (img_pil.width, header_h))
        combined.paste(res2, (img_pil.width * 2, header_h))
        
        # 寫上標籤
        draw_text(combined, "Input", img_pil.width // 2)
        draw_text(combined, name1, img_pil.width + img_pil.width // 2)
        draw_text(combined, name2, img_pil.width * 2 + img_pil.width // 2)
        
        # 存檔
        save_name = os.path.join(args.out_path, f"compare_{img_name}")
        combined.save(save_name)

    print("All done!")

if __name__ == '__main__':
    main()