import os
import argparse
import random
import numpy as np
import torch
import csv
import sys
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn.functional as F

# 專案內部引用
from options.options import parse
from archs import create_model, create_optim_scheduler
from losses import create_loss, calculate_loss
from data.dataset_reader.datapipeline import MyDataset_Crop

def pad_tensor(tensor, multiple=8):
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    return F.pad(tensor, (0, pad_w, 0, pad_h), value=0)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # --- [DEBUG MARKER] ---
    print("\n" + "="*50)
    print("🚀 Code Updated! VGG Safety Check Active")
    print("==================================================\n")
    # ----------------------

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    args = parser.parse_args()
    
    opt = parse(args.opt)
    set_random_seed(opt.get('manual_seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Start training [{opt['network']['name']}] on {device}")

    # 1. 準備數據
    def get_imgs(path): 
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset path not found: {path}")
        return sorted([os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print("Loading datasets...")
    train_ds = MyDataset_Crop(
        get_imgs(opt['datasets']['train']['dataroot_lq']),
        get_imgs(opt['datasets']['train']['dataroot_gt']),
        cropsize=opt['datasets']['train']['gt_size'],
        tensor_transform=transforms.ToTensor(),
        test=False, crop_type='Random'
    )
    train_loader = DataLoader(train_ds, batch_size=opt['datasets']['train']['batch_size'], shuffle=True, num_workers=2, pin_memory=True)

    val_ds = MyDataset_Crop(
        get_imgs(opt['datasets']['val']['dataroot_lq']),
        get_imgs(opt['datasets']['val']['dataroot_gt']),
        cropsize=None, tensor_transform=transforms.ToTensor(), test=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    # 2. 建立模型
    model, _, _ = create_model(opt['network'], rank=0)
    optimizer, scheduler = create_optim_scheduler(opt['train'], model)
    criterion = create_loss(opt['train'], rank=0)

    # 3. Log
    os.makedirs(opt['path']['models'], exist_ok=True)
    log_path = os.path.join(opt['path']['models'], 'training_log.csv')
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'Loss', 'PSNR'])

    # 4. 訓練迴圈
    total_epochs = opt['train']['epochs']
    print(f"Total Epochs: {total_epochs}")

    for epoch in range(1, total_epochs + 1):
        model.train()
        loss_meter = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{total_epochs}")
        
        for gt, lq in pbar:
            gt, lq = gt.to(device), lq.to(device)
            optimizer.zero_grad()
            
            if opt['train']['enhance']:
                side_out, output = model(lq, side_loss=True)
                
                # --- [強制防呆機制] ---
                # 不管原本設定多少，只要特徵圖小於 32x32，一律強制放大
                # 這樣 VGG 絕對不會崩潰
                MIN_SIZE = 32
                if side_out.size(-1) < MIN_SIZE or side_out.size(-2) < MIN_SIZE:
                    side_out = F.interpolate(side_out, size=(MIN_SIZE, MIN_SIZE), mode='bilinear', align_corners=False)
                    # 重新計算 scale_factor
                    current_scale = MIN_SIZE / gt.size(-1)
                    loss = calculate_loss(criterion, output, gt, outside_batch=side_out, scale_factor=current_scale)
                else:
                    # 正常情況
                    loss = calculate_loss(criterion, output, gt, outside_batch=side_out, scale_factor=0.125)
            else:
                output = model(lq)
                loss = calculate_loss(criterion, output, gt)
            
            loss.backward()
            optimizer.step()
            loss_meter += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        avg_loss = loss_meter / len(train_loader)
        
        # 5. 驗證與存檔
        current_psnr = 0.0
        if epoch % opt['train']['val_freq'] == 0:
            model.eval()
            psnr_val = 0
            with torch.no_grad():
                for gt, lq in val_loader:
                    lq = pad_tensor(lq.to(device))
                    out = model(lq, side_loss=False)
                    if isinstance(out, tuple): out = out[1]
                    
                    mse = torch.mean((out.cpu() - gt)**2)
                    if mse == 0: psnr_val += 100
                    else: psnr_val += 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            current_psnr = psnr_val / len(val_loader)
            print(f"Val PSNR: {current_psnr:.4f} dB")
            
            save_path = os.path.join(opt['path']['models'], f'epoch_{epoch}.pth')
            torch.save({'model_state_dict': model.state_dict()}, save_path)

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, avg_loss, current_psnr])

if __name__ == '__main__':
    main()