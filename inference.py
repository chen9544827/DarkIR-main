import os
from PIL import Image
import cv2 as cv
from options.options import parse
import argparse
# from archs.retinexformer import RetinexFormer # 這裡似乎沒用到，create_model 會處理
# from torch.nn.parallel import DistributedDataParallel as DDP # 移除 DDP

# PyTorch library
import torch
import torch.optim
# import torch.multiprocessing as mp # 移除多進程庫
from tqdm import tqdm
from torchvision import transforms # 補上這行，原本可能漏了或是隱藏在 datapipeline
import torch.nn.functional as F # 補上這行，原本 pad_tensor 用到

from data.dataset_reader.datapipeline import *
from archs import *
from losses import *
from data import *
from utils.test_utils import *
from ptflops import get_model_complexity_info

# --- Argument Parsing ---
parser = argparse.ArgumentParser(description="Script for prediction")
parser.add_argument('-p', '--config', type=str, default='./options/inference/LOLBlur.yml', help = 'Config file of prediction')
parser.add_argument('-i', '--inp_path', type=str, default='./images/inputs', help="Folder path")
args = parser.parse_args()

path_options = args.config
opt = parse(path_options)

# 設定顯卡
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# --- Auxiliary Functions ---
pil_to_tensor = transforms.ToTensor()
tensor_to_pil = transforms.ToPILImage()

def path_to_tensor(path):
    img = Image.open(path).convert('RGB')
    img = pil_to_tensor(img).unsqueeze(0)
    return img

def normalize_tensor(tensor):
    max_value = torch.max(tensor)
    min_value = torch.min(tensor)
    output = (tensor - min_value)/(max_value)
    return output

def save_tensor(tensor, path):
    tensor = tensor.squeeze(0)
    # print(tensor.shape, tensor.dtype, torch.max(tensor), torch.min(tensor)) # 註解掉以保持輸出乾淨
    img = tensor_to_pil(tensor)
    img.save(path)

def pad_tensor(tensor, multiple = 8):
    '''pad the tensor to be multiple of some number'''
    multiple = multiple
    _, _, H, W = tensor.shape
    pad_h = (multiple - H % multiple) % multiple
    pad_w = (multiple - W % multiple) % multiple
    tensor = F.pad(tensor, (0, pad_w, 0, pad_h), value = 0)
    return tensor

def load_model(model, path_weights):
    map_location = device # 直接載入到 GPU
    checkpoints = torch.load(path_weights, map_location=map_location, weights_only=False)
    
    weights = checkpoints['params']
    
    # --- 關鍵修改：權重名稱處理 ---
    # 原本的代碼強制加了 'module.'，現在我們要反過來：
    # 1. 如果權重本身沒有 'module.' (單卡訓練的)，直接用。
    # 2. 如果權重有 'module.' (DDP訓練的)，要把它去掉。
    
    new_weights = {}
    for key, value in weights.items():
        if key.startswith('module.'):
            new_weights[key.replace('module.', '')] = value
        else:
            new_weights[key] = value
            
    # 移除原本強制添加 module. 的那一行
    # weights = {'module.' + key: value for key, value in weights.items()} 

    # 計算 FLOPs (可選，若報錯可註解掉)
    try:
        macs, params = get_model_complexity_info(model, (3, 256, 256), print_per_layer_stat=False, verbose=False)
        print(f"MACs: {macs}, Params: {params}")
    except:
        pass

    model.load_state_dict(new_weights, strict=True)
    print('Loaded weights correctly')
    
    return model

#parameters for saving model
PATH_MODEL = opt['save']['path']
resize = opt['Resize']

# --- Main Inference Function (Single Process) ---
def predict_folder():
    # 移除 setup(...)
    
    print(f"Creating model: {opt['network']}")
    
    # 直接呼叫，不要 try-except
    # rank=0 代表使用第一張顯卡
    model, _, _ = create_model(opt['network'], rank=0)
    
    
    model = model.to(device) # 確保模型在 GPU 上

    model = load_model(model, path_weights = opt['save']['path'])
    
    # Create paths
    PATH_IMAGES= args.inp_path
    PATH_RESULTS = './images/results'

    if not os.path.isdir(PATH_RESULTS):
        os.mkdir(PATH_RESULTS)

    path_images = [os.path.join(PATH_IMAGES, path) for path in os.listdir(PATH_IMAGES) if path.endswith(('.png', '.PNG', '.jpg', '.JPEG'))]
    path_images = [file for file in path_images if not file.endswith('.csv') and not file.endswith('.txt')]
   
    model.eval()
    
    # 只需要一個進程的進度條
    pbar = tqdm(total = len(path_images))
        
    for path_img in path_images:
            tensor = path_to_tensor(path_img).to(device)
            _, _, H, W = tensor.shape
            
            # =========== 這裡開始是修改的部分 ===========
            # 強制縮放邏輯 (避免 OOM)
            # 你的圖片是 6000x4000，必須縮小才能在 12GB 顯存跑
            MAX_SIZE = 1024  # 設定長邊不超過 1024 (若還是OOM可改 800)
            
            if H > MAX_SIZE or W > MAX_SIZE:
                scale = MAX_SIZE / max(H, W)
                new_H, new_W = int(H * scale), int(W * scale)
                
                # 確保長寬是 8 的倍數 (模型要求)
                new_H = (new_H // 8) * 8
                new_W = (new_W // 8) * 8
                
                # 使用 transforms.Resize 進行縮放
                downsample = transforms.Resize((new_H, new_W), antialias=True)
                print(f"Resizing {os.path.basename(path_img)} from ({H}, {W}) to ({new_H}, {new_W})")
            else:
                # 如果圖片本來就夠小，就不動
                # 也要確保是 8 的倍數，以免報錯
                new_H = (H // 8) * 8
                new_W = (W // 8) * 8
                if new_H != H or new_W != W:
                    downsample = transforms.Resize((new_H, new_W), antialias=True)
                else:
                    downsample = torch.nn.Identity()
            
            # 執行縮放
            tensor_in = downsample(tensor)
            # =========== 修改結束 ===========

            tensor_in = pad_tensor(tensor_in)

            with torch.no_grad():
                output = model(tensor_in, side_loss=False) 
                
                if isinstance(output, (list, tuple)):
                    output = output[0]

            # 這裡決定輸出要不要放大回原圖大小
            # 如果顯存夠，可以嘗試放大回去；如果不夠，就輸出縮小後的圖
            # 建議先輸出縮小後的圖，確保能跑完
            # if resize:
            #    upsample = transforms.Resize((H, W)) 
            # else: 
            upsample = torch.nn.Identity() # 暫時不放大回去，避免最後一步爆顯存
                
            output = upsample(output)
            output = torch.clamp(output, 0., 1.)
            # output = output[:,:, :H, :W] # 如果沒放大回去，這行要註解掉，不然會切錯
            
            save_name = os.path.join(PATH_RESULTS, os.path.basename(path_img))
            save_tensor(output, save_name)

            pbar.update(1)


    print('Finished inference!')
    pbar.close()   
    # 移除 cleanup()

def main():
    # 移除 mp.spawn
    predict_folder()

if __name__ == '__main__':
    main()