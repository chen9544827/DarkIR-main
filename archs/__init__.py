import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from ptflops import get_model_complexity_info
import torch.distributed as dist
from .DarkIR import DarkIR 

# --- [新增] 嘗試匯入 DarkMamba ---
try:
    from .DarkMamba import DarkMamba
except ImportError:
    DarkMamba = None
    pass
# -------------------------------

def create_model(opt, rank, adapter = False):
    name = opt['name']

    # --- [修改] 加入模型切換邏輯 ---
    if name == 'DarkMamba':
        if DarkMamba is None:
            raise ImportError("Cannot import DarkMamba. Please check archs/DarkMamba.py")
        print(f"Initializing DarkMamba with width={opt['width']}...")
        model = DarkMamba(img_channel=opt['img_channels'], 
                        width=opt['width'], 
                        middle_blk_num_enc=opt['middle_blk_num_enc'],
                        middle_blk_num_dec=opt['middle_blk_num_dec'], 
                        enc_blk_nums=opt['enc_blk_nums'],
                        dec_blk_nums=opt['dec_blk_nums'], 
                        dilations=opt['dilations'],
                        extra_depth_wise=opt['extra_depth_wise'])
    else:
        # DarkIR (Baseline)
        model = DarkIR(img_channel=opt['img_channels'], 
                        width=opt['width'], 
                        middle_blk_num_enc=opt['middle_blk_num_enc'],
                        middle_blk_num_dec=opt['middle_blk_num_dec'], 
                        enc_blk_nums=opt['enc_blk_nums'],
                        dec_blk_nums=opt['dec_blk_nums'], 
                        dilations=opt['dilations'],
                        extra_depth_wise=opt['extra_depth_wise'])
    # -----------------------------------

    # --- [關鍵修改] 先設定 Device 並把模型搬過去 ---
    # 這是為了解決 Mamba 組件 (causal_conv1d) 必須在 CUDA 上才能運算的問題
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')
    
    model = model.to(device)
    # -------------------------------------------

    # --- 再計算 FLOPs (這時候模型已經在 GPU 了) ---
    if rank == 0:
        print(f'Using {name} network')
        try:
            # 嘗試計算複雜度
            # ptflops 會自動偵測模型所在的 device，所以現在傳入 GPU model 是安全的
            input_size = (3, 256, 256)
            macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
            print(f'Computational complexity: {macs}, Params: {params}')    
        except Exception as e:
            print(f"Complexity info skipped (Normal for Mamba structure if input type mismatch): {e}")
            macs, params = "Unknown", "Unknown"
    else:
        macs, params = None, None

    # --- DDP 設定 ---
    if dist.is_initialized():
        model = DDP(model, device_ids=[rank], find_unused_parameters=adapter)
    else:
        if rank == 0:
            print("Warning: DDP is not initialized. Running in single-GPU mode.")
        
    return model, macs, params

# 以下保留原有的輔助函式，確保相容性
def create_optim_scheduler(opt, model):
    optim = torch.optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()) , 
                            lr = opt['lr_initial'],
                            weight_decay = opt['weight_decay'],
                            betas = opt['betas'])
    
    if opt['lr_scheme'] == 'CosineAnnealing':
        scheduler = CosineAnnealingLR(optim, T_max=opt['epochs'], eta_min=opt['eta_min'])
    else: 
        raise NotImplementedError('scheduler not implemented')    
    return optim, scheduler

def load_weights(model, old_weights):
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in old_weights.items() if k in new_weights})
    model.load_state_dict(new_weights)
    return model

def load_optim(optim, optim_weights):
    optim_new_weights = optim.state_dict()
    optim_new_weights.update({k:v for k, v in optim_weights.items() if k in optim_new_weights})
    return optim

def resume_model(model, optim, scheduler, path_model, rank, resume:str=None):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    if resume:
        checkpoints = torch.load(path_model, map_location=map_location, weights_only=False)
        weights = checkpoints.get('model_state_dict', checkpoints) # 兼容性修正
        model = load_weights(model, old_weights=weights)
        if 'optimizer_state_dict' in checkpoints:
            optim = load_optim(optim, optim_weights = checkpoints['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoints:
            scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
        start_epochs = checkpoints.get('epoch', 0)
        if rank == 0: print('Loaded weights')
    else:
        start_epochs = 0
        if rank==0: print('Starting from zero the training')
    return model, optim, scheduler, start_epochs

def save_checkpoint(model, optim, scheduler, metrics_eval, metrics_train, paths, adapter = False, rank = None):
    if rank!=0: return metrics_train['best_psnr']
    
    # 簡化了原本複雜的 metric 結構處理，增加防呆
    if not isinstance(metrics_eval, dict):
         metrics_eval = {'metrics': metrics_eval}

    weights = model.state_dict()
    model_to_save = {
        'epoch': metrics_train['epoch'],
        'model_state_dict': weights,
        'optimizer_state_dict': optim.state_dict(),
        'loss': metrics_train['train_loss'],
        'scheduler_state_dict': scheduler.state_dict()
    }

    try:
        torch.save(model_to_save, paths['new'])
        # 嘗試從 metrics_eval 中提取 psnr
        current_val = 0.0
        try:
            # 針對不同的 dict 結構做嘗試
            if 'valid_psnr' in metrics_eval:
                current_val = metrics_eval['valid_psnr']
            else:
                first_val = next(iter(metrics_eval.values()))
                current_val = first_val['valid_psnr'] if isinstance(first_val, dict) else first_val
        except:
            pass

        if current_val >= metrics_train['best_psnr']:
            torch.save(model_to_save, paths['best'])
            metrics_train['best_psnr'] = current_val
    except Exception as e:
        print(f"Error saving model: {e}")
    return metrics_train['best_psnr']

# 確保所有需要的函式都有導出
__all__ = ['create_model', 'resume_model', 'create_optim_scheduler', 'save_checkpoint',
           'load_optim', 'load_weights']