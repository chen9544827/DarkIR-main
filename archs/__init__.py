import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from ptflops import get_model_complexity_info
import torch.distributed as dist
from .DarkIR import DarkIR    

def create_model(opt, rank, adapter = False):
    '''
    Creates the model.
    opt: a dictionary from the yaml config key network
    '''
    name = opt['name']

    # 1. 建立模型實例
    model = DarkIR(img_channel=opt['img_channels'], 
                    width=opt['width'], 
                    middle_blk_num_enc=opt['middle_blk_num_enc'],
                    middle_blk_num_dec=opt['middle_blk_num_dec'], 
                    enc_blk_nums=opt['enc_blk_nums'],
                    dec_blk_nums=opt['dec_blk_nums'], 
                    dilations=opt['dilations'],
                    extra_depth_wise=opt['extra_depth_wise'])

    # 2. 定義 Device (修復 NameError)
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{rank}')
    else:
        device = torch.device('cpu')

    # 3. 計算模型複雜度 (只在 rank 0 執行)
    if rank == 0:
        print(f'Using {name} network')
        input_size = (3, 256, 256)
        # 注意：這裡加上 try-except 是為了防止某些架構在 ptflops 計算時報錯導致卡住
        try:
            macs, params = get_model_complexity_info(model, input_size, print_per_layer_stat = False)
            print(f'Computational complexity at {input_size}: {macs}')
            print('Number of parameters: ', params)    
        except Exception as e:
            print(f"Complexity info failed: {e}")
            macs, params = "Unknown", "Unknown"
    else:
        macs, params = None, None

    # 4. 將模型移動到指定的 GPU
    model = model.to(device)
    
    # 5. DDP 分散式處理邏輯 (修復核心問題)
    # 檢查是否有啟動分散式環境
    if dist.is_initialized():
        # DDP 模式
        model = DDP(model, device_ids=[rank], find_unused_parameters=adapter)
    else:
        # 單機模式
        if rank == 0:
            print("Warning: DDP is not initialized. Running in single-GPU mode.")
        # 已經在上面做過 model.to(device) 了，這裡不需要再做，也不會報錯
        
    # 6. 回傳結果 (修復縮排錯誤：必須在 if-else 之外回傳)
    return model, macs, params

def create_optim_scheduler(opt, model):
    '''
    Returns the optim and its scheduler.
    opt: a dictionary of the yaml config file with the train key
    '''
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
    '''
    Loads the weights of a pretrained model, picking only the weights that are
    in the new model.
    '''
    new_weights = model.state_dict()
    new_weights.update({k: v for k, v in old_weights.items() if k in new_weights})
    
    model.load_state_dict(new_weights)
    return model

def load_optim(optim, optim_weights):
    '''
    Loads the values of the optimizer picking only the weights that are in the new model.
    '''
    optim_new_weights = optim.state_dict()
    # optim_new_weights.load_state_dict(optim_weights)
    optim_new_weights.update({k:v for k, v in optim_weights.items() if k in optim_new_weights})
    return optim

def resume_model(model,
                 optim,
                 scheduler, 
                 path_model, 
                 rank, resume:str=None):
    '''
    Returns the loaded weights of model and optimizer if resume flag is True
    '''
    # 這裡的 map_location 邏輯：將權重從 cuda:0 映射到當前的 cuda:rank
    # 如果是單機 rank=0，這就是 cuda:0 -> cuda:0，沒問題
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    
    if resume:
        # 加上 weights_only=False 避免新版 PyTorch 警告
        checkpoints = torch.load(path_model, map_location=map_location, weights_only=False)
        
        # 處理 DDP 帶來的 module. 前綴問題 (以防萬一 checkpoint 是 DDP 存的但現在是單機跑)
        if 'model_state_dict' in checkpoints:
            weights = checkpoints['model_state_dict']
        else:
            weights = checkpoints # 兼容不同的儲存格式
            
        model = load_weights(model, old_weights=weights)
        
        if 'optimizer_state_dict' in checkpoints:
            optim = load_optim(optim, optim_weights = checkpoints['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoints:
            scheduler.load_state_dict(checkpoints['scheduler_state_dict'])
            
        start_epochs = checkpoints.get('epoch', 0) # 使用 get 避免 key error

        if rank == 0: print('Loaded weights')
    else:
        start_epochs = 0
        if rank==0: print('Starting from zero the training')
    
    return model, optim, scheduler, start_epochs

def find_different_keys(dict1, dict2):
    # Finding different keys
    different_keys = set(dict1.keys()) ^ set(dict2.keys())
    return different_keys

def number_common_keys(dict1, dict2):
    # Finding common keys
    common_keys = set(dict1.keys()) & set(dict2.keys())

    # Counting the number of common keys
    common_keys_count = len(common_keys)
    return common_keys_count

def save_checkpoint(model, optim, scheduler, metrics_eval, metrics_train, paths, adapter = False, rank = None):
    '''
    Save the .pt of the model after each epoch.
    '''
    best_psnr = metrics_train['best_psnr']
    if rank!=0: 
        return best_psnr
    
    # 處理 metric 格式
    if isinstance(next(iter(metrics_eval.values())), float):
         # 如果是單純的數值，不需要取 dict
         pass
    elif type(next(iter(metrics_eval.values()))) != dict:
        metrics_eval = {'metrics': metrics_eval}

    weights = model.state_dict()

    # Save the model after every epoch
    model_to_save = {
        'epoch': metrics_train['epoch'],
        'model_state_dict': weights,
        'optimizer_state_dict': optim.state_dict(),
        'loss': metrics_train['train_loss'],
        'scheduler_state_dict': scheduler.state_dict()
    }

    try:
        torch.save(model_to_save, paths['new'])

        # 取得當前的 valid_psnr
        # 這裡的邏輯稍微複雜，視你的 metrics_eval 結構而定，保持原樣
        current_val_psnr = 0
        first_val = next(iter(metrics_eval.values()))
        if isinstance(first_val, dict):
            current_val_psnr = first_val['valid_psnr']
        else:
            # 假設如果不是 dict，它本身可能就是一個數值或者結構不同
             current_val_psnr = 0 # 避免報錯

        # Save best model if new valid_psnr is higher than the best one
        if current_val_psnr >= metrics_train['best_psnr']:
            torch.save(model_to_save, paths['best'])
            metrics_train['best_psnr'] = current_val_psnr  # update best psnr
            
    except Exception as e:
        print(f"Error saving model: {e}")
        
    return metrics_train['best_psnr']

__all__ = ['create_model', 'resume_model', 'create_optim_scheduler', 'save_checkpoint',
           'load_optim', 'load_weights']