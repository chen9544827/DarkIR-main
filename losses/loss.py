import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import pytorch_msssim
import numpy as np
from losses.loss_utils import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

@weighted_loss
def log_mse_loss(pred, target):
    return torch.log(F.mse_loss(pred, target, reduction='none'))

@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)

@weighted_loss
def psnr_loss(pred, target): # NCHW
    mseloss = F.mse_loss(pred, target, reduction='none').mean((1,2,3))
    psnr_val = 10 * torch.log10(1 / mseloss).mean().item()
    return psnr_val

class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * psnr_loss(pred, target, weight) * -1.0 

class L1Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}.')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)

class FrequencyLoss(nn.Module):
    def __init__(self, loss_weight = 0.01, criterion ='l2', reduction = 'mean'):
        super(FrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')

    def forward(self, pred, target, weight=None, **kwargs):
        pred_freq = self.get_fft_amplitude(pred)
        target_freq = self.get_fft_amplitude(target)
        return self.loss_weight * self.criterion(pred_freq, target_freq)

    def get_fft_amplitude(self, inp):
        inp_freq = torch.fft.rfft2(inp, norm='backward')
        amp = torch.abs(inp_freq)
        return amp

class CharbonnierLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)

class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, loss_weight=1.0, criterion = 'l1', reduction='mean'):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19().cuda()
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss')
        
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.weight = loss_weight

    def forward(self, x, y):
        # [防呆機制] 如果輸入太小，強制放大到 32x32
        MIN_SIZE = 32
        if x.size(-1) < MIN_SIZE or x.size(-2) < MIN_SIZE:
            x = F.interpolate(x, size=(MIN_SIZE, MIN_SIZE), mode='bilinear', align_corners=False)
        if y.size(-1) < MIN_SIZE or y.size(-2) < MIN_SIZE:
            y = F.interpolate(y, size=(MIN_SIZE, MIN_SIZE), mode='bilinear', align_corners=False)

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return self.weight * loss

class EdgeLoss(nn.Module):
    def __init__(self, rank, loss_weight=1.0, criterion = 'l2',reduction='mean'):
        super(EdgeLoss, self).__init__()
        self.weight = loss_weight
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).to(rank)
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.criterion(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight

def SSIM_loss(pred_img, real_img, data_range):
    return pytorch_msssim.ssim(pred_img, real_img, data_range = data_range)

class SSIM(nn.Module):
    def __init__(self, loss_weight=1.0, data_range = 1.):
        super(SSIM, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range
    def forward(self, pred, target, **kwargs):
        return self.loss_weight * SSIM_loss(pred, target, self.data_range)

class EnhanceLoss(nn.Module):
    '''
    這是關鍵修正的地方！
    '''
    def __init__(self, loss_weight=1.0, criterion = 'l1', reduction='mean'):
        super(EnhanceLoss, self).__init__()
        self.loss_weight = loss_weight
        if criterion == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == 'l2':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError('Unsupported criterion loss') 
        
        self.vgg19 = VGGLoss(loss_weight = 0.01, criterion = criterion, reduction = 'mean')
        
    def forward(self, gt, enhanced, scale_factor = None):
        # [終極修復] 自動對齊尺寸
        # 直接讀取模型輸出 (enhanced) 的尺寸，強制將 gt 縮放到一樣大
        target_size = (enhanced.size(2), enhanced.size(3))
        
        gt_low_res = F.interpolate(gt, size=target_size, mode='bilinear', align_corners=False)
        
        # 如果尺寸太小，VGGLoss 內部會自己處理放大，這裡只要確保兩者尺寸一致即可
        return self.vgg19(gt_low_res, enhanced) + self.loss_weight * self.criterion(gt_low_res, enhanced)