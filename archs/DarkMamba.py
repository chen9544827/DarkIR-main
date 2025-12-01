import torch
import torch.nn as nn
import torch.nn.functional as F
from .arch_util import CustomSequential, LayerNorm2d

# 嘗試導入 Mamba，如果環境沒裝好會報錯提醒
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    print("[Warning] 'mamba_ssm' not found. DarkMamba will not work without it.")

class VisionMambaBlock(nn.Module):
    """
    這是 DarkMamba 的核心創新組件。
    它將圖片視為序列 (Sequence)，利用 Mamba 進行全域特徵掃描。
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super(VisionMambaBlock, self).__init__()
        if Mamba is None:
            raise ImportError("Please install 'mamba_ssm' to use this block.")
            
        self.norm = nn.LayerNorm(dim)
        
        # Mamba 核心層
        self.mamba = Mamba(
            d_model=dim,      # Model dimension (C)
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=expand,    # Block expansion factor
        )
        
        # 為了穩定訓練，加入一個簡單的投影層
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        
        # 1. Reshape for Mamba: [B, C, H, W] -> [B, H*W, C]
        x_in = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x_in = self.norm(x_in)
        
        # 2. Mamba Global Context Modeling
        x_out = self.mamba(x_in)
        x_out = self.proj(x_out)
        
        # 3. Reshape back: [B, H*W, C] -> [B, C, H, W]
        x_out = x_out.view(B, H, W, C).permute(0, 3, 1, 2)
        
        # 4. Residual Connection
        return x + x_out

class DarkMamba(nn.Module):
    """
    DarkMamba: 基於 Mamba 的低光照影像修復網路
    架構模仿 DarkIR 的 U-Net 結構，但將 Encoder/Decoder 的核心運算單元換成 VisionMambaBlock。
    """
    def __init__(self, img_channel=3, 
                 width=32, 
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2, 
                 enc_blk_nums=[1, 2, 3], 
                 dec_blk_nums=[3, 1, 1],  
                 dilations=[1, 4, 9], 
                 extra_depth_wise=True):
        super(DarkMamba, self).__init__()
        
        # 入口卷積 (Feature Extraction)
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        # 出口卷積 (Image Reconstruction)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1, bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks_enc = nn.ModuleList()
        self.middle_blks_dec = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        
        chan = width
        
        # --- Encoder Path (下採樣路徑) ---
        for num in enc_blk_nums:
            # 這裡我們用 VisionMambaBlock 替換原本的 EBlock
            self.encoders.append(
                CustomSequential(
                    *[VisionMambaBlock(chan) for _ in range(num)]
                )
            )
            # 下採樣 (Downsampling)
            self.downs.append(
                nn.Conv2d(chan, 2*chan, kernel_size=2, stride=2)
            )
            chan = chan * 2

        # --- Bottleneck (中間層 - 全域資訊交換最密集的地方) ---
        self.middle_blks_enc = \
            CustomSequential(
                *[VisionMambaBlock(chan) for _ in range(middle_blk_num_enc)]
            )
        
        self.middle_blks_dec = \
            CustomSequential(
                *[VisionMambaBlock(chan) for _ in range(middle_blk_num_dec)]
            )

        # --- Decoder Path (上採樣路徑) ---
        for num in dec_blk_nums:
            # 上採樣 (Upsampling)
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            
            # Decoder Block
            self.decoders.append(
                CustomSequential(
                    *[VisionMambaBlock(chan) for _ in range(num)]
                )
            )
            
        self.padder_size = 2 ** len(self.encoders)
        
        # Side Loss Layer (保持與 DarkIR 一致，用於訓練穩定性)
        self.side_out = nn.Conv2d(in_channels=width * 2**len(self.encoders), out_channels=img_channel, 
                                kernel_size=3, stride=1, padding=1)
        
    def forward(self, input, side_loss=False, use_adapter=None):
        _, _, H, W = input.shape

        input = self.check_image_size(input)
        x = self.intro(input)
        
        skips = []
        # Encoder Forward
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # Bottleneck Forward
        x_light = self.middle_blks_enc(x)
        
        if side_loss:
            out_side = self.side_out(x_light)
            
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        # Decoder Forward
        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input # Residual learning (只學習殘差)
        out = x[:, :, :H, :W] 
        
        if side_loss:
            return out_side, out
        else:        
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value = 0)
        return x