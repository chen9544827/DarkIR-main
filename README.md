# 🚀 DarkMamba: Integrating State Space Models for Low-Light Image Restoration

**Course:** ZA5010701 電腦視覺實務與深度學習 (Computer Vision Practice and Deep Learning)  
**Task:** Final Project - Group Presentation  
**Base Paper:** DarkIR (ICCV 2025 / CVPR 2025)

---

## 📖 專案簡介 (Introduction)

本專案旨在解決現有基於 CNN 的低光照影像修復 (Low-Light Image Restoration, LLIE) 模型在處理**高動態範圍 (HDR)** 場景時的局限性。

我們復現了 SOTA 模型 **DarkIR**，發現其雖然在局部細節恢復上表現優異，但受限於卷積神經網路 (CNN) 的**局部感受野 (Local Receptive Field)**，在面對大面積逆光或極亮天空與極暗前景並存的場景時，容易出現**全域光照不一致 (Inconsistent Global Illumination)** 的問題（例如：太陽過曝、暗部物體缺乏立體感）。

因此，我們提出 **DarkMamba**，將 **State Space Models (Mamba)** 引入修復架構中。利用 Mamba 的**全域感受野 (Global Receptive Field)** 與**線性計算複雜度**，在不增加顯著運算成本的前提下，提升模型對全域光影分佈的理解能力。

---

## 🛠️ 核心架構與創新 (Methodology)

為了確保比較的公平性，我們設計了嚴格的 A/B Test 實驗：

### 1. Baseline: DarkIR-m (Lightweight)
- **架構：** 基於 Efficient CNN 與 Frequency Domain Learning。
- **設定：** 考慮到單卡 12GB VRAM 的限制，我們將模型通道數 (Width) 調整為 32 (原論文為 64)。
- **狀態：** 已完成復現與訓練 (Best PSNR on LOLv2-real: 19.66 dB)。

### 2. Ours: DarkMamba
- **創新點：** 將 Encoder 與 Decoder 中的核心卷積模組替換為 Vision Mamba Block (Vim)。
- **預期優勢：** 更好的全域特徵聚合能力，能夠抑制亮部過曝並精準提亮暗部。

---

## 📊 實驗結果 (Results)

我們使用真實世界數據集 **LOLv2-real** 進行訓練與評估。

| Model               | Width | Params  | FLOPs  | PSNR (dB) | SSIM |
|---------------------|-------|---------|--------|-----------|------|
| DarkIR (Baseline)   | 32    | 3.31 M  | 7.25 G | 19.66     | -    |
| DarkMamba (Ours)    | 32    | TBD     | TBD    | Running...| -    |

> **註：** Baseline 數據取自 Epoch 100 最佳權重

---

## ⚙️ 環境安裝 (Installation)

本專案基於 PyTorch 構建。請確保您的環境滿足以下要求：

```bash
# 1. Clone 本專案
git clone <your-repo-url>
cd DarkIR-main

# 2. 安裝依賴套件
pip install -r requirements.txt

# 3. 安裝 Mamba 相關庫 (核心步驟)
# 注意：Windows 使用者若安裝失敗，請參考 mamba-ssm 官方 issue 或使用 WSL2
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

---

## 📂 數據集準備 (Dataset Preparation)

我們使用 **LOLv2-real** 數據集。請將下載後的數據依照以下結構放置：

```
datasets/
└── LOLv2/
    └── Real_captured/
        ├── Train/
        │   ├── Low/    # 訓練用低光照圖
        │   └── Normal/ # 訓練用 GT 圖
        └── Test/
            ├── Low/    # 驗證用低光照圖
            └── Normal/ # 驗證用 GT 圖
```

---

## 🚀 執行指南 (Usage)

### 1. 訓練 (Training)

我們提供了針對不同模型的訓練設定檔。

**訓練 Baseline (DarkIR):**
```bash
python train.py -opt options/train/train_LOLv2.yml
```

**訓練 Innovation (DarkMamba):**
```bash
python train.py -opt options/train/train_LOLv2_Mamba.yml
```

訓練過程的 Loss 與 PSNR 會自動記錄於 `experiments/DarkIR_LOLv2/models/training_log.csv`，可用 Excel 直接繪製曲線圖。

### 2. 推論與測試 (Inference)

使用訓練好的權重對單張圖片或資料夾進行修復：

```bash
# 修改 options/inference/LOLBlur.yml 中的 save.path 指向你的 .pth 檔
# 並確保 width 設定正確 (例如 width: 32)
python inference.py -p options/inference/LOLBlur.yml -i ./demo/inputs/
```

---

## 📝 團隊成員 (Team Members)

- **組長：** [姓名] - [學號]
- **組員：** [姓名] - [學號]
- **組員：** [姓名] - [學號]

---

## 📎 參考文獻 (Acknowledgements)

- **DarkIR:** Detect Anything 3D in the Wild / Robust Low-Light Image Restoration (ICCV/CVPR 2025 context).
- **Mamba:** Mamba: Linear-Time Sequence Modeling with Selective State Spaces (Gu et al., 2023).
- **LOLv2 Dataset:** Low-Light Image and Video Enhancement Using Deep Learning: A Survey (Li et al.).