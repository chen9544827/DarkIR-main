# 🚀 DarkMamba: Integrating State Space Models for Low-Light Image Restoration

**Course:** ZA5010701 電腦視覺實務與深度學習 (Computer Vision Practice and Deep Learning)  
**Task:** Final Project - Group Presentation  
**Base Paper:** DarkIR (ICCV 2025 / CVPR 2025 context)

---

## 📖 專案簡介 (Introduction)

本專案旨在解決現有基於 CNN 的低光照影像修復 (Low-Light Image Restoration, LLIE) 模型在處理**高動態範圍 (HDR)** 場景時的局限性。

我們復現了 SOTA 模型 **DarkIR**,發現其雖然在局部細節恢復上表現優異,但受限於卷積神經網路 (CNN) 的**局部感受野 (Local Receptive Field)**,在面對大面積逆光或光照極度不均的場景時,容易出現全域光照不一致的問題。

因此,我們提出 **DarkMamba**,將 **State Space Models (Mamba)** 引入修復架構中。利用 Mamba 的**全域感受野 (Global Receptive Field)** 與線性計算複雜度,在不顯著增加運算成本的前提下,大幅提升模型對光影分佈的理解能力。

---

## 📊 實驗結果 (Experimental Results)

我們在真實世界數據集 **LOLv2-real** 上進行了嚴格的 A/B Testing。

### 1. 量化評估 (Quantitative Comparison)

| Model | Width | Params | Best Epoch | Best PSNR (dB) | Improvement |
|:------|:------|:-------|:-----------|:---------------|:------------|
| **DarkIR (Baseline)** | 32 | 3.31 M | ~100 | 19.66 | - |
| **DarkMamba (Ours)** | 32 | 3.3 M* | **170** | **20.24** | **+0.58 dB** 🔺 |

> **分析：**
> - DarkMamba 在參數量相近的情況下,PSNR 提升了 **0.58 dB**。
> - **收斂速度驚人：** DarkMamba 在第 50 個 Epoch 時 PSNR 已達 19.10 dB,展現了 SSM 架構極佳的特徵提取效率。

### 2. 視覺化比較 (Visual Comparison)

我們提供了腳本可直接生成 "Input | DarkIR | DarkMamba" 的對比圖。

![DarkIR vs DarkMamba Comparison](assets/compare.jpg)

---

## 🛠️ 核心架構 (Methodology)

我們基於 U-Net 架構進行改良:

1. **Baseline (DarkIR):** 使用基於 CNN 的 EBlock 與 DBlock。
2. **Ours (DarkMamba):** 將 Encoder 與 Decoder 的核心特徵提取層替換為 **Vision Mamba Block (Vim)**,引入全域掃描機制。

---

## ⚙️ 環境安裝 (Installation)

本專案基於 PyTorch 與 Mamba 構建。由於 Mamba 對 CUDA 版本有特定要求,請依照以下順序安裝:

```bash
# 1. Clone 本專案
git clone <your-repo-url>
cd DarkIR-main

# 2. 安裝基礎依賴
pip install -r requirements.txt

# 3. 安裝 Mamba 相關庫 (核心步驟)
# 注意:請確保系統已安裝 CUDA (建議 11.8 或 12.1)
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

---

## 📂 數據集準備 (Dataset)

目前的實驗基於 LOLv2-real 數據集。

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

```bash
# 訓練 DarkMamba (Ours)
python train.py -opt options/train/train_LOLv2_Mamba.yml
```

### 2. 推論與比較 (Inference & Comparison)

我們提供了一個自動化腳本,可同時載入 Baseline 與 Ours 模型,並生成拼接對比圖:

```bash
# 比較 DarkIR 與 DarkMamba 的效果
# 請確保 options/inference/ 下的 .yml 檔中 'save.path' 指向正確的權重檔
python inference_compare.py \
  -opt1 options/inference/LOLBlur.yml \
  -opt2 options/inference/DarkMamba.yml \
  -i ./demo/inputs \
  -o ./images/comparison_results
```

---

## 🔮 未來工作 (Future Work)

為了進一步驗證 DarkMamba 的泛化能力,我們計畫進行以下擴充實驗:

- [ ] 擴充訓練數據集 (LSWR, SID)。
- [ ] 加入 Early Stopping 機制以解決後期過擬合 (Overfitting) 問題。
- [ ] 探討不同 Mamba 掃描策略 (Bi-directional vs Cross-scan) 的影響。

---

## 📝 團隊成員 (Team)

- **組長：** [姓名]
- **組員：** [姓名]
- **組員：** [姓名]

---

## 📎 致謝 (Acknowledgements)

- **DarkIR:** [Paper Link / Repository]
- **Mamba:** [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) (Gu et al., 2023)

---

## 📄 License

[在此添加您的授權資訊]

## 📧 Contact

如有任何問題,歡迎透過 [email] 或 GitHub Issues 聯繫我們。