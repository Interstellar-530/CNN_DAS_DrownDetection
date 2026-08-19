# Milestone Models (01-13)

- 01: Initial baseline (lightweight CNN)
- 02: V2 architecture + FSD50K screams
- 03: Tempered version (macro-F1, stronger regularization)
- 04: Re-balanced dataset
- 05: Aligned labels, balanced data
- 06: Reference recipe + balanced data
- 07: + real FSD50K water clips
- 08: + fresh FSD50K dev screams
- 09: Dual-recall objective
- 10: 08 recipe, 80 epochs
- 11: Data cleaning + architecture optimization retrain
- 12: Probability fusion of 07 and 08
- 13: Fine-tuned with real water

## Why 13

11 has better macro metrics because they average the four classes equally, but the joint drowning alarm does not: water recall is the safety-critical half. 11's water recall is 0.52 (0.638 real-water), 13's is 0.65 (0.724 real-water); joint recall rises from 0.491 to 0.617. For the alarm application, 13 is the better trade.

## Dataset versions

Original: LibriSpeech dev-clean + Kaggle screams + ESC-50 water approximations (4450 clips); v2: + FSD50K screams; v23/v24: expansion and cleaning; v3: balanced to ~1200 per class with aligned labels; v3.5: + 89 real water clips in training, 58 held out as test; v3.6: + 55 fresh dev screams; fresh test set: 175 FSD50K eval clips; FP probe: 303 clips.

## 中文说明

### 版本（01–13）

01 初始基线（轻量 CNN）· 02 V2 结构 + FSD50K 尖叫 · 03 回火版（macro-F1、加强正则）· 04 数据重平衡 · 05 标签对齐 + 均衡数据 · 06 reference 配方 + 均衡数据 · 07 + 真实水声 · 08 + 全新 dev 尖叫 · 09 双召回目标 · 10 08 配方 80 epoch · 11 数据清洗 + 结构优化重训 · 12 07 与 08 概率融合 · 13 真实水声微调

### 为什么选 13

11 的宏观指标更好是因为对四类等权平均，而联合报警并不等权：落水召回是安全关键半边。11 的落水召回 0.52（真实水声 0.638），13 为 0.65（真实水声 0.724）；联合召回从 0.491 升到 0.617。对报警应用，13 更合理。

### 数据集版本说明

原始：LibriSpeech dev-clean + Kaggle 尖叫 + ESC-50 水声近似（4450 条）；v2：+ FSD50K 尖叫；v23/v24：扩充与清洗；v3：每类均衡至约 1200 条、标签对齐；v3.5：训练 + 89 条真实水声、留出 58 条作测试；v3.6：+ 55 条全新 dev 尖叫；全新测试集：175 条 FSD50K eval；误报探针：303 条。

Details: `model_info.json` in each folder.
