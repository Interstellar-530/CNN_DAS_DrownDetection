# CNN_v1.0 - Drowning-Acoustic Detection

Four-class CNN for audio event detection (normal_speech / scream / water_splash / other_non_scream) on 128x128 log-mel spectrograms at 16 kHz.

## Model versions (01-13)

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

## Why version 13

11 looks stronger on the macro table (macro precision/recall 0.823/0.766 vs 0.801/0.761, probe FP 0.063 vs 0.109) because the macro metrics average the four classes equally, whereas the joint drowning alarm does not - a missed water detection is the safety-critical half. On the reconstructed fresh test set (n = 175, low-quality label mismatches replaced with verified same-class clips), 11's water recall is only 0.55 (0.638 on the 58 real-water holdout); 13 raises it to 0.683 (0.724) and joint recall from 0.522 to 0.649 (0.665 at threshold 0.4), while accuracy rises to 0.754 (11: 0.743). For the alarm application, 13 is the better trade.

## Dataset versions

- Original: LibriSpeech dev-clean (normal speech) + Kaggle human screams (scream/non-scream) + ESC-50 water approximations (water_splash); 4450 training clips.
- v2: + FSD50K pure screams.
- v23/v24: expanded and cleaned data (more water and hard negatives).
- v3: classes re-balanced to about 1200 each; labels aligned.
- v3.5: + 89 real FSD50K water clips in training; 58 real FSD50K water clips held out as an unseen test.
- v3.6: + 55 never-used FSD50K dev screams.
- Fresh four-class test set: 175 FSD50K eval clips (scream 40 / speech 35 / water 60 / other 40), unseen by every version.
- False-positive probe: 303 non-scream clips (birds, horns, children, alarms, pets, screech).

## Contents

- `code/` - training/evaluation/UI code
- `model/` - model checkpoint (best_model_13.pt) + config + metrics
- `milestones/` - version checkpoints 01-13
- `docs/` - bilingual reports (MD + PDF) and figures

## 中文说明

四分类 CNN（normal_speech / scream / water_splash / other_non_scream），输入 128×128 log-mel 谱图，16 kHz。

### 模型版本（01–13）

- 01：初始基线（轻量 CNN）
- 02：V2 结构 + FSD50K 尖叫
- 03：回火版（macro-F1、加强正则）
- 04：数据重平衡
- 05：标签对齐 + 均衡数据
- 06：reference 配方 + 均衡数据
- 07：+ 真实水声
- 08：+ 全新 dev 尖叫
- 09：双召回目标
- 10：08 配方 80 epoch
- 11：数据清洗 + 结构优化重训
- 12：07 与 08 概率融合
- 13：真实水声微调

### 为什么选 13

表格里 11 的宏观指标更好看（宏观精确率/召回率 0.823/0.766 对 0.801/0.761、探针误报 0.063 对 0.109），因为宏观指标对四个类等权平均；而联合溺水报警并不等权——落水检出是安全关键半边。在重建后的全新测试集（n=175，已剔除货不对板并用验证过的同类样本替换）上，11 的落水召回仅 0.55（58 条真实水声基准 0.638），13 提升到 0.683（0.724），联合召回从 0.522 升到 0.649（@0.4 为 0.665），准确率升至 0.754（11 为 0.743）。对溺水报警应用，13 是更合理的取舍。

### 数据集版本说明

- 原始：LibriSpeech dev-clean（正常语音）+ Kaggle 人类尖叫（尖叫/非尖叫）+ ESC-50 水声近似（落水声），训练 4450 条。
- v2：+ FSD50K 纯尖叫。
- v23/v24：数据扩充与清洗（更多水声与硬负样本）。
- v3：类别均衡到每类约 1200 条，标签对齐。
- v3.5：训练 + 89 条真实 FSD50K 水声；留出 58 条真实 FSD50K 水声作全新测试。
- v3.6：+ 55 条从未使用的 FSD50K dev 尖叫。
- 全新四类测试集：175 条 FSD50K eval（尖叫 40 / 语音 35 / 落水 60 / 其他 40），任何版本均未见。
- 误报探针：303 条非尖叫（鸟鸣、鸣笛、儿童、警笛、宠物、screech）。
