# 持续学习课程项目：LibContinual 方法迁移与 CLIP/Fly-CL 实验

本仓库是《机器学习导论》持续学习课程项目代码库。项目围绕类别增量持续学习展开，包含两个层次：

- **Level-1：方法迁移与复现**  
  选择 LibContinual 框架外的 2026 年持续学习方法 **SABER**，将其机制迁移到 LibContinual 风格的图像类别增量学习代码中，并在 CIFAR-10、CIFAR-100、Tiny-ImageNet-200 上进行实验。

- **Level-2：科研探索与工程优化**  
  复现 Fly-CL 风格的稀疏高维随机投影原型分类方法，并加入多个 baseline、消融和改进版 Adaptive Sparse Fly-CL，从准确率、运行时间、投影稀疏度等角度分析。

当前主实验采用 **CLIP ViT-B/16** 冻结特征。历史 ResNet-18 结果保留为 backbone 对比。

## 论文与项目链接

- SABER 论文：[*SABER: Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning*](https://arxiv.org/abs/2606.01379)
- SABER 官方项目：[OptMN-Lab/SABER-ICML-2026](https://github.com/OptMN-Lab/SABER-ICML-2026)
- LibContinual 框架：[RL-MIND/LibContinual](https://github.com/RL-MIND/LibContinual)

## SABER 算法简介与迁移形式

SABER 的全称是 **Selective Backward Refinement**。它关注参数高效持续学习中的一个问题：模型在学习新任务后，是否可以“回头”更新一部分历史任务模块，从而提升旧任务或相关任务表现，同时不引入灾难性遗忘。

原论文中的主要思想包括：

- **任务特定参数**：每个任务维护独立的 prompt/PEFT 参数，而不是全量微调整个预训练模型。
- **梯度子空间保护**：根据历史任务训练过程中的梯度构造受保护子空间，用于判断哪些更新方向可能破坏旧知识。
- **任务相关性选择**：并非所有历史任务都参与回退更新，只选择与当前任务梯度关系较强的历史任务。
- **Selective Backward Refinement**：在任务结束后，对被选中的历史任务模块做受约束的 backward refinement，使更新尽量避开干扰旧知识的方向。

原始 SABER 主要面向 NLP 参数高效持续学习，backbone 包括 T5、LLaMA、Qwen 等语言模型，任务包括 Long Sequence、SuperNI 等文本任务。由于本课程项目要求基于 LibContinual，并且 LibContinual 主要服务于图像持续学习，本项目没有直接复用原论文的 NLP 训练管线，而是将 SABER 的机制迁移到图像类别增量学习中。

本项目的迁移形式如下：

| SABER 原论文 | 本项目迁移实现 |
| --- | --- |
| 语言模型 backbone，如 T5/LLaMA/Qwen | 冻结图像 backbone，如 ResNet-18、CLIP ViT-B/16 |
| 文本任务 prompt/PEFT 参数 | 每个图像增量任务的 feature-space prompt 向量 |
| 任务特定 PEFT head/module | 每个任务独立的分类 head |
| 梯度子空间保护历史任务参数 | 使用 prompt 梯度构造历史任务受保护子空间 |
| backward refinement 更新历史任务模块 | 在图像特征空间中尝试更新历史任务 prompt |
| NLP 持续学习评价 | CIFAR-10、CIFAR-100、Tiny-ImageNet 类别增量评价 |

代码中对应的迁移版本称为 `SABERFeaturePrompt`，核心位置包括：

- `src/methods.py`：快速实验 runner 使用的特征空间版本；
- `_external_LibContinual/core/model/saber_feature_prompt.py`：迁移到 LibContinual 风格生命周期的版本；
- `configs/level1_saber_*_clip_vit_b16.yaml`：CLIP ViT-B/16 下的 Level-1 实验配置。

实验发现，SABER 的任务特定 prompt/head 机制在 CLIP 特征上表现稳定，但原始 backward refinement 思路直接迁移到图像 CLIP 特征时会产生负迁移。因此，本项目将 `with backward refinement` 作为消融实验，将 `no-backward` 版本作为 CLIP backbone 下的主结果。

## 1. 代码库结构

```text
ml_project/
├── configs/                         # 每个实验的 YAML 配置
├── reports/                         # 中文报告与实验汇总表
│   ├── project_report.md
│   ├── experiment_summary.md
│   └── experiment_summary.csv
├── scripts/                         # 一键运行、依赖安装、结果汇总脚本
│   ├── run_clip_vit_b16.ps1
│   ├── run_all.ps1
│   ├── run_multidataset.ps1
│   ├── summarize_results.py
│   └── install_libcontinual_extra_deps.ps1
├── src/                             # 快速特征缓存实验框架
│   ├── datasets.py
│   ├── feature_cache.py
│   ├── methods.py
│   ├── metrics.py
│   └── runner.py
├── _external_LibContinual/           # LibContinual 源码与迁移方法
├── environment.yml                   # Conda 环境说明
├── requirements-libcontinual-extra.txt
└── README.md
```



## 2. 环境配置

### 2.1 推荐环境

本项目实验环境：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5090 |

### 2.2 从零创建环境

进入项目目录：

```powershell
cd .\ml_project
```

创建 conda 环境：

```powershell
conda create -p ".conda_env" python=3.10 numpy=2.2.6 scipy=1.15.2 scikit-learn=1.7.2 pandas=2.3.3 matplotlib=3.10.9 tqdm=4.68.3 pyyaml=6.0.3 psutil=7.2.2 -c conda-forge -y
```

安装 PyTorch CUDA 版本：

```powershell
conda run -p ".conda_env" python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

安装 LibContinual/CLIP 额外依赖：

```powershell
conda run -p ".conda_env" python -m pip install timm==0.6.7 ftfy==6.3.1 regex==2024.11.6 continuum==1.2.7 diffdist==0.1 easydict==1.13
```


## 3. 数据集与模型权重

### 3.1 数据集放置

| 数据集 | 下载文件 | 放置位置 |
| --- | --- | --- |
| CIFAR-10 | `cifar-10-python.tar.gz` | `data/cifar-10-python.tar.gz` |
| CIFAR-100 | `cifar-100-python.tar.gz` | `data/cifar-100-python.tar.gz` |
| Tiny-ImageNet-200 | `tiny-imagenet-200.zip` 解压后目录 | `data/tiny-imagenet-200/` |



### 3.2 CLIP ViT-B/16 权重

下载官方 OpenAI CLIP TorchScript 权重：

```text
https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

放到：

```text
models/ViT-B-16.pt
```


## 4. 运行命令

### 4.1 运行完整 CLIP 实验

```powershell
.\scripts\run_clip_vit_b16.ps1
```

该脚本会运行：

- Level-1 NCM baseline
- Level-1 SABER with backward refinement
- Level-1 SABER no-backward
- Level-2 task-agnostic NCM baseline
- Level-2 Sparse Fly-CL
- Level-2 Adaptive Sparse Fly-CL
- 结果汇总脚本

### 4.2 运行单个实验

示例：运行 Level-1 CIFAR-100 主结果：

```powershell
conda run -p ".conda_env" python -m src.runner --config configs/level1_saber_no_backward_clip_vit_b16.yaml
```

示例：运行 Level-2 CIFAR-100 Sparse Fly-CL：

```powershell
conda run -p ".conda_env" python -m src.runner --config configs/level2_flycl_sparse_clip_vit_b16.yaml
```

示例：运行 Level-2 CIFAR-100 Adaptive Fly-CL：

```powershell
conda run -p ".conda_env" python -m src.runner --config configs/level2_flycl_adaptive_cifar100_clip_vit_b16.yaml
```

### 4.3 汇总结果

```powershell
conda run -p ".conda_env" python scripts/summarize_results.py --results results --output reports/experiment_summary.md
```

输出文件：

```text
reports/experiment_summary.md
reports/experiment_summary.csv
```

## 5. 复现结果对比表

完整结果见：

```text
reports/experiment_summary.md
reports/project_report.md
```

### 5.1 Level-1：CLIP ViT-B/16 主结果

| 数据集 | 方法 | Backward refinement | Final Acc. | Avg. Inc. Acc. | Forgetting |
| --- | --- | --- | ---: | ---: | ---: |
| CIFAR-10 | NCM baseline | - | 98.71 | 98.51 | 0.00 |
| CIFAR-10 | SABERFeaturePrompt | 关闭 | 98.52 | 97.32 | 0.00 |
| CIFAR-100 | NCM baseline | - | 90.65 | 90.91 | 0.00 |
| CIFAR-100 | SABERFeaturePrompt | 关闭 | 93.37 | 93.61 | 0.00 |
| Tiny-ImageNet-200 | NCM baseline | - | 86.08 | 86.85 | 0.00 |
| Tiny-ImageNet-200 | SABERFeaturePrompt | 关闭 | 88.86 | 88.65 | 0.00 |

结论：CLIP ViT-B/16 特征下，SABERFeaturePrompt no-backward 在 CIFAR-100 和 Tiny-ImageNet 上超过 NCM baseline，并保持零遗忘。

### 5.2 Level-1：backward refinement 消融

| 数据集 | 变体 | Final Acc. | Avg. Inc. Acc. | Forgetting | BWT |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | with backward refinement | 98.50 | 97.30 | 0.02 | -0.02 |
| CIFAR-10 | without backward refinement | 98.52 | 97.32 | 0.00 | 0.00 |
| CIFAR-100 | with backward refinement | 43.07 | 64.44 | 55.97 | -55.89 |
| CIFAR-100 | without backward refinement | 93.37 | 93.61 | 0.00 | 0.00 |
| Tiny-ImageNet-200 | with backward refinement | 31.07 | 53.50 | 64.21 | -64.21 |
| Tiny-ImageNet-200 | without backward refinement | 88.86 | 88.65 | 0.00 | 0.00 |

结论：直接将 ResNet-18 上的 backward refinement 超参迁移到 CLIP 特征会产生负迁移。CLIP 特征已经高度可分，历史任务 prompt 的回退更新反而破坏旧任务边界。

### 5.3 Level-2：task-agnostic NCM baseline

| 数据集 | Backbone | 方法 | Final Acc. | Avg. Inc. Acc. | Forgetting | Runtime (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | CLIP ViT-B/16 | NCM | 92.17 | 94.34 | 4.74 | 0.25 |
| CIFAR-100 | CLIP ViT-B/16 | NCM | 70.02 | 77.47 | 9.56 | 0.40 |
| Tiny-ImageNet-200 | CLIP ViT-B/16 | NCM | 65.85 | 74.07 | 9.10 | 0.57 |

### 5.4 Level-2：Sparse Fly-CL vs NCM

| 数据集 | NCM Final Acc. | Sparse Fly-CL Final Acc. | 提升 | NCM Runtime (s) | Fly-CL Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 92.17 | 91.23 | -0.94 | 0.25 | 4.11 |
| CIFAR-100 | 70.02 | 72.71 | +2.69 | 0.40 | 6.00 |
| Tiny-ImageNet-200 | 65.85 | 69.61 | +3.76 | 0.57 | 8.43 |

结论：Sparse Fly-CL 在类别更多、更难的 CIFAR-100 和 Tiny-ImageNet 上优于 NCM；在 CIFAR-10 上，原始 CLIP 特征已经足够可分，随机投影略有扰动。

### 5.5 Level-2：backbone 工程优化对比

| 数据集 | ResNet-18 + Sparse Fly-CL | CLIP ViT-B/16 + Sparse Fly-CL | 提升 |
| --- | ---: | ---: | ---: |
| CIFAR-10 | 74.98 | 91.23 | +16.25 |
| CIFAR-100 | 53.59 | 72.71 | +19.12 |
| Tiny-ImageNet-200 | 54.70 | 69.61 | +14.91 |

结论：更强的冻结 backbone 是最有效的工程优化之一。由于特征被缓存，CLIP 的较高前向成本只出现在首次特征抽取阶段。

### 5.6 Level-2：whitening 消融

| 数据集 | Backbone | 方法 | Final Acc. | Avg. Inc. Acc. | Forgetting |
| --- | --- | --- | ---: | ---: | ---: |
| CIFAR-100 | ResNet-18 | Sparse Fly-CL without whitening | 51.94 | 62.60 | 12.68 |
| CIFAR-100 | ResNet-18 | Sparse Fly-CL with whitening | 53.59 | 63.90 | 13.12 |

结论：whitening 对最终准确率有小幅提升，但没有明显降低 forgetting。

### 5.7 Level-2：算法改进 Adaptive Sparse Fly-CL

改进点：

- 累计更新 whitening 统计量；
- 每次更新统计量后重建历史类别原型；
- 加入 WTA top-k 稀疏激活，只保留投影后最强的 512 个维度。

| 数据集 | Sparse Fly-CL Final Acc. | Adaptive Fly-CL Final Acc. | 差值 | Sparse Runtime (s) | Adaptive Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 91.23 | 92.58 | +1.35 | 4.11 | 14.35 |
| CIFAR-100 | 72.71 | 72.11 | -0.60 | 6.00 | 18.24 |
| Tiny-ImageNet-200 | 69.61 | 68.89 | -0.72 | 8.43 | 29.47 |

结论：Adaptive Fly-CL 只在 CIFAR-10 上小幅提升，在 CIFAR-100 和 Tiny-ImageNet 上略低于原始 Sparse Fly-CL，且运行时间明显增加。因此最终主方法采用 CLIP ViT-B/16 + Sparse Fly-CL，Adaptive 版本作为改进尝试和反例分析。

## 6. 实验与算法问题分析

本节记录实验设计和算法实现过程中遇到的问题现象以及原因分析。

### 6.1 CLIP 下 SABER backward refinement 出现负迁移

现象：

| 数据集 | with backward Final Acc. | no-backward Final Acc. | 差异 |
| --- | ---: | ---: | ---: |
| CIFAR-100 | 43.07 | 93.37 | -50.30 |
| Tiny-ImageNet-200 | 31.07 | 88.86 | -57.79 |

原因分析：

SABER 的 backward refinement 会用当前任务数据对历史任务 prompt 做熵最小化式更新。该机制在原论文的参数高效语言模型持续学习中用于挖掘任务间正迁移，但在本项目的 CLIP 图像特征空间中，历史任务类别已经高度可分。继续用当前任务样本更新历史 prompt，容易把历史任务决策边界拉向当前任务分布，从而造成严重负迁移。


### 6.2 task-aware 与 task-agnostic 结果不能直接横向比较

现象：

Level-1 的 SABER no-backward 在 CIFAR-100 上达到 93.37%，而 Level-2 的 Sparse Fly-CL 在 CIFAR-100 上为 72.71%。如果直接比较，会误以为 Fly-CL 明显更弱。

原因分析：

| 部分 | 设置 | 测试时类别范围 |
| --- | --- | --- |
| Level-1 | task-aware | 只在当前任务类别内预测 |
| Level-2 | task-agnostic | 在所有已见类别中预测 |

Level-2 的 task-agnostic 设置难度更高，因为模型没有任务标识，需要区分所有已见类别。


### 6.3 NCM baseline 过强，Fly-CL 在简单数据集上不占优

现象：

| 数据集 | NCM Final Acc. | Sparse Fly-CL Final Acc. |
| --- | ---: | ---: |
| CIFAR-10 | 92.17 | 91.23 |
| CIFAR-100 | 70.02 | 72.71 |
| Tiny-ImageNet-200 | 65.85 | 69.61 |

在 CIFAR-10 上，Sparse Fly-CL 低于直接使用 CLIP 特征的 NCM baseline。

原因分析：

CLIP ViT-B/16 的原始特征已经具有很强的语义可分性。对于 CIFAR-10 这种类别少、难度低的数据集，随机投影可能引入额外扰动，反而不如直接在原始特征空间做原型分类。


### 6.4 whitening 带来有限收益，但不能解决遗忘

现象：

| 方法 | CIFAR-100 Final Acc. | Forgetting |
| --- | ---: | ---: |
| Sparse Fly-CL without whitening | 51.94 | 12.68 |
| Sparse Fly-CL with whitening | 53.59 | 13.12 |

whitening 提升了最终准确率，但 forgetting 没有下降。

原因分析：

whitening 能改善投影特征尺度差异，使原型分类边界更稳定，因此 final accuracy 有提升。但它不存储旧样本，也不显式约束新旧类别边界，所以不能从机制上消除 task-agnostic 设置中的历史类别混淆。


### 6.5 Adaptive Fly-CL 改进不稳定

现象：

| 数据集 | Sparse Fly-CL | Adaptive Fly-CL | 差异 |
| --- | ---: | ---: | ---: |
| CIFAR-10 | 91.23 | 92.58 | +1.35 |
| CIFAR-100 | 72.71 | 72.11 | -0.60 |
| Tiny-ImageNet-200 | 69.61 | 68.89 | -0.72 |

Adaptive Fly-CL 只在 CIFAR-10 上提升，在 CIFAR-100 和 Tiny-ImageNet 上略低于原始 Sparse Fly-CL，并且运行时间更长。

原因分析：

Adaptive Fly-CL 会累计更新 whitening 统计量，并在每次任务后重建历史类别原型。这使特征空间随任务变化而整体重标定，可能导致历史原型发生漂移。WTA top-k 稀疏激活也可能丢弃细粒度分类所需的弱响应维度。


### 6.6 backbone 升级收益显著

现象：

| 数据集 | ResNet-18 + Sparse Fly-CL | CLIP ViT-B/16 + Sparse Fly-CL | 提升 |
| --- | ---: | ---: | ---: |
| CIFAR-10 | 74.98 | 91.23 | +16.25 |
| CIFAR-100 | 53.59 | 72.71 | +19.12 |
| Tiny-ImageNet-200 | 54.70 | 69.61 | +14.91 |

原因分析：

CLIP ViT-B/16 的语义特征远强于 ResNet-18，因此性能提升并不完全来自 Fly-CL 算法本身，而是来自 backbone 表征能力增强。


## 7. 主要文件说明

| 文件 | 作用 |
| --- | --- |
| `src/feature_cache.py` | 数据集读取、CLIP/ResNet/ViT 特征提取与缓存 |
| `src/methods.py` | NCM、Sparse Fly-CL、Adaptive Fly-CL、SABERFeaturePrompt |
| `src/runner.py` | 通用实验入口，读取 YAML 配置并输出结果 |
| `scripts/summarize_results.py` | 汇总 `results/*.json` 到 Markdown/CSV |
| `reports/project_report.md` | 实验报告 |
| `_external_LibContinual/core/model/saber_feature_prompt.py` | 迁移到 LibContinual 风格的 SABER 方法 |
