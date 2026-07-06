# 持续学习课程项目实验报告

## 1. 项目目标

本项目围绕类别增量持续学习展开，按照课程要求完成两个层次的实验：

- Level-1：选择 LibContinual 框架外的 2026 年顶级会议持续学习方法 SABER，将其思想迁移到 LibContinual 风格的代码结构中，并在经典图像数据集上完成复现、对比和消融。
- Level-2：选择较轻量的 Fly-CL 风格方向，复现稀疏高维随机投影与原型分类方法，并从效率、运行时间和投影稀疏度等工程维度进行分析。

实验使用 CIFAR-10、CIFAR-100 和 Tiny-ImageNet-200 三个数据集，避免只在单一数据集上得出结论。后续 backbone 从 ResNet-18 升级为 CLIP ViT-B/16，并重新完成主要实验。

## 2. 方法迁移

Level-1 选择的方法是 **SABER: Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning**。该方法来自 LibContinual 框架外，官方仓库为 `OptMN-Lab/SABER-ICML-2026`。

SABER 原始方法主要面向 NLP 参数高效持续学习，使用 T5、LLaMA、Qwen 等预训练语言模型。由于 LibContinual 主要围绕图像持续学习，本项目将 SABER 的核心思想迁移到图像类别增量场景：

- 使用冻结的预训练 backbone 提取图像特征；
- 每个增量任务维护独立的 prompt 向量和分类头；
- 训练当前任务时只更新当前任务的 prompt/head；
- 根据 prompt 梯度构造受保护子空间；
- 在任务结束后尝试对历史任务做 selective backward refinement；
- 额外加入 no-backward 消融，用于分析 backward refinement 是否产生负迁移。

LibContinual 集成位置：

- `_external_LibContinual/core/model/saber_feature_prompt.py`
- `_external_LibContinual/config/saber_feature_prompt.yaml`
- `_external_LibContinual/reproduce/saber_feature_prompt/README.md`

项目中的快速实验 runner 位于 `src/`，遵循相同的持续学习任务生命周期，并通过特征缓存降低重复实验开销。

## 3. 与原论文实验的关系

需要说明的是，本项目没有直接复现 SABER 原论文的 NLP 表格。原因是原论文任务、模型和评价协议与 LibContinual 图像分类框架差异较大：原论文主要使用 Long Sequence、SuperNI 等 NLP 任务，并基于 T5/LLaMA/Qwen 等语言模型；本项目则将方法机制迁移到 CIFAR-10、CIFAR-100 和 Tiny-ImageNet-200 的图像类别增量学习。

因此，本文结论应理解为“框架外方法迁移与图像持续学习适配实验”，而不是对原论文 NLP 数值表格的逐项复现。报告中只比较同一图像设置下的 NCM、SABER 变体和 Fly-CL 变体，不将不同任务和不同 backbone 的数值直接并列为原论文复现结果。

## 4. 实验设置

### 数据集

| 数据集 | 类别数 | 训练样本 | 测试/验证样本 | 任务划分 |
| --- | ---: | ---: | ---: | --- |
| CIFAR-10 | 10 | 50000 | 10000 | 5 个任务，每任务 2 类 |
| CIFAR-100 | 100 | 50000 | 10000 | 10 个任务，每任务 10 类 |
| Tiny-ImageNet-200 | 200 | 100000 | 10000 | 10 个任务，每任务 20 类 |

### Backbone

实验先使用 ImageNet 预训练 ResNet-18 作为基础版本，随后升级为 **CLIP ViT-B/16**。CLIP 权重文件为：

`models/ViT-B-16.pt`

该文件 SHA256 为：

`5806E77CD80F8B59890B7E101EABD078D9FB84E6937F9E85E4ECB61988DF416F`

CLIP 特征维度为 512，所有特征已缓存到 `data/features/`，后续实验无需重复抽取 backbone 特征。

## 5. Level-1 结果

### 5.1 CLIP ViT-B/16 主结果

| 数据集 | 方法 | Backward refinement | Final Acc. | Avg. Inc. Acc. | Forgetting |
| --- | --- | --- | ---: | ---: | ---: |
| CIFAR-10 | NCM baseline | - | 98.71 | 98.51 | 0.00 |
| CIFAR-10 | SABERFeaturePrompt | 关闭 | 98.52 | 97.32 | 0.00 |
| CIFAR-100 | NCM baseline | - | 90.65 | 90.91 | 0.00 |
| CIFAR-100 | SABERFeaturePrompt | 关闭 | 93.37 | 93.61 | 0.00 |
| Tiny-ImageNet-200 | NCM baseline | - | 86.08 | 86.85 | 0.00 |
| Tiny-ImageNet-200 | SABERFeaturePrompt | 关闭 | 88.86 | 88.65 | 0.00 |

CLIP ViT-B/16 显著增强了冻结特征质量。在 CIFAR-100 上，SABERFeaturePrompt no-backward 相比 NCM baseline 从 90.65% 提升到 93.37%，提升 2.72 个百分点；在 Tiny-ImageNet-200 上从 86.08% 提升到 88.86%，提升 2.78 个百分点。三个数据集上的 forgetting 均为 0，说明任务特定 prompt/head 在强特征上较稳定。

### 5.2 Backward refinement 消融

| 数据集 | 变体 | Final Acc. | Avg. Inc. Acc. | Forgetting | BWT |
| --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | with backward refinement | 98.50 | 97.30 | 0.02 | -0.02 |
| CIFAR-10 | without backward refinement | 98.52 | 97.32 | 0.00 | 0.00 |
| CIFAR-100 | with backward refinement | 43.07 | 64.44 | 55.97 | -55.89 |
| CIFAR-100 | without backward refinement | 93.37 | 93.61 | 0.00 | 0.00 |
| Tiny-ImageNet-200 | with backward refinement | 31.07 | 53.50 | 64.21 | -64.21 |
| Tiny-ImageNet-200 | without backward refinement | 88.86 | 88.65 | 0.00 | 0.00 |

消融结果表明，直接沿用 ResNet-18 上的 backward refinement 超参到 CLIP 特征时会产生明显负迁移。CLIP 特征本身已经具有很强的类别可分性，使用当前任务数据对历史任务 prompt 做熵最小化更新，会破坏历史任务的决策边界。因此，CLIP backbone 下采用 no-backward 版本作为主结果更合理；with-backward 版本保留为重要消融，说明 SABER 的 backward refinement 对 backbone 和特征分布较敏感。

### 5.3 ResNet-18 对照

| 数据集 | 方法 | Backbone | Final Acc. | Avg. Inc. Acc. |
| --- | --- | --- | ---: | ---: |
| CIFAR-100 | NCM baseline | ResNet-18 | 82.84 | 83.16 |
| CIFAR-100 | SABERFeaturePrompt | ResNet-18 | 89.63 | 90.43 |
| CIFAR-100 | NCM baseline | CLIP ViT-B/16 | 90.65 | 90.91 |
| CIFAR-100 | SABERFeaturePrompt no-backward | CLIP ViT-B/16 | 93.37 | 93.61 |

Backbone 升级带来明显收益。仅将 NCM baseline 从 ResNet-18 换成 CLIP ViT-B/16，CIFAR-100 final accuracy 就从 82.84% 提升到 90.65%；进一步使用 SABERFeaturePrompt no-backward 后达到 93.37%。

## 6. Level-2 结果

Level-2 复现 Fly-CL 风格的稀疏高维随机投影和原型分类方法。该部分全部采用 task-agnostic 设置，即测试时不给出任务标识，模型需要在所有已见类别中直接决策。为了使对比更严谨，报告中使用四类 baseline 或对照：

1. 原始特征空间 NCM：不做随机投影，直接用 CLIP 特征的类别均值分类。
2. Sparse Fly-CL：使用稀疏随机投影和 whitening 的主方法。
3. 无 whitening 消融：检验去相关/标准化是否有效。
4. Adaptive Sparse Fly-CL：本文额外改进版，使用累计 whitening 统计量并重建历史类别原型，同时加入 WTA top-k 稀疏激活。

### 6.1 Baseline 1：原始特征空间 NCM

该 baseline 用于回答：Fly-CL 的稀疏高维投影是否比直接在 CLIP 特征空间做最近类均值分类更好。

| 数据集 | Backbone | 方法 | Final Acc. | Avg. Inc. Acc. | Forgetting | Runtime (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CIFAR-10 | CLIP ViT-B/16 | Task-agnostic NCM | 92.17 | 94.34 | 4.74 | 0.25 |
| CIFAR-100 | CLIP ViT-B/16 | Task-agnostic NCM | 70.02 | 77.47 | 9.56 | 0.40 |
| Tiny-ImageNet-200 | CLIP ViT-B/16 | Task-agnostic NCM | 65.85 | 74.07 | 9.10 | 0.57 |

NCM 是非常强的轻量 baseline，几乎没有训练成本。在强 CLIP 特征下，直接用类别原型已经能达到较高准确率。因此，Fly-CL 是否有效，不能只和 ResNet-18 版本比较，还必须和这个原始特征原型 baseline 比较。

### 6.2 Baseline 2：Sparse Fly-CL 相对 NCM 的收益

| 数据集 | NCM Final Acc. | Sparse Fly-CL Final Acc. | 提升 | NCM Runtime (s) | Fly-CL Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 92.17 | 91.23 | -0.94 | 0.25 | 4.11 |
| CIFAR-100 | 70.02 | 72.71 | +2.69 | 0.40 | 6.00 |
| Tiny-ImageNet-200 | 65.85 | 69.61 | +3.76 | 0.57 | 8.43 |

结果显示，Sparse Fly-CL 在类别数更多、任务更难的 CIFAR-100 和 Tiny-ImageNet-200 上优于 NCM，说明稀疏高维投影有助于增强类别原型可分性。但在较简单的 CIFAR-10 上，NCM 反而略高，说明当 CLIP 特征本身已经足够可分时，随机投影可能带来不必要的信息扰动。

### 6.3 Baseline 3：Backbone 对比

该对比固定算法为 Sparse Fly-CL，只更换 backbone，用于分析工程优化中“更强冻结特征”的贡献。

| 数据集 | ResNet-18 + Sparse Fly-CL | CLIP ViT-B/16 + Sparse Fly-CL | 提升 | ResNet Runtime (s) | CLIP Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 74.98 | 91.23 | +16.25 | 3.82 | 4.11 |
| CIFAR-100 | 53.59 | 72.71 | +19.12 | 5.35 | 6.00 |
| Tiny-ImageNet-200 | 54.70 | 69.61 | +14.91 | 8.11 | 8.43 |

CLIP ViT-B/16 在三个数据集上均显著优于 ResNet-18，且方法阶段运行时间只小幅增加。由于本项目缓存的是 backbone 输出特征，CLIP 的额外前向开销主要发生在首次特征抽取阶段，后续持续学习算法对比可以复用缓存。

### 6.4 Baseline 4：whitening 消融

| 数据集 | Backbone | 方法 | Final Acc. | Avg. Inc. Acc. | Forgetting |
| --- | --- | --- | ---: | ---: | ---: |
| CIFAR-100 | ResNet-18 | Sparse Fly-CL without whitening | 51.94 | 62.60 | 12.68 |
| CIFAR-100 | ResNet-18 | Sparse Fly-CL with whitening | 53.59 | 63.90 | 13.12 |

在 ResNet-18 特征上，whitening 使 CIFAR-100 final accuracy 从 51.94% 提升到 53.59%，说明去相关/标准化对稀疏随机投影后的原型分类有一定帮助。但提升幅度不大，且 forgetting 指标没有同步下降，说明 whitening 主要改善最终分类边界，而不是从根本上解决 task-agnostic 设置下的历史类别混淆。

### 6.5 算法改进：Adaptive Sparse Fly-CL

原始 Sparse Fly-CL 的 whitening 统计量只在第一次任务上估计，后续任务到来时不再更新。这在类别分布逐步扩展的持续学习场景中可能不合理。因此，本项目实现了 Adaptive Sparse Fly-CL：

- 使用所有已见任务的投影特征累计更新 mean/std；
- 每次统计量更新后重建所有已见类别原型；
- 加入 WTA top-k 稀疏激活，只保留投影后最强的 512 个维度；
- 目标是在保持稀疏性的同时，让 whitening 更适应不断扩展的类别分布。

| 数据集 | Sparse Fly-CL Final Acc. | Adaptive Fly-CL Final Acc. | 差值 | Sparse Runtime (s) | Adaptive Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| CIFAR-10 | 91.23 | 92.58 | +1.35 | 4.11 | 14.35 |
| CIFAR-100 | 72.71 | 72.11 | -0.60 | 6.00 | 18.24 |
| Tiny-ImageNet-200 | 69.61 | 68.89 | -0.72 | 8.43 | 29.47 |

Adaptive Fly-CL 在 CIFAR-10 上取得小幅提升，但在 CIFAR-100 和 Tiny-ImageNet-200 上略低于原始 Sparse Fly-CL，并且运行时间明显增加。这个结果说明，累计 whitening 和 WTA top-k 并不是稳定有效的通用改进。原因可能是：历史类别原型在统计量更新后会被整体重标定，而高维随机投影本身已经提供了足够的扩展表示；额外的 WTA 截断会丢弃部分对细粒度类别有用的弱激活。

因此，本文最终采用 **CLIP ViT-B/16 + Sparse Fly-CL** 作为 Level-2 主方法，Adaptive Fly-CL 作为算法改进尝试和反例分析。该实验也体现了工程优化中的重要原则：更复杂的自适应机制不一定带来更好的准确率，必须用多数据集和强 baseline 验证。

## 7. 工程优化

项目中使用了以下工程优化：

- 使用冻结 backbone + 特征缓存，避免每个实验重复前向抽特征；
- CLIP、ResNet-18、CIFAR-10、CIFAR-100、Tiny-ImageNet 特征均缓存为 `.npz`；
- Fly-CL 使用稀疏投影矩阵，显著降低投影存储与计算开销；
- 设置 `KMP_DUPLICATE_LIB_OK` 和 `OMP_NUM_THREADS`，解决 Windows 下 OpenMP 冲突；
- 设置 `MPLCONFIGDIR=.mplconfig`，避免 Matplotlib 写入用户目录失败导致导入变慢；
- 将 CLIP 权重放在 `models/ViT-B-16.pt`，保证实验可离线复现。

## 8. 结论

本项目完成了课程要求的两个层次：

- Level-1：将 LibContinual 框架外的 2026 方法 SABER 迁移到图像持续学习场景，并集成到 LibContinual 风格代码中。实验表明，在 CLIP ViT-B/16 特征下，SABERFeaturePrompt no-backward 在 CIFAR-100 和 Tiny-ImageNet-200 上均超过 NCM baseline。
- Level-2：复现 Fly-CL 风格稀疏投影原型分类方法，并在三个数据集上完成 task-agnostic 对比。CLIP backbone 显著提升了 Fly-CL 准确率，同时保持较低运行时间和极低投影密度。

一个重要发现是：SABER 的 backward refinement 并非在所有 backbone 上都稳定。对于强 CLIP 特征，直接回退更新历史任务 prompt 会产生负迁移；关闭 backward refinement 后反而得到更高准确率和零遗忘。这说明持续学习方法迁移到新 backbone 时，不能只复用旧超参，还需要结合特征分布做消融验证。
