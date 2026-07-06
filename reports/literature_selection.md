# Literature Selection

## Level-1

Selected paper:

**Turning Back Without Forgetting: Selective Backward Refinement for Parameter-Efficient Continual Learning**

Reason:

- The released repository is named `OptMN-Lab/SABER-ICML-2026`.
- SABER is a continual-learning method, not included in the cloned LibContinual method list.
- The core contribution is algorithmic and portable: task-correlation selection by prompt-gradient geometry / loss-distribution similarity, followed by safe backward refinement in non-interfering prompt directions.
- The original repository targets PEFT for language models. For this course project, I migrate the idea into LibContinual's image classification interface with frozen image features, task-specific prompts, and task heads.

Primary sources:

- Paper page: https://arxiv.org/abs/2606.01379
- Official code release: https://github.com/OptMN-Lab/SABER-ICML-2026/

## Level-2

Selected direction:

**Fly-CL / brain-inspired efficient decorrelation**

Reason:

- It is the lowest-compute Level-2 direction in the assignment screenshot.
- It can be explored without training a foundation model.
- It naturally supports engineering optimization: dense vs sparse projection, whitening/decorrelation ablation, runtime, memory, and projection density.
