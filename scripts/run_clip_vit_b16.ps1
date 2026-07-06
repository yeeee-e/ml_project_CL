$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:MPLCONFIGDIR = "C:\Users\12503\Desktop\chaos\ml_project\.mplconfig"

$Conda = "C:\Users\12503\Documents\Codex\2026-07-05\jian\work\miniforge3\Scripts\conda.exe"
$EnvPath = "C:\Users\12503\Desktop\chaos\ml_project\.conda_env"

& $Conda run -p $EnvPath python -m src.runner --config configs/level1_ncm_baseline_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level1_ncm_baseline_cifar10_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level1_ncm_baseline_tinyimagenet_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level1_saber_feature_prompt_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level1_saber_cifar10_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level1_saber_tinyimagenet_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level2_ncm_cifar10_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level2_ncm_cifar100_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level2_ncm_tinyimagenet_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level2_flycl_sparse_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level2_flycl_cifar10_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level2_flycl_tinyimagenet_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level2_flycl_adaptive_cifar100_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level2_flycl_adaptive_cifar10_clip_vit_b16.yaml
& $Conda run -p $EnvPath python -m src.runner --config configs/level2_flycl_adaptive_tinyimagenet_clip_vit_b16.yaml
& $Conda run -p $EnvPath python scripts/summarize_results.py --results results --output reports/experiment_summary.md
