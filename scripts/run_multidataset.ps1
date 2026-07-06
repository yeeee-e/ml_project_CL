$ErrorActionPreference = "Stop"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:OMP_NUM_THREADS = "1"
$env:MPLCONFIGDIR = "C:\Users\12503\Desktop\chaos\ml_project\.mplconfig"
$Conda = "C:\Users\12503\Documents\Codex\2026-07-05\jian\work\miniforge3\Scripts\conda.exe"
$EnvPath = (Resolve-Path (Join-Path $PSScriptRoot "..\.conda_env")).Path

$Configs = @(
  "configs/level1_ncm_baseline_resnet18.yaml",
  "configs/level1_saber_feature_prompt_resnet18.yaml",
  "configs/level1_saber_cifar10_resnet18.yaml",
  "configs/level1_saber_tinyimagenet_resnet18.yaml",
  "configs/level2_flycl_sparse_resnet18.yaml",
  "configs/level2_flycl_cifar10_resnet18.yaml",
  "configs/level2_flycl_tinyimagenet_resnet18.yaml"
)

foreach ($Config in $Configs) {
  & $Conda run -p $EnvPath python -m src.runner --config $Config
}

& $Conda run -p $EnvPath python scripts/summarize_results.py --results results --output reports/experiment_summary.md
