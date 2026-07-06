$ErrorActionPreference = "Stop"

$Conda = "C:\Users\12503\Documents\Codex\2026-07-05\jian\work\miniforge3\Scripts\conda.exe"
$EnvPath = "C:\Users\12503\Desktop\chaos\ml_project\.conda_env"
$WheelDir = "C:\Users\12503\Desktop\chaos\ml_project\wheels"

if (Test-Path $WheelDir) {
  & $Conda run -p $EnvPath python -m pip install --no-index --find-links $WheelDir -r requirements-libcontinual-extra.txt
} else {
  & $Conda run -p $EnvPath python -m pip install -r requirements-libcontinual-extra.txt
}

& $Conda run -p $EnvPath python -c "import timm, ftfy, regex, continuum, diffdist, easydict; print('LibContinual extra dependencies OK')"
