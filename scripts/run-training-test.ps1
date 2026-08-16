param(
    [Parameter(Mandatory = $true)] [string] $Raw2025,
    [Parameter(Mandatory = $true)] [string] $Zip2026,
    [Parameter(Mandatory = $true)] [string] $MortalPythonRoot,
    [int] $Workers = 4,
    [int] $BatchSize = 256
)

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$manifests = Join-Path $project "data\manifests"
$train = Join-Path $project "data\processed-v2\train-2025-1of20"
$validation = Join-Path $project "data\processed-v2\validation-2026"
$run = Join-Path $project "runs\train-2025-1of20-v2-test"
$weights = Join-Path $project "weights\riichi-analysis-2025-1of20-v2-test.pt"
$log = Join-Path $project "runs\train-2025-1of20-v2-test.log"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Create .venv and install the training dependencies first."
}

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
$Host.UI.RawUI.WindowTitle = "Riichi Analysis Engine - convert and train"
$env:PYTHONUNBUFFERED = "1"
Start-Transcript -LiteralPath $log -Force

try {
    & $python -m riichi_analysis_engine.prepare `
        --raw-2025 $Raw2025 `
        --zip-2026 $Zip2026 `
        --output $manifests
    if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed." }

    & $python -m riichi_analysis_engine.convert `
        --manifest (Join-Path $manifests "validation-2026.jsonl") `
        --output $validation `
        --mortal-python-root $MortalPythonRoot `
        --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Validation-data conversion failed." }

    & $python -m riichi_analysis_engine.convert `
        --manifest (Join-Path $manifests "train-2025-1of20.jsonl") `
        --output $train `
        --mortal-python-root $MortalPythonRoot `
        --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Training-data conversion failed." }

    & $python -m riichi_analysis_engine.train `
        --train $train `
        --validation $validation `
        --run $run `
        --epochs 1 `
        --batch-size $BatchSize `
        --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Training failed." }

    & $python -m riichi_analysis_engine.export_weights `
        (Join-Path $run "checkpoint-epoch-1.pt") `
        $weights
    if ($LASTEXITCODE -ne 0) { throw "Weight export failed." }

    Write-Host "Conversion, training, and weight export completed."
}
finally {
    Stop-Transcript
}
