param(
    [Parameter(Mandatory = $true)] [string] $Raw2025,
    [Parameter(Mandatory = $true)] [string] $Zip2026,
    [Parameter(Mandatory = $true)] [string] $MortalPythonRoot,
    [int] $Workers = 4,
    [int] $TrainGames = 1024,
    [int] $BatchSize = 32,
    [int] $MaxSteps = 5000,
    [string] $RunName = "v6-default-g1024-s5000-b32"
)

$ErrorActionPreference = "Stop"
if ($TrainGames -le 0) { throw "TrainGames must be positive." }
if ($BatchSize -le 0) { throw "BatchSize must be positive." }
if ($MaxSteps -le 0) { throw "MaxSteps must be positive for a controlled comparison." }
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$manifests = Join-Path $project "data\manifests"
$train = Join-Path $project "data\processed-v4\train-2025"
$validation = Join-Path $project "data\processed-v4\validation-2026"
$run = Join-Path $project "runs\$RunName"
$weights = Join-Path $project "weights\riichi-analysis-$RunName.pt"
$log = Join-Path $project "runs\$RunName.log"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Create .venv and install the training dependencies first."
}

function Reset-GeneratedDirectory([string] $Path) {
    $projectFull = [IO.Path]::GetFullPath($project).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
    $targetFull = [IO.Path]::GetFullPath($Path)
    if (-not $targetFull.StartsWith($projectFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a directory outside this engine project: $targetFull"
    }
    if (Test-Path -LiteralPath $targetFull) {
        Remove-Item -LiteralPath $targetFull -Recurse -Force
    }
}

# Converted shards and checkpoints are derived outputs. Start this controlled
# test from a clean conversion rather than silently reusing an earlier schema.
Reset-GeneratedDirectory $train
Reset-GeneratedDirectory $validation
Reset-GeneratedDirectory $run
Reset-GeneratedDirectory (Join-Path $project "data\processed")
Reset-GeneratedDirectory (Join-Path $project "data\processed-v2")

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
        --manifest (Join-Path $manifests "train-2025.jsonl") `
        --output $train `
        --mortal-python-root $MortalPythonRoot `
        --workers $Workers `
        --max-games $TrainGames
    if ($LASTEXITCODE -ne 0) { throw "Training-data conversion failed." }

    & $python -m riichi_analysis_engine.train `
        --train $train `
        --validation $validation `
        --run $run `
        --epochs 20 `
        --batch-size $BatchSize `
        --max-steps $MaxSteps `
        --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Training failed." }

    & $python -m riichi_analysis_engine.export_weights `
        (Join-Path $run "checkpoint-step-$MaxSteps.pt") `
        $weights
    if ($LASTEXITCODE -ne 0) { throw "Weight export failed." }

    Write-Host "Conversion, training, and weight export completed."
}
finally {
    Stop-Transcript
}
