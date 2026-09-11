param(
    [Parameter(Mandatory = $true)] [string] $Raw2025,
    [Parameter(Mandatory = $true)] [string] $Zip2026,
    [Parameter(Mandatory = $true)] [string] $MortalPythonRoot,
    [int] $Workers = 4,
    [int] $PackWorkers = 8,
    [int] $TrainGames = 1024,
    [int] $BatchSize = 32,
    [int] $PackSamples = 65536,
    [int] $MaxSteps = 5000,
    [string] $RunName = "v7-default-g1024-s5000-b32"
)

$ErrorActionPreference = "Stop"
if ($BatchSize -le 0) { throw "BatchSize must be positive." }
if ($MaxSteps -le 0) { throw "MaxSteps must be positive for a controlled comparison." }
if ($PackSamples -le 0 -or $PackSamples % $BatchSize -ne 0) {
    throw "PackSamples must be a positive multiple of BatchSize."
}
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"
$manifests = Join-Path $project "data\manifests"
$stagedTrain = Join-Path $project "data\processed-v4\staged-train-2025"
$stagedValidation = Join-Path $project "data\processed-v4\staged-validation-2026"
$packsTrain = Join-Path $project "data\processed-v4\packs-train-2025"
$packsValidation = Join-Path $project "data\processed-v4\packs-validation-2026"
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

# Converted games, packs and checkpoints are derived outputs. Start this
# controlled test from a clean conversion rather than silently reusing an
# earlier schema or an earlier training order.
Reset-GeneratedDirectory $stagedTrain
Reset-GeneratedDirectory $stagedValidation
Reset-GeneratedDirectory $packsTrain
Reset-GeneratedDirectory $packsValidation
Reset-GeneratedDirectory $run

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $log) | Out-Null
$Host.UI.RawUI.WindowTitle = "Riichi Analysis Engine - convert, pack and train"
$env:PYTHONUNBUFFERED = "1"
Start-Transcript -LiteralPath $log -Force

$trainGames = @()
if ($TrainGames -gt 0) { $trainGames = @("--max-games", "$TrainGames") }

try {
    & $python -m riichi_analysis_engine.prepare `
        --raw-2025 $Raw2025 `
        --zip-2026 $Zip2026 `
        --output $manifests
    if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed." }

    & $python -m riichi_analysis_engine.convert `
        --manifest (Join-Path $manifests "validation-2026.jsonl") `
        --output $stagedValidation `
        --mortal-python-root $MortalPythonRoot `
        --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Validation-data conversion failed." }

    & $python -m riichi_analysis_engine.convert `
        --manifest (Join-Path $manifests "train-2025.jsonl") `
        --output $stagedTrain `
        --mortal-python-root $MortalPythonRoot `
        --workers $Workers @trainGames
    if ($LASTEXITCODE -ne 0) { throw "Training-data conversion failed." }

    # Packing decides the training order and audits it from the written packs.
    & $python (Join-Path $PSScriptRoot "pack_global.py") `
        --stage $stagedValidation `
        --output $packsValidation `
        --workers $PackWorkers `
        --pack-samples $PackSamples
    if ($LASTEXITCODE -ne 0) { throw "Validation packing failed." }

    & $python (Join-Path $PSScriptRoot "pack_global.py") `
        --stage $stagedTrain `
        --output $packsTrain `
        --workers $PackWorkers `
        --pack-samples $PackSamples
    if ($LASTEXITCODE -ne 0) { throw "Training packing failed." }

    & $python (Join-Path $PSScriptRoot "check_packs.py") `
        --packs $packsTrain `
        --stage $stagedTrain `
        --batch-size $BatchSize
    if ($LASTEXITCODE -ne 0) { throw "Training packs failed verification." }

    # One epoch is one pass over the mixed corpus; the step budget ends the run.
    & $python -m riichi_analysis_engine.train `
        --train $packsTrain `
        --validation $packsValidation `
        --run $run `
        --epochs 1 `
        --batch-size $BatchSize `
        --max-steps $MaxSteps `
        --device cuda
    if ($LASTEXITCODE -ne 0) { throw "Training failed." }

    & $python -m riichi_analysis_engine.export_weights `
        (Join-Path $run "checkpoint-step-$MaxSteps.pt") `
        $weights
    if ($LASTEXITCODE -ne 0) { throw "Weight export failed." }

    Write-Host "Conversion, packing, training, and weight export completed."
}
finally {
    Stop-Transcript
}
