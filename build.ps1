[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $MortalPythonRoot
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$OutputRoot = Join-Path $ProjectRoot 'dist'
$WorkRoot = Join-Path $ProjectRoot 'build'
$Libriichi = Join-Path $MortalPythonRoot 'libriichi.pyd'

if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Missing .venv. Install the build dependencies first.'
}
if (-not (Test-Path -LiteralPath $Libriichi)) {
    throw "Missing libriichi.pyd under $MortalPythonRoot"
}

foreach ($Path in @($OutputRoot, $WorkRoot)) {
    $FullPath = [System.IO.Path]::GetFullPath($Path)
    $RootPath = [System.IO.Path]::GetFullPath($ProjectRoot)
    if (-not $FullPath.StartsWith($RootPath, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean a path outside the project: $FullPath"
    }
    if (Test-Path -LiteralPath $FullPath) {
        Remove-Item -LiteralPath $FullPath -Recurse -Force
    }
}

& $Python -m PyInstaller `
    --noconfirm `
    --clean `
    --onedir `
    --name riichi-analysis-engine `
    --distpath (Join-Path $WorkRoot 'pyinstaller') `
    --workpath (Join-Path $WorkRoot 'work') `
    --specpath (Join-Path $WorkRoot 'spec') `
    --paths (Join-Path $ProjectRoot 'src') `
    --paths $MortalPythonRoot `
    --hidden-import libriichi `
    --add-binary "$Libriichi;." `
    (Join-Path $ProjectRoot 'engine.py')
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller failed.'
}

$PackageRoot = Join-Path $OutputRoot 'riichi-analysis-engine'
$RuntimeRoot = Join-Path $PackageRoot 'runtime'
New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null
Copy-Item -Path (Join-Path $WorkRoot 'pyinstaller\riichi-analysis-engine\*') -Destination $RuntimeRoot -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'engine.json') -Destination $PackageRoot
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'LICENSE') -Destination $PackageRoot
Copy-Item -LiteralPath (Join-Path $ProjectRoot 'THIRD_PARTY_NOTICES.md') -Destination $PackageRoot
& $Python -m piplicenses `
    --format plain-vertical `
    --with-license-file `
    --no-license-path `
    --output-file (Join-Path $PackageRoot 'THIRD_PARTY_LICENSES.txt')
if ($LASTEXITCODE -ne 0) {
    throw 'Third-party license collection failed.'
}
$PythonLicense = & $Python -c "import pathlib, sys; print(pathlib.Path(sys.base_prefix) / 'LICENSE.txt')"
if (Test-Path -LiteralPath $PythonLicense) {
    Copy-Item -LiteralPath $PythonLicense -Destination (Join-Path $PackageRoot 'PYTHON_LICENSE.txt')
}

Write-Host "Built runtime package: $PackageRoot"
Write-Host 'Model weights are not included.'
