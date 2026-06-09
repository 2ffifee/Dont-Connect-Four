$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Set-Location $ProjectRoot

if (-not (Test-Path $VenvPython)) {
    python -m venv .venv
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e ".[dev,analysis]"
