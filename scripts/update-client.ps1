[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$rowanRoot = Split-Path -Parent $PSScriptRoot
$rowanPython = Join-Path $rowanRoot '.venv\Scripts\python.exe'
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Install Git for Windows first.' }
if (-not (Test-Path -LiteralPath (Join-Path $rowanRoot '.git'))) { throw 'Automatic updates require a Git clone. See README: Updating.' }
if (-not (Test-Path -LiteralPath $rowanPython)) { throw 'Run setup-client.bat first.' }
$env:PYTHONUTF8 = '1'
Push-Location $rowanRoot
try {
    $rowanChanges = & git status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0) { throw 'Could not inspect the Git checkout.' }
    if ($rowanChanges) { throw 'Source files have local changes. Save them before updating; config.yaml and data are not affected.' }
    Write-Host 'Close the Rowan client before updating. Start it again with start-client.bat afterward.'
    & git pull --ff-only
    if ($LASTEXITCODE -ne 0) { throw 'Git update failed. No reset or cleanup was attempted.' }
    & $rowanPython -m pip install -r client/requirements.txt -r client/requirements-browser.txt -r client/requirements-overlay.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Retry update-client.bat.' }
    $rowanCamera = & $rowanPython -m client.setup --camera-enabled
    if ($LASTEXITCODE -ne 0) { throw 'Could not read local camera settings.' }
    if ($rowanCamera -eq 'true') {
        & $rowanPython -m pip install -r client/requirements-camera.txt
        if ($LASTEXITCODE -ne 0) { throw 'Camera dependencies failed. Retry update-client.bat.' }
    }
    & $rowanPython -m client.setup --check
    if ($LASTEXITCODE -ne 0) { throw 'Local configuration needs attention. Run setup-client.bat.' }
    Write-Host 'Updated. Run start-client.bat.' -ForegroundColor Green
} finally {
    Pop-Location
}
