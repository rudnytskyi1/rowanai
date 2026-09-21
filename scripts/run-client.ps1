[CmdletBinding()]
param([string]$Config)
$ErrorActionPreference = 'Stop'
$rowanRoot = Split-Path -Parent $PSScriptRoot
$rowanPython = Join-Path $rowanRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $rowanPython)) { throw 'Run setup-client.bat first.' }
if (-not $Config) { $Config = Join-Path $rowanRoot 'config.yaml' }
if (-not (Test-Path -LiteralPath $Config)) { throw 'Configuration is missing. Run setup-client.bat first.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
Push-Location $rowanRoot
try {
    & $rowanPython -m client.main --config $Config
    $rowanExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $rowanExit
