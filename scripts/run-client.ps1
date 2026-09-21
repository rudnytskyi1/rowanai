[CmdletBinding()]
param([string]$Config, [string]$Python, [switch]$Conda)
$ErrorActionPreference = 'Stop'
$rowanRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot 'client-python.ps1')
$rowanPython = Resolve-RowanPython -Root $rowanRoot -Python $Python -Conda:$Conda
Enable-RowanPythonEnvironment -Python $rowanPython
if ($Conda -or $Python) { Save-RowanPython -Root $rowanRoot -Python $rowanPython }
if (-not $Config) { $Config = Join-Path $rowanRoot 'config.yaml' }
if (-not (Test-Path -LiteralPath $Config)) { throw 'Configuration is missing. Run setup-client.bat first.' }
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'
Write-Host "Using Python: $rowanPython"
Push-Location $rowanRoot
try {
    & $rowanPython -m client.main --config $Config
    $rowanExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $rowanExit
