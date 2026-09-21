[CmdletBinding()]
param([string]$Python)
$ErrorActionPreference = 'Stop'
$rowanRoot = Split-Path -Parent $PSScriptRoot
$rowanPython = Join-Path $rowanRoot '.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONUNBUFFERED = '1'

function Invoke-RowanPython {
    param([string[]]$Arguments)
    & $rowanPython @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
}

Push-Location $rowanRoot
try {
    if (-not (Test-Path -LiteralPath $rowanPython)) {
        if ($Python) {
            & $Python -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) and sys.maxsize > 2**32 else 1)'
            if ($LASTEXITCODE -ne 0) { throw 'Use Python 3.11 or 3.12 (64-bit).' }
            & $Python -m venv .venv
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
        } else {
            $rowanLauncher = Get-Command py -ErrorAction SilentlyContinue
            if (-not $rowanLauncher) { throw 'Install Python 3.11 or 3.12 (64-bit) from python.org, including the py launcher, then run setup again.' }
            $rowanVersion = $null
            foreach ($version in @('-3.11', '-3.12')) {
                try {
                    & $rowanLauncher.Source $version -c 'import sys; sys.exit(0 if sys.maxsize > 2**32 else 1)' 2>$null
                    if ($LASTEXITCODE -eq 0) { $rowanVersion = $version; break }
                } catch {
                    # Windows PowerShell may turn a missing version's stderr
                    # into a terminating error. Continue to the next version.
                    continue
                }
            }
            if (-not $rowanVersion) { throw 'Python 3.11 or 3.12 (64-bit) was not found. Install it from python.org.' }
            & $rowanLauncher.Source $rowanVersion -m venv .venv
            if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
        }
    }
    Invoke-RowanPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-RowanPython -Arguments @('-m', 'pip', 'install', '-r', 'client/requirements.txt', '-r', 'client/requirements-browser.txt', '-r', 'client/requirements-overlay.txt')
    Invoke-RowanPython -Arguments @('-m', 'client.setup')
    $rowanCamera = & $rowanPython -m client.setup --camera-enabled
    if ($LASTEXITCODE -ne 0) { throw 'Could not read camera settings.' }
    if ($rowanCamera -eq 'true') {
        Invoke-RowanPython -Arguments @('-m', 'pip', 'install', '-r', 'client/requirements-camera.txt')
    }
    Invoke-RowanPython -Arguments @('-m', 'client.setup', '--download-wake-model')
    Write-Host 'Ready. Double-click start-client.bat to start Rowan.' -ForegroundColor Green
} finally {
    Pop-Location
}
