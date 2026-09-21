# Shared interpreter selection for setup, start and updates.
function Test-RowanPython {
    param([Parameter(Mandatory = $true)][string]$Python)
    if (-not [System.IO.Path]::IsPathRooted($Python) -or -not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        throw 'The selected Python executable is missing. Activate the intended environment and run setup-client-conda.bat, or pass -Python with its full python.exe path.'
    }
    $rowanResolved = (Get-Item -LiteralPath $Python).FullName
    & $rowanResolved -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) and sys.maxsize > 2**32 else 1)'
    if ($LASTEXITCODE -ne 0) { throw 'The selected environment must use Python 3.11 or 3.12 (64-bit).' }
    return $rowanResolved
}

function Resolve-RowanPython {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [string]$Python,
        [switch]$Conda,
        [switch]$AllowMissingDefault
    )
    if ($Python) { return Test-RowanPython -Python $Python }
    if ($Conda) {
        if (-not $env:CONDA_PREFIX) {
            throw 'Open Anaconda Prompt, run "conda activate YOUR_ENV", then "setup-client-conda.bat" or "start-client-conda.bat". Alternatively, pass -Python "C:\Path\To\env\python.exe" to the .bat file.'
        }
        return Test-RowanPython -Python (Join-Path $env:CONDA_PREFIX 'python.exe')
    }
    $rowanSelection = Join-Path $Root '.rowan-python'
    if (Test-Path -LiteralPath $rowanSelection) {
        $rowanSaved = ([string](Get-Content -LiteralPath $rowanSelection -Raw -Encoding UTF8)).Trim()
        if (-not $rowanSaved) { throw 'The saved Python selection is empty. Run setup-client-conda.bat to select the intended environment again.' }
        return Test-RowanPython -Python $rowanSaved
    }
    $rowanDefault = Join-Path $Root '.venv\Scripts\python.exe'
    if ($AllowMissingDefault -and -not (Test-Path -LiteralPath $rowanDefault)) { return $rowanDefault }
    if (-not (Test-Path -LiteralPath $rowanDefault)) { throw 'Run setup-client.bat or setup-client-conda.bat first.' }
    return Test-RowanPython -Python $rowanDefault
}

function Save-RowanPython {
    param(
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$Python
    )
    $rowanSelection = Join-Path $Root '.rowan-python'
    if (Test-Path -LiteralPath $rowanSelection) {
        $rowanPrevious = [string](Get-Content -LiteralPath $rowanSelection -Raw -Encoding UTF8)
        if ($rowanPrevious.Trim() -eq $Python) { return }
        Copy-Item -LiteralPath $rowanSelection -Destination ($rowanSelection + '.bak-' + [guid]::NewGuid().ToString('N'))
    }
    [System.IO.File]::WriteAllText($rowanSelection, $Python + [Environment]::NewLine, [System.Text.UTF8Encoding]::new($false))
}

function Enable-RowanPythonEnvironment {
    param([Parameter(Mandatory = $true)][string]$Python)
    $rowanPrefix = Split-Path -Parent $Python
    # Conda DLLs and subprocess tools also need the selected environment on PATH
    # when ordinary start-client.bat is opened outside Anaconda Prompt.
    if (Test-Path -LiteralPath (Join-Path $rowanPrefix 'conda-meta') -PathType Container) {
        $rowanPaths = @($rowanPrefix)
        foreach ($rowanChild in @('Library\mingw-w64\bin', 'Library\usr\bin', 'Library\bin', 'Scripts', 'bin')) {
            $rowanPaths += Join-Path $rowanPrefix $rowanChild
        }
        $env:PATH = ($rowanPaths -join [System.IO.Path]::PathSeparator) + [System.IO.Path]::PathSeparator + $env:PATH
        $env:CONDA_PREFIX = $rowanPrefix
    }
}
