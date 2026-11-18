param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $projectRoot
$venvPython = Join-Path $projectRoot '.venv\Scripts\python.exe'
$preparedPython = Join-Path $env:USERPROFILE 'anaconda3\python.exe'
if ($env:PATHSNAP_PYTHON) {
    $pythonExe = $env:PATHSNAP_PYTHON
} elseif (Test-Path -LiteralPath $venvPython) {
    $pythonExe = $venvPython
} elseif ((Test-Path -LiteralPath (Join-Path $projectRoot '.runtime')) -and (Test-Path -LiteralPath $preparedPython)) {
    $pythonExe = $preparedPython
    $env:PYTHONPATH = (Join-Path $projectRoot '.runtime') + [IO.Path]::PathSeparator + $projectRoot
} else {
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
}
$env:PYTHONIOENCODING = 'utf-8'
& $pythonExe (Join-Path $PSScriptRoot 'run_server.py') --host 0.0.0.0 --port $Port
exit $LASTEXITCODE
