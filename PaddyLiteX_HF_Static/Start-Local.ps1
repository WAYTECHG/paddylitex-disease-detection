param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
$site = $PSScriptRoot
$python = $null

if ($env:CONDA_PREFIX) {
    $candidate = Join-Path $env:CONDA_PREFIX 'python.exe'
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $python = $candidate
    }
}

if (-not $python) {
    $command = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($command) {
        $python = $command.Source
    }
}

if (-not $python) {
    throw 'Python was not found. Activate your paddy_yolo environment first.'
}

Write-Host "PaddyLiteX local website: http://localhost:$Port"
Write-Host 'Keep this terminal open. Press Ctrl+C to stop.'
& $python -m http.server $Port --bind 127.0.0.1 --directory $site

if ($LASTEXITCODE -ne 0) {
    throw 'Local server stopped with an error. Check whether the port is already in use.'
}
