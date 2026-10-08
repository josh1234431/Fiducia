# Builds the Fiducia Windows installer.
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#
# 1. Builds the interface (vite)            -> dist\
# 2. Freezes the engine (PyInstaller)       -> <build>\engine\fiducia-engine\
# 3. Packages both into an installer        -> <build>\release\Fiducia-Setup-<version>.exe
#
# <build> is FIDUCIA_BUILD_DIR, or a Fiducia-build folder beside the
# checkout. Not under AppData: a program started from a packaged app (such as
# an editor installed from the Microsoft Store) has its AppData writes
# redirected to a private folder, and the installer would be invisible to
# Explorer. Needs the engine's virtualenv (FIDUCIA_PYTHON, or
# %LOCALAPPDATA%\Fiducia\venv) with requirements.txt and pyinstaller installed.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$build = if ($env:FIDUCIA_BUILD_DIR) { $env:FIDUCIA_BUILD_DIR } else { Join-Path (Split-Path -Parent $root) 'Fiducia-build' }
$env:FIDUCIA_BUILD_DIR = $build   # read by electron-builder.config.cjs
$python = if ($env:FIDUCIA_PYTHON) { $env:FIDUCIA_PYTHON } else { Join-Path $env:LOCALAPPDATA 'Fiducia\venv\Scripts\python.exe' }

Push-Location $root
try {
    Write-Host '== Interface' -ForegroundColor Cyan
    npx vite build
    if ($LASTEXITCODE) { throw 'Interface build failed' }

    Write-Host '== Engine' -ForegroundColor Cyan
    Push-Location engine
    & $python -m PyInstaller fiducia-engine.spec --noconfirm `
        --distpath (Join-Path $build 'engine') --workpath (Join-Path $build 'work')
    $code = $LASTEXITCODE
    Pop-Location
    if ($code) { throw 'Engine build failed' }

    Write-Host '== Installer' -ForegroundColor Cyan
    npx electron-builder --win --config electron-builder.config.cjs
    if ($LASTEXITCODE) { throw 'Installer build failed' }

    Write-Host "Done: $(Join-Path $build 'release')" -ForegroundColor Green
}
finally {
    Pop-Location
}
