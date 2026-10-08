# PyInstaller recipe for the packaged engine. Build with packaging/build.ps1,
# which puts the output outside OneDrive.
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

# Packages whose data files (GDAL and PROJ grids, Rust extensions) or
# plug-in modules are loaded by name at run time, where static analysis
# cannot see them.
for package in ("rasterio", "pyproj", "laspy", "lazrs", "cv2", "pypdfium2", "pypdfium2_raw", "winrt"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("fiducia")
hiddenimports += ["anthropic", "openai", "multipart", "python_multipart"]


a = Analysis(
    ["packaged_main.py"],
    pathex=["."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib", "IPython", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fiducia-engine",
    console=True,  # Electron starts it with windowsHide, so no window shows
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="fiducia-engine", upx=False)
