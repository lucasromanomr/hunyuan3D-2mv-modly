"""
Modly extension setup for Hunyuan3D-2mv.

Creates an isolated environment using a Conda env (for the CUDA toolkit
matching PyTorch's bundled CUDA version) plus a regular Python venv
wrapper that lives at the Modly-expected path:

    <ext_dir>/venv/Scripts/python.exe   (Windows)
    <ext_dir>/venv/bin/python           (Linux/Mac)

Layout:
    <ext_dir>/
        .conda_env/      # conda env with python + cuda-toolkit
        venv/            # regular venv whose Python is the conda Python
        Hunyuan3D-2/     # cloned upstream
        .bin/            # bundled micromamba binary (if downloaded)

Why both: Modly's runner hard-codes the venv layout. Conda envs have a
different layout, so we use the conda env strictly to ship a matching
CUDA toolkit + a Python base, then create a regular venv on top so Modly
can locate the interpreter at the expected path.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"
MACHINE = platform.machine().lower()


# --------------------------------------------------------------------- #
# Micromamba bootstrap
# --------------------------------------------------------------------- #

MICROMAMBA_URLS = {
    ("Windows", "amd64"):  "https://micro.mamba.pm/api/micromamba/win-64/latest",
    ("Windows", "x86_64"): "https://micro.mamba.pm/api/micromamba/win-64/latest",
    ("Linux",   "x86_64"): "https://micro.mamba.pm/api/micromamba/linux-64/latest",
    ("Linux",   "aarch64"):"https://micro.mamba.pm/api/micromamba/linux-aarch64/latest",
    ("Darwin",  "x86_64"): "https://micro.mamba.pm/api/micromamba/osx-64/latest",
    ("Darwin",  "arm64"):  "https://micro.mamba.pm/api/micromamba/osx-arm64/latest",
}


def _ensure_micromamba(ext_dir):
    """Return (exe_path, kind). Prefer system conda/mamba/micromamba.
    Falls back to downloading a private micromamba binary inside ext_dir."""
    for binary in ("micromamba", "mamba", "conda"):
        exe = shutil.which(binary)
        if exe:
            print("[setup] Using system %s: %s" % (binary, exe))
            return exe, binary

    bin_dir = ext_dir / ".bin"
    bin_dir.mkdir(exist_ok=True)
    target = bin_dir / ("micromamba.exe" if IS_WIN else "micromamba")
    if target.exists():
        return str(target), "micromamba"

    key = (platform.system(), MACHINE)
    url = MICROMAMBA_URLS.get(key)
    if url is None:
        raise RuntimeError("No micromamba binary for %s %s" % key)

    archive = bin_dir / "micromamba.tar.bz2"
    print("[setup] Downloading micromamba: %s" % url)
    urllib.request.urlretrieve(url, archive)

    with tarfile.open(archive, "r:bz2") as tar:
        for m in tar.getmembers():
            name = m.name.replace("\\", "/")
            if name.endswith("micromamba.exe") or name.endswith("bin/micromamba"):
                with tar.extractfile(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                break
    archive.unlink(missing_ok=True)

    if not target.exists():
        raise RuntimeError("micromamba binary not found in downloaded archive")
    if not IS_WIN:
        os.chmod(target, 0o755)
    print("[setup] micromamba installed: %s" % target)
    return str(target), "micromamba"


# --------------------------------------------------------------------- #
# Conda env creation
# --------------------------------------------------------------------- #

def _cuda_dotted(cuda_version):
    """128 -> '12.8', 121 -> '12.1', 118 -> '11.8'."""
    s = str(cuda_version)
    return "%s.%s" % (s[:-1], s[-1])


def _conda_python(conda_prefix):
    if IS_WIN:
        return conda_prefix / "python.exe"
    return conda_prefix / "bin" / "python"


def _conda_cuda_home(conda_prefix):
    """Path that holds bin/, lib/, include/ for the CUDA toolkit."""
    if IS_WIN:
        return conda_prefix / "Library"
    return conda_prefix


def _create_conda_env(mamba_exe, mamba_kind, conda_prefix, cuda_version):
    cuda_str = _cuda_dotted(cuda_version)
    print("[setup] Creating conda env at %s (python=3.11, cuda-toolkit=%s)" %
          (conda_prefix, cuda_str))

    if mamba_kind == "micromamba":
        cmd = [mamba_exe, "create", "-y", "-p", str(conda_prefix),
               "-c", "nvidia/label/cuda-%s.0" % cuda_str,
               "-c", "conda-forge",
               "python=3.11", "cuda-toolkit"]
    elif mamba_kind == "mamba":
        cmd = [mamba_exe, "create", "-y", "-p", str(conda_prefix),
               "-c", "nvidia", "-c", "conda-forge",
               "python=3.11", "cuda-toolkit=%s" % cuda_str]
    else:  # conda
        cmd = [mamba_exe, "create", "-y", "-p", str(conda_prefix),
               "--no-default-packages",
               "-c", "nvidia", "-c", "conda-forge",
               "python=3.11", "cuda-toolkit=%s" % cuda_str]

    subprocess.run(cmd, check=True)


# --------------------------------------------------------------------- #
# venv wrapper + pip helpers
# --------------------------------------------------------------------- #

def _venv_python(venv_dir):
    if IS_WIN:
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _site_packages(venv_dir):
    if IS_WIN:
        return venv_dir / "Lib" / "site-packages"
    return sorted((venv_dir / "lib").glob("python*/site-packages"))[-1]


def _create_venv(conda_python, venv_dir):
    print("[setup] Creating venv at %s (base: %s)" % (venv_dir, conda_python))
    # --system-site-packages exposes the conda env's pythonXX.dll/site-packages,
    # but we want isolation for installed deps. Plain venv suffices because
    # pythonXX.dll resolves next to python.exe via the venvlauncher trick.
    subprocess.run([str(conda_python), "-m", "venv", str(venv_dir)], check=True)


def pip(venv_dir, *args, env=None):
    py = _venv_python(venv_dir)
    subprocess.run([str(py), "-m", "pip", *args], check=True, env=env)


# --------------------------------------------------------------------- #
# Runtime DLL/PATH hook
# --------------------------------------------------------------------- #

def _write_runtime_pth(venv_dir, cuda_home):
    """Drop a .pth file so importing anything in the venv auto-adds the
    conda env's CUDA bin dir to the DLL search path / LD path."""
    sp = _site_packages(venv_dir)
    sp.mkdir(parents=True, exist_ok=True)

    if IS_WIN:
        body = (
            "import os\n"
            "_p = r'%s'\n"
            "try:\n"
            "    os.add_dll_directory(_p)\n"
            "except Exception:\n"
            "    pass\n"
            "os.environ.setdefault('CUDA_HOME', r'%s')\n"
            "os.environ.setdefault('CUDA_PATH', r'%s')\n"
            "os.environ['PATH'] = _p + os.pathsep + os.environ.get('PATH', '')\n"
        ) % (str(cuda_home / "bin"), str(cuda_home), str(cuda_home))
        # .pth files must be one logical line per entry; use exec-on-import trick
        pth = "import modly_cuda_hook\n"
        (sp / "modly_cuda.pth").write_text(pth, encoding="utf-8")
        (sp / "modly_cuda_hook.py").write_text(body, encoding="utf-8")
    else:
        lib_dir = str(cuda_home / "lib")
        body = (
            "import os\n"
            "_p = r'%s'\n"
            "os.environ.setdefault('CUDA_HOME', r'%s')\n"
            "os.environ['LD_LIBRARY_PATH'] = _p + os.pathsep + os.environ.get('LD_LIBRARY_PATH', '')\n"
        ) % (lib_dir, str(cuda_home))
        (sp / "modly_cuda.pth").write_text("import modly_cuda_hook\n", encoding="utf-8")
        (sp / "modly_cuda_hook.py").write_text(body, encoding="utf-8")


# --------------------------------------------------------------------- #
# Compiler / MSVC discovery (Windows)
# --------------------------------------------------------------------- #

def _find_cl_exe():
    candidates = []
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    pf86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    for root in (pf, pf86):
        for edition in ("BuildTools", "Community", "Professional", "Enterprise"):
            base = Path(root) / "Microsoft Visual Studio" / "2022" / edition / "VC" / "Tools" / "MSVC"
            if base.exists():
                candidates += list(base.glob("*/bin/Hostx64/x64/cl.exe"))
            base19 = Path(root) / "Microsoft Visual Studio" / "2019" / edition / "VC" / "Tools" / "MSVC"
            if base19.exists():
                candidates += list(base19.glob("*/bin/Hostx64/x64/cl.exe"))
    return candidates[0] if candidates else None


# --------------------------------------------------------------------- #
# custom_rasterizer build
# --------------------------------------------------------------------- #

def _build_rasterizer(venv_dir, rast_dir, cuda_home):
    if not rast_dir.exists():
        print("[setup] rast_dir not found: %s" % rast_dir)
        return False

    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)

    if IS_WIN:
        bin_dir = cuda_home / "bin"
        env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")
        env["PATH"] = str(_venv_python(venv_dir).parent) + os.pathsep + env["PATH"]
        if shutil.which("cl", path=env["PATH"]) is None:
            cl = _find_cl_exe()
            if cl:
                env["PATH"] = str(cl.parent) + os.pathsep + env["PATH"]
                print("[setup] Using cl.exe: %s" % cl)
            else:
                print("[setup] WARNING: cl.exe not found; build will likely fail")

    print("[setup] Building custom_rasterizer (CUDA_HOME=%s)" % cuda_home)
    result = subprocess.run(
        [str(_venv_python(venv_dir)), "setup.py", "build_ext", "--inplace"],
        cwd=str(rast_dir),
        env=env,
    )
    if result.returncode != 0:
        print("[setup] custom_rasterizer build exited with code %d" % result.returncode)
        return False

    built = (list(rast_dir.glob("custom_rasterizer_kernel*.pyd")) +
             list(rast_dir.glob("custom_rasterizer_kernel*.so")))
    if not built:
        print("[setup] no built artifact found in %s" % rast_dir)
        return False

    artifact = built[0]
    sp = _site_packages(venv_dir)
    shutil.copy2(str(artifact), str(sp / artifact.name))
    print("[setup] Built rasterizer: %s -> %s" % (artifact.name, sp))

    # Save back to extension root for future installs
    ext_dest = Path(__file__).parent / artifact.name
    try:
        shutil.copy2(str(artifact), str(ext_dest))
    except Exception:
        pass
    return True


def _use_prebuilt_rasterizer(venv_dir, rast_dir):
    ext_dir = Path(__file__).parent
    prebuilt = list(ext_dir.glob("custom_rasterizer_kernel*.pyd"))
    if not prebuilt:
        return False
    art = prebuilt[0]
    print("[setup] Falling back to prebuilt kernel: %s" % art.name)
    rast_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(art), str(rast_dir / art.name))
    sp = _site_packages(venv_dir)
    shutil.copy2(str(art), str(sp / art.name))
    return True


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def setup(python_exe, ext_dir, gpu_sm, cuda_version=128):
    ext_dir = Path(ext_dir).resolve()
    conda_prefix = ext_dir / ".conda_env"
    venv_dir = ext_dir / "venv"

    # ----- 1. micromamba ----------------------------------------------- #
    mamba_exe, mamba_kind = _ensure_micromamba(ext_dir)

    # ----- 2. conda env with cuda-toolkit ------------------------------ #
    conda_python = _conda_python(conda_prefix)
    if not conda_python.exists():
        _create_conda_env(mamba_exe, mamba_kind, conda_prefix, cuda_version)
    else:
        print("[setup] Conda env exists at %s" % conda_prefix)

    if not conda_python.exists():
        raise RuntimeError("Conda env created but python missing at %s" % conda_python)

    cuda_home = _conda_cuda_home(conda_prefix)

    # ----- 3. venv wrapper on top of conda python --------------------- #
    if not _venv_python(venv_dir).exists():
        _create_venv(conda_python, venv_dir)
    else:
        print("[setup] venv exists at %s" % venv_dir)

    # ----- 4. runtime PATH hook --------------------------------------- #
    _write_runtime_pth(venv_dir, cuda_home)

    # ----- 5. build tools --------------------------------------------- #
    pip(venv_dir, "install", "--upgrade", "pip")
    pip(venv_dir, "install", "ninja", "setuptools", "wheel")

    # ----- 6. pytorch + xformers -------------------------------------- #
    cuda_str = "cu%d" % cuda_version
    torch_index = "https://download.pytorch.org/whl/%s" % cuda_str

    if cuda_version >= 128:
        torch_pkgs = ["torch>=2.7.0", "torchvision>=0.22.0", "torchaudio>=2.7.0"]
        xformers_pkg = "xformers==0.0.30"
    elif cuda_version >= 124:
        torch_pkgs = ["torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0"]
        xformers_pkg = "xformers==0.0.30"
    else:
        torch_pkgs = ["torch==2.5.1", "torchvision==0.20.1", "torchaudio==2.5.1"]
        xformers_pkg = "xformers==0.0.28.post2"

    print("[setup] Installing PyTorch (%s)" % cuda_str)
    pip(venv_dir, "install", *torch_pkgs, "--index-url", torch_index)

    print("[setup] Installing %s" % xformers_pkg)
    if gpu_sm >= 70:
        pip(venv_dir, "install", xformers_pkg, "--index-url", torch_index)
    else:
        pip(venv_dir, "install", "xformers==0.0.28.post2",
            "--index-url", "https://download.pytorch.org/whl/cu118")

    # ----- 7. core deps ----------------------------------------------- #
    print("[setup] Installing core dependencies")
    pip(venv_dir, "install",
        "accelerate", "omegaconf", "einops", "Pillow", "numpy", "scipy",
        "trimesh", "pymeshlab", "pygltflib", "opencv-python-headless",
        "tqdm", "safetensors", "rembg")

    if not IS_WIN:
        try:
            pip(venv_dir, "install", "triton")
        except subprocess.CalledProcessError:
            print("[setup] triton skip (non-fatal)")

    print("[setup] Installing onnxruntime")
    if gpu_sm >= 70:
        try:
            pip(venv_dir, "install", "onnxruntime-gpu")
        except subprocess.CalledProcessError:
            pip(venv_dir, "install", "onnxruntime")
    else:
        pip(venv_dir, "install", "onnxruntime")

    # ----- 8. clone Hunyuan3D-2 --------------------------------------- #
    repo_dir = ext_dir / "Hunyuan3D-2"
    if not repo_dir.exists():
        print("[setup] Cloning Hunyuan3D-2")
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git",
             str(repo_dir)],
            check=True,
        )
    else:
        print("[setup] Hunyuan3D-2 already present")

    # ----- 9. build custom_rasterizer (or fall back to prebuilt) ------ #
    rast_dir = repo_dir / "hy3dgen" / "texgen" / "custom_rasterizer"
    rast_ok = _build_rasterizer(venv_dir, rast_dir, cuda_home)
    if not rast_ok:
        rast_ok = _use_prebuilt_rasterizer(venv_dir, rast_dir)
        if not rast_ok:
            print("[setup] *** custom_rasterizer unavailable — texture step will fail ***")

    # ----- 10. install Python packages -------------------------------- #
    print("[setup] Installing hy3dgen (editable)")
    subprocess.run(
        [str(_venv_python(venv_dir)), "-m", "pip", "install", "-e", str(repo_dir)],
        check=True,
    )

    print("[setup] Installing custom_rasterizer Python package")
    env = os.environ.copy()
    env["CUDA_HOME"] = str(cuda_home)
    env["CUDA_PATH"] = str(cuda_home)
    if IS_WIN:
        env["PATH"] = str(cuda_home / "bin") + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        [str(_venv_python(venv_dir)), "-m", "pip", "install", "-e", str(rast_dir)],
        env=env,
    )
    if result.returncode != 0:
        print("[setup] pip install -e rast_dir failed; copying Python sources")
        py_src = rast_dir / "custom_rasterizer"
        if py_src.is_dir():
            sp = _site_packages(venv_dir)
            shutil.copytree(str(py_src), str(sp / "custom_rasterizer"), dirs_exist_ok=True)

    print("[setup] DONE")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        try:
            args = json.loads(sys.argv[1])
            setup(
                python_exe=args["python_exe"],
                ext_dir=args["ext_dir"],
                gpu_sm=int(args["gpu_sm"]),
                cuda_version=int(args.get("cuda_version", 128)),
            )
        except (json.JSONDecodeError, KeyError) as e:
            print("Bad JSON args: %s" % e, file=sys.stderr)
            sys.exit(1)
    elif len(sys.argv) >= 4:
        cuda_version = int(sys.argv[4]) if len(sys.argv) >= 5 else 128
        setup(
            python_exe=sys.argv[1],
            ext_dir=sys.argv[2],
            gpu_sm=int(sys.argv[3]),
            cuda_version=cuda_version,
        )
    else:
        print('Usage: setup.py <python_exe> <ext_dir> <gpu_sm> [cuda_version]')
        print('   or: setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89,"cuda_version":128}\'')
        sys.exit(1)
