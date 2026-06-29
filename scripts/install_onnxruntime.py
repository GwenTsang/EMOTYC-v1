from __future__ import annotations

import argparse
import importlib.metadata
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Installer exactement un paquet ONNX Runtime")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--auto", action="store_true", help="Choisir CPU sauf si l'etat CUDA/GPU est ambigu")
    group.add_argument("--cpu", action="store_true", help="Install onnxruntime")
    group.add_argument("--gpu", action="store_true", help="Install onnxruntime-gpu")
    args = parser.parse_args()

    if args.auto:
        package = choose_auto_package()
    else:
        package = "onnxruntime-gpu" if args.gpu else "onnxruntime"
    ensure_single_runtime(package)
    if package_installed(package):
        print(f"{package} est deja installe.")
        return
    print(f"Installation de {package}.")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def choose_auto_package() -> str:
    if shutil.which("nvidia-smi") is None:
        print("Aucun signal GPU/CUDA detecte. Choix de ONNX Runtime CPU.")
        return "onnxruntime"
    raise SystemExit(
        "Un GPU est visible, mais la compatibilite CUDA/cuDNN n'a pas ete verifiee. "
        "Lancez explicitement avec --cpu ou --gpu."
    )


def ensure_single_runtime(target: str) -> None:
    installed_cpu = package_installed("onnxruntime")
    installed_gpu = package_installed("onnxruntime-gpu")
    if target == "onnxruntime" and installed_gpu:
        raise SystemExit("onnxruntime-gpu est deja installe. Supprimez-le avant onnxruntime.")
    if target == "onnxruntime-gpu" and installed_cpu:
        raise SystemExit("onnxruntime est deja installe. Supprimez-le avant onnxruntime-gpu.")


def package_installed(name: str) -> bool:
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


if __name__ == "__main__":
    main()
