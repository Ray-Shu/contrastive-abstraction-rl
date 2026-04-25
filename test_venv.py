#!/usr/bin/env python3
"""Quick venv sanity check — verifies all key dependencies are installed."""

import sys


def test_imports():
    """Test that all key dependencies import cleanly."""
    deps = {
        "torch": "PyTorch",
        "pytorch_lightning": "PyTorch Lightning",
        "gymnasium": "Gymnasium",
        "wandb": "Weights & Biases",
        "numpy": "NumPy",
        "sklearn": "Scikit-learn",
        "umap": "UMAP",
        "hflayers": "Hopfield Layers",
    }

    print("Testing imports...")
    failed = []
    for module, name in deps.items():
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append(name)

    return len(failed) == 0, failed


def test_torch():
    """Test PyTorch and CUDA availability."""
    import torch

    print("\nTesting PyTorch...")
    print(f"  Version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  Device count: {torch.cuda.device_count()}")
        print(f"  Device: {torch.cuda.get_device_name(0)}")

        # CUDA smoke test
        try:
            x = torch.randn(10, 32, device="cuda")
            y = x @ x.T
            print(f"  CUDA smoke test: {x.shape} @ {x.T.shape} -> {y.shape} [OK]")
        except Exception as e:
            print(f"  CUDA smoke test failed: {e}")
    else:
        print("  (Will run on CPU)")

    # CPU smoke test
    x = torch.randn(10, 32, device="cpu")
    y = x @ x.T
    print(f"  CPU smoke test: {x.shape} @ {x.T.shape} -> {y.shape} [OK]")


def test_project_imports():
    """Test that project modules import cleanly."""
    import sys
    import os

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    print("\nTesting project module imports...")
    modules = [
        "src.models.cl_model",
        "src.models.beta_model",
        "src.data.trajectories",
        "src.data.sampler",
    ]

    failed = []
    for mod in modules:
        try:
            __import__(mod)
            print(f"  [OK] {mod}")
        except Exception as e:
            print(f"  [FAIL] {mod}: {e}")
            failed.append(mod)

    return len(failed) == 0, failed


def main():
    print("=" * 60)
    print("Environment Setup Check")
    print("=" * 60)

    all_ok = True

    ok, failed = test_imports()
    if not ok:
        print(f"\n[WARNING] Missing packages: {', '.join(failed)}")
        print("   Run: uv sync")
        all_ok = False

    try:
        test_torch()
    except Exception as e:
        print(f"\n[FAIL] PyTorch test failed: {e}")
        all_ok = False

    try:
        ok, failed = test_project_imports()
        if not ok:
            print(f"\n[WARNING] Project module errors: {', '.join(failed)}")
            all_ok = False
    except Exception as e:
        print(f"\n[FAIL] Project import test failed: {e}")
        all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("[OK] All checks passed! Environment is ready.")
        return 0
    else:
        print("[FAIL] Some checks failed. See above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
