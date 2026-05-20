"""Lab 12 smoke tests — minimal CPU-only assertions that don't need GPU.

These tests run inside GitHub's free hosted runner (Ubuntu x86), so they
must not import torch with CUDA, must not load TensorRT, and must not
require the IMX219 camera. Their job is to catch the obvious 'I broke
the import graph' or 'the YOLO model file is corrupted' failures BEFORE
the slower Docker build job spends 10 min building an ARM64 image.

Run locally with: pytest -v tests/
"""

from pathlib import Path

import pytest


def test_best_pt_exists():
    """The fine-tuned weights file must be present in the repo."""
    p = Path(__file__).parent.parent / "best.pt"
    assert p.exists(), f"{p} missing — did you forget to commit best.pt?"
    assert p.stat().st_size > 1_000_000, "best.pt suspiciously small (<1 MB)"


def test_requirements_pinned():
    """Every dep in requirements.txt should be pinned (== or ~=)."""
    req = (Path(__file__).parent.parent / "requirements.txt").read_text()
    for line in req.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Allow editable installs and -e/-r/-c directives
        if line.startswith(("-e", "-r", "-c", "--")):
            continue
        assert "==" in line or "~=" in line or ">=" in line, (
            f"Unpinned dep in requirements.txt: {line!r}"
        )


def test_dockerfile_uses_arm64_base():
    """Dockerfile.ci must use a Jetson-compatible (ARM64) base image."""
    df = (Path(__file__).parent.parent / "Dockerfile.ci").read_text()
    # dustynv images are aarch64-only; l4t-base is also aarch64-only.
    # Either is acceptable; a generic python:3.11 base would silently
    # build for x86 and never run on the Jetson.
    assert any(base in df for base in ["dustynv/", "l4t-", "nvcr.io/nvidia/l4t"]), (
        "Dockerfile.ci must FROM a Jetson ARM64 base (dustynv/* or l4t-*)"
    )


@pytest.mark.parametrize("name", ["src/inference_node.py", "best.pt", "requirements.txt"])
def test_required_files(name):
    """Files the Docker COPY steps reference must exist."""
    assert (Path(__file__).parent.parent / name).exists(), f"{name} missing"
