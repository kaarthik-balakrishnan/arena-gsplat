"""
run_colmap.py — Stage 2: Structure-from-Motion with COLMAP.

This is ONE clean solve (no model merging — that was the bug in the old repo).
The matcher is chosen automatically based on your data:

  - video / sequential capture     -> sequential matcher (+ loop closure)
  - small image set (<= threshold) -> exhaustive matcher (most accurate)
  - large image set (>  threshold) -> vocab-tree matcher (scales to thousands)

After mapping we run image_undistorter, which produces a PINHOLE dataset that
gsplat / 3DGS trainers consume directly. The final layout is:

  <work_dir>/colmap/dense/
       images/            undistorted images
       sparse/0/          cameras.bin, images.bin, points3D.bin

which is exactly what scripts/train.py points the trainer at.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import List, Optional

# Pretrained vocabulary tree for large-scene matching (downloaded on demand).
VOCAB_TREE_URL = (
    "https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin"
)


def _run(cmd: List[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd)


def _count_images(images_dir: Path) -> int:
    return len([p for p in images_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])


def _registered_images(sparse_model: Path) -> int:
    """Count registered cameras in a sparse model (binary or text)."""
    txt = sparse_model / "images.txt"
    if txt.exists():
        n = sum(1 for ln in txt.read_text().splitlines()
                if ln.strip() and not ln.startswith("#"))
        return n // 2
    # binary: ask COLMAP to convert, then count
    try:
        _run(["colmap", "model_converter",
              "--input_path", str(sparse_model),
              "--output_path", str(sparse_model),
              "--output_type", "TXT"])
        return _registered_images(sparse_model)
    except Exception:
        return -1


def choose_matcher(n_images: int, is_sequential: bool,
                   exhaustive_max: int = 150) -> str:
    if is_sequential:
        return "sequential"
    if n_images <= exhaustive_max:
        return "exhaustive"
    return "vocab_tree"


def run_colmap(work_dir: str | Path,
               camera_model: str = "OPENCV",
               single_camera: bool = True,
               is_sequential: bool = False,
               matcher: Optional[str] = None,
               use_gpu: bool = True,
               exhaustive_max: int = 150) -> Path:
    """
    camera_model : OPENCV is a safe default for an unknown phone/camera.
                   Use SIMPLE_RADIAL if you know it's a clean single-lens phone.
    single_camera: True if every image came from the same physical camera.
    is_sequential: True for video frames or a continuous walk-through.
    matcher      : override auto-selection ('exhaustive'|'sequential'|'vocab_tree').
    Returns the path to the undistorted dense dir (images/ + sparse/0/).
    """
    work_dir = Path(work_dir)
    images_dir = work_dir / "images"
    col = work_dir / "colmap"
    col.mkdir(parents=True, exist_ok=True)
    db = col / "database.db"
    sparse = col / "sparse"
    sparse.mkdir(exist_ok=True)

    n = _count_images(images_dir)
    if matcher is None:
        matcher = choose_matcher(n, is_sequential, exhaustive_max)
    gpu = "1" if use_gpu else "0"
    print(f"COLMAP on {n} images | camera={camera_model} | matcher={matcher} | gpu={gpu}")

    # 1) Feature extraction -------------------------------------------
    _run(["colmap", "feature_extractor",
          "--database_path", str(db),
          "--image_path", str(images_dir),
          "--ImageReader.camera_model", camera_model,
          "--ImageReader.single_camera", "1" if single_camera else "0",
          "--SiftExtraction.use_gpu", gpu])

    # 2) Matching ------------------------------------------------------
    if matcher == "exhaustive":
        _run(["colmap", "exhaustive_matcher",
              "--database_path", str(db),
              "--SiftMatching.use_gpu", gpu])
    elif matcher == "sequential":
        # Sequential + loop closure via vocab tree catches revisited areas.
        vt = col / "vocab_tree.bin"
        if not vt.exists():
            print("  downloading vocab tree for loop closure ...")
            urllib.request.urlretrieve(VOCAB_TREE_URL, vt)
        _run(["colmap", "sequential_matcher",
              "--database_path", str(db),
              "--SequentialMatching.overlap", "10",
              "--SequentialMatching.loop_detection", "1",
              "--SequentialMatching.vocab_tree_path", str(vt),
              "--SiftMatching.use_gpu", gpu])
    elif matcher == "vocab_tree":
        vt = col / "vocab_tree.bin"
        if not vt.exists():
            print("  downloading vocab tree ...")
            urllib.request.urlretrieve(VOCAB_TREE_URL, vt)
        _run(["colmap", "vocab_tree_matcher",
              "--database_path", str(db),
              "--VocabTreeMatching.vocab_tree_path", str(vt),
              "--SiftMatching.use_gpu", gpu])
    else:
        raise ValueError(f"Unknown matcher: {matcher}")

    # 3) Mapping (sparse reconstruction) -------------------------------
    _run(["colmap", "mapper",
          "--database_path", str(db),
          "--image_path", str(images_dir),
          "--output_path", str(sparse)])

    # COLMAP may produce several sub-models in sparse/0, sparse/1, ...
    # Pick the one with the MOST registered images (the dominant reconstruction).
    submodels = sorted([d for d in sparse.iterdir() if d.is_dir()])
    if not submodels:
        raise RuntimeError("COLMAP mapping produced no model. "
                           "Likely too little image overlap.")
    best = max(submodels, key=_registered_images)
    reg = _registered_images(best)
    print(f"  best sub-model: {best.name} with {reg}/{n} images registered")
    if reg < 0.5 * n:
        print(f"  WARNING: only {reg}/{n} registered. Your capture probably has "
              f"weak overlap between viewpoints. More results won't fix this; "
              f"more overlapping coverage will.")

    # 4) Undistort -> clean PINHOLE dataset for the trainer ------------
    dense = col / "dense"
    if dense.exists():
        shutil.rmtree(dense)
    _run(["colmap", "image_undistorter",
          "--image_path", str(images_dir),
          "--input_path", str(best),
          "--output_path", str(dense),
          "--output_type", "COLMAP"])

    # image_undistorter writes sparse/ (not sparse/0/). gsplat wants sparse/0/.
    und_sparse = dense / "sparse"
    target = dense / "sparse" / "0"
    if und_sparse.exists() and not target.exists():
        tmp = dense / "_sparse0"
        shutil.move(str(und_sparse), str(tmp))
        (dense / "sparse").mkdir(exist_ok=True)
        shutil.move(str(tmp), str(target))

    print(f"COLMAP done. Trainer-ready dataset: {dense}")
    return dense, reg, matcher
