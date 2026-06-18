"""
ingest.py — Stage 1: turn your input into a clean folder of frames.

Accepts EITHER:
  - a folder / zip of images, or
  - a video file (mp4/mov/...).

For video we extract frames with ffmpeg at a target FPS and (optionally) drop
the blurriest frames, because motion-blurred frames poison Structure-from-Motion.

Output: <work_dir>/images/  containing 0001.jpg, 0002.jpg, ...
All frames are renamed to a zero-padded sequence so COLMAP's sequential matcher
(used for video) sees them in capture order.
"""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import List

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _run(cmd: List[str]) -> None:
    print("  $", " ".join(str(c) for c in cmd))
    subprocess.check_call(cmd)


def _blur_score(path: Path) -> float:
    """Variance of the Laplacian: higher = sharper. Used to drop blurry frames."""
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _maybe_resize(img_dir: Path, max_dim: int) -> None:
    """Downscale the long edge to max_dim. 1600px is a good 3DGS sweet spot:
    big enough for detail, small enough that COLMAP + training stay fast."""
    if max_dim <= 0:
        return
    import cv2
    for p in sorted(img_dir.iterdir()):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        if scale < 1.0:
            img = cv2.resize(img, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 92])


def ingest(source: str | Path,
           work_dir: str | Path,
           is_video: bool | None = None,
           fps: float = 2.0,
           max_images: int = 0,
           blur_drop_fraction: float = 0.0,
           resize_max_dim: int = 1600) -> Path:
    """
    source            : path to a video, an image folder, or a .zip of images.
    work_dir          : project working dir; images land in work_dir/images.
    is_video          : force video handling; auto-detected from extension if None.
    fps               : frames/sec to extract from video.
    max_images        : if >0, evenly subsample down to this many frames.
    blur_drop_fraction: 0.0-0.9; fraction of blurriest frames to discard.
    resize_max_dim    : long-edge cap in px (0 = keep original).
    """
    source = Path(source)
    work_dir = Path(work_dir)
    img_dir = work_dir / "images"
    if img_dir.exists():
        shutil.rmtree(img_dir)
    img_dir.mkdir(parents=True)

    raw = work_dir / "_raw_frames"
    if raw.exists():
        shutil.rmtree(raw)
    raw.mkdir()

    if is_video is None:
        is_video = source.suffix.lower() in VIDEO_EXTS

    # --- gather raw frames into `raw` ---------------------------------
    if is_video:
        print(f"Extracting frames from video at {fps} fps ...")
        _run(["ffmpeg", "-hide_banner", "-loglevel", "error",
              "-i", str(source),
              "-vf", f"fps={fps}",
              "-qscale:v", "2",
              str(raw / "frame_%05d.jpg")])
    else:
        if source.suffix.lower() == ".zip":
            print("Unzipping image archive ...")
            shutil.unpack_archive(str(source), str(raw))
        elif source.is_dir():
            for p in source.rglob("*"):
                if p.suffix.lower() in IMAGE_EXTS:
                    shutil.copy2(p, raw / p.name)
        else:
            raise ValueError(f"Unsupported source: {source}")

    frames = sorted([p for p in raw.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
    if not frames:
        raise RuntimeError("No frames found after ingest. Check your source file.")
    print(f"  collected {len(frames)} raw frames")

    # --- optional: drop blurriest frames ------------------------------
    if blur_drop_fraction > 0.0 and len(frames) > 10:
        scored = sorted(frames, key=_blur_score)            # blurriest first
        n_drop = int(len(frames) * blur_drop_fraction)
        keep = set(scored[n_drop:])
        frames = [f for f in frames if f in keep]
        print(f"  dropped {n_drop} blurry frames -> {len(frames)} remain")

    # --- optional: cap total count (even subsample) -------------------
    if max_images and len(frames) > max_images:
        step = len(frames) / max_images
        frames = [frames[int(i * step)] for i in range(max_images)]
        print(f"  subsampled to {len(frames)} frames")

    # --- write final, sequentially named images -----------------------
    for i, f in enumerate(frames, 1):
        shutil.copy2(f, img_dir / f"{i:05d}.jpg")
    shutil.rmtree(raw)

    _maybe_resize(img_dir, resize_max_dim)
    n = len(list(img_dir.glob("*.jpg")))
    print(f"Ingest done: {n} images in {img_dir}")
    return img_dir
