"""
train.py — Stage 3: Gaussian Splatting training via gsplat's simple_trainer.

We deliberately do NOT hand-write the training loop. We call gsplat's official
example trainer, which is the reference implementation of adaptive density
control, opacity reset, SH scheduling and optimiser handling. (Re-implementing
those by hand is what produced the exploding-Gaussian / frozen-SH-colour bugs in
the previous project.)

Checkpoints and PLYs are written into result_dir, which lives on Google Drive,
at every save step. So if Colab drops mid-training you keep the latest artifact.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _latest_ckpt(result_dir: Path) -> Optional[Path]:
    ckpts = list(result_dir.rglob("ckpt_*.pt")) + list(result_dir.rglob("*.pt"))
    if not ckpts:
        return None
    # newest by step number embedded in filename, else by mtime
    def step_of(p: Path) -> int:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else int(p.stat().st_mtime)
    return max(ckpts, key=step_of)


def latest_ply(result_dir: Path) -> Optional[Path]:
    plys = list(result_dir.rglob("*.ply"))
    if not plys:
        return None
    def step_of(p: Path) -> int:
        digits = "".join(ch for ch in p.stem if ch.isdigit())
        return int(digits) if digits else int(p.stat().st_mtime)
    return max(plys, key=step_of)


def train(dense_dir: str | Path,
          result_dir: str | Path,
          gsplat_dir: str | Path,
          max_steps: int = 30000,
          data_factor: int = 1,
          save_steps: Optional[List[int]] = None,
          strategy: str = "default",
          extra_args: Optional[List[str]] = None) -> Path:
    """
    dense_dir   : COLMAP undistorted output (has images/ + sparse/0/).
    result_dir  : where checkpoints/PLYs are written (put this on Drive!).
    gsplat_dir  : path to the cloned gsplat repo (for examples/simple_trainer.py).
    max_steps   : 30000 is the paper default and a good production target.
    data_factor : 1 = full res. Set 2 or 4 to downsample if you hit VRAM limits.
    strategy    : 'default' (paper densification) or 'mcmc' (fixed budget).
    """
    dense_dir = Path(dense_dir)
    result_dir = Path(result_dir)
    gsplat_dir = Path(gsplat_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    trainer = gsplat_dir / "examples" / "simple_trainer.py"
    if not trainer.exists():
        raise FileNotFoundError(
            f"simple_trainer.py not found at {trainer}. "
            f"Clone gsplat first: git clone https://github.com/nerfstudio-project/gsplat")

    if save_steps is None:
        # Frequent saves = a recoverable artifact if Colab disconnects.
        save_steps = sorted({max_steps // 4, max_steps // 2,
                             3 * max_steps // 4, max_steps})

    cmd = [sys.executable, str(trainer), strategy,
           "--data-dir", str(dense_dir),
           "--data-factor", str(data_factor),
           "--result-dir", str(result_dir),
           "--max-steps", str(max_steps),
           "--save-steps", *[str(s) for s in save_steps],
           "--disable-viewer"]
    if extra_args:
        cmd += extra_args

    print("Launching gsplat trainer:")
    print("  $", " ".join(cmd))
    print("  (flag names can differ slightly between gsplat versions; if it "
          "errors on an argument, run the trainer with --help to check.)")
    subprocess.check_call(cmd, cwd=str(gsplat_dir))

    ply = latest_ply(result_dir)
    print(f"Training finished. Latest PLY: {ply}")
    return result_dir
