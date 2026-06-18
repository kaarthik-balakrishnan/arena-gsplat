"""
export.py — Stage 4: collect the final PLY into a tidy outputs/ folder.

Prefers a .ply that the trainer already wrote. If only a checkpoint .pt exists
(e.g. training was interrupted), it converts the newest checkpoint to PLY so you
always end up with something viewable.
"""
from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional

from .train import latest_ply, _latest_ckpt
from .ckpt_to_ply import ckpt_to_ply


def export(result_dir: str | Path,
           work_dir: str | Path,
           project: str = "scene") -> Optional[Path]:
    result_dir = Path(result_dir)
    out_dir = Path(work_dir) / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{project}.ply"

    ply = latest_ply(result_dir)
    if ply is not None:
        shutil.copy2(ply, final)
        print(f"Exported trained PLY -> {final}")
        return final

    ckpt = _latest_ckpt(result_dir)
    if ckpt is not None:
        print(f"No PLY found; converting latest checkpoint {ckpt.name} ...")
        ckpt_to_ply(ckpt, final)
        return final

    print("Nothing to export — no PLY or checkpoint found in result dir.")
    return None
