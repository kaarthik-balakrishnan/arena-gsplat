"""
pipeline.py — the orchestrator.

Runs the four stages in order, each one resumable. Re-running after a Colab
disconnect picks up exactly where you left off because every finished stage is
recorded in state.json on Drive and skipped on the next run.

Typical use (from the notebook or CLI):

    from scripts.pipeline import Pipeline
    pipe = Pipeline(project="my_scene", drive_root="/content/drive/MyDrive/gsplat",
                    gsplat_dir="/content/gsplat")
    pipe.run(source="/content/input.mp4", is_video=True)

Force a single stage to re-run:
    pipe.state.reset("colmap"); pipe.run(...)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

from .state import State
from . import ingest as ingest_mod
from . import run_colmap as colmap_mod
from . import train as train_mod
from . import export as export_mod


class Pipeline:
    def __init__(self, project: str, drive_root: str | Path,
                 gsplat_dir: str | Path):
        self.project = project
        self.work_dir = Path(drive_root) / project
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.gsplat_dir = Path(gsplat_dir)
        self.result_dir = self.work_dir / "gsplat_result"
        self.state = State(self.work_dir, project)
        print(self.state.summary())

    # ---- individual stages ------------------------------------------
    def stage_ingest(self, source, is_video=None, fps=2.0, max_images=0,
                     blur_drop_fraction=0.0, resize_max_dim=1600, force=False):
        if self.state.is_done("ingest") and not force:
            print("[skip] ingest already done"); return
        ingest_mod.ingest(source, self.work_dir, is_video=is_video, fps=fps,
                          max_images=max_images,
                          blur_drop_fraction=blur_drop_fraction,
                          resize_max_dim=resize_max_dim)
        n = len(list((self.work_dir / "images").glob("*.jpg")))
        self.state.mark_done("ingest", n_images=n,
                             images_dir=str(self.work_dir / "images"))

    def stage_colmap(self, camera_model="OPENCV", single_camera=True,
                     is_sequential=False, matcher=None, use_gpu=True,
                     exhaustive_max=150, force=False):
        if self.state.is_done("colmap") and not force:
            print("[skip] colmap already done"); return
        dense, reg, used = colmap_mod.run_colmap(
            self.work_dir, camera_model=camera_model,
            single_camera=single_camera, is_sequential=is_sequential,
            matcher=matcher, use_gpu=use_gpu, exhaustive_max=exhaustive_max)
        self.state.mark_done("colmap", dense_dir=str(dense),
                             registered=reg, matcher=used)

    def stage_train(self, max_steps=30000, data_factor=1, strategy="default",
                    force=False):
        if self.state.is_done("train") and not force:
            print("[skip] train already done"); return
        dense = self.state.get("colmap", "dense_dir")
        if not dense:
            raise RuntimeError("Run COLMAP before training.")
        train_mod.train(dense, self.result_dir, self.gsplat_dir,
                        max_steps=max_steps, data_factor=data_factor,
                        strategy=strategy)
        ply = train_mod.latest_ply(self.result_dir)
        self.state.mark_done("train", result_dir=str(self.result_dir),
                             last_ply=str(ply) if ply else None,
                             max_steps=max_steps)

    def stage_export(self, force=False):
        if self.state.is_done("export") and not force:
            print("[skip] export already done"); return
        final = export_mod.export(self.result_dir, self.work_dir, self.project)
        self.state.mark_done("export", final_ply=str(final) if final else None)
        return final

    # ---- full run ----------------------------------------------------
    def run(self, source, is_video=None, fps=2.0, max_images=0,
            blur_drop_fraction=0.0, resize_max_dim=1600,
            camera_model="OPENCV", single_camera=True, matcher=None,
            max_steps=30000, data_factor=1, strategy="default"):
        # If the source is a video, treat the capture as sequential for COLMAP.
        seq = bool(is_video) if matcher is None else (matcher == "sequential")
        self.stage_ingest(source, is_video=is_video, fps=fps,
                          max_images=max_images,
                          blur_drop_fraction=blur_drop_fraction,
                          resize_max_dim=resize_max_dim)
        self.stage_colmap(camera_model=camera_model, single_camera=single_camera,
                          is_sequential=seq, matcher=matcher)
        self.stage_train(max_steps=max_steps, data_factor=data_factor,
                         strategy=strategy)
        final = self.stage_export()
        print("\n" + self.state.summary())
        return final
