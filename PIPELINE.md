# Pipeline (stage by stage)

This explains what each stage does and what it leaves on disk, so the pipeline
stays interpretable and debuggable. Everything is written under
`DRIVE_ROOT/PROJECT/` on your Google Drive.

```
DRIVE_ROOT/PROJECT/
├── state.json            # which stages are done (resume lives here)
├── images/               # [1] final frames, 00001.jpg ...
├── colmap/
│   ├── database.db       # [2] features + matches
│   ├── sparse/0/         # [2] sparse reconstruction
│   └── dense/
│       ├── images/       # [2] undistorted images   ← trainer input
│       └── sparse/0/     # [2] undistorted poses     ← trainer input
├── gsplat_result/        # [3] checkpoints (.pt) + intermediate .ply
└── outputs/PROJECT.ply   # [4] final splat
```

## Stage 1 — ingest (`scripts/ingest.py`)

Takes images (folder/zip) or a video and produces a clean, sequentially-named
`images/` folder.

- **Video** → `ffmpeg` extracts frames at `fps`.
- **Blur filter** (optional) drops the blurriest frames by variance-of-Laplacian;
  motion-blurred frames hurt SfM.
- **Resize** caps the long edge (default 1600 px) — the 3DGS detail/speed sweet spot.

Sequential naming matters: COLMAP's sequential matcher (used for video) relies on
frames being in capture order.

## Stage 2 — colmap (`scripts/run_colmap.py`)

One clean Structure-from-Motion solve:

1. `feature_extractor` — SIFT features per image (GPU).
2. matching — auto-selected: `sequential` (video) / `exhaustive` (≤150) /
   `vocab_tree` (>150).
3. `mapper` — builds the sparse model. If COLMAP splits the scene into several
   sub-models, the one with the **most registered images** is kept.
4. `image_undistorter` — produces a PINHOLE dataset (`dense/images` +
   `dense/sparse/0`) that gsplat reads directly.

The stage prints how many images registered. **If far fewer than your total
register, that's a capture problem** (weak overlap), not a settings problem — see
`troubleshooting.md`.

## Stage 3 — train (`scripts/train.py`)

Invokes gsplat's `examples/simple_trainer.py` on the undistorted dataset. We pass
the data dir, result dir (on Drive), step budget, and frequent `--save-steps`.
No custom training code runs — the reference implementation handles all the
density-control and SH logic.

Checkpoints and PLYs land in `gsplat_result/` at quarter, half, three-quarter and
full step counts, so an interrupted run is recoverable.

## Stage 4 — export (`scripts/export.py`)

Copies the newest trained `.ply` to `outputs/PROJECT.ply`. If only a checkpoint
exists (training interrupted), it converts the latest `.pt` to a standard
62-property INRIA PLY via `ckpt_to_ply.py` — so you always get a viewable file.

## The resume mechanism (`scripts/state.py`)

`state.json` records `{done: true/false, ...}` per stage. The orchestrator skips
done stages. Writes are atomic (temp file + rename) so a disconnect mid-write
can't corrupt the file. `pipe.state.reset("<stage>")` forces a re-run.
