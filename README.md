# Arena GSplat

Turn **photos or a video** into a 3D Gaussian Splat, entirely on **Google Colab**,
with every stage saved to Google Drive so a disconnect never costs you more than
the current step.

```
input (images .zip OR video)
        │
        ▼
  [1] ingest      frames extracted, deblurred, resized      → images/
  [2] colmap      ONE clean SfM solve, auto-chosen matcher   → colmap/dense/
  [3] train       gsplat simple_trainer (reference impl.)    → gsplat_result/
  [4] export      final point cloud                          → outputs/<project>.ply
```

This is a clean rebuild of an earlier pipeline. Two deliberate design choices fix
the problems that produced unconvincing results before:

1. **No hand-written training loop.** Training calls gsplat's official
   `simple_trainer.py` — the reference implementation of densification, opacity
   reset, SH scheduling and optimiser handling. (Re-implementing those by hand is
   what caused the exploding-Gaussian and frozen-SH-colour bugs previously.)
2. **No COLMAP model merging.** One clean solve, with the matcher chosen for your
   data. Merging two reconstructions produced globally inconsistent camera poses,
   which no trainer can recover from.

## Resuming after a Colab disconnect

Every stage records completion in `state.json` on Drive. Re-open the notebook,
run the setup cells, and re-run — finished stages are skipped. Training also
checkpoints to Drive at quarter-step intervals; `export` will convert the latest
checkpoint to a viewable `.ply` even if training was interrupted.

To force a stage to re-run (e.g. you re-shot the scene):

```python
pipe.state.reset("colmap")
pipe.stage_colmap(...)
```

## Quick start (Colab)

1. Push this repo to your GitHub (see below).
2. Open `arena_gsplat_colab.ipynb` in Colab (`Runtime → Change runtime type → GPU`).
3. Edit the variables in Cell 1 (`GITHUB_USER`, `PROJECT`), then run top to bottom.

## Choosing settings

| Setting | When to change it |
|---|---|
| `camera_model` | `OPENCV` is safe for unknown cameras; `SIMPLE_RADIAL` for a clean phone lens. |
| `matcher` | Leave `null` to auto-select. Force `exhaustive` for small, hard scenes. |
| `fps` (video) | Higher = more frames = slower COLMAP. 2 fps is a good start. |
| `blur_drop_fraction` | Raise to `0.15`+ for handheld video with motion blur. |
| `max_steps` | `7000` quick test, `30000` production. |
| `data_factor` | Raise to `2`/`4` if you hit out-of-memory during training. |

## How the matcher is chosen

| Input | Matcher | Why |
|---|---|---|
| video / sequential | `sequential` + loop closure | consecutive frames overlap; loop closure rejoins revisited spots |
| ≤ 150 images | `exhaustive` | compares every pair — most accurate |
| > 150 images | `vocab_tree` | scales to thousands without O(n²) blow-up |

## Push this repo to GitHub

From the folder containing this README:

```bash
# 1. Create an empty repo on github.com first (no README/license), then:
git init
git add .
git commit -m "Initial commit: arena-gsplat pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/arena-gsplat.git
git push -u origin main
```

If `git push` asks for a password, use a **Personal Access Token** (GitHub →
Settings → Developer settings → Personal access tokens), not your account
password.

To host an input dataset in the same repo, commit a `.zip` of images or a video
file and use its **raw** URL in the notebook's Cell 3:

```
https://github.com/YOUR_USERNAME/arena-gsplat/raw/main/scene.zip
```

> Note: GitHub blocks single files over 100 MB. For large videos, host on Drive
> or Git LFS and point the notebook at that URL instead. `.gitignore` already
> excludes generated `.ply`/`.pt`/`images/` so you don't accidentally commit
> gigabytes of outputs.

## Repo layout

```
arena-gsplat/
├── arena_gsplat_colab.ipynb   # main notebook (one cell per stage)
├── config.yaml                # default settings
├── requirements.txt
├── scripts/
│   ├── pipeline.py            # orchestrator (resumable)
│   ├── state.py              # Drive-backed stage state
│   ├── ingest.py             # images/video → frames
│   ├── run_colmap.py         # one clean SfM solve + auto matcher
│   ├── train.py              # wrapper around gsplat simple_trainer
│   ├── ckpt_to_ply.py        # checkpoint → viewer-ready PLY
│   └── export.py             # collect final PLY
└── docs/
    └── troubleshooting.md
```

## Requirements

A Colab GPU runtime (free T4 works; Colab Pro gives longer sessions and faster
GPUs). Everything else is installed by the notebook.

## Credits

- 3D Gaussian Splatting — Kerbl et al. 2023 (arXiv:2308.04079)
- [gsplat](https://github.com/nerfstudio-project/gsplat) — CUDA rasterizer + reference trainer
- [COLMAP](https://colmap.github.io/) — Structure-from-Motion
