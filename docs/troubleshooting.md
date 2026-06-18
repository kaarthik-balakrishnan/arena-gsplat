# Troubleshooting

## Few images register in COLMAP (e.g. 34 of 91)

This is the single most common reason for a bad splat, and it's almost always a
**capture** problem, not a settings one. COLMAP can only place a camera if it
shares enough features with images already in the model.

Fixes, in order of impact:

1. **More overlap.** Adjacent photos should share ~60–70% of their view. For a
   multi-room/chamber scene, shoot a continuous path *through* doorways, not
   isolated shots per room. Connections between areas are where reconstructions
   break.
2. **Force exhaustive matching** for small, hard scenes: set `matcher="exhaustive"`.
3. **Avoid blur.** For video, raise `blur_drop_fraction` to 0.15–0.3.
4. **Texture.** Blank walls/floors give no features. Mixed, textured surfaces
   register far better.

More training steps will **not** fix low registration. Re-shoot with more overlap.

## Suspiciously perfect reprojection error (~0.001 px)

A real COLMAP solve has mean reprojection error around 0.3–1.0 px. A near-zero
value usually means the model wasn't produced by a single clean bundle adjustment
(e.g. it was stitched/merged by hand). This pipeline does one clean solve
specifically to avoid that. If you import an external COLMAP model, sanity-check
its reprojection error before trusting the poses.

## `CUDA available: False`

`Runtime → Change runtime type → GPU`, then re-run the setup cells.

## Out of memory during training

- Set `data_factor=2` (or `4`) to train at half/quarter resolution.
- Use `strategy="mcmc"` and a capped Gaussian budget.
- Reduce input resolution via `resize_max_dim` in ingest.

## Training flag errors (`unrecognized arguments`)

gsplat's CLI flag names occasionally change between versions. Check the current
names with:

```bash
python /content/gsplat/examples/simple_trainer.py default --help
```

then adjust the flags in `scripts/train.py`.

## COLMAP feature extraction is very slow

The apt build of COLMAP may fall back to CPU SIFT. It still works but is slower.
GPU matching is enabled by default (`use_gpu=True`); if it errors, set it to
`False` to force CPU and at least complete the solve.

## The result looks like fog / giant blobs

Usually one of:

- **Low image registration** (see top of this file) — too few/poorly-posed cameras.
- **Too few training steps** — run the full 30,000, not the 7,000 quick test.
- **Bad camera model** — try `SIMPLE_RADIAL` vs `OPENCV` if the lens is unusual.

## My Drive is filling up

Generated checkpoints are large. After exporting the final PLY you can delete
`gsplat_result/` and `colmap/dense/` for finished projects; `state.json` plus
`outputs/` are enough to keep.
