"""
ckpt_to_ply.py — turn any gsplat checkpoint into a viewer-ready PLY.

Why this exists: gsplat's simple_trainer saves checkpoints during training. If
Colab disconnects at step 14000, you still have ckpt_14000.pt on Drive. This
script converts that into a standard INRIA-format .ply so an interrupted run is
never wasted — you get a (slightly lower quality) splat you can actually view.

The PLY column order matches what SuperSplat, PlayCanvas and the Unity viewer
expect: x y z, nx ny nz, f_dc_0..2, f_rest_0..44, opacity, scale_0..2, rot_0..3.
Scales are written as log and opacity as logit (the INRIA convention — viewers
apply exp()/sigmoid() themselves), which is exactly how gsplat stores them.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def _load_splats(ckpt_path: Path):
    import torch
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    # gsplat saves {"step": int, "splats": state_dict-or-ParameterDict, ...}
    splats = ckpt.get("splats", ckpt)
    if hasattr(splats, "state_dict"):
        splats = splats.state_dict()
    out = {}
    for k, v in splats.items():
        out[k] = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
    step = int(ckpt.get("step", -1))
    return out, step


def ckpt_to_ply(ckpt_path: str | Path, out_path: str | Path) -> Path:
    splats, step = _load_splats(Path(ckpt_path))

    means = splats["means"].astype(np.float32)              # (N, 3)
    scales = splats["scales"].astype(np.float32)            # (N, 3) log-space
    quats = splats["quats"].astype(np.float32)              # (N, 4)
    opac = splats["opacities"].astype(np.float32).reshape(-1, 1)  # (N,1) logit
    sh0 = splats["sh0"].astype(np.float32)                  # (N, 1, 3)
    shN = splats.get("shN")                                 # (N, K, 3) or None

    n = means.shape[0]
    # f_dc: (N,1,3) -> transpose to channel-major -> (N,3)
    f_dc = np.transpose(sh0, (0, 2, 1)).reshape(n, -1)      # (N, 3)
    # f_rest: always pad/truncate to 45 cols (degree-3) for viewer compatibility.
    if shN is not None and shN.size > 0:
        f_rest = np.transpose(shN, (0, 2, 1)).reshape(n, -1)  # (N, 3*K)
    else:
        f_rest = np.zeros((n, 0), dtype=np.float32)
    if f_rest.shape[1] < 45:
        f_rest = np.concatenate(
            [f_rest, np.zeros((n, 45 - f_rest.shape[1]), np.float32)], axis=1)
    else:
        f_rest = f_rest[:, :45]

    normals = np.zeros((n, 3), np.float32)

    # Assemble columns in the canonical order.
    cols = [means, normals, f_dc, f_rest, opac, scales, quats]
    names = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + [f"f_rest_{i}" for i in range(45)]
             + ["opacity"]
             + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)])
    data = np.concatenate(cols, axis=1).astype(np.float32)
    assert data.shape[1] == len(names), (data.shape, len(names))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_ply(out_path, names, data)
    print(f"Wrote {n:,} Gaussians (from step {step}) -> {out_path}")
    return out_path


def _write_ply(path: Path, names, data: np.ndarray) -> None:
    n = data.shape[0]
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}"]
    header += [f"property float {nm}" for nm in names]
    header += ["end_header"]
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(np.ascontiguousarray(data, dtype="<f4").tobytes())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="path to gsplat checkpoint .pt")
    ap.add_argument("-o", "--out", default="model.ply")
    args = ap.parse_args()
    ckpt_to_ply(args.ckpt, args.out)
