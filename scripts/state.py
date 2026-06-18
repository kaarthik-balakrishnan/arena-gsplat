"""
state.py — Drive-backed pipeline state.

The whole point of this file: every stage records what it finished and where it
put its outputs, in a single JSON file that lives on Google Drive. When Colab
disconnects (it will), you re-run the notebook and each stage looks here first.
If its output already exists, it is skipped. Nothing is recomputed needlessly.

State file layout (project_dir/state.json):
{
  "project":   "my_scene",
  "created":   "2026-06-18T10:00:00",
  "updated":   "2026-06-18T10:42:00",
  "stages": {
     "ingest":  {"done": true, "n_images": 248, "images_dir": ".../images"},
     "colmap":  {"done": true, "matcher": "vocab_tree", "registered": 240, ...},
     "train":   {"done": false, "last_step": 14000, "ckpt": ".../ckpt_14000.pt"},
     "export":  {"done": false}
  }
}
"""
from __future__ import annotations
import json
import datetime
from pathlib import Path
from typing import Any, Dict, Optional

STAGES = ["ingest", "colmap", "train", "export"]


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class State:
    def __init__(self, project_dir: str | Path, project_name: str = "scene"):
        self.project_dir = Path(project_dir)
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.project_dir / "state.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text())
        else:
            self.data = {
                "project": project_name,
                "created": _now(),
                "updated": _now(),
                "stages": {s: {"done": False} for s in STAGES},
            }
            self._flush()

    # ---- persistence -------------------------------------------------
    def _flush(self) -> None:
        self.data["updated"] = _now()
        # Atomic-ish write so a disconnect mid-write can't corrupt the file.
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        tmp.replace(self.path)

    # ---- queries -----------------------------------------------------
    def is_done(self, stage: str) -> bool:
        return bool(self.data["stages"].get(stage, {}).get("done", False))

    def get(self, stage: str, key: str, default: Any = None) -> Any:
        return self.data["stages"].get(stage, {}).get(key, default)

    # ---- mutations ---------------------------------------------------
    def update(self, stage: str, **fields: Any) -> None:
        """Merge fields into a stage record without marking it done."""
        self.data["stages"].setdefault(stage, {})
        self.data["stages"][stage].update(fields)
        self._flush()

    def mark_done(self, stage: str, **fields: Any) -> None:
        self.update(stage, done=True, **fields)

    def reset(self, stage: str) -> None:
        """Force a stage to re-run next time (e.g. you re-shot the scene)."""
        self.data["stages"][stage] = {"done": False}
        self._flush()

    def summary(self) -> str:
        lines = [f"Project: {self.data['project']}  (updated {self.data['updated']})"]
        for s in STAGES:
            rec = self.data["stages"].get(s, {})
            tick = "DONE" if rec.get("done") else "  - "
            extra = {k: v for k, v in rec.items() if k != "done"}
            lines.append(f"  [{tick}] {s:8s} {extra if extra else ''}")
        return "\n".join(lines)
