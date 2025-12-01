from __future__ import annotations

from pathlib import Path

from invarlock.cli.commands.run import _persist_ref_masks


class _Obj:
    def __init__(self, data):
        self.edit = data


def test_persist_ref_masks_from_object(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    payload = {
        "artifacts": {
            "mask_payload": {
                "keep": [0, 2, 4],
                "meta": {"note": "ok"},
            }
        }
    }
    obj = _Obj(payload)
    p = _persist_ref_masks(obj, run_dir)
    assert p is not None and p.exists()
