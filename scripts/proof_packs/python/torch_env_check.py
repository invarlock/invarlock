from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"ERROR: failed to import torch: {exc}", file=sys.stderr)
        return 1

    print("torch", torch.__version__)
    if not torch.cuda.is_available():
        print("CUDA not available in torch", file=sys.stderr)
        return 1

    print("cuda", torch.version.cuda)
    print("gpus", torch.cuda.device_count())
    print("gpu0", torch.cuda.get_device_name(0))
    print("cc0", torch.cuda.get_device_capability(0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
