from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"WARNING: failed to import torch: {exc}", file=sys.stderr)
        return 0

    if not torch.cuda.is_available():
        return 0

    arch_list = torch.cuda.get_arch_list()
    has_sm100 = any(("sm_100" in a) or ("compute_100" in a) for a in arch_list)
    if has_sm100:
        return 0

    print("WARNING: PyTorch does not report sm_100 (B200) support.")
    print("Install a build with CUDA 12.8+ / sm_100 support, for example:")
    print(
        "  pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
