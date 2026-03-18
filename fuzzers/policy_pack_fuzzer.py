from __future__ import annotations

import sys

import atheris

with atheris.instrument_imports():
    from invarlock.fuzzing import exercise_policy_pack_bytes


def test_one_input(data: bytes) -> None:
    exercise_policy_pack_bytes(data)


def main() -> None:
    atheris.Setup(sys.argv, test_one_input)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
