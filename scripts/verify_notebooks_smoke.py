#!/usr/bin/env python3
"""
Smoke-run Jupyter notebooks by extracting code cells into temp .py scripts.

This is a lightweight verifier intended for CI/dev sanity checks when full
notebook execution (via Jupyter kernels) isn't available.

What it does:
  - Reads `notebooks/*.ipynb`
  - Writes a runnable `.py` per notebook into a temp run directory
  - Converts Jupyter shell escapes (`!cmd`) into `bash -c ...` subprocess calls
  - Converts `%%bash` cells into `bash -c ...` subprocess calls
  - Runs each generated script in an isolated temp working directory

Defaults:
  - Skips `pip install ...` lines inside notebooks (assumes deps already present)
  - Sets a cautious env for HF/transformers + InvarLock demos
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _iter_code_cells(nb: dict) -> list[tuple[int, str]]:
    cells = nb.get("cells", [])
    out: list[tuple[int, str]] = []
    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        out.append((idx, src))
    return out


def _is_pip_install(cmd: str) -> bool:
    tokens = cmd.strip().split()
    if not tokens:
        return False
    if tokens[0] == "pip":
        return "install" in tokens[1:]
    if (
        len(tokens) >= 3
        and tokens[0] in {"python", "python3"}
        and tokens[1] == "-m"
        and tokens[2] == "pip"
    ):
        return "install" in tokens[3:]
    return False


def _convert_cell(
    *,
    cell_index: int,
    cell_source: str,
    notebook_name: str,
    skip_pip: bool,
) -> list[str]:
    lines = cell_source.splitlines()
    if not lines:
        return []

    first = lines[0].lstrip()
    if first.startswith("%%bash"):
        script = "\n".join(lines[1:]).rstrip() + "\n"
        return [
            f"print({(f'[{notebook_name}] cell {cell_index} (%%bash)').__repr__()})\n",
            f"_run_bash({script!r})\n",
            "\n",
        ]

    out: list[str] = [f"print({(f'[{notebook_name}] cell {cell_index}').__repr__()})\n"]
    for raw in lines:
        stripped = raw.lstrip()
        if stripped.startswith("!"):
            cmd = stripped[1:].strip()
            if skip_pip and _is_pip_install(cmd):
                out.append(f"print({(f'  (skip) {cmd}').__repr__()})\n")
                continue
            out.append(f"_run_bash({cmd!r})\n")
            continue
        out.append(raw + "\n")
    out.append("\n")
    return out


def write_script(*, nb_path: Path, out_py: Path, skip_pip: bool) -> None:
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code_cells = _iter_code_cells(nb)

    header = f"""\
#!/usr/bin/env python3
# Generated from: {nb_path}
# Generated at: {datetime.now(tz=UTC).isoformat()}

from __future__ import annotations

import os
import subprocess


def _run_bash(cmd: str) -> None:
    # Use bash so notebook-style commands (pipes, heredocs, exports) behave.
    # Avoid `bash -l` (login shell), which can reset PATH and break venv/conda
    # command resolution for `invarlock`, `python`, etc.
    subprocess.run(["bash", "-c", cmd], check=True, env=os.environ.copy())


def main() -> None:
"""

    body: list[str] = []
    notebook_name = nb_path.name
    for cell_index, cell_source in code_cells:
        body.extend(
            _convert_cell(
                cell_index=cell_index,
                cell_source=cell_source,
                notebook_name=notebook_name,
                skip_pip=skip_pip,
            )
        )

    footer = """\


if __name__ == "__main__":
    main()
"""

    # Indent body under main().
    indented_body = []
    for line in body:
        if line.strip():
            indented_body.append("    " + line)
        else:
            indented_body.append(line)

    out_py.write_text(header + "".join(indented_body) + footer, encoding="utf-8")


def _env_for_run() -> dict[str, str]:
    env = os.environ.copy()
    # Prefer local repo code when invoked from source checkout.
    env["PYTHONPATH"] = str(ROOT / "src") + (
        (os.pathsep + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""
    )
    env.setdefault("INVARLOCK_ALLOW_NETWORK", "1")
    env.setdefault("INVARLOCK_DEDUP_TEXTS", "1")
    env.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


def run_script(*, script_path: Path, cwd: Path, timeout_s: int) -> None:
    env = _env_for_run()
    stdout_path = cwd / "stdout.txt"
    stderr_path = cwd / "stderr.txt"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(cwd),
            env=env,
            stdout=out,
            stderr=err,
            timeout=timeout_s,
        )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Notebook smoke failed: {script_path.name} (exit={proc.returncode})\n"
            f"  cwd: {cwd}\n"
            f"  stdout: {stdout_path}\n"
            f"  stderr: {stderr_path}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebooks",
        nargs="*",
        help="Notebook paths (default: notebooks/*.ipynb)",
    )
    parser.add_argument(
        "--out-root",
        default="",
        help="Output root (default: /tmp/invarlock_notebook_smoke_<ts>)",
    )
    parser.add_argument(
        "--timeout-s",
        type=int,
        default=3600,
        help="Per-notebook timeout in seconds (default: 3600)",
    )
    parser.add_argument(
        "--run-pip",
        action="store_true",
        help="Do not skip `pip install ...` lines from notebooks.",
    )
    args = parser.parse_args(argv)

    ts = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else Path("/tmp") / f"invarlock_notebook_smoke_{ts}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    nb_paths = [Path(p) for p in args.notebooks] if args.notebooks else []
    if not nb_paths:
        nb_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    nb_paths = [p for p in nb_paths if p.exists()]
    if not nb_paths:
        raise SystemExit("No notebooks found.")

    print(f"Output root: {out_root}")
    print(f"Notebooks: {len(nb_paths)}")

    skip_pip = not bool(args.run_pip)
    for nb in nb_paths:
        run_dir = out_root / nb.stem
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = run_dir / f"{nb.stem}.py"
        write_script(nb_path=nb, out_py=script_path, skip_pip=skip_pip)
        print(f"Running: {nb.name}")
        run_script(script_path=script_path, cwd=run_dir, timeout_s=int(args.timeout_s))
        print(f"OK: {nb.name}")

    print("All notebook smoke runs passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
