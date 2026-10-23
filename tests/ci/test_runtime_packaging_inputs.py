"""Runtime source stages must carry the package's declared distribution inputs."""

import shlex
import tomllib
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILES = sorted((ROOT / "runtime").glob("Dockerfile*"))


@pytest.mark.parametrize("dockerfile", DOCKERFILES, ids=lambda path: path.name)
def test_runtime_project_copies_include_declared_metadata_files(dockerfile):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    required = {project["readme"]["file"], "LICENSE", "THIRD_PARTY_NOTICES.md"}
    copied = {}
    stage = dockerfile.read_text().split("FROM scratch AS runtime-inputs", 1)[1]
    stage = stage.split("\nFROM ", 1)[0]
    for line in stage.splitlines():
        if not line.startswith("COPY "):
            continue
        *sources, destination = shlex.split(line)[1:]
        if not destination.startswith("/project/"):
            continue
        target = PurePosixPath(destination).relative_to("/project")
        for source in sources:
            if (ROOT / source).is_dir():
                for required_path in required:
                    relative = PurePosixPath(required_path)
                    if relative.is_relative_to(source):
                        copied[str(target / relative.relative_to(source))] = (
                            required_path
                        )
            else:
                output = (
                    target / Path(source).name if destination.endswith("/") else target
                )
                copied[str(output)] = source
    for path in required:
        assert copied.get(path) == path, f"{dockerfile.name} omits or relocates {path}"
