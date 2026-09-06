"""Authenticate the whole upstream Expat release used by the native image."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path

VERSION = "2.8.4"
PACKAGE_VERSION = "2.8.4-0invarlock1"
PRIMARY_KEY = "3176EF7DB2367F1FCA4F306B1F9B0E909AF37285"
SIGNING_KEY = "CB8DE70A90CFBF6C3BF5CC5696262ACFFBD3AEC6"
INPUT_HASHES = {
    "expat-2.8.4.tar.xz": "656ae1cc8da3b4ea513bb4e254f33e6243938084c0ec6239da873376b09985a7",
    "expat-2.8.4.tar.xz.asc": "05bc77a3b59d3a02c135ea8b3c1997aa586115f62075fa2148fc66857a98120b",
    "hartwork.gpg": "efca77908aa6662eee79f2db08276aab57575f1a5ed2114200cf1914b8c5b9b2",
}


def read(path, limit=8 * 1024 * 1024):
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("Expat input must be a bounded regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
    finally:
        os.close(descriptor)
    if len(data) > limit:
        raise ValueError("Expat input exceeds size bound")
    return data


def valid_signer(status):
    records = re.findall(r"^\[GNUPG:\] VALIDSIG (.+)$", status.decode(), re.MULTILINE)
    if len(records) != 1:
        raise ValueError("Expat release requires exactly one valid signature")
    fields = records[0].split()
    if (
        len(fields) != 10
        or fields[0] != SIGNING_KEY
        or fields[-1] != PRIMARY_KEY
        or fields[7] not in {"8", "9", "10"}
        or re.search(
            r"^\[GNUPG:\] (?:BADSIG|ERRSIG|EXPSIG|EXPKEYSIG|REVKEYSIG|FAILURE)\b",
            status.decode(),
            re.MULTILINE,
        )
    ):
        raise ValueError("Expat release signer or signature status differs")


def authenticate(bundle):
    payloads = {name: read(bundle / name) for name in INPUT_HASHES}
    if any(
        hashlib.sha256(payloads[name]).hexdigest() != digest
        for name, digest in INPUT_HASHES.items()
    ):
        raise ValueError("Expat source, signature, or public key identity differs")
    with tempfile.TemporaryDirectory(prefix="k2-expat-signature-") as temporary:
        root = Path(temporary).resolve()
        for name, data in payloads.items():
            (root / name).write_bytes(data)
        command = [
            "gpg",
            "--no-options",
            "--homedir",
            str(root),
            "--batch",
            "--no-tty",
            "--no-autostart",
            "--no-auto-key-retrieve",
        ]
        subprocess.run(
            [*command, "--import", str(root / "hartwork.gpg")],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        checked = subprocess.run(
            [
                *command,
                "--status-fd",
                "1",
                "--verify",
                str(root / "expat-2.8.4.tar.xz.asc"),
                str(root / "expat-2.8.4.tar.xz"),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        if len(checked.stdout) > 65536 or len(checked.stderr) > 65536:
            raise ValueError("Expat signature output exceeds size bound")
        valid_signer(checked.stdout)
    return payloads


def prepared_inputs(bundle):
    import json

    payloads = authenticate(bundle)
    report = {
        "format": "invarlock/k2-expat-source-v1",
        "source_version": VERSION,
        "package_version": PACKAGE_VERSION,
        "input_sha256": INPUT_HASHES,
        "primary_key": PRIMARY_KEY,
        "signing_key": SIGNING_KEY,
    }
    payloads["source-authentication.json"] = (
        json.dumps(report, sort_keys=True, indent=2) + "\n"
    ).encode()
    return {"expat/" + name: data for name, data in payloads.items()}


LIBDIR = Path("usr/lib/x86_64-linux-gnu")
PACKAGES = ("libexpat1", "libexpat1-dev")
ABI_SOURCE = r"""
#include <expat.h>
#include <string.h>
static int elements, text_ok;
static void start(void *data, const XML_Char *name, const XML_Char **attrs) {
  const char *expected = "urn:test|r";
  unsigned i;
  (void)data; (void)attrs;
  for (i = 0; expected[i]; ++i) if (name[i] != expected[i]) return;
  if (name[i] == 0) ++elements;
}
static void text(void *data, const XML_Char *value, int length) {
  (void)data;
  if (length == 2 && value[0] == 'o' && value[1] == 'k') ++text_ok;
}
int main(void) {
  const XML_Feature *f;
  int char_ok = 0, lchar_ok = 0;
  XML_Parser parser;
  const char *xml = "<!DOCTYPE r [<!ENTITY x 'ok'>]><r xmlns='urn:test'>&x;</r>";
  if (strcmp(XML_ExpatVersion(), "expat_2.8.4")) return 1;
  for (f = XML_GetFeatureList(); f->feature; ++f) {
    if (f->feature == XML_FEATURE_SIZEOF_XML_CHAR) char_ok = f->value == sizeof(XML_Char);
    if (f->feature == XML_FEATURE_SIZEOF_XML_LCHAR) lchar_ok = f->value == 1;
  }
  if (!char_ok || !lchar_ok) return 2;
  parser = XML_ParserCreateNS(NULL, '|');
  if (!parser) return 3;
  XML_SetElementHandler(parser, start, NULL);
  XML_SetCharacterDataHandler(parser, text);
  if (XML_Parse(parser, xml, (int)strlen(xml), XML_TRUE) != XML_STATUS_OK) return 4;
  XML_ParserFree(parser);
  return elements == 1 && text_ok == 1 ? 0 : 5;
}
"""


def execute(command, *, cwd=None, env=None, timeout=300, log=None):
    if log is not None:
        with log.open("ab") as stream:
            subprocess.run(
                command,
                cwd=cwd,
                env=env,
                check=True,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=timeout,
            )
        return b""
    # Keep warning text out of parsed identity-query stdout.
    return subprocess.run(
        command, cwd=cwd, env=env, check=True, capture_output=True, timeout=timeout
    ).stdout


def file_inventory(root):
    """Record package payload bytes and links, excluding package control files."""
    result = {}
    for path in sorted(root.rglob("*")):
        name = str(path.relative_to(root))
        if name.startswith("DEBIAN/"):
            continue
        if path.is_symlink():
            target = os.readlink(path)
            if "/" in target or target in {".", ".."}:
                raise ValueError(
                    "Expat package link must stay in its library directory"
                )
            result[name] = {"symlink": target}
        elif path.is_dir():
            continue
        else:
            result[name] = {"sha256": hashlib.sha256(read(path)).hexdigest()}
    return result


def unpack_source(data, root):
    import io
    import tarfile

    with tarfile.open(fileobj=io.BytesIO(data), mode="r:xz") as archive:
        members = archive.getmembers()
        if len(members) > 1000 or sum(item.size for item in members) > 16 * 1024 * 1024:
            raise ValueError("Expat archive exceeds size or member bound")
        seen = set()
        for item in members:
            path = Path(item.name)
            if (
                (
                    not item.name.startswith("expat-2.8.4/")
                    and not (item.name.rstrip("/") == "expat-2.8.4" and item.isdir())
                )
                or path.is_absolute()
                or ".." in path.parts
                or str(path) != item.name.rstrip("/")
                or item.name in seen
                or not (item.isdir() or item.isfile())
            ):
                raise ValueError("Expat archive contains an unsupported member")
            seen.add(item.name)
            destination = root / path
            if item.isdir():
                destination.mkdir(parents=True, exist_ok=True)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(archive.extractfile(item).read())
                destination.chmod(0o755 if item.mode & 0o111 else 0o644)
    return root / "expat-2.8.4"


def validate_package_identity(fields, name):
    expected = {
        "Package": name,
        "Version": PACKAGE_VERSION,
        "Architecture": "amd64",
        "Source": f"expat ({PACKAGE_VERSION})",
    }
    if any(fields.get(key) != value for key, value in expected.items()):
        raise ValueError("derived Expat package identity differs")
    if (
        name == "libexpat1-dev"
        and fields.get("Depends") != f"libexpat1 (= {PACKAGE_VERSION})"
    ):
        raise ValueError(
            "Expat development package must bind the exact runtime version"
        )


def package_metadata(path):
    data = execute(["dpkg-deb", "--field", str(path)]).decode()
    return dict(
        line.split(": ", 1)
        for line in data.splitlines()
        if ": " in line and not line.startswith(" ")
    )


def symbols(path):
    data = execute(
        ["nm", "--dynamic", "--defined-only", "--format=posix", str(path)]
    ).decode()
    return {line.split()[0] for line in data.splitlines() if line.startswith("XML_")}


def check_abi(source, library, output, *, wide):
    command = ["cc", "-Wall", "-Werror", "-I", str(source / "lib")]
    if wide:
        command.append("-DXML_UNICODE")
    command += [str(output.parent / "abi.c"), str(library), "-lm", "-o", str(output)]
    if ".so" in library.name:
        command += ["-Wl,-rpath," + str(library.parent)]
    execute(command)
    execute([str(output)])


def build_install(bundle, output):
    import json
    import platform
    import shutil

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Expat packages require Linux x86_64")
    payloads = authenticate(bundle)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    env = {**os.environ, "SOURCE_DATE_EPOCH": "1788182984"}
    with tempfile.TemporaryDirectory(prefix="k2-expat-build-") as temporary:
        work = Path(temporary).resolve()
        source = unpack_source(payloads["expat-2.8.4.tar.xz"], work)
        (work / "abi.c").write_text(ABI_SOURCE)
        runtime, development = work / "libexpat1", work / "libexpat1-dev"
        stages = {}
        old_symbols = {
            stem: symbols(Path("/") / LIBDIR / (stem + ".so.1"))
            for stem in ("libexpat", "libexpatw")
        }
        for wide in (False, True):
            stem = "libexpatw" if wide else "libexpat"
            for shared in (True, False):
                name = stem + ("-shared" if shared else "-static")
                build, stage = work / name, work / (name + "-stage")
                options = [
                    "cmake",
                    "-S",
                    str(source),
                    "-B",
                    str(build),
                    "-DCMAKE_BUILD_TYPE=Release",
                    "-DCMAKE_C_FLAGS_RELEASE=-O2 -DNDEBUG -fstack-protector-strong -Wformat -Werror=format-security -D_FORTIFY_SOURCE=3",
                    "-DCMAKE_SHARED_LINKER_FLAGS=-Wl,-z,relro,-z,now",
                    "-DCMAKE_INSTALL_PREFIX=/usr",
                    "-DCMAKE_INSTALL_LIBDIR=lib/x86_64-linux-gnu",
                    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                    "-DEXPAT_BUILD_TOOLS=OFF",
                    "-DEXPAT_BUILD_EXAMPLES=OFF",
                    "-DEXPAT_BUILD_DOCS=OFF",
                    "-DEXPAT_BUILD_PKGCONFIG=ON",
                    "-DEXPAT_SYMBOL_VERSIONING=OFF",
                    "-DEXPAT_DTD=ON",
                    "-DEXPAT_GE=ON",
                    "-DEXPAT_NS=ON",
                    "-DEXPAT_CHAR_TYPE=" + ("ushort" if wide else "char"),
                    "-DEXPAT_SHARED_LIBS=" + ("ON" if shared else "OFF"),
                    "-DEXPAT_BUILD_TESTS=" + ("OFF" if wide else "ON"),
                ]
                log = output / (name + ".log")
                execute(options, env=env, log=log)
                execute(
                    ["cmake", "--build", str(build), "--parallel", "2"],
                    env=env,
                    timeout=900,
                    log=log,
                )
                if not wide:
                    execute(
                        ["ctest", "--test-dir", str(build), "--output-on-failure"],
                        env=env,
                        timeout=900,
                        log=log,
                    )
                execute(
                    ["cmake", "--install", str(build)],
                    env={**env, "DESTDIR": str(stage)},
                    log=log,
                )
                library = stage / LIBDIR / (stem + (".so.1.12.4" if shared else ".a"))
                check_abi(source, library, work / (name + "-abi"), wide=wide)
                if shared:
                    dynamic = execute(["readelf", "--dynamic", str(library)]).decode()
                    if f"[{stem}.so.1]" not in dynamic or not old_symbols[
                        stem
                    ] <= symbols(library):
                        raise ValueError(
                            "Expat shared library ABI or exported symbols differ"
                        )
                stages[name] = stage
        # Preserve the narrow CMake/pkg-config export, public headers, and both
        # library widths; wide exports cannot overwrite the narrow target metadata.
        shutil.copytree(
            stages["libexpat-shared"] / "usr", development / "usr", symlinks=True
        )
        # The minimal base excludes ordinary /usr/share/doc payloads. Retain the
        # explicitly owned copyright files below, rather than recording filtered
        # upstream AUTHORS/README files as installed package content.
        docs = development / "usr/share/doc"
        if docs.exists():
            shutil.rmtree(docs)
        for stem in ("libexpat", "libexpatw"):
            shared = stages[stem + "-shared"] / LIBDIR
            (runtime / LIBDIR).mkdir(parents=True, exist_ok=True)
            (development / LIBDIR).mkdir(parents=True, exist_ok=True)
            for suffix in (".so.1.12.4", ".so.1"):
                target = runtime / LIBDIR / (stem + suffix)
                shutil.copy2(shared / (stem + suffix), target, follow_symlinks=False)
                copied = development / LIBDIR / (stem + suffix)
                if copied.exists() or copied.is_symlink():
                    copied.unlink()
            link = development / LIBDIR / (stem + ".so")
            if not link.is_symlink():
                link.symlink_to(stem + ".so.1")
            shutil.copy2(
                stages[stem + "-static"] / LIBDIR / (stem + ".a"),
                development / LIBDIR / (stem + ".a"),
            )
        config = development / "usr/include/x86_64-linux-gnu/expat_config.h"
        config.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work / "libexpat-shared/expat_config.h", config)
        # Ask Debian tooling for the real ELF dependency floor of both libraries.
        (work / "debian").mkdir()
        (work / "debian/control").write_text(
            "Source: expat\nSection: libs\nPriority: optional\nMaintainer: InvarLock maintainers\nStandards-Version: 4.6.2\n\nPackage: libexpat1\nArchitecture: amd64\nDescription: Locally built Expat\n"
        )
        dependencies = (
            execute(
                [
                    "dpkg-shlibdeps",
                    "-O",
                    *[
                        "-e" + str(runtime / LIBDIR / (stem + ".so.1.12.4"))
                        for stem in ("libexpat", "libexpatw")
                    ],
                ],
                cwd=work,
            )
            .decode()
            .strip()
        )
        if not dependencies.startswith("shlibs:Depends=") or "\n" in dependencies:
            raise ValueError("Expat ELF dependency metadata is ambiguous")
        for name, tree in zip(PACKAGES, (runtime, development), strict=True):
            control = tree / "DEBIAN"
            control.mkdir()
            depends = (
                dependencies.removeprefix("shlibs:Depends=")
                if name == "libexpat1"
                else f"libexpat1 (= {PACKAGE_VERSION})"
            )
            (control / "control").write_text(
                f"Package: {name}\nSource: expat ({PACKAGE_VERSION})\nVersion: {PACKAGE_VERSION}\nArchitecture: amd64\nMulti-Arch: same\nMaintainer: InvarLock maintainers\nDepends: {depends}\nDescription: Locally built whole-release Expat {VERSION}\n"
            )
            if name == "libexpat1":
                (control / "shlibs").write_text(
                    "\n".join(
                        f"{stem} 1 libexpat1 (>= {PACKAGE_VERSION})"
                        for stem in ("libexpat", "libexpatw")
                    )
                    + "\n"
                )
                (control / "triggers").write_text("activate-noawait ldconfig\n")
            doc = tree / "usr/share/doc" / name
            doc.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / "COPYING", doc / "copyright")
            artifact = output / f"{name}_{PACKAGE_VERSION}_amd64.deb"
            execute(
                ["dpkg-deb", "--root-owner-group", "--build", str(tree), str(artifact)],
                env=env,
            )
            validate_package_identity(package_metadata(artifact), name)
        expected_files = {**file_inventory(runtime), **file_inventory(development)}
        artifacts = {
            path.name: hashlib.sha256(read(path)).hexdigest()
            for path in output.glob("*.deb")
        }
        report = {
            "format": "invarlock/k2-expat-build-v1",
            "source_version": VERSION,
            "package_version": PACKAGE_VERSION,
            "input_sha256": INPUT_HASHES,
            "recipe_sha256": hashlib.sha256(read(Path(__file__))).hexdigest(),
            "package_sha256": artifacts,
            "installed_files": expected_files,
            "native_parser_checks": [
                "narrow-shared",
                "narrow-static",
                "wide-shared",
                "wide-static",
            ],
            "upstream_tests": ["narrow-shared", "narrow-static"],
            "wide_upstream_tests": "Unsupported by upstream ushort configuration; independent parser and ABI checks passed.",
        }
        (output / "build-report.json").write_text(
            json.dumps(report, sort_keys=True, indent=2) + "\n"
        )
        execute(
            ["dpkg", "--install", *[str(output / name) for name in sorted(artifacts)]]
        )
        execute(["ldconfig"])
    return verify_installed(output)


def verify_installed(output, root=Path("/")):
    import ctypes
    import json
    import sys

    report = json.loads(read(output / "build-report.json"))
    if (
        report.get("format") != "invarlock/k2-expat-build-v1"
        or report.get("source_version") != VERSION
        or report.get("package_version") != PACKAGE_VERSION
        or report.get("input_sha256") != INPUT_HASHES
        or report.get("recipe_sha256")
        != hashlib.sha256(read(Path(__file__))).hexdigest()
    ):
        raise ValueError("Expat build report identity differs")
    expected_names = {f"{name}_{PACKAGE_VERSION}_amd64.deb" for name in PACKAGES}
    if set(report["package_sha256"]) != expected_names:
        raise ValueError("Expat package artifact set differs")
    for name, digest in report["package_sha256"].items():
        artifact = output / name
        if hashlib.sha256(read(artifact)).hexdigest() != digest:
            raise ValueError("Expat package artifact bytes differ")
        package = name.split("_", 1)[0]
        validate_package_identity(package_metadata(artifact), package)
        fields = execute(
            [
                "dpkg-query",
                "-W",
                "-f=${Version} ${Architecture} ${db:Status-Status}",
                package,
            ]
        ).decode()
        if fields != PACKAGE_VERSION + " amd64 installed":
            raise ValueError("installed Expat package identity differs")
    with tempfile.TemporaryDirectory(prefix="k2-expat-payload-") as temporary:
        extracted_files = {}
        for name in sorted(expected_names):
            extracted = Path(temporary).resolve() / name
            execute(["dpkg-deb", "--extract", str(output / name), str(extracted)])
            files = file_inventory(extracted)
            if extracted_files.keys() & files.keys():
                raise ValueError("Expat package payloads overlap")
            extracted_files.update(files)
        required = {
            str(LIBDIR / (stem + suffix))
            for stem in ("libexpat", "libexpatw")
            for suffix in (".so", ".so.1", ".so.1.12.4", ".a")
        }
        if (
            extracted_files != report["installed_files"]
            or not required <= extracted_files.keys()
        ):
            raise ValueError("Expat package payload differs from recorded files")
    for name, expected in report["installed_files"].items():
        path = root / name
        if (
            not name.startswith("usr/")
            or ".." in Path(name).parts
            or (
                "symlink" in expected
                and (not path.is_symlink() or os.readlink(path) != expected["symlink"])
            )
            or (
                "sha256" in expected
                and hashlib.sha256(read(path)).hexdigest() != expected["sha256"]
            )
        ):
            raise ValueError("installed Expat file differs")
    library_files = {p.name for p in (root / LIBDIR).glob("libexpat*.so.*")}
    if library_files != {
        stem + suffix
        for stem in ("libexpat", "libexpatw")
        for suffix in (".so.1", ".so.1.12.4")
    }:
        raise ValueError("unexpected older Expat shared library remains")
    for stem in ("libexpat", "libexpatw"):
        library = ctypes.CDLL(str(root / LIBDIR / (stem + ".so.1")))
        library.XML_ExpatVersion.restype = ctypes.c_char_p
        if library.XML_ExpatVersion() != b"expat_2.8.4":
            raise ValueError("loaded Expat library version differs")
    pyexpat = (
        execute([sys.executable, "-c", "import pyexpat; print(pyexpat.EXPAT_VERSION)"])
        .decode()
        .strip()
    )
    if pyexpat != "expat_2.8.4":
        raise ValueError("Python loaded Expat version differs")
    return {
        "source_version": VERSION,
        "package_version": PACKAGE_VERSION,
        "build_report_sha256": hashlib.sha256(
            read(output / "build-report.json")
        ).hexdigest(),
        "package_sha256": report["package_sha256"],
        "pyexpat_version": pyexpat,
        "installed_file_count": len(report["installed_files"]),
    }


def main(argv=None):
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-install", "verify"))
    parser.add_argument(
        "--bundle", type=Path, default=Path("/usr/share/invarlock-k2/expat")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("/usr/share/invarlock-k2/expat-built")
    )
    args = parser.parse_args(argv)
    result = (
        build_install(args.bundle, args.output)
        if args.command == "build-install"
        else verify_installed(args.output)
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
