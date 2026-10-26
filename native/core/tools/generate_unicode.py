"""Regenerate committed scoring tables using the reference Unicode release.

Build-time maintenance only. The Rust implementation never invokes Python.
Run with CPython 3.12 (Unicode 15.0.0).
"""
from pathlib import Path
import unicodedata as unicode

assert unicode.unidata_version == "15.0.0", unicode.unidata_version
TARGET = Path(__file__).resolve().parents[1] / "src/unicode_tables.rs"
lines = [
    "// Unicode 15.0.0 tables generated from CPython 3.12 unicodedata.",
    "// Regenerate using native/core/tools/generate_unicode.py.",
    'pub const UNICODE_VERSION: &str = "15.0.0";',
]
for name, transform in [("DECOMPOSITION", lambda c: unicode.normalize("NFKD", c)),
                        ("CASEFOLD", str.casefold)]:
    lines.append(f"pub const {name}: &[(u32, &[u32])] = &[")
    for codepoint in range(0x110000):
        if 0xD800 <= codepoint <= 0xDFFF or name == "DECOMPOSITION" and 0xAC00 <= codepoint <= 0xD7A3:
            continue
        char = chr(codepoint)
        transformed = transform(char)
        if transformed != char:
            lines.append(f'({codepoint},&[{",".join(str(ord(c)) for c in transformed)}]),')
    lines.append("];")
for name, function in [("COMBINING", unicode.combining), ("DECIMAL_DIGITS", lambda c: unicode.decimal(c, -1))]:
    lines.append(f"pub const {name}: &[(u32, u8)] = &[")
    for codepoint in range(0x110000):
        value = function(chr(codepoint))
        if value > 0 if name == "COMBINING" else value >= 0:
            lines.append(f"({codepoint},{value}),")
    lines.append("];")
lines.append("pub const COMPOSITION: &[((u32,u32),u32)] = &[")
compositions = []
for codepoint in range(0x110000):
    decomposition = unicode.decomposition(chr(codepoint)).split()
    if len(decomposition) == 2 and not decomposition[0].startswith("<"):
        pair = tuple(int(part, 16) for part in decomposition)
        if unicode.normalize("NFC", "".join(map(chr, pair))) == chr(codepoint):
            compositions.append((pair, codepoint))
for (a, b), codepoint in sorted(compositions):
    lines.append(f"(({a},{b}),{codepoint}),")
lines.append("];")
TARGET.write_text("\n".join(lines) + "\n")
