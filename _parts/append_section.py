"""Idempotent append helper for the frontend design report.

Usage:
    python _parts/append_section.py <target.md> <section_part.md>

Appends section_part.md to target.md only if the part's first non-empty
heading line is NOT already present in the target. Safe to re-run.
"""
import io
import sys


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python _parts/append_section.py <target.md> <section_part.md>")
        sys.exit(2)
    target, part = sys.argv[1], sys.argv[2]
    with io.open(part, "r", encoding="utf-8") as f:
        content = f.read()
    # find the first heading line ("#"/"##"/"###") in the part
    marker = None
    for line in content.splitlines():
        s = line.strip()
        if s.startswith("#") and not s.startswith("## ") and len(s) > 3:
            marker = s
            break
    with io.open(target, "r", encoding="utf-8") as f:
        target_text = f.read()
    if marker and marker in target_text:
        print(f"SKIP (already present): {marker} in {target}")
        return
    with io.open(target, "a", encoding="utf-8") as f:
        f.write(content)
        f.write("\n")
    with io.open(target, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"APPENDED {part} -> {target}; target now {len(lines)} lines")


if __name__ == "__main__":
    main()
