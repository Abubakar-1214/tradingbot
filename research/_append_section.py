"""Append a part file's content into the report before the APPEND marker."""
import io
import sys

TARGET = r"e:\Desktop\NeoMind\Bazz\autonoumuse_trader\research\FINAL_AUTONOMOUS_TRADING_AI_STUDY.md"
MARKER = "<!-- APPEND-HERE -->"


def main(part_path: str) -> None:
    with io.open(TARGET, "r", encoding="utf-8") as f:
        target = f.read()
    with io.open(part_path, "r", encoding="utf-8") as f:
        part = f.read()
    if MARKER in target:
        target = target.replace(MARKER, part.rstrip() + "\n\n" + MARKER, 1)
    else:
        target = target.rstrip() + "\n\n" + part.rstrip() + "\n"
    with io.open(TARGET, "w", encoding="utf-8", newline="\n") as f:
        f.write(target)
    print(f"appended {part_path}: {len(part)} chars -> target {len(target)} chars")


if __name__ == "__main__":
    main(sys.argv[1])
