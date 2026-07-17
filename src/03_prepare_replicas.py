#!/usr/bin/env python3
"""Extract the production starts listed in the tracked manifest."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_frames(path: Path, natoms: int) -> list[list[str]]:
    lines = path.read_text().splitlines()
    block = natoms + 2
    if len(lines) % block:
        raise ValueError(f"{path} does not contain complete XYZ frames")
    return [lines[index : index + block] for index in range(0, len(lines), block)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/production_starts/manifest.csv"),
    )
    parser.add_argument("--natoms", type=int, default=176)
    args = parser.parse_args()

    frames = read_frames(args.trajectory, args.natoms)
    with args.manifest.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        frame = int(row["source_frame"])
        output = Path(row["xyz"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(frames[frame]) + "\n")
        print(f"Wrote {output} from frame {frame}")


if __name__ == "__main__":
    main()
