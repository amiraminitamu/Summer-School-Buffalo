#!/usr/bin/env python3
from pathlib import Path

trajectory = Path("output_aimd_100fs/aimd.md.xyz")
outdir = Path("test_snapshots")
outdir.mkdir(exist_ok=True)

natoms = 176
block_size = natoms + 2

lines = trajectory.read_text().splitlines()

if len(lines) % block_size != 0:
    raise ValueError(
        f"Trajectory has {len(lines)} lines, which is not divisible "
        f"by the expected frame size {block_size}."
    )

nframes = len(lines) // block_size
selected = [0, nframes // 2, nframes - 1]

print(f"Found {nframes} frames")

for frame in selected:
    start = frame * block_size
    block = lines[start:start + block_size]

    time_fs = frame * 0.5
    filename = outdir / f"snapshot_{frame:03d}_t{time_fs:05.1f}fs.xyz"

    filename.write_text("\n".join(block) + "\n")
    print(f"Wrote {filename}")
