"""Create a ZIP archive for Part 2 deliverables."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


FILES = [
    "utils_seed.py",
    "data_synthetic.py",
    "data_mnist01.py",
    "model_shallow.py",
    "losses.py",
    "metrics_part2.py",
    "optim_gd.py",
    "optim_sgd.py",
    "optim_kfac.py",
    "run_all_part2.py",
    "README.md",
]


def make_zip(name: str, student_id: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / f"{name}-{student_id}.zip"
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for filename in FILES:
            path = Path(filename)
            if not path.exists():
                raise FileNotFoundError(f"Missing required file: {filename}")
            zf.write(path, arcname=path.name)
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Package Part 2 files into a ZIP archive.")
    parser.add_argument("--name", required=True, help="Your name for the zip filename")
    parser.add_argument("--id", required=True, help="Your ID for the zip filename")
    parser.add_argument("--out_dir", default=".", help="Output directory for the ZIP")
    args = parser.parse_args()

    make_zip(args.name, args.id, Path(args.out_dir))


if __name__ == "__main__":
    main()
