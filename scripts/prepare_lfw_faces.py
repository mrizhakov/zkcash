import argparse
import os
import random
import shutil
import tarfile
from pathlib import Path
from urllib.request import urlretrieve


DEFAULT_URL = "https://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare LFW dataset as identity folders for training."
    )
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="URL for lfw-deepfunneled.tgz",
    )
    parser.add_argument(
        "--archive-path",
        type=str,
        default="./datasets/raw/lfw-deepfunneled.tgz",
        help="Where to store the downloaded archive.",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        default="./datasets/raw",
        help="Directory where archive is extracted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./datasets/faces",
        help="Output directory with identity subfolders.",
    )
    parser.add_argument(
        "--min-images-per-identity",
        type=int,
        default=8,
        help="Keep only identities with at least this many images.",
    )
    parser.add_argument(
        "--max-identities",
        type=int,
        default=120,
        help="Maximum number of identities to keep (most frequent first).",
    )
    parser.add_argument(
        "--max-images-per-identity",
        type=int,
        default=20,
        help="Cap images per identity to keep training balanced and faster.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--clean-output",
        action="store_true",
        help="Delete output directory before writing prepared data.",
    )
    return parser.parse_args()


def download_archive(url: str, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if archive_path.exists() and archive_path.stat().st_size > 0:
        print(f"Archive already exists: {archive_path}")
        return
    print(f"Downloading {url}")
    urlretrieve(url, str(archive_path))
    print(f"Saved archive to {archive_path}")


def extract_archive(archive_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        root_name = tar.getmembers()[0].name.split("/")[0]
        root_dir = extract_dir / root_name
        if root_dir.exists() and any(root_dir.iterdir()):
            print(f"Archive already extracted: {root_dir}")
            return root_dir
        print(f"Extracting archive to {extract_dir}")
        tar.extractall(extract_dir)
        print("Extraction complete")
        return root_dir


def prepare_faces(
    source_root: Path,
    output_dir: Path,
    min_images_per_identity: int,
    max_identities: int,
    max_images_per_identity: int,
    seed: int,
    clean_output: bool,
) -> None:
    rng = random.Random(seed)

    identities = []
    for person_dir in sorted(source_root.iterdir()):
        if not person_dir.is_dir():
            continue
        images = sorted(
            p
            for p in person_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if len(images) >= min_images_per_identity:
            identities.append((person_dir.name, images))

    if not identities:
        raise RuntimeError(
            "No identities matched min-images criteria. Lower --min-images-per-identity."
        )

    identities.sort(key=lambda it: len(it[1]), reverse=True)
    selected = identities[: max(1, max_identities)]

    if clean_output and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    for person_name, images in selected:
        sampled = list(images)
        rng.shuffle(sampled)
        sampled = sampled[: max(1, max_images_per_identity)]

        person_out = output_dir / person_name
        person_out.mkdir(parents=True, exist_ok=True)

        for src in sampled:
            dst = person_out / src.name
            shutil.copy2(src, dst)
            total_images += 1

    print("Prepared dataset summary")
    print(f"Identities: {len(selected)}")
    print(f"Images: {total_images}")
    print(f"Output: {output_dir}")


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive_path)
    extract_dir = Path(args.extract_dir)
    output_dir = Path(args.output_dir)

    download_archive(args.url, archive_path)
    source_root = extract_archive(archive_path, extract_dir)
    if not source_root.exists():
        raise RuntimeError(f"Expected extracted directory not found: {source_root}")

    prepare_faces(
        source_root=source_root,
        output_dir=output_dir,
        min_images_per_identity=args.min_images_per_identity,
        max_identities=args.max_identities,
        max_images_per_identity=args.max_images_per_identity,
        seed=args.seed,
        clean_output=args.clean_output,
    )


if __name__ == "__main__":
    main()