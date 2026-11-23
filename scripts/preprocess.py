"""
Rice leaf disease dataset preprocessing (reconstructed reproducible pipeline).

Designed to match the Chapter 3 workflow:
  1. Collect and verify source images.
  2. Remove corrupted and exact-duplicate images BEFORE splitting.
  3. Convert polygon or YOLO box annotations to valid YOLO boxes.
  4. Validate class IDs, box coordinates, and annotation consistency.
  5. Group related Roboflow image variants by original image ID.
  6. Perform reproducible approximate dominant-class stratified group splitting.
  7. Check leakage, write images/labels, data.yaml, and audit reports.

Important: This is a reconstruction, not the recovered original preprocessing code.
It cannot guarantee the original historical training/validation/test membership.
It NEVER deletes or overwrites an existing output directory.

Usage (Python 3.10+, run from the repository root):
  python scripts/preprocess.py
  python scripts/preprocess.py --source /path/to/rice/rice --output /path/to/new_output

By default the raw Kaggle export is read from rice/rice/ (images/ and labels/ inside) and a
NEW folder datasets/rice_yolo_reconstructed_seed42/ is written. The experiments used the
subsets in datasets/rice_yolo_split/, which were prepared beforehand (thesis Section 4.2.2).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, UnidentifiedImageError

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10
RATIOS = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}
SPLITS = ("train", "val", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = [
    "Rice__BacterialLeafBlight",
    "Rice__BrownSpot",
    "Rice__Healthy",
    "Rice__Hispa",
    "Rice__LeafBlast",
    "Rice__LeafScald",
    "Rice__LeafSmut",
    "Rice__NarrowBrownLeafSpot",
    "Rice__NeckBlast",
]

# Historical thesis counts: reference values only, NEVER fabricated or forced.
HISTORICAL_RAW = 8665
HISTORICAL_CLEANED = 7957
HISTORICAL_SPLITS = {"train": 5567, "val": 1587, "test": 803}


@dataclass
class ImageRecord:
    path: Path
    label_path: Path
    original_id: str
    content_hash: str
    labels: tuple[str, ...]
    classes: tuple[int, ...]
    had_missing_label: bool
    discarded_label_lines: int
    clipped_boxes: int


def original_image_id(filename: str) -> str:
    """Normalize Roboflow IDs to group augmented copies of one original image."""
    stem = Path(filename).stem
    stem = stem.split(".rf.", 1)[0]
    stem = re.sub(r"_(?:train|val|test)_", "_", stem, flags=re.IGNORECASE)
    return stem.casefold()


def source_label_path(image_path: Path, source_root: Path) -> Path:
    """Mirror images/<any subfolders>/photo.jpg -> labels/<same>/photo.txt."""
    relative = image_path.relative_to(source_root / "images")
    return (source_root / "labels" / relative).with_suffix(".txt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_is_readable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            im.load()
            return im.width > 0 and im.height > 0
    except (OSError, ValueError, UnidentifiedImageError):
        return False


def clean_annotation_line(raw_line: str) -> tuple[str, int, bool] | None:
    """Parse YOLO bbox/polygon; clip actual box corners to [0,1]."""
    parts = raw_line.strip().split()
    if len(parts) < 5:
        return None
    try:
        cls_number = float(parts[0])
        if not math.isfinite(cls_number) or not cls_number.is_integer():
            return None
        cls = int(cls_number)
        values = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    if not 0 <= cls < len(CLASS_NAMES):
        return None
    if not all(math.isfinite(v) for v in values):
        return None

    if len(values) == 4:
        x, y, width, height = values
        if width <= 0 or height <= 0:
            return None
        xmin, xmax = x - width / 2, x + width / 2
        ymin, ymax = y - height / 2, y + height / 2
    elif len(values) >= 6 and len(values) % 2 == 0:
        xmin, xmax = min(values[::2]), max(values[::2])
        ymin, ymax = min(values[1::2]), max(values[1::2])
    else:
        return None

    old_corners = (xmin, ymin, xmax, ymax)
    xmin = min(1., max(0., xmin))
    xmax = min(1., max(0., xmax))
    ymin = min(1., max(0., ymin))
    ymax = min(1., max(0., ymax))
    if xmax - xmin <= 1e-8 or ymax - ymin <= 1e-8:
        return None
    clipped = any(abs(a - b) > 1e-10 for a, b in zip(old_corners, (xmin, ymin, xmax, ymax)))
    x, y = (xmin + xmax) / 2, (ymin + ymax) / 2
    w, h = xmax - xmin, ymax - ymin
    return f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}", cls, clipped


def read_and_clean_labels(label_path: Path) -> tuple[tuple[str, ...], tuple[int, ...], int, int, bool]:
    if not label_path.is_file():
        return (), (), 0, 0, True
    cleaned, classes = [], []
    rejected = clipped = 0
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeError:
        lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for raw_line in lines:
        if not raw_line.strip():
            continue
        result = clean_annotation_line(raw_line)
        if result is None:
            rejected += 1
            continue
        line, cls, was_clipped = result
        cleaned.append(line)
        classes.append(cls)
        clipped += int(was_clipped)
    return tuple(cleaned), tuple(classes), rejected, clipped, False


def collect_and_clean_images(source_root: Path):
    """Validate image bytes and labels, then deduplicate all source images."""
    image_root = source_root / "images"
    if not image_root.is_dir():
        raise FileNotFoundError(f"Image root not found: {image_root}")
    images = sorted(
        (p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.as_posix().casefold(),
    )
    if not images:
        raise ValueError(f"No supported images found under {image_root}")

    corrupt = []
    candidates = defaultdict(list)
    rejected_rows = clipped_boxes = missing_labels = 0
    for index, path in enumerate(images, 1):
        if not image_is_readable(path):
            corrupt.append(path.relative_to(source_root).as_posix())
            continue
        label = source_label_path(path, source_root)
        lines, classes, bad, clipped, missing = read_and_clean_labels(label)
        rejected_rows += bad
        clipped_boxes += clipped
        missing_labels += int(missing)
        record = ImageRecord(
            path=path,
            label_path=label,
            original_id=original_image_id(path.name),
            content_hash=sha256_file(path),
            labels=lines,
            classes=classes,
            had_missing_label=missing,
            discarded_label_lines=bad,
            clipped_boxes=clipped,
        )
        candidates[record.content_hash].append(record)
        if index % 1000 == 0:
            print(f"  Checked {index:,}/{len(images):,} source images")

    unique = []
    duplicate_examples = []
    duplicate_count = 0
    for content_hash in sorted(candidates):
        matches = sorted(
            candidates[content_hash],
            key=lambda r: (-len(r.labels), r.path.as_posix().casefold()),
        )
        unique.append(matches[0])
        for extra in matches[1:]:
            duplicate_count += 1
            if len(duplicate_examples) < 30:
                duplicate_examples.append({
                    "removed": extra.path.relative_to(source_root).as_posix(),
                    "retained": matches[0].path.relative_to(source_root).as_posix(),
                })
    unique.sort(key=lambda r: r.path.as_posix().casefold())
    if not unique:
        raise ValueError("No valid images remained after corruption and duplicate checks")
    report = {
        "source_images": len(images),
        "corrupt_images": len(corrupt),
        "corrupt_examples": corrupt[:30],
        "exact_duplicates_removed": duplicate_count,
        "duplicate_examples": duplicate_examples,
        "cleaned_images": len(unique),
        "images_missing_source_label": missing_labels,
        "invalid_annotation_rows_discarded": rejected_rows,
        "boxes_clipped_to_image_bounds": clipped_boxes,
        "note": "Images with missing labels are retained as background with an empty .txt file. Inspect missing-label cases.",
    }
    return unique, report


def group_by_original_image(records: list[ImageRecord]):
    groups = defaultdict(list)
    for record in records:
        groups[record.original_id].append(record)
    return groups


def dominant_class(records: list[ImageRecord]) -> int:
    counter = Counter(cls for record in records for cls in record.classes)
    if not counter:
        return -1
    # Deterministic class-ID tie break.
    return min(counter, key=lambda cls: (-counter[cls], cls))


def stratified_group_split(groups: dict[str, list[ImageRecord]]):
    """Allocate whole groups, matching approximate image-based class proportions."""
    rng = random.Random(SEED)
    by_class = defaultdict(list)
    for group_id, items in sorted(groups.items()):
        by_class[dominant_class(items)].append((group_id, items))

    allocated: dict[str, list[ImageRecord]] = {split: [] for split in SPLITS}
    image_counts = Counter()
    total_images = sum(len(items) for items in groups.values())
    global_targets = {split: total_images * ratio for split, ratio in RATIOS.items()}
    for cls in sorted(by_class):
        class_records = by_class[cls]
        rng.shuffle(class_records)
        class_records.sort(key=lambda g: len(g[1]), reverse=True)  # stable tie shuffle
        class_total = sum(len(items) for _, items in class_records)
        class_targets = {s: class_total * RATIOS[s] for s in SPLITS}
        class_allocated = Counter()
        for _, items in class_records:
            # Prioritize remaining image-count quota for this class, then globally.
            chosen = max(
                SPLITS,
                key=lambda s: (
                    class_targets[s] - class_allocated[s],
                    global_targets[s] - image_counts[s],
                ),
            )
            allocated[chosen].extend(items)
            class_allocated[chosen] += len(items)
            image_counts[chosen] += len(items)

    for split in SPLITS:
        allocated[split].sort(key=lambda r: r.path.as_posix().casefold())
    return allocated


def verify_split(split_records: dict[str, list[ImageRecord]]):
    seen_paths, seen_hashes = set(), set()
    group_owner, hash_owner = {}, {}
    class_rows, present_images = {}, {}
    for split in SPLITS:
        class_rows[split] = Counter()
        present_images[split] = Counter()
        for rec in split_records[split]:
            assert rec.path not in seen_paths, "An image was assigned to multiple splits"
            seen_paths.add(rec.path)
            if rec.original_id in group_owner and group_owner[rec.original_id] != split:
                raise RuntimeError(f"Group leakage: {rec.original_id}")
            group_owner[rec.original_id] = split
            if rec.content_hash in hash_owner and hash_owner[rec.content_hash] != split:
                raise RuntimeError(f"Exact-image leakage: {rec.path}")
            hash_owner[rec.content_hash] = split
            seen_hashes.add(rec.content_hash)
            class_rows[split].update(rec.classes)
            present_images[split].update(set(rec.classes))
            for line in rec.labels:
                parts = line.split()
                assert len(parts) == 5
                x, y, w, h = map(float, parts[1:])
                if not (-1e-7 <= x-w/2 and x+w/2 <= 1+1e-7 and
                        -1e-7 <= y-h/2 and y+h/2 <= 1+1e-7 and w > 0 and h > 0):
                    raise RuntimeError(f"Out-of-bounds box: {rec.path} {line}")

    total_count = sum(len(split_records[s]) for s in SPLITS)
    if total_count != len(seen_paths):
        raise RuntimeError("Count mismatch in assigned images")
    return {
        "group_leakage": 0,
        "exact_duplicate_leakage": 0,
        "images_total": total_count,
        "split_images": {s: len(split_records[s]) for s in SPLITS},
        "split_group_counts": {s: len(set(r.original_id for r in split_records[s])) for s in SPLITS},
        "object_annotation_counts": {s: sum(class_rows[s].values()) for s in SPLITS},
        "per_class_object_counts": {s: {str(i): class_rows[s][i] for i in range(len(CLASS_NAMES))} for s in SPLITS},
        "per_class_image_presence": {s: {str(i): present_images[s][i] for i in range(len(CLASS_NAMES))} for s in SPLITS},
    }


def unique_output_name(record: ImageRecord, used_names: set[str]) -> str:
    name = record.path.name
    if Path(name).stem.casefold() not in used_names:
        used_names.add(Path(name).stem.casefold())
        return name
    short_hash = hashlib.sha256(record.path.as_posix().encode("utf-8")).hexdigest()[:10]
    stem, ext = record.path.stem, record.path.suffix
    name = f"{stem}__src{short_hash}{ext}"
    n = 2
    while Path(name).stem.casefold() in used_names:
        name = f"{stem}__src{short_hash}_{n}{ext}"
        n += 1
    used_names.add(Path(name).stem.casefold())
    return name


def write_outputs(output_root: Path, source_root: Path, split_records, report):
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_root}\n"
            "Choose a NEW --output directory. Never overwrite your historical thesis dataset."
        )
    # Fail if the selected output is nested in source to avoid accidentally rescanning generated files.
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("Output directory must not be inside the source directory")
    output_root.mkdir(parents=True, exist_ok=False)
    manifest = []
    for split in SPLITS:
        img_dir = output_root / "images" / split
        lbl_dir = output_root / "labels" / split
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        used_names: set[str] = set()
        for record in split_records[split]:
            name = unique_output_name(record, used_names)
            destination_image = img_dir / name
            destination_label = lbl_dir / f"{Path(name).stem}.txt"
            shutil.copy2(record.path, destination_image)
            destination_label.write_text(
                "\n".join(record.labels) + ("\n" if record.labels else ""),
                encoding="utf-8",
            )
            manifest.append({
                "subset": split,
                "source_image": record.path.relative_to(source_root).as_posix(),
                "output_image": destination_image.relative_to(output_root).as_posix(),
                "group_id": record.original_id,
                "sha256": record.content_hash,
                "annotation_count": len(record.labels),
                "missing_source_label": int(record.had_missing_label),
            })

    # Use double-quoted YAML string for Windows paths (forward slashes).
    yaml_lines = [
        f"path: {json.dumps(output_root.resolve().as_posix())}",
        "train: images/train", "val: images/val", "test: images/test",
        f"nc: {len(CLASS_NAMES)}", "names:",
    ]
    yaml_lines += [f"  {i}: {json.dumps(name)}" for i, name in enumerate(CLASS_NAMES)]
    (output_root / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    with (output_root / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    with (output_root / "class_distribution.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class_name", "train_objects", "val_objects", "test_objects", "train_pct", "val_pct", "test_pct"])
        for cls, name in enumerate(CLASS_NAMES):
            amounts = [report["per_class_object_counts"][split][str(cls)] for split in SPLITS]
            n = sum(amounts)
            writer.writerow([cls, name, *amounts, *[f"{100 * x / n:.2f}" if n else "0" for x in amounts]])

    (output_root / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )


def main():
    default_project = Path(__file__).resolve().parent.parent  # repository root
    parser = argparse.ArgumentParser(description="Safe reconstructed rice leaf preprocessing, seed 42")
    parser.add_argument("--source", type=Path, default=default_project / "rice" / "rice",
                        help="Original export containing images/ and labels/ subfolders")
    parser.add_argument("--output", type=Path,
                        default=default_project / "datasets" / "rice_yolo_reconstructed_seed42",
                        help="NEW destination directory, must not exist already")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing directory: {output}")
    if source == output or source in output.parents:
        parser.error("Output directory must not be inside source directory")

    print("Dataset preprocessing reconstruction — seed =", SEED)
    print("Source:", source)
    print("Output:", output)
    print("\n[1/4] Scanning images, validating, removing corrupted/duplicate files...")
    records, cleaning = collect_and_clean_images(source)
    print(f"  Source images: {cleaning['source_images']:,}")
    print(f"  Corrupt: {cleaning['corrupt_images']:,}")
    print(f"  Exact duplicates removed: {cleaning['exact_duplicates_removed']:,}")
    print(f"  Retained: {cleaning['cleaned_images']:,}")
    print(f"  Missing labels: {cleaning['images_missing_source_label']:,}")
    print(f"  Invalid label rows: {cleaning['invalid_annotation_rows_discarded']:,}")
    print("\n[2/4] Grouping original IDs and stratifying grouped images...")
    groups = group_by_original_image(records)
    allocated = stratified_group_split(groups)
    stats = verify_split(allocated)
    if stats["images_total"] != len(records):
        raise RuntimeError("Not all cleaned images appear in the final split")
    for split in SPLITS:
        print(f"  {split}: {stats['split_images'][split]:,} images; {stats['split_group_counts'][split]:,} groups")

    report = {
        "seed": SEED,
        "ratios": RATIOS,
        "source_root": str(source),
        "output_root": str(output),
        "historical_reference_only": {
            "source_images": HISTORICAL_RAW,
            "cleaned_images": HISTORICAL_CLEANED,
            "split_counts": HISTORICAL_SPLITS,
        },
        "cleaning": cleaning,
        **stats,
        "all_classes_present_in_each_subset": all(
            stats["per_class_object_counts"][s][str(i)] > 0
            for s in SPLITS for i in range(len(CLASS_NAMES))
        ),
        "historical_counts_matched": (
            cleaning["source_images"] == HISTORICAL_RAW
            and cleaning["cleaned_images"] == HISTORICAL_CLEANED
            and stats["split_images"] == HISTORICAL_SPLITS
        ),
    }
    if not report["historical_counts_matched"]:
        print("\nNOTE: reconstructed counts/splits differ from historical thesis records.")
        print("This does not mean historical results are invalid; do not replace the original test set.")
    if not report["all_classes_present_in_each_subset"]:
        print("WARNING: at least one class is absent from a subset; inspect source class coverage.")
    print("\n[3/4] Writing NEW dataset and audit reports...")
    write_outputs(output, source, allocated, report)
    print("\n[4/4] Finished. Leakage checks: PASS for exact duplicates and original-ID groups")
    print("Created:", output)
    print("  data.yaml, preprocessing_report.json, split_manifest.csv, class_distribution.csv")
    print("NOTE: Historical equivalence must be verified separately using the old dataset files.")


if __name__ == "__main__":
    main()
