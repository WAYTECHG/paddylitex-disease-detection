from pathlib import Path
import shutil
import random
import re
import hashlib
import math
from collections import defaultdict, Counter

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\User\Documents\Thesis_Code")

# Earliest available public export
SRC_ROOT = PROJECT_ROOT / "rice" / "rice"

# Final cleaned dataset name
OUT_ROOT = PROJECT_ROOT / "datasets" / "rice_leaf_disease"
OUT_YAML = PROJECT_ROOT / "data" / "rice_leaf_disease.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

TRAIN_RATIO = 0.70
VAL_RATIO = 0.20
TEST_RATIO = 0.10

SEED = 42

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

# ============================================================
# BASIC HELPERS
# ============================================================

def get_original_id(filename: str) -> str:
    """
    Extract original image identity before Roboflow .rf. hash.

    Example:
    r6k_train_img_brown_32_JPG_jpg.rf.abc123.jpg
    becomes:
    r6k_img_brown_32_JPG_jpg
    """
    stem = Path(filename).stem

    if ".rf." in stem:
        stem = stem.split(".rf.")[0]

    # Remove old split words inside filename
    stem = re.sub(r"_(train|val|test)_", "_", stem, flags=re.IGNORECASE)

    return stem.lower()


def get_label_path(img_path: Path) -> Path:
    """
    Convert:
    .../images/train/xxx.jpg
    to:
    .../labels/train/xxx.txt
    """
    parts = list(img_path.parts)

    if "images" not in parts:
        raise ValueError(f"'images' not found in path: {img_path}")

    parts[parts.index("images")] = "labels"
    return Path(*parts).with_suffix(".txt")


def file_hash(path: Path) -> str:
    h = hashlib.md5()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)

    return h.hexdigest()


def make_unique_path(dst_path: Path) -> Path:
    """
    Avoid accidental overwrite if the same filename appears from different source folders.
    """
    if not dst_path.exists():
        return dst_path

    parent = dst_path.parent
    stem = dst_path.stem
    suffix = dst_path.suffix

    i = 1
    while True:
        new_path = parent / f"{stem}_{i}{suffix}"
        if not new_path.exists():
            return new_path
        i += 1


# ============================================================
# LABEL PROCESSING
# ============================================================

def convert_label_line_to_bbox(line: str):
    """
    Supports normal YOLO bbox:
    cls x_center y_center width height

    Also supports polygon format:
    cls x1 y1 x2 y2 x3 y3 ...
    and converts it to YOLO bbox.
    """
    parts = line.strip().split()

    if len(parts) < 5:
        return None

    try:
        cls = int(float(parts[0]))
        nums = [float(x) for x in parts[1:]]
    except Exception:
        return None

    # Already YOLO bbox format
    if len(nums) == 4:
        x, y, w, h = nums
        return cls, x, y, w, h

    # Polygon format
    if len(nums) >= 6 and len(nums) % 2 == 0:
        xs = nums[0::2]
        ys = nums[1::2]

        x_min = max(0.0, min(xs))
        y_min = max(0.0, min(ys))
        x_max = min(1.0, max(xs))
        y_max = min(1.0, max(ys))

        w = x_max - x_min
        h = y_max - y_min

        if w <= 0 or h <= 0:
            return None

        x = x_min + w / 2
        y = y_min + h / 2

        return cls, x, y, w, h

    return None


def write_clean_label(src_label: Path, dst_label: Path):
    """
    Convert/copy labels into clean YOLO bbox format.
    """
    dst_label.parent.mkdir(parents=True, exist_ok=True)

    if not src_label.exists():
        dst_label.write_text("", encoding="utf-8")
        return

    cleaned_lines = []

    for line in src_label.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue

        result = convert_label_line_to_bbox(line)

        if result is None:
            continue

        cls, x, y, w, h = result

        if cls < 0 or cls >= len(CLASS_NAMES):
            continue

        x = min(max(x, 0.0), 1.0)
        y = min(max(y, 0.0), 1.0)
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)

        if w <= 0 or h <= 0:
            continue

        cleaned_lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    dst_label.write_text(
        "\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""),
        encoding="utf-8"
    )


def get_classes_from_label(label_path: Path):
    """
    Read class IDs from label file.
    """
    classes = []

    if not label_path.exists():
        return classes

    for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()

        if not parts:
            continue

        try:
            cls = int(float(parts[0]))
        except Exception:
            continue

        if 0 <= cls < len(CLASS_NAMES):
            classes.append(cls)

    return classes


def get_group_dominant_class(img_paths):
    """
    Use the most common class in an original-image group for stratified splitting.
    """
    counter = Counter()

    for img_path in img_paths:
        label_path = get_label_path(img_path)
        counter.update(get_classes_from_label(label_path))

    if not counter:
        return -1

    return counter.most_common(1)[0][0]


def count_boxes(label_path: Path) -> int:
    if not label_path.exists():
        return 0

    lines = label_path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    return len([line for line in lines if line.strip()])


# ============================================================
# COLLECT SOURCE DATA
# ============================================================

def collect_source_images():
    """
    Collect all images from SRC_ROOT/images/*.
    The old source train/val/test folders are ignored as official split meaning.
    """
    image_root = SRC_ROOT / "images"

    if not image_root.exists():
        raise FileNotFoundError(f"Cannot find source image folder: {image_root}")

    source_folders = [p.name for p in image_root.iterdir() if p.is_dir()]
    print("Detected source image folders:", source_folders)

    images = []

    for folder in source_folders:
        img_dir = image_root / folder

        for img_path in img_dir.rglob("*"):
            if img_path.suffix.lower() in IMAGE_EXTS:
                images.append(img_path)

    return images


def build_groups(images):
    """
    Group Roboflow variants by original identity before .rf. hash.
    """
    groups = defaultdict(list)

    for img_path in images:
        group_id = get_original_id(img_path.name)
        groups[group_id].append(img_path)

    return groups


# ============================================================
# GROUPED-ALL SPLIT
# ============================================================

def split_groups_stratified_by_image_count(groups):
    """
    Grouped-all split.

    Goals:
    1. Same original group never crosses train/val/test.
    2. All augmented variants are retained inside the assigned split.
    3. Image-file ratio stays close to 70:20:10.
    4. Class distribution is approximately preserved using dominant group class.
    """
    random.seed(SEED)

    split_records = {
        "train": [],
        "val": [],
        "test": [],
    }

    split_image_counts = Counter()
    split_group_counts = Counter()

    class_to_records = defaultdict(list)

    for group_id, img_paths in groups.items():
        dominant_class = get_group_dominant_class(img_paths)

        record = {
            "group_id": group_id,
            "img_paths": sorted(img_paths, key=lambda p: p.name.lower()),
            "dominant_class": dominant_class,
            "group_size": len(img_paths),
        }

        class_to_records[dominant_class].append(record)

    for cls, records in class_to_records.items():
        random.shuffle(records)

        # Put larger groups first so file-count ratio becomes more balanced
        records.sort(key=lambda r: r["group_size"], reverse=True)

        total_class_images = sum(r["group_size"] for r in records)

        target = {
            "train": total_class_images * TRAIN_RATIO,
            "val": total_class_images * VAL_RATIO,
            "test": total_class_images * TEST_RATIO,
        }

        current = {
            "train": 0,
            "val": 0,
            "test": 0,
        }

        for record in records:
            remaining_need = {
                split: target[split] - current[split]
                for split in ["train", "val", "test"]
            }

            chosen_split = max(remaining_need, key=remaining_need.get)

            split_records[chosen_split].append(record)
            current[chosen_split] += record["group_size"]

            split_image_counts[chosen_split] += record["group_size"]
            split_group_counts[chosen_split] += 1

    return split_records, split_image_counts, split_group_counts


# ============================================================
# COPY DATASET
# ============================================================

def prepare_output_folders():
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)

    for split in ["train", "val", "test"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)


def copy_dataset(split_records):
    prepare_output_folders()

    for split, records in split_records.items():
        for record in records:
            for src_img in record["img_paths"]:
                src_label = get_label_path(src_img)

                dst_img = OUT_ROOT / "images" / split / src_img.name
                dst_img = make_unique_path(dst_img)

                dst_label = OUT_ROOT / "labels" / split / f"{dst_img.stem}.txt"

                shutil.copy2(src_img, dst_img)
                write_clean_label(src_label, dst_label)


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_exact_duplicate_images():
    """
    Remove exact duplicate image-content files across the output dataset.

    For each exact image duplicate group:
    - Keep one image.
    - Prefer image with more bounding boxes.
    - If tie, use alphabetical filename.
    - Remove the duplicate image and its label.

    This removes:
    - exact duplicate images
    - exact duplicate image+label pairs
    - same image with slightly different label files
    """
    image_hash_to_items = defaultdict(list)

    for split in ["train", "val", "test"]:
        img_dir = OUT_ROOT / "images" / split
        lbl_dir = OUT_ROOT / "labels" / split

        for img_path in img_dir.rglob("*"):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            img_hash = file_hash(img_path)

            image_hash_to_items[img_hash].append({
                "split": split,
                "img_path": img_path,
                "lbl_path": lbl_path,
                "box_count": count_boxes(lbl_path),
            })

    removed = []

    for img_hash, items in image_hash_to_items.items():
        if len(items) <= 1:
            continue

        # Prefer more annotated boxes, then stable alphabetical name
        items = sorted(
            items,
            key=lambda item: (
                -item["box_count"],
                item["img_path"].name.lower()
            )
        )

        keep = items[0]
        duplicates = items[1:]

        for dup in duplicates:
            img_path = dup["img_path"]
            lbl_path = dup["lbl_path"]

            if img_path.exists():
                img_path.unlink()

            if lbl_path.exists():
                lbl_path.unlink()

            removed.append((dup["split"], img_path.name, keep["img_path"].name))

    print(f"\nRemoved exact duplicate images: {len(removed)}")

    if removed:
        print("Examples removed:")
        for split, removed_name, kept_name in removed[:10]:
            print(f"  [{split}] removed {removed_name} | kept {kept_name}")


# ============================================================
# YAML
# ============================================================

def write_yaml():
    yaml_text = f"""path: {OUT_ROOT.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

nc: {len(CLASS_NAMES)}
names:
"""

    for i, name in enumerate(CLASS_NAMES):
        yaml_text += f"  {i}: {name}\n"

    OUT_YAML.write_text(yaml_text, encoding="utf-8")


# ============================================================
# CACHE CLEANUP
# ============================================================

def remove_yolo_cache_files():
    removed = 0

    for split in ["train", "val", "test"]:
        cache_path = OUT_ROOT / "labels" / f"{split}.cache"

        if cache_path.exists():
            cache_path.unlink()
            removed += 1

    print(f"\nRemoved YOLO cache files: {removed}")


