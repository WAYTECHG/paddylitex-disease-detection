from pathlib import Path
from collections import Counter, defaultdict

root = Path("datasets/rice_leaf_disease")
splits = ["train", "val", "test"]

names = {
    0: "BacterialLeafBlight",
    1: "BrownSpot",
    2: "Healthy",
    3: "Hispa",
    4: "LeafBlast",
    5: "LeafScald",
    6: "LeafSmut",
    7: "NarrowBrownLeafSpot",
    8: "NeckBlast",
}

def filename_class(fname):
    fname = fname.lower()

    # Check specific names before generic Blast
    if "neckblast" in fname or "neck_blast" in fname or "neck-blast" in fname:
        return 8
    if "narrowbrownleafspot" in fname or "narrow_brown" in fname:
        return 7
    if "bacterialleafblight" in fname or "bacterial_leaf_blight" in fname:
        return 0
    if "brownspot" in fname or "brown_spot" in fname:
        return 1
    if "healthy" in fname:
        return 2
    if "hispa" in fname:
        return 3
    if "leafscald" in fname or "leaf_scald" in fname:
        return 5
    if "leafsmut" in fname or "leaf_smut" in fname:
        return 6
    if "leafblast" in fname or "leaf_blast" in fname or "blast" in fname:
        return 4

    return None

suspicious = []
pair_counter = Counter()
split_counter = Counter()
label_id_counter = Counter()

for split in splits:
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split

    for img_path in img_dir.glob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        lines = lbl_path.read_text().strip().splitlines()
        if not lines:
            continue

        label_classes = {int(line.split()[0]) for line in lines if line.strip()}
        fcls = filename_class(img_path.name)

        for lcls in label_classes:
            label_id_counter[(split, lcls)] += 1

        if fcls is not None and fcls not in label_classes:
            for lcls in label_classes:
                suspicious.append((split, img_path.name, fcls, lcls))
                pair_counter[(names[fcls], names[lcls])] += 1
                split_counter[split] += 1

print("Suspicious filename-vs-label cases:", len(suspicious))
print("\nBy split:")
for k, v in split_counter.items():
    print(k, v)

print("\nBy mismatch type:")
for (file_class, label_class), count in pair_counter.most_common():
    print(f"{file_class} filename -> {label_class} label: {count}")

print("\nFirst 50 suspicious cases:")
for split, fname, fcls, lcls in suspicious[:50]:
    print(split, fname, "filename_class:", names[fcls], "label_class:", names[lcls])