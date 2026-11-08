import os
import json
from collections import Counter
from sklearn.model_selection import train_test_split

KEYPOINT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"
GEOMETRIC_DIR = r"D:\AI-Sign-Language-Recognition\geometric_features"
OUTPUT_FILE = r"D:\AI-Sign-Language-Recognition\scripts\grouped_split_61class.json"

EXPECTED_CLASSES = 61
TEST_SIZE = 0.20
RANDOM_STATE = 42


def discover_classes():
    classes = []
    for name in sorted(os.listdir(KEYPOINT_DIR), key=str.casefold):
        folder = os.path.join(KEYPOINT_DIR, name)
        if os.path.isdir(folder) and any(
            f.lower().endswith(".npz") for f in os.listdir(folder)
        ):
            classes.append(name)
    if len(classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes, found {len(classes)}: {classes}"
        )
    return classes


def base_recording(filename):
    for suffix in ("_left_tilt.npz", "_right_tilt.npz"):
        if filename.endswith(suffix):
            return filename[:-len(suffix)] + ".npz"
    return filename


def main():
    classes = discover_classes()
    groups = []

    for label in classes:
        kp_dir = os.path.join(KEYPOINT_DIR, label)
        geo_dir = os.path.join(GEOMETRIC_DIR, label)
        if not os.path.isdir(geo_dir):
            raise FileNotFoundError(f"Missing feature folder: {geo_dir}")

        files = sorted(
            f for f in os.listdir(kp_dir) if f.lower().endswith(".npz")
        )
        if not files:
            raise ValueError(f"No keypoint files for {label}")

        by_base = {}
        for filename in files:
            by_base.setdefault(base_recording(filename), []).append(filename)

        for base, variants in by_base.items():
            for filename in variants:
                if not os.path.isfile(os.path.join(geo_dir, filename)):
                    raise FileNotFoundError(
                        f"Missing geometric feature: {label}/{filename}"
                    )
            groups.append({
                "label": label,
                "base": base,
                "files": sorted(variants)
            })

    counts = Counter(g["label"] for g in groups)
    print("Recording groups per class:")
    for label in classes:
        print(f"{label}: {counts[label]}")
    if set(counts) != set(classes) or any(counts[x] < 2 for x in classes):
        raise RuntimeError("Each class must have at least two recording groups.")

    train_groups, test_groups = train_test_split(
        groups,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=[g["label"] for g in groups]
    )

    def expand(selected):
        return [
            {"label": group["label"], "filename": filename}
            for group in selected
            for filename in group["files"]
        ]

    train_samples = expand(train_groups)
    test_samples = expand(test_groups)

    train_ids = {(g["label"], g["base"]) for g in train_groups}
    test_ids = {(g["label"], g["base"]) for g in test_groups}
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(f"Recording leakage: {overlap}")

    split = {
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "classes": classes,
        "train_groups": train_groups,
        "test_groups": test_groups,
        "train": train_samples,
        "test": test_samples
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    print("\n61-CLASS GROUPED SPLIT CREATED")
    print("Classes:", len(classes))
    print("Train groups:", len(train_groups))
    print("Test groups:", len(test_groups))
    print("Train samples:", len(train_samples))
    print("Test samples:", len(test_samples))
    print("Recording overlap:", len(overlap))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
