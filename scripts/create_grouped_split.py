import os
import glob
import json
from sklearn.model_selection import train_test_split

KEYPOINT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"
GEOMETRIC_DIR = r"D:\AI-Sign-Language-Recognition\geometric_features"

OUTPUT_FILE = r"D:\AI-Sign-Language-Recognition\scripts\grouped_split.json"

# ------------------------------------------------------------
# Classes
# ------------------------------------------------------------

classes = sorted([
    d for d in os.listdir(KEYPOINT_DIR)
    if os.path.isdir(os.path.join(KEYPOINT_DIR, d))
    and len(glob.glob(os.path.join(KEYPOINT_DIR, d, "*.npz"))) > 0
])

print("Classes:", len(classes))


# ------------------------------------------------------------
# Create recording groups
# ------------------------------------------------------------

groups = []

for label in classes:

    files = glob.glob(
        os.path.join(KEYPOINT_DIR, label, "*.npz")
    )

    grouped = {}

    for path in files:

        filename = os.path.basename(path)

        # Remove augmentation suffix
        if filename.endswith("_left_tilt.npz"):
            base = filename.replace(
                "_left_tilt.npz",
                ".npz"
            )

        elif filename.endswith("_right_tilt.npz"):
            base = filename.replace(
                "_right_tilt.npz",
                ".npz"
            )

        else:
            base = filename

        grouped.setdefault(base, [])

        grouped[base].append(filename)

    for base, variants in grouped.items():

        # Verify geometric files exist
        valid_variants = []

        for filename in variants:

            geometric_path = os.path.join(
                GEOMETRIC_DIR,
                label,
                filename
            )

            if os.path.exists(geometric_path):
                valid_variants.append(filename)

        if valid_variants:

            groups.append({
                "label": label,
                "base": base,
                "files": valid_variants
            })


print("Recording groups:", len(groups))


# ------------------------------------------------------------
# Group-level stratified split
# ------------------------------------------------------------

group_labels = [
    group["label"]
    for group in groups
]

train_groups, test_groups = train_test_split(
    groups,
    test_size=0.20,
    random_state=42,
    stratify=group_labels
)


# ------------------------------------------------------------
# Expand groups into samples
# ------------------------------------------------------------

train_samples = []
test_samples = []

for group in train_groups:

    for filename in group["files"]:

        train_samples.append({
            "label": group["label"],
            "filename": filename
        })


for group in test_groups:

    for filename in group["files"]:

        test_samples.append({
            "label": group["label"],
            "filename": filename
        })


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

split = {
    "random_state": 42,
    "test_size": 0.20,
    "classes": classes,
    "train_groups": train_groups,
    "test_groups": test_groups,
    "train": train_samples,
    "test": test_samples
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(split, f, indent=2)


print("\nGrouped split created.")

print("Training groups:", len(train_groups))
print("Testing groups :", len(test_groups))

print("\nTraining samples:", len(train_samples))
print("Testing samples :", len(test_samples))

print("\nSaved to:")
print(OUTPUT_FILE)