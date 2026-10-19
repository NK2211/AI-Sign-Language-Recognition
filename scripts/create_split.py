import os
import glob
import json
from sklearn.model_selection import train_test_split

KEYPOINT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"
GEOMETRIC_DIR = r"D:\AI-Sign-Language-Recognition\geometric_features"

OUTPUT_FILE = r"D:\AI-Sign-Language-Recognition\scripts\split.json"

# ------------------------------------------------------------
# Find all usable classes
# ------------------------------------------------------------

classes = sorted([
    d for d in os.listdir(KEYPOINT_DIR)
    if os.path.isdir(os.path.join(KEYPOINT_DIR, d))
    and len(glob.glob(os.path.join(KEYPOINT_DIR, d, "*.npz"))) > 0
])

print("Classes:", len(classes))

# ------------------------------------------------------------
# Create paired samples
# ------------------------------------------------------------

samples = []

for label in classes:

    spatial_files = glob.glob(
        os.path.join(KEYPOINT_DIR, label, "*.npz")
    )

    for spatial_path in spatial_files:

        filename = os.path.basename(spatial_path)

        geometric_path = os.path.join(
            GEOMETRIC_DIR,
            label,
            filename
        )

        if os.path.exists(geometric_path):

            samples.append({
                "label": label,
                "filename": filename
            })

print("Total samples:", len(samples))

# ------------------------------------------------------------
# Labels for stratified split
# ------------------------------------------------------------

labels = [sample["label"] for sample in samples]

train_samples, test_samples = train_test_split(
    samples,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

# ------------------------------------------------------------
# Save split
# ------------------------------------------------------------

split = {
    "random_state": 42,
    "test_size": 0.20,
    "classes": classes,
    "train": train_samples,
    "test": test_samples
}

with open(OUTPUT_FILE, "w") as f:
    json.dump(split, f, indent=2)

print("\nSplit created successfully.")
print("Training samples:", len(train_samples))
print("Testing samples :", len(test_samples))
print("Saved to:", OUTPUT_FILE)