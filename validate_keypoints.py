from pathlib import Path
import numpy as np

ROOT = Path("keypoints")

CLASSES = [
    "Bear", "Break", "Brinjal", "Budget", "Busy", "Cabbage",
    "Carrot", "Cauliflower", "Chilli", "Clean", "Close", "Come",
    "Cook", "Crocodile", "Cry", "Cucumber", "Deer", "Drink",
    "Elephant", "Exam", "Fedup", "Fever", "Giraffe", "Thank you"
]

bad_files = []

for class_name in CLASSES:
    folder = ROOT / class_name
    files = list(folder.glob("*.npz"))

    for file in files:
        if class_name == "Bear" and file.name == "WIN_20261031_17_50_20_Pro.npz":
            continue

        try:
            with np.load(file) as data:
                landmarks = data["landmarks"]
                label = str(data["label"].item())

                if landmarks.shape != (32, 48, 3):
                    bad_files.append((str(file), f"Wrong shape: {landmarks.shape}"))
                elif label != class_name:
                    bad_files.append((str(file), f"Wrong label: {label}"))
                elif not np.isfinite(landmarks).all():
                    bad_files.append((str(file), "Contains NaN or infinity"))

        except Exception as e:
            bad_files.append((str(file), str(e)))

print("\nVALIDATION COMPLETE")
print("Bad files:", len(bad_files))

for file, reason in bad_files:
    print(file, "->", reason)