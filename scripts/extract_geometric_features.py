import os
import numpy as np

INPUT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"
OUTPUT_DIR = r"D:\AI-Sign-Language-Recognition\geometric_features"
EXPECTED_CLASSES = 61

os.makedirs(OUTPUT_DIR, exist_ok=True)

def distance(a, b):
    return np.linalg.norm(a - b)

def angle(a, b, c):
    ba, bc = a - b, c - b
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator < 1e-8:
        return 0.0
    cosine = np.dot(ba, bc) / denominator
    return np.arccos(np.clip(cosine, -1.0, 1.0))

def extract_features(frame):
    ls, rs, le, re, lw, rw = frame[:6]
    features = [
        distance(ls, rs), distance(ls, le), distance(le, lw),
        distance(ls, lw), distance(rs, re), distance(re, rw),
        distance(rs, rw), distance(lw, rw),
        angle(ls, le, lw), angle(rs, re, rw),
        distance(lw, rs), distance(rw, ls)
    ]

    for i in range(6, 27):
        features.append(distance(lw, frame[i]))
    for i in range(27, 48):
        features.append(distance(rw, frame[i]))

    features.extend((lw - ls).tolist())
    features.extend((rw - rs).tolist())

    result = np.asarray(features, dtype=np.float32)
    if result.shape != (60,):
        raise ValueError(f"Expected 60 features, got {result.shape}")
    return result


def discover_classes():
    classes = []
    for name in sorted(os.listdir(INPUT_DIR), key=str.casefold):
        folder = os.path.join(INPUT_DIR, name)
        if os.path.isdir(folder) and any(
            f.lower().endswith(".npz") for f in os.listdir(folder)
        ):
            classes.append(name)
    if len(classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} keypoint folders, found {len(classes)}"
        )
    return classes


def main():
    classes = discover_classes()
    processed = failed = 0

    for ci, label in enumerate(classes, 1):
        src_dir = os.path.join(INPUT_DIR, label)
        dst_dir = os.path.join(OUTPUT_DIR, label)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".npz"))
        print(f"\n[{ci}/{len(classes)}] {label}: {len(files)} files")

        for filename in files:
            src = os.path.join(src_dir, filename)
            dst = os.path.join(dst_dir, filename)
            try:
                with np.load(src, allow_pickle=True) as data:
                    landmarks = data["landmarks"].astype(np.float32)
                    stored_label = str(data["label"].item())

                if stored_label != label:
                    raise ValueError(f"Label mismatch: {stored_label}")
                if landmarks.shape != (32, 48, 3):
                    raise ValueError(f"Unexpected shape: {landmarks.shape}")
                if not np.isfinite(landmarks).all():
                    raise ValueError("Non-finite landmark values")

                geometric = np.asarray(
                    [extract_features(frame) for frame in landmarks],
                    dtype=np.float32
                )
                if geometric.shape != (32, 60):
                    raise ValueError(f"Unexpected feature shape: {geometric.shape}")

                np.savez_compressed(
                    dst, geometric_features=geometric, label=label
                )
                processed += 1
            except Exception as exc:
                failed += 1
                print(f"FAILED {label}/{filename}: {exc}")

    print("\nGEOMETRIC EXTRACTION FINISHED")
    print("Processed:", processed, "| Failed:", failed)
    if failed:
        print("Fix failed files before creating the split.")


if __name__ == "__main__":
    main()
