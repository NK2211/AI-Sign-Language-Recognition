import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATASET_DIR = r"D:\AI-Sign-Language-Recognition\dataset\Video_Dataset\Video_Dataset"
MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\holistic_landmarker.task"
OUTPUT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"
EXPECTED_CLASSES = 61
SEQUENCE_LENGTH = 32
POSE_INDICES = [11, 12, 13, 14, 15, 16]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def discover_classes():
    classes = []
    for name in sorted(os.listdir(DATASET_DIR), key=str.casefold):
        folder = os.path.join(DATASET_DIR, name)
        if os.path.isdir(folder) and any(
            f.lower().endswith(".mp4") for f in os.listdir(folder)
        ):
            classes.append(name)

    if len(classes) != EXPECTED_CLASSES:
        raise RuntimeError(
            f"Expected {EXPECTED_CLASSES} classes, found {len(classes)}: {classes}"
        )
    return classes


def create_landmarker():
    options = vision.HolisticLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH)
    )
    return vision.HolisticLandmarker.create_from_options(options)


def normalize_landmarks(pose, left_hand, right_hand):
    landmarks = np.zeros((48, 3), dtype=np.float32)

    if pose:
        for i, idx in enumerate(POSE_INDICES):
            if idx < len(pose):
                landmarks[i] = [pose[idx].x, pose[idx].y, pose[idx].z]

    if left_hand:
        for i, lm in enumerate(left_hand[:21]):
            landmarks[6 + i] = [lm.x, lm.y, lm.z]

    if right_hand:
        for i, lm in enumerate(right_hand[:21]):
            landmarks[27 + i] = [lm.x, lm.y, lm.z]

    ls, rs = landmarks[0], landmarks[1]
    center = (ls + rs) / 2.0
    shoulder_distance = np.linalg.norm(ls - rs)
    if shoulder_distance < 1e-6:
        shoulder_distance = 1.0

    return ((landmarks - center) / shoulder_distance).astype(np.float32)


def resample_sequence(sequence):
    sequence = np.asarray(sequence, dtype=np.float32)
    if len(sequence) == 0:
        return None
    indices = np.round(
        np.linspace(0, len(sequence) - 1, SEQUENCE_LENGTH)
    ).astype(int)
    return sequence[indices]


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return None

    landmarker = None
    sequence = []
    try:
        landmarker = create_landmarker()
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(image)

            sequence.append(normalize_landmarks(
                result.pose_landmarks or None,
                result.left_hand_landmarks or None,
                result.right_hand_landmarks or None
            ))
    except Exception as exc:
        print(f"[ERROR] {video_path}: {exc}")
        return None
    finally:
        cap.release()
        if landmarker is not None:
            landmarker.close()

    return resample_sequence(sequence)


def valid_existing(path, label):
    try:
        with np.load(path, allow_pickle=True) as data:
            arr = data["landmarks"]
            stored_label = str(data["label"].item())
            return (
                arr.shape == (32, 48, 3)
                and stored_label == label
                and np.isfinite(arr).all()
            )
    except Exception:
        return False


def main():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    classes = discover_classes()
    print(f"Found {len(classes)} classes")
    for label in classes:
        print(" -", label)

    processed = skipped = failed = 0

    for ci, label in enumerate(classes, 1):
        source = os.path.join(DATASET_DIR, label)
        destination = os.path.join(OUTPUT_DIR, label)
        os.makedirs(destination, exist_ok=True)

        videos = sorted(
            f for f in os.listdir(source) if f.lower().endswith(".mp4")
        )
        print(f"\n[{ci}/{len(classes)}] {label}: {len(videos)} videos")

        for vi, filename in enumerate(videos, 1):
            stem = os.path.splitext(filename)[0]
            output_path = os.path.join(destination, stem + ".npz")

            if valid_existing(output_path, label):
                skipped += 1
                continue

            sequence = process_video(os.path.join(source, filename))
            if sequence is None:
                failed += 1
                continue

            np.savez_compressed(
                output_path, landmarks=sequence, label=label
            )
            processed += 1
            print(f"  {vi}/{len(videos)} saved: {stem}.npz")

    print("\nLANDMARK EXTRACTION FINISHED")
    print("New:", processed, "| Existing valid:", skipped, "| Failed:", failed)
    if failed:
        print("Fix failed videos before continuing.")


if __name__ == "__main__":
    main()
