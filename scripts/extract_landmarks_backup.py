import os
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# PATHS
# =========================

DATASET_DIR = r"D:\AI-Sign-Language-Recognition\dataset\Video_Dataset\Video_Dataset"
MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\holistic_landmarker.task"
OUTPUT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# CONFIGURATION
# =========================

# Number of frames used for every video sequence
SEQUENCE_LENGTH = 32

# Upper-body pose landmarks:
# 11 = left shoulder
# 12 = right shoulder
# 13 = left elbow
# 14 = right elbow
# 15 = left wrist
# 16 = right wrist
POSE_INDICES = [11, 12, 13, 14, 15, 16]


# =========================
# CREATE HOLISTIC LANDMARKER
# =========================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HolisticLandmarkerOptions(
    base_options=base_options
)

landmarker = vision.HolisticLandmarker.create_from_options(options)


# =========================
# NORMALIZATION
# =========================

def normalize_landmarks(pose, left_hand, right_hand):
    """
    Normalize landmarks relative to the shoulder center.

    Scale is based on the distance between the shoulders.
    This reduces the effect of person's position and body size.
    """

    # Create fixed-size array:
    # 6 pose landmarks + 21 left hand + 21 right hand
    # = 48 landmarks
    landmarks = np.zeros((48, 3), dtype=np.float32)

    # -------------------------
    # Pose
    # -------------------------

    if pose:
        for i, idx in enumerate(POSE_INDICES):
            if idx < len(pose):
                landmarks[i] = [
                    pose[idx].x,
                    pose[idx].y,
                    pose[idx].z
                ]

    # -------------------------
    # Left hand
    # -------------------------

    if left_hand:
        for i, lm in enumerate(left_hand):
            landmarks[6 + i] = [
                lm.x,
                lm.y,
                lm.z
            ]

    # -------------------------
    # Right hand
    # -------------------------

    if right_hand:
        for i, lm in enumerate(right_hand):
            landmarks[27 + i] = [
                lm.x,
                lm.y,
                lm.z
            ]

    # -------------------------
    # Shoulder center
    # -------------------------

    left_shoulder = landmarks[0]
    right_shoulder = landmarks[1]

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2.0

    # Shoulder distance
    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder
    )

    # Prevent division by zero
    if shoulder_distance < 1e-6:
        shoulder_distance = 1.0

    # Normalize
    landmarks = (
        landmarks - shoulder_center
    ) / shoulder_distance

    return landmarks


# =========================
# RESAMPLE SEQUENCE
# =========================

def resample_sequence(sequence, target_length):
    """
    Convert a variable-length video sequence
    into a fixed-length sequence.
    """

    sequence = np.array(sequence, dtype=np.float32)

    if len(sequence) == 0:
        return np.zeros(
            (target_length, 48, 3),
            dtype=np.float32
        )

    if len(sequence) == target_length:
        return sequence

    indices = np.linspace(
        0,
        len(sequence) - 1,
        target_length
    )

    indices = np.round(indices).astype(int)

    return sequence[indices]


# =========================
# PROCESS ONE VIDEO
# =========================

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return None

    sequence = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # OpenCV BGR → RGB
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        # Run Holistic
        result = landmarker.detect(image)

        # Extract landmarks
        pose = result.pose_landmarks \
            if result.pose_landmarks else None

        left_hand = result.left_hand_landmarks \
            if result.left_hand_landmarks else None

        right_hand = result.right_hand_landmarks \
            if result.right_hand_landmarks else None

        # Normalize
        landmarks = normalize_landmarks(
            pose,
            left_hand,
            right_hand
        )

        sequence.append(landmarks)

    cap.release()

    # Convert to fixed length
    sequence = resample_sequence(
        sequence,
        SEQUENCE_LENGTH
    )

    return sequence


# =========================
# PROCESS DATASET
# =========================

classes = sorted([
    folder
    for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(
        os.path.join(DATASET_DIR, folder)
    )
])

print("=" * 60)
print("AI SIGN LANGUAGE RECOGNITION")
print("LANDMARK EXTRACTION")
print("=" * 60)

print(f"Classes found: {len(classes)}")
print(f"Sequence length: {SEQUENCE_LENGTH}")
print(f"Output directory: {OUTPUT_DIR}")
print()


total_videos = 0
processed_videos = 0


for class_index, class_name in enumerate(classes):

    class_path = os.path.join(
        DATASET_DIR,
        class_name
    )

    output_class_path = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    os.makedirs(
        output_class_path,
        exist_ok=True
    )

    videos = sorted([
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(".mp4")
    ])

    print(
        f"[{class_index + 1}/{len(classes)}] "
        f"{class_name}: {len(videos)} videos"
    )

    for video_index, video_file in enumerate(videos):

        video_path = os.path.join(
            class_path,
            video_file
        )

        # Remove .mp4
        base_name = os.path.splitext(
            video_file
        )[0]

        output_file = os.path.join(
            output_class_path,
            base_name + ".npz"
        )

        # Skip if already processed
        if os.path.exists(output_file):
            processed_videos += 1
            continue

        sequence = process_video(
            video_path
        )

        if sequence is not None:

            np.savez_compressed(
                output_file,
                landmarks=sequence,
                label=class_name
            )

            processed_videos += 1

        total_videos += 1

        print(
            f"    {video_index + 1}/{len(videos)} "
            f"-> {base_name}.npz"
        )


# =========================
# CLEANUP
# =========================

landmarker.close()


print()
print("=" * 60)
print("EXTRACTION COMPLETE")
print("=" * 60)
print(f"Processed: {processed_videos}")
print(f"Output: {OUTPUT_DIR}")