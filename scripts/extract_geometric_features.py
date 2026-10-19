import os
import numpy as np


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"

OUTPUT_DIR = r"D:\AI-Sign-Language-Recognition\geometric_features"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LANDMARK INDICES
# ============================================================

LEFT_SHOULDER = 0
RIGHT_SHOULDER = 1

LEFT_ELBOW = 2
RIGHT_ELBOW = 3

LEFT_WRIST = 4
RIGHT_WRIST = 5

LEFT_HAND_START = 6
LEFT_HAND_END = 27

RIGHT_HAND_START = 27
RIGHT_HAND_END = 48


# ============================================================
# DISTANCE
# ============================================================

def distance(a, b):

    return np.linalg.norm(a - b)


# ============================================================
# ANGLE
# ============================================================

def angle(a, b, c):

    # Angle ABC

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba)
        * np.linalg.norm(bc)
    )

    if denominator < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / denominator

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.arccos(cosine)


# ============================================================
# GEOMETRIC FEATURES FOR ONE FRAME
# ============================================================

def extract_features(frame):

    features = []

    # --------------------------------------------------------
    # BASIC BODY DISTANCES
    # --------------------------------------------------------

    ls = frame[LEFT_SHOULDER]
    rs = frame[RIGHT_SHOULDER]

    le = frame[LEFT_ELBOW]
    re = frame[RIGHT_ELBOW]

    lw = frame[LEFT_WRIST]
    rw = frame[RIGHT_WRIST]


    # Shoulder distance
    features.append(
        distance(ls, rs)
    )

    # Left arm
    features.append(
        distance(ls, le)
    )

    features.append(
        distance(le, lw)
    )

    features.append(
        distance(ls, lw)
    )

    # Right arm
    features.append(
        distance(rs, re)
    )

    features.append(
        distance(re, rw)
    )

    features.append(
        distance(rs, rw)
    )

    # Between hands
    features.append(
        distance(lw, rw)
    )


    # --------------------------------------------------------
    # BODY ANGLES
    # --------------------------------------------------------

    # Left elbow angle
    features.append(
        angle(ls, le, lw)
    )

    # Right elbow angle
    features.append(
        angle(rs, re, rw)
    )


    # --------------------------------------------------------
    # HAND → BODY DISTANCES
    # --------------------------------------------------------

    features.append(
        distance(lw, rs)
    )

    features.append(
        distance(rw, ls)
    )


    # --------------------------------------------------------
    # HAND LANDMARK DISTANCES
    # --------------------------------------------------------

    # Wrist to each landmark in the same hand

    for i in range(
        LEFT_HAND_START,
        LEFT_HAND_END
    ):

        features.append(
            distance(lw, frame[i])
        )


    for i in range(
        RIGHT_HAND_START,
        RIGHT_HAND_END
    ):

        features.append(
            distance(rw, frame[i])
        )


    # --------------------------------------------------------
    # RELATIVE HAND POSITIONS
    # --------------------------------------------------------

    # Left hand relative to left shoulder
    relative_left = lw - ls

    features.extend(
        relative_left.tolist()
    )

    # Right hand relative to right shoulder
    relative_right = rw - rs

    features.extend(
        relative_right.tolist()
    )


    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(input_file, output_file):

    data = np.load(
        input_file,
        allow_pickle=True
    )

    landmarks = data["landmarks"]

    label = data["label"]

    geometric_sequence = []

    for frame in landmarks:

        features = extract_features(
            frame
        )

        geometric_sequence.append(
            features
        )

    geometric_sequence = np.array(
        geometric_sequence,
        dtype=np.float32
    )

    np.savez_compressed(
        output_file,
        geometric_features=geometric_sequence,
        label=label
    )


# ============================================================
# PROCESS DATASET
# ============================================================

classes = sorted([
    folder
    for folder in os.listdir(INPUT_DIR)
    if os.path.isdir(
        os.path.join(INPUT_DIR, folder)
    )
])


print("=" * 60)
print("GEOMETRIC FEATURE EXTRACTION")
print("=" * 60)

print(
    f"Classes found: {len(classes)}"
)

print(
    f"Input: {INPUT_DIR}"
)

print(
    f"Output: {OUTPUT_DIR}"
)

print()


total = 0


for class_index, class_name in enumerate(classes):

    input_class_dir = os.path.join(
        INPUT_DIR,
        class_name
    )

    output_class_dir = os.path.join(
        OUTPUT_DIR,
        class_name
    )

    os.makedirs(
        output_class_dir,
        exist_ok=True
    )


    files = sorted([
        f
        for f in os.listdir(input_class_dir)
        if f.endswith(".npz")
    ])


    print(
        f"[{class_index + 1}/{len(classes)}] "
        f"{class_name}: {len(files)} files"
    )


    for file in files:

        input_file = os.path.join(
            input_class_dir,
            file
        )

        output_file = os.path.join(
            output_class_dir,
            file
        )


        process_file(
            input_file,
            output_file
        )

        total += 1


print()
print("=" * 60)
print("GEOMETRIC EXTRACTION COMPLETE")
print("=" * 60)

print(
    f"Files processed: {total}"
)

print(
    f"Output: {OUTPUT_DIR}"
)