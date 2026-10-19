import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque, Counter


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\fusion_grouped_23class.pth"
HOLISTIC_MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\holistic_landmarker.task"


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 32

# Stability settings
CONFIDENCE_THRESHOLD = 70.0
VOTING_WINDOW = 7
MIN_VOTES = 3

CLASSES = [
    "Bear",
    "Break",
    "Brinjal",
    "Budget",
    "Busy",
    "Cabbage",
    "Carrot",
    "Cauliflower",
    "Chilli",
    "Clean",
    "Close",
    "Come",
    "Cook",
    "Crocodile",
    "Cry",
    "Cucumber",
    "Deer",
    "Drink",
    "Elephant",
    "Exam",
    "Fedup",
    "Fever",
    "Giraffe"
]


# ============================================================
# MODEL
# ============================================================

class FusionModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        # ----------------------------------------------------
        # Spatial branch
        # ----------------------------------------------------

        self.spatial_bigru = nn.GRU(
            input_size=144,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )

        # ----------------------------------------------------
        # Geometric branch
        # ----------------------------------------------------

        self.geometric_gru = nn.GRU(
            input_size=60,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        # ----------------------------------------------------
        # Geometric projection
        # ----------------------------------------------------

        self.geometric_projection = nn.Linear(
            128,
            256
        )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=4,
            batch_first=True
        )

        # ----------------------------------------------------
        # Skip connection
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(256)

        self.dropout = nn.Dropout(0.3)

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(
        self,
        spatial,
        geometric
    ):

        # ====================================================
        # Spatial branch
        # ====================================================

        spatial_out, _ = self.spatial_bigru(
            spatial
        )

        # Shape:
        # [batch, 32, 256]


        # ====================================================
        # Geometric branch
        # ====================================================

        geometric_out, _ = self.geometric_gru(
            geometric
        )

        # Shape:
        # [batch, 32, 128]


        geometric_out = self.geometric_projection(
            geometric_out
        )

        # Shape:
        # [batch, 32, 256]


        # ====================================================
        # Feature fusion
        # ====================================================

        fused = (
            spatial_out +
            geometric_out
        )


        # ====================================================
        # Attention
        # ====================================================

        attended, _ = self.attention(
            fused,
            fused,
            fused
        )


        # ====================================================
        # Skip connection
        # ====================================================

        fused = self.norm(
            fused + attended
        )


        # ====================================================
        # Temporal pooling
        # ====================================================

        pooled = fused.mean(
            dim=1
        )

        pooled = self.dropout(
            pooled
        )


        # ====================================================
        # Classification
        # ====================================================

        output = self.classifier(
            pooled
        )

        return output


# ============================================================
# LOAD MODEL
# ============================================================

device = torch.device("cpu")

model = FusionModel(
    num_classes=len(CLASSES)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

# Support either a raw state_dict
# or a checkpoint containing model_state_dict

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.to(device)
model.eval()


print("=" * 60)
print("23-CLASS FUSION MODEL LOADED")
print("=" * 60)

print(
    f"Model: {MODEL_PATH}"
)

print(
    f"Classes: {len(CLASSES)}"
)

print(
    f"Device: {device}"
)

print("=" * 60)


# ============================================================
# MEDIAPIPE HOLISTIC
# ============================================================

BaseOptions = mp.tasks.BaseOptions

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)

HolisticLandmarker = (
    mp.tasks.vision.HolisticLandmarker
)

HolisticLandmarkerOptions = (
    mp.tasks.vision.HolisticLandmarkerOptions
)


options = HolisticLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HOLISTIC_MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO
)


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def get_landmark_xyz(landmark):

    return np.array(
        [
            landmark.x,
            landmark.y,
            landmark.z
        ],
        dtype=np.float32
    )


def extract_frame_landmarks(result):

    # ========================================================
    # 48 landmarks
    #
    # 0-5   = body
    # 6-26  = left hand
    # 27-47 = right hand
    # ========================================================

    landmarks = np.zeros(
        (48, 3),
        dtype=np.float32
    )


    # ========================================================
    # Pose landmarks
    # ========================================================

    pose = result.pose_landmarks

    if (
        pose is not None
        and len(pose) >= 17
    ):

        pose_indices = [
            11, 12,
            13, 14,
            15, 16
        ]

        for i, pose_index in enumerate(
            pose_indices
        ):

            landmarks[i] = (
                get_landmark_xyz(
                    pose[pose_index]
                )
            )


    # ========================================================
    # Left hand
    # ========================================================

    if (
        result.left_hand_landmarks
        is not None
        and len(result.left_hand_landmarks) >= 21
    ):

        for i in range(21):

            landmarks[
                6 + i
            ] = get_landmark_xyz(
                result.left_hand_landmarks[i]
            )


    # ========================================================
    # Right hand
    # ========================================================

    if (
        result.right_hand_landmarks
        is not None
        and len(result.right_hand_landmarks) >= 21
    ):

        for i in range(21):

            landmarks[
                27 + i
            ] = get_landmark_xyz(
                result.right_hand_landmarks[i]
            )


    return landmarks


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_landmarks(landmarks):

    normalized = landmarks.copy()

    left_shoulder = normalized[0]
    right_shoulder = normalized[1]


    # --------------------------------------------------------
    # Shoulder center
    # --------------------------------------------------------

    shoulder_center = (
        left_shoulder +
        right_shoulder
    ) / 2.0


    # --------------------------------------------------------
    # Shoulder distance
    # --------------------------------------------------------

    shoulder_distance = np.linalg.norm(
        left_shoulder -
        right_shoulder
    )


    # --------------------------------------------------------
    # Prevent division by zero
    # --------------------------------------------------------

    if shoulder_distance < 1e-6:

        return np.zeros_like(
            normalized
        )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = (
        normalized -
        shoulder_center
    ) / shoulder_distance


    return normalized.astype(
        np.float32
    )


# ============================================================
# GEOMETRIC FEATURE INDICES
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

    return np.linalg.norm(
        a - b
    )


# ============================================================
# ANGLE
# ============================================================

def angle(a, b, c):

    # Angle ABC

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )


    if denominator < 1e-8:

        return 0.0


    cosine = (
        np.dot(ba, bc) /
        denominator
    )


    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )


    return np.arccos(
        cosine
    )


# ============================================================
# GEOMETRIC FEATURES
# EXACTLY MATCHES TRAINING
# ============================================================

def extract_geometric_features(frame):

    features = []


    # ========================================================
    # Body landmarks
    # ========================================================

    ls = frame[LEFT_SHOULDER]
    rs = frame[RIGHT_SHOULDER]

    le = frame[LEFT_ELBOW]
    re = frame[RIGHT_ELBOW]

    lw = frame[LEFT_WRIST]
    rw = frame[RIGHT_WRIST]


    # ========================================================
    # Basic body distances
    # ========================================================

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

    # Hand-to-hand
    features.append(
        distance(lw, rw)
    )


    # ========================================================
    # Body angles
    # ========================================================

    features.append(
        angle(ls, le, lw)
    )

    features.append(
        angle(rs, re, rw)
    )


    # ========================================================
    # Hand -> body distances
    # ========================================================

    features.append(
        distance(lw, rs)
    )

    features.append(
        distance(rw, ls)
    )


    # ========================================================
    # Left wrist -> left hand landmarks
    # ========================================================

    for i in range(
        LEFT_HAND_START,
        LEFT_HAND_END
    ):

        features.append(
            distance(
                lw,
                frame[i]
            )
        )


    # ========================================================
    # Right wrist -> right hand landmarks
    # ========================================================

    for i in range(
        RIGHT_HAND_START,
        RIGHT_HAND_END
    ):

        features.append(
            distance(
                rw,
                frame[i]
            )
        )


    # ========================================================
    # Relative left hand position
    # ========================================================

    relative_left = (
        lw - ls
    )

    features.extend(
        relative_left.tolist()
    )


    # ========================================================
    # Relative right hand position
    # ========================================================

    relative_right = (
        rw - rs
    )

    features.extend(
        relative_right.tolist()
    )


    return np.array(
        features,
        dtype=np.float32
    )


# ============================================================
# VERIFY GEOMETRIC FEATURE SIZE
# ============================================================

dummy_frame = np.zeros(
    (48, 3),
    dtype=np.float32
)

dummy_features = (
    extract_geometric_features(
        dummy_frame
    )
)

print(
    f"Geometric features per frame: "
    f"{len(dummy_features)}"
)

if len(dummy_features) != 60:

    raise RuntimeError(
        "ERROR: Geometric feature size "
        f"is {len(dummy_features)}, "
        "but trained model expects 60."
    )


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# ============================================================
# ROLLING BUFFERS
# ============================================================

spatial_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)

geometric_buffer = deque(
    maxlen=SEQUENCE_LENGTH
)


# ============================================================
# PREDICTION VARIABLES
# ============================================================

prediction = "Waiting..."

raw_prediction = "Waiting..."

confidence = 0.0

prediction_history = deque(
    maxlen=VOTING_WINDOW
)

frame_timestamp = 0


# ============================================================
# RUN MEDIAPIPE
# ============================================================

with HolisticLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        # ====================================================
        # READ FRAME
        # ====================================================

        ret, frame = cap.read()


        if not ret:

            print(
                "Could not read webcam frame."
            )

            break


        # ====================================================
        # MIRROR IMAGE
        # ====================================================

        frame = cv2.flip(
            frame,
            1
        )


        # ====================================================
        # BGR -> RGB
        # ====================================================

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # MEDIAPIPE IMAGE
        # ====================================================

        mp_image = mp.Image(
            image_format=(
                mp.ImageFormat.SRGB
            ),
            data=rgb
        )


        # ====================================================
        # VIDEO TIMESTAMP
        # ====================================================

        frame_timestamp += 33


        # ====================================================
        # HOLISTIC LANDMARK DETECTION
        # ====================================================

        result = landmarker.detect_for_video(
            mp_image,
            frame_timestamp
        )


        # ====================================================
        # EXTRACT 48 LANDMARKS
        # ====================================================

        landmarks = (
            extract_frame_landmarks(
                result
            )
        )


        # ====================================================
        # NORMALIZATION
        # ====================================================

        normalized = (
            normalize_landmarks(
                landmarks
            )
        )


        # ====================================================
        # SPATIAL FEATURES
        #
        # 48 landmarks × 3 coordinates = 144
        # ====================================================

        spatial_features = (
            normalized.reshape(
                144
            )
        )


        # ====================================================
        # GEOMETRIC FEATURES
        #
        # 60 features
        # ====================================================

        geometric_features = (
            extract_geometric_features(
                normalized
            )
        )


        # ====================================================
        # UPDATE ROLLING BUFFERS
        # ====================================================

        spatial_buffer.append(
            spatial_features
        )

        geometric_buffer.append(
            geometric_features
        )


        # ====================================================
        # PREDICT AFTER 32 FRAMES
        # ====================================================

        if (
            len(spatial_buffer)
            == SEQUENCE_LENGTH
        ):

            # ------------------------------------------------
            # Convert sequence to NumPy
            # ------------------------------------------------

            spatial_sequence = np.array(
                spatial_buffer,
                dtype=np.float32
            )

            geometric_sequence = np.array(
                geometric_buffer,
                dtype=np.float32
            )


            # ------------------------------------------------
            # Convert to tensors
            # ------------------------------------------------

            spatial_tensor = torch.tensor(
                spatial_sequence,
                dtype=torch.float32
            ).unsqueeze(0).to(device)


            geometric_tensor = torch.tensor(
                geometric_sequence,
                dtype=torch.float32
            ).unsqueeze(0).to(device)


            # ------------------------------------------------
            # Model prediction
            # ------------------------------------------------

            with torch.no_grad():

                logits = model(
                    spatial_tensor,
                    geometric_tensor
                )


                probabilities = torch.softmax(
                    logits,
                    dim=1
                )


                confidence_value, predicted_index = (
                    torch.max(
                        probabilities,
                        dim=1
                    )
                )


            # ------------------------------------------------
            # Raw prediction
            # ------------------------------------------------

            raw_prediction = CLASSES[
                predicted_index.item()
            ]


            confidence = (
                confidence_value.item()
                * 100.0
            )


            # =================================================
            # CONFIDENCE FILTER
            # =================================================

            if confidence >= CONFIDENCE_THRESHOLD:

                prediction_history.append(
                    raw_prediction
                )


                # =============================================
                # TEMPORAL MAJORITY VOTING
                # =============================================

                if (
                    len(prediction_history)
                    >= MIN_VOTES
                ):

                    vote_counts = Counter(
                        prediction_history
                    )


                    stable_prediction, vote_count = (
                        vote_counts.most_common(1)[0]
                    )


                    prediction = (
                        stable_prediction
                    )


        # ====================================================
        # DISPLAY HEADER
        # ====================================================

        cv2.rectangle(
            frame,
            (0, 0),
            (frame.shape[1], 135),
            (0, 0, 0),
            -1
        )


        # ====================================================
        # STABLE PREDICTION
        # ====================================================

        cv2.putText(
            frame,
            f"Prediction: {prediction}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


        # ====================================================
        # CONFIDENCE
        # ====================================================

        cv2.putText(
            frame,
            f"Confidence: {confidence:.1f}%",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        # ====================================================
        # RAW PREDICTION
        # ====================================================

        cv2.putText(
            frame,
            f"Raw: {raw_prediction}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2
        )


        # ====================================================
        # QUIT MESSAGE
        # ====================================================

        cv2.putText(
            frame,
            "Press Q to quit",
            (
                20,
                frame.shape[0] - 20
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        # ====================================================
        # SHOW WINDOW
        # ====================================================

        cv2.imshow(
            "AI Sign Language Recognition - 23 Class Fusion",
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print(
    "Webcam demo stopped."
)