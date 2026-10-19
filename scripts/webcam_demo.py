import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\bigru_5class.pth"
HOLISTIC_MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\holistic_landmarker.task"

SEQUENCE_LENGTH = 32

CLASSES = [
    "Bear",
    "Cry",
    "Cucumber",
    "Drink",
    "Elephant"
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# BiGRU MODEL
# ============================================================

class BiGRU(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.dropout = nn.Dropout(0.3)

        self.fc = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        output, _ = self.gru(x)

        x = output[:, -1, :]

        x = self.dropout(x)

        return self.fc(x)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

input_size = checkpoint["input_size"]

model = BiGRU(
    input_size,
    len(CLASSES)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("BiGRU model loaded.")
print("Input features:", input_size)
print("Classes:", CLASSES)


# ============================================================
# MEDIAPIPE
# ============================================================

base_options = python.BaseOptions(
    model_asset_path=HOLISTIC_MODEL_PATH
)

options = vision.HolisticLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO
)

landmarker = vision.HolisticLandmarker.create_from_options(
    options
)

print("MediaPipe Holistic loaded.")


# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def get_xyz(lm):

    return [
        lm.x,
        lm.y,
        lm.z
    ]


def extract_landmarks(result):

    landmarks = []

    # --------------------------------------------------------
    # LEFT HAND — 21
    # --------------------------------------------------------

    if result.left_hand_landmarks:

        for lm in result.left_hand_landmarks:

            landmarks.append(
                get_xyz(lm)
            )

    else:

        landmarks.extend(
            [[0.0, 0.0, 0.0]] * 21
        )


    # --------------------------------------------------------
    # RIGHT HAND — 21
    # --------------------------------------------------------

    if result.right_hand_landmarks:

        for lm in result.right_hand_landmarks:

            landmarks.append(
                get_xyz(lm)
            )

    else:

        landmarks.extend(
            [[0.0, 0.0, 0.0]] * 21
        )


    # --------------------------------------------------------
    # POSE — shoulders, elbows, wrists = 6
    # --------------------------------------------------------

    if result.pose_landmarks:

        pose = result.pose_landmarks

        for index in [11, 12, 13, 14, 15, 16]:

            lm = pose[index]

            landmarks.append(
                get_xyz(lm)
            )

    else:

        landmarks.extend(
            [[0.0, 0.0, 0.0]] * 6
        )


    return np.array(
        landmarks,
        dtype=np.float32
    )


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    raise SystemExit


sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction = "Collecting..."
confidence = 0.0

timestamp_ms = 0

print()
print("========================================")
print("AI SIGN LANGUAGE RECOGNITION")
print("========================================")
print("Signs:")
print("Bear")
print("Cry")
print("Cucumber")
print("Drink")
print("Elephant")
print()
print("Press Q to quit.")
print("========================================")


# ============================================================
# MAIN LOOP
# ============================================================

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(
        frame,
        1
    )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp_ms += 33

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # --------------------------------------------------------
    # EXTRACT LANDMARKS
    # --------------------------------------------------------

    landmarks = extract_landmarks(
        result
    )

    if landmarks.shape == (48, 3):

        sequence.append(
            landmarks
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if len(sequence) == SEQUENCE_LENGTH:

        data = np.array(
            sequence,
            dtype=np.float32
        )

        data = data.reshape(
            1,
            SEQUENCE_LENGTH,
            -1
        )

        tensor = torch.tensor(
            data,
            dtype=torch.float32
        ).to(DEVICE)

        with torch.no_grad():

            output = model(
                tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence_tensor, predicted = torch.max(
                probabilities,
                dim=1
            )

        prediction = CLASSES[
            predicted.item()
        ]

        confidence = confidence_tensor.item()


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (560, 115),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "SIGN: " + prediction,
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "AI Sign Language Recognition",
        frame
    )


    # --------------------------------------------------------
    # QUIT
    # --------------------------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

landmarker.close()

cv2.destroyAllWindows()

print("\nDemo closed.")