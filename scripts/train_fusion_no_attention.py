import os
import json
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# CONFIG
# ============================================================

KEYPOINT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"
GEOMETRIC_DIR = r"D:\AI-Sign-Language-Recognition\geometric_features"
SPLIT_FILE = r"D:\AI-Sign-Language-Recognition\scripts\split.json"

MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\fusion_no_attention_23class.pth"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# ============================================================
# LOAD FIXED SPLIT
# ============================================================

with open(SPLIT_FILE, "r") as f:
    split = json.load(f)

classes = split["classes"]
train_samples = split["train"]
test_samples = split["test"]

label_to_idx = {
    label: idx
    for idx, label in enumerate(classes)
}

NUM_CLASSES = len(classes)

print("\nClasses:", NUM_CLASSES)
print("Training samples:", len(train_samples))
print("Testing samples :", len(test_samples))


# ============================================================
# DATASET
# ============================================================

class FusionDataset(Dataset):

    def __init__(self, samples):

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        label = sample["label"]
        filename = sample["filename"]

        spatial_path = os.path.join(
            KEYPOINT_DIR,
            label,
            filename
        )

        geometric_path = os.path.join(
            GEOMETRIC_DIR,
            label,
            filename
        )

        # ----------------------------------------------------
        # Spatial landmarks
        # ----------------------------------------------------

        spatial_data = np.load(spatial_path)

        landmarks = spatial_data["landmarks"].astype(
            np.float32
        )

        # (32, 48, 3) -> (32, 144)

        landmarks = landmarks.reshape(
            32,
            144
        )

        # ----------------------------------------------------
        # Geometric features
        # ----------------------------------------------------

        geometric_data = np.load(geometric_path)

        geometric = geometric_data[
            "geometric_features"
        ].astype(np.float32)

        # (32, 60)

        return (
            torch.tensor(
                landmarks,
                dtype=torch.float32
            ),
            torch.tensor(
                geometric,
                dtype=torch.float32
            ),
            torch.tensor(
                label_to_idx[label],
                dtype=torch.long
            )
        )


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = FusionDataset(
    train_samples
)

test_dataset = FusionDataset(
    test_samples
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

class FusionNoAttention(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Spatial BiGRU
        # ----------------------------------------------------

        self.spatial_bigru = nn.GRU(
            input_size=144,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Output = 256


        # ----------------------------------------------------
        # Geometric GRU
        # ----------------------------------------------------

        self.geometric_gru = nn.GRU(
            input_size=60,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.3
        )

        # Output = 128


        # ----------------------------------------------------
        # Project geometric → 256
        # ----------------------------------------------------

        self.geometric_projection = nn.Linear(
            128,
            256
        )


        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                NUM_CLASSES
            )
        )


    def forward(
        self,
        spatial,
        geometric
    ):

        # ----------------------------------------------------
        # Spatial branch
        # ----------------------------------------------------

        spatial_out, _ = self.spatial_bigru(
            spatial
        )

        # (batch, 32, 256)


        # ----------------------------------------------------
        # Geometric branch
        # ----------------------------------------------------

        geometric_out, _ = self.geometric_gru(
            geometric
        )

        # (batch, 32, 128)


        # Project geometric branch

        geometric_out = self.geometric_projection(
            geometric_out
        )

        # (batch, 32, 256)


        # ----------------------------------------------------
        # SIMPLE FUSION
        # ----------------------------------------------------

        fused = (
            spatial_out +
            geometric_out
        )

        # ----------------------------------------------------
        # Temporal average
        # ----------------------------------------------------

        fused = fused.mean(
            dim=1
        )

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        output = self.classifier(
            fused
        )

        return output


# ============================================================
# CREATE MODEL
# ============================================================

model = FusionNoAttention().to(
    DEVICE
)

print("\nModel:")
print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for spatial, geometric, labels in train_loader:

        spatial = spatial.to(DEVICE)
        geometric = geometric.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward

        outputs = model(
            spatial,
            geometric
        )

        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Statistics

        running_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_accuracy = (
        100 * correct / total
    )

    average_loss = (
        running_loss /
        len(train_loader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}%"
    )


# ============================================================
# EVALUATION
# ============================================================

print("\nEvaluating model...\n")

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for spatial, geometric, labels in test_loader:

        spatial = spatial.to(DEVICE)
        geometric = geometric.to(DEVICE)

        outputs = model(
            spatial,
            geometric
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.numpy()
        )


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("=" * 60)

print(
    f"TEST ACCURACY: {accuracy * 100:.2f}%"
)

print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=classes,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:\n")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)


# ============================================================
# SAVE MODEL
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "label_to_idx": label_to_idx
    },
    MODEL_PATH
)

print("\nModel saved to:")

print(MODEL_PATH)