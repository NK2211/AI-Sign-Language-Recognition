import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================

KEYPOINT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"
GEOMETRIC_DIR = r"D:\AI-Sign-Language-Recognition\geometric_features"
MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\fusion_23class.pth"

SEQUENCE_LENGTH = 32
SPATIAL_FEATURES = 48 * 3      # 144
GEOMETRIC_FEATURES = 60

HIDDEN_SIZE = 128
NUM_CLASSES = 23

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class FusionDataset(Dataset):

    def __init__(self, samples, label_to_idx):

        self.samples = samples
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        spatial_path, geometric_path, label = self.samples[idx]

        # Load spatial landmarks
        spatial_data = np.load(spatial_path)

        landmarks = spatial_data["landmarks"].astype(np.float32)

        # (32, 48, 3) -> (32, 144)
        landmarks = landmarks.reshape(SEQUENCE_LENGTH, -1)

        # Load geometric features
        geometric_data = np.load(geometric_path)

        geometric = geometric_data["geometric_features"].astype(np.float32)

        # Safety check
        assert landmarks.shape == (32, 144)
        assert geometric.shape == (32, 60)

        y = self.label_to_idx[label]

        return (
            torch.tensor(landmarks, dtype=torch.float32),
            torch.tensor(geometric, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )


# ============================================================
# LOAD DATA
# ============================================================

classes = sorted([
    d for d in os.listdir(KEYPOINT_DIR)
    if os.path.isdir(os.path.join(KEYPOINT_DIR, d))
])

# Remove empty classes
classes = [
    c for c in classes
    if len(glob.glob(os.path.join(KEYPOINT_DIR, c, "*.npz"))) > 0
]

label_to_idx = {
    label: idx for idx, label in enumerate(classes)
}

idx_to_label = {
    idx: label for label, idx in label_to_idx.items()
}

NUM_CLASSES = len(classes)

print("\nClasses:")
for i, c in enumerate(classes):
    print(i, "->", c)

print("\nNumber of classes:", NUM_CLASSES)


# ============================================================
# CREATE PAIRED SAMPLES
# ============================================================

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

            samples.append(
                (spatial_path, geometric_path, label)
            )


print("\nTotal paired samples:", len(samples))


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

labels = [s[2] for s in samples]

train_samples, test_samples = train_test_split(
    samples,
    test_size=0.20,
    random_state=42,
    stratify=labels
)

print("Training samples:", len(train_samples))
print("Testing samples :", len(test_samples))


# ============================================================
# DATA LOADERS
# ============================================================

train_dataset = FusionDataset(
    train_samples,
    label_to_idx
)

test_dataset = FusionDataset(
    test_samples,
    label_to_idx
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
# PROPOSED FUSION MODEL
# ============================================================

class FusionModel(nn.Module):

    def __init__(
        self,
        spatial_input,
        geometric_input,
        hidden_size,
        num_classes
    ):

        super().__init__()

        # ----------------------------------------------------
        # SPATIAL BRANCH
        # ----------------------------------------------------

        self.spatial_bigru = nn.GRU(
            input_size=spatial_input,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        # Output = 128 × 2 = 256
        spatial_dim = hidden_size * 2

        # ----------------------------------------------------
        # GEOMETRIC BRANCH
        # ----------------------------------------------------

        self.geometric_gru = nn.GRU(
            input_size=geometric_input,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=False,
            dropout=0.3
        )

        geometric_dim = hidden_size

        # ----------------------------------------------------
        # PROJECT GEOMETRIC FEATURES
        # TO SAME DIMENSION AS SPATIAL FEATURES
        # ----------------------------------------------------

        self.geometric_projection = nn.Linear(
            geometric_dim,
            spatial_dim
        )

        # ----------------------------------------------------
        # ATTENTION FUSION
        # ----------------------------------------------------

        self.attention = nn.MultiheadAttention(
            embed_dim=spatial_dim,
            num_heads=4,
            batch_first=True
        )

        # ----------------------------------------------------
        # SKIP CONNECTION
        # ----------------------------------------------------

        self.norm = nn.LayerNorm(spatial_dim)

        self.dropout = nn.Dropout(0.3)

        # ----------------------------------------------------
        # CLASSIFIER
        # ----------------------------------------------------

        self.classifier = nn.Sequential(
            nn.Linear(spatial_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )


    def forward(self, spatial, geometric):

        # ====================================================
        # SPATIAL BRANCH
        # ====================================================

        spatial_out, _ = self.spatial_bigru(spatial)

        # (batch, 32, 256)


        # ====================================================
        # GEOMETRIC BRANCH
        # ====================================================

        geometric_out, _ = self.geometric_gru(geometric)

        # (batch, 32, 128)


        # Project to 256
        geometric_out = self.geometric_projection(
            geometric_out
        )

        # (batch, 32, 256)


        # ====================================================
        # FUSION
        # ====================================================

        fused = spatial_out + geometric_out

        # ====================================================
        # ATTENTION
        # ====================================================

        attended, _ = self.attention(
            fused,
            fused,
            fused
        )

        # ====================================================
        # SKIP CONNECTION
        # ====================================================

        fused = self.norm(
            fused + attended
        )

        # ====================================================
        # TEMPORAL POOLING
        # ====================================================

        fused = fused.mean(dim=1)

        # ====================================================
        # CLASSIFIER
        # ====================================================

        fused = self.dropout(fused)

        output = self.classifier(fused)

        return output


# ============================================================
# CREATE MODEL
# ============================================================

model = FusionModel(
    spatial_input=SPATIAL_FEATURES,
    geometric_input=GEOMETRIC_FEATURES,
    hidden_size=HIDDEN_SIZE,
    num_classes=NUM_CLASSES
).to(DEVICE)

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

    for spatial, geometric, labels_batch in train_loader:

        spatial = spatial.to(DEVICE)

        geometric = geometric.to(DEVICE)

        labels_batch = labels_batch.to(DEVICE)

        # Forward
        outputs = model(
            spatial,
            geometric
        )

        loss = criterion(
            outputs,
            labels_batch
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels_batch.size(0)

        correct += (
            predicted == labels_batch
        ).sum().item()

    train_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
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

    for spatial, geometric, labels_batch in test_loader:

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
            labels_batch.numpy()
        )


# ============================================================
# METRICS
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

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:\n")

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
        "label_to_idx": label_to_idx,
        "sequence_length": SEQUENCE_LENGTH,
        "spatial_features": SPATIAL_FEATURES,
        "geometric_features": GEOMETRIC_FEATURES
    },
    MODEL_PATH
)

print("\nModel saved to:")

print(MODEL_PATH)