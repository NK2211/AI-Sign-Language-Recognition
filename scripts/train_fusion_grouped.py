import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

KEYPOINT_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"
GEOMETRIC_DIR = r"D:\AI-Sign-Language-Recognition\geometric_features"
SPLIT_FILE = r"D:\AI-Sign-Language-Recognition\scripts\grouped_split_61class.json"
MODEL_PATH = r"D:\AI-Sign-Language-Recognition\models\fusion_grouped_61class.pth"

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split = json.load(f)

classes = split["classes"]
train_samples = split["train"]
test_samples = split["test"]
NUM_CLASSES = len(classes)
label_to_idx = {label: idx for idx, label in enumerate(classes)}

if NUM_CLASSES != 61:
    raise RuntimeError(f"Expected 61 classes, got {NUM_CLASSES}")
if not train_samples or not test_samples:
    raise RuntimeError("Training or test set is empty.")

class FusionDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label, filename = sample["label"], sample["filename"]
        kp_path = os.path.join(KEYPOINT_DIR, label, filename)
        geo_path = os.path.join(GEOMETRIC_DIR, label, filename)

        with np.load(kp_path, allow_pickle=True) as d:
            landmarks = d["landmarks"].astype(np.float32)
            kp_label = str(d["label"].item())
        with np.load(geo_path, allow_pickle=True) as d:
            geometric = d["geometric_features"].astype(np.float32)
            geo_label = str(d["label"].item())

        if kp_label != label or geo_label != label:
            raise ValueError(f"Label mismatch: {label}/{filename}")
        if landmarks.shape != (32, 48, 3):
            raise ValueError(f"Bad landmark shape: {label}/{filename}")
        if geometric.shape != (32, 60):
            raise ValueError(f"Bad geometric shape: {label}/{filename}")
        if not np.isfinite(landmarks).all() or not np.isfinite(geometric).all():
            raise ValueError(f"Non-finite values: {label}/{filename}")

        return (
            torch.tensor(landmarks.reshape(32, 144), dtype=torch.float32),
            torch.tensor(geometric, dtype=torch.float32),
            torch.tensor(label_to_idx[label], dtype=torch.long)
        )

train_loader = DataLoader(
    FusionDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    FusionDataset(test_samples), batch_size=BATCH_SIZE, shuffle=False
)

class FusionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_bigru = nn.GRU(
            144, 128, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.3
        )
        self.geometric_gru = nn.GRU(
            60, 128, num_layers=2, batch_first=True,
            bidirectional=False, dropout=0.3
        )
        self.geometric_projection = nn.Linear(128, 256)
        self.attention = nn.MultiheadAttention(
            256, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, spatial, geometric):
        spatial_out, _ = self.spatial_bigru(spatial)
        geometric_out, _ = self.geometric_gru(geometric)
        geometric_out = self.geometric_projection(geometric_out)
        fused = spatial_out + geometric_out
        attended, _ = self.attention(fused, fused, fused)
        fused = self.norm(fused + attended)
        fused = fused.mean(dim=1)
        return self.classifier(self.dropout(fused))

model = FusionModel().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"\nTraining {NUM_CLASSES} classes")
print("Train samples:", len(train_samples))
print("Test samples:", len(test_samples))

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = total = 0

    for spatial, geometric, labels in train_loader:
        spatial, geometric, labels = (
            spatial.to(DEVICE), geometric.to(DEVICE), labels.to(DEVICE)
        )
        outputs = model(spatial, geometric)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {running_loss / len(train_loader):.4f} "
        f"Train Acc: {100 * correct / total:.2f}%"
    )

model.eval()
all_predictions, all_labels = [], []
with torch.no_grad():
    for spatial, geometric, labels in test_loader:
        outputs = model(spatial.to(DEVICE), geometric.to(DEVICE))
        all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_predictions)
print(f"\nGROUPED TEST ACCURACY: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=classes,
    digits=4,
    zero_division=0
))
print("\nConfusion Matrix:\n")
print(confusion_matrix(
    all_labels, all_predictions, labels=list(range(NUM_CLASSES))
))

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "classes": classes,
    "label_to_idx": label_to_idx,
    "num_classes": NUM_CLASSES
}, MODEL_PATH)

print("\n61-class model saved to:")
print(MODEL_PATH)
