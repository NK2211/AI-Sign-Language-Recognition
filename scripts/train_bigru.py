import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

DATA_DIR = r"D:\AI-Sign-Language-Recognition\keypoints"

CLASSES = [
    "Bear",
    "Cry",
    "Cucumber",
    "Drink",
    "Elephant"
]

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD DATA
# =========================

X = []
y = []

print("\nLoading dataset...\n")

for label, class_name in enumerate(CLASSES):

    class_dir = os.path.join(DATA_DIR, class_name)

    files = [
        f for f in os.listdir(class_dir)
        if f.endswith(".npz")
    ]

    print(f"{class_name}: {len(files)} samples")

    for file in files:

        path = os.path.join(class_dir, file)

        data = np.load(path)

        array = data[data.files[0]]

        X.append(array)
        y.append(label)


X = np.array(X, dtype=np.float32)
y = np.array(y)

# Convert (samples, frames, landmarks, xyz)
# into (samples, frames, features)

X = X.reshape(
    X.shape[0],
    X.shape[1],
    -1
)

print("\nDataset shape:", X.shape)
print("Labels shape:", y.shape)


# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# =========================
# DATASET
# =========================

class SignDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


train_dataset = SignDataset(
    X_train,
    y_train
)

test_dataset = SignDataset(
    X_test,
    y_test
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


# =========================
# BiGRU MODEL
# =========================

class BiGRU(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes
    ):

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
            128 * 2,
            num_classes
        )

    def forward(self, x):

        output, _ = self.gru(x)

        x = output[:, -1, :]

        x = self.dropout(x)

        x = self.fc(x)

        return x


# =========================
# MODEL SETUP
# =========================

input_size = X.shape[2]

model = BiGRU(
    input_size,
    len(CLASSES)
).to(DEVICE)

print("\nInput features:", input_size)
print("Device:", DEVICE)


criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================
# TRAINING
# =========================

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()


    # Evaluation

    model.eval()

    predictions = []
    actual = []

    with torch.no_grad():

        for inputs, labels in test_loader:

            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            preds = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                preds.cpu().numpy()
            )

            actual.extend(
                labels.numpy()
            )


    accuracy = accuracy_score(
        actual,
        predictions
    )

    print(
        f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
        f"Loss: {total_loss / len(train_loader):.4f} "
        f"Accuracy: {accuracy * 100:.2f}%"
    )


# =========================
# FINAL RESULTS
# =========================

print("\n==============================")
print("FINAL RESULTS")
print("==============================")

print(
    classification_report(
        actual,
        predictions,
        target_names=CLASSES
    )
)


# =========================
# SAVE MODEL
# =========================

MODEL_DIR = r"D:\AI-Sign-Language-Recognition\models"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "bigru_5class.pth"
)

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": CLASSES,
        "input_size": input_size,
        "sequence_length": X.shape[1]
    },
    MODEL_PATH
)

print("\nModel saved to:")
print(MODEL_PATH)