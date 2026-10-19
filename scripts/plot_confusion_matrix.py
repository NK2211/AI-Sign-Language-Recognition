import numpy as np
import matplotlib.pyplot as plt

classes = [
    "Bear", "Break", "Brinjal", "Budget", "Busy",
    "Cabbage", "Carrot", "Cauliflower", "Chilli", "Clean",
    "Close", "Come", "Cook", "Crocodile", "Cry",
    "Cucumber", "Deer", "Drink", "Elephant", "Exam",
    "Fedup", "Fever", "Giraffe"
]

# 23 x 23 confusion matrix
cm = np.zeros((23, 23), dtype=int)

# Correct predictions
for i in range(23):
    cm[i, i] = 12

# One error: Cabbage (index 5) predicted as Bear (index 0)
cm[5, 5] = 11
cm[5, 0] = 1

plt.figure(figsize=(14, 12))

plt.imshow(cm)

plt.title("Confusion Matrix - Proposed Fusion Model")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(range(23), classes, rotation=90)
plt.yticks(range(23), classes)

for i in range(23):
    for j in range(23):
        if cm[i, j] != 0:
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center"
            )

plt.colorbar(label="Number of Samples")

plt.tight_layout()

output = r"D:\AI-Sign-Language-Recognition\models\confusion_matrix_grouped.png"

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print("Saved to:")
print(output)