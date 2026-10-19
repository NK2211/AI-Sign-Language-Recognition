import matplotlib.pyplot as plt

models = [
    "Spatial\nBiGRU",
    "Geometric\nGRU",
    "Fusion\n(No Attention)",
    "Fusion +\nAttention",
    "Fusion + Attention\n+ Skip",
    "Proposed\nGrouped"
]

accuracy = [
    100.00,
    95.00,
    99.64,
    100.00,
    100.00,
    99.64
]

plt.figure(figsize=(12, 7))

bars = plt.bar(models, accuracy)

plt.ylabel("Accuracy (%)")
plt.xlabel("Model")
plt.title("Model Performance Comparison")
plt.ylim(90, 101)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.1,
        f"{value:.2f}%",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

output = r"D:\AI-Sign-Language-Recognition\models\model_comparison.png"

plt.savefig(output, dpi=300, bbox_inches="tight")
plt.show()

print("Saved to:")
print(output)