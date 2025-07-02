import matplotlib.pyplot as plt

# Replace with your numbers
domains = ["Register", "Product"]
accuracies = [85.94, 81.82]

plt.figure(figsize=(6, 4))
bars = plt.bar(domains, accuracies, color=["steelblue", "seagreen"])
plt.ylim(0, 100)
plt.ylabel("Few-Shot Accuracy (%)")
plt.title("Few-Shot Logo Recognition: Register vs Product")

# Add value labels on bars
for bar, acc in zip(bars, accuracies):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{acc:.2f}%", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("results/register_vs_product.png")
print("Plot saved to results/register_vs_product.png")
plt.show()
