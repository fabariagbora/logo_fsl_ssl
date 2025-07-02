import os

# 📌 Read actual results from files
register_result = open("results/eval_register.txt").read().strip()
product_result = open("results/eval_product.txt").read().strip()
ssl_rotation = open("results/rotation_validator.txt").read().strip()
ssl_domain = open("results/domain_invariance.txt").read().strip()

# 📌 Extract numeric values for drop calculation
# Assuming format: "Average Accuracy: 85.94%"
register_acc = float(register_result.split(":")[1].strip().replace("%", ""))
product_acc = float(product_result.split(":")[1].strip().replace("%", ""))

drop = register_acc - product_acc

# 📌 Build lines
lines = [
    "===== FEW-SHOT LOGO RECOGNITION RESULTS =====\n",
    f"Register Domain:\n  - {register_result}\n",
    f"Product Domain:\n  - {product_result}\n",
    f"Difference:\n  - Drop: {drop:.2f} percentage points\n",
    "----------------------------------------------\n",
    f"N-Way: 5, K-Shot: 5, M-Query: 5\n",
    f"Episodes: 1000 (train), 200 (eval)\n",
    "Pretrained Backbone: ResNet-18\n",
    "\n===== SELF-SUPERVISED VALIDATION RESULTS =====\n",
    f"Rotation Invariance:\n  - {ssl_rotation}\n",
    f"Domain Invariance:\n  - {ssl_domain}\n",
]

# 📌 Save to summary.txt
os.makedirs("results", exist_ok=True)
with open("results/summary.txt", "w") as f:
    f.writelines(lines)

print("Results summary saved to results/summary.txt")
