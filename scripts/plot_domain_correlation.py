import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load domain invariance scores
domain_df = pd.read_csv("results/ssl_domain.csv")  # mark_id, ssl_domain

# Load few-shot scores
fewshot_df = pd.read_csv("results/product_per_class.csv")  # mark_id, fewshot_acc

# Merge on mark_id
merged = pd.merge(domain_df, fewshot_df, on="mark_id")

print(merged.head())

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(merged["ssl_domain"], merged["fewshot_acc"], alpha=0.7)
plt.xlabel("SSL Domain Invariance")
plt.ylabel("Few-Shot Accuracy")
plt.title("Domain Invariance vs Few-Shot Accuracy")

# Pearson correlation
r, p = pearsonr(merged["ssl_domain"], merged["fewshot_acc"])
print(f"[PRODUCT] Domain-FewShot Pearson r: {r:.4f} (p = {p:.4e})")

# Save plot
plt.savefig("results/domain_vs_fewshot.png")
print("✅ Saved: results/domain_vs_fewshot.png")

# Save merged data + correlation to CSV
merged["pearson_r"] = r  # same r for all rows, for reference if needed
merged["pearson_p"] = p
merged.to_csv("results/domain_vs_fewshot.csv", index=False)
print("✅ Saved merged correlation CSV: results/domain_vs_fewshot.csv")
