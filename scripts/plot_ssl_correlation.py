import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

# ---------------------------------------------
# 1️⃣ Load CSVs
# ---------------------------------------------
ssl_df = pd.read_csv("results/ssl_rotation.csv")

product_df = pd.read_csv("results/product_per_class.csv")
register_df = pd.read_csv("results/register_per_class.csv")

# ---------------------------------------------
# 2️⃣ Merge on mark_id
# ---------------------------------------------
product_merged = product_df.merge(ssl_df, on="mark_id")
register_merged = register_df.merge(ssl_df, on="mark_id")

# ---------------------------------------------
# 3️⃣ Correlation + Trendline: PRODUCT
# ---------------------------------------------
prod_ssl = product_merged["ssl_rotation"]
prod_acc = product_merged["fewshot_acc"]

prod_corr, prod_pval = stats.pearsonr(prod_ssl, prod_acc)
print(f"✅ [PRODUCT] Pearson correlation: {prod_corr:.4f}")

plt.figure(figsize=(6, 5))
plt.scatter(prod_ssl, prod_acc, alpha=0.7)
m, b = np.polyfit(prod_ssl, prod_acc, 1)
plt.plot(prod_ssl, m * prod_ssl + b, color="red", label=f"r = {prod_corr:.2f}")
plt.xlabel("SSL Rotation Invariance")
plt.ylabel("Few-Shot Accuracy (Product)")
plt.title("Rotation Invariance vs Few-Shot Accuracy (Product)")
plt.legend()
plt.tight_layout()
plt.savefig("results/ssl_rotation_vs_fewshot_product.png")
plt.show()

# ---------------------------------------------
# 4️⃣ Correlation + Trendline: REGISTER
# ---------------------------------------------
reg_ssl = register_merged["ssl_rotation"]
reg_acc = register_merged["fewshot_acc"]

reg_corr, reg_pval = stats.pearsonr(reg_ssl, reg_acc)
print(f"✅ [REGISTER] Pearson correlation: {reg_corr:.4f}")

plt.figure(figsize=(6, 5))
plt.scatter(reg_ssl, reg_acc, alpha=0.7)
m, b = np.polyfit(reg_ssl, reg_acc, 1)
plt.plot(reg_ssl, m * reg_ssl + b, color="green", label=f"r = {reg_corr:.2f}")
plt.xlabel("SSL Rotation Invariance")
plt.ylabel("Few-Shot Accuracy (Register)")
plt.title("Rotation Invariance vs Few-Shot Accuracy (Register)")
plt.legend()
plt.tight_layout()
plt.savefig("results/ssl_rotation_vs_fewshot_register.png")
plt.show()

# ---------------------------------------------
# 5️⃣ Save correlation summary to CSV
# ---------------------------------------------
correlation_summary = pd.DataFrame([
    {"domain": "product", "metric": "rotation", "pearson_r": prod_corr, "p_value": prod_pval},
    {"domain": "register", "metric": "rotation", "pearson_r": reg_corr, "p_value": reg_pval}
])

correlation_summary.to_csv("results/ssl_rotation_correlations.csv", index=False)
print("✅✅✅ Saved correlation summary: results/ssl_rotation_correlations.csv")
print("✅✅✅ Done! Both plots + CSV saved in results/")
