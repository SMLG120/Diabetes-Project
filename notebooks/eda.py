"""
=============================================================================
VALENCIA — DATA ENGINEER: eda.py
=============================================================================
Exploratory Data Analysis for the Diabetes Prediction Challenge.
Run with:  python notebooks/eda.py
(Or copy cells into a Jupyter notebook: eda.ipynb)
=============================================================================
"""


# %% [markdown]
# # 🩺 Diabetes Prediction — Exploratory Data Analysis
# **Member 1 — Data Engineer**


# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set_theme(style="whitegrid", palette="muted")
os.makedirs("outputs/figures", exist_ok=True)


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# %% [markdown]
# ## 1. Load Data


# %%
train_df = pd.read_csv("data/train.csv")
test_df  = pd.read_csv("data/test.csv")


print("Train shape:", train_df.shape)
print("Test  shape:", test_df.shape)
print("\nColumn names:\n", train_df.columns.tolist())


# %% [markdown]
# ## 2. Basic Info


# %%
print("\n[dtypes]\n",         train_df.dtypes)
print("\n[Describe]\n",       train_df.describe().T)
print("\n[Missing values]\n", train_df.isnull().sum())
print("\n[Target dist]\n",    train_df["diagnosed_diabetes"].value_counts(normalize=True))


# %% [markdown]
# ## 3. Target Distribution


# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))


# Count plot
vc = train_df["diagnosed_diabetes"].value_counts()
axes[0].bar(["No Diabetes (0)", "Diabetes (1)"], vc.values,
            color=["#4C72B0", "#DD8452"], edgecolor="white", linewidth=0.8)
axes[0].set_title("Target Distribution (Counts)", fontsize=13)
axes[0].set_ylabel("Count")
for i, v in enumerate(vc.values):
    axes[0].text(i, v + 50, str(v), ha="center", fontsize=10)


# Pie chart
axes[1].pie(vc.values, labels=["No Diabetes", "Diabetes"],
            autopct="%1.1f%%", colors=["#4C72B0", "#DD8452"],
            startangle=90, wedgeprops={"edgecolor": "white"})
axes[1].set_title("Target Distribution (%)", fontsize=13)


plt.tight_layout()
fig.savefig("outputs/figures/target_distribution.png", dpi=150)
plt.close(fig)
print("[EDA] Target distribution saved.")


# %% [markdown]
# ## 4. Feature Distributions


# %%
num_cols = train_df.select_dtypes(include=[np.number]).columns.drop(["id", "diagnosed_diabetes"]).tolist()


n_cols = 3
n_rows = (len(num_cols) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()


for i, col in enumerate(num_cols):
    axes[i].hist(train_df[col].dropna(), bins=40, color="#4C72B0",
                 edgecolor="white", alpha=0.85)
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Count")


# Hide unused subplots
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)


plt.suptitle("Feature Distributions", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig("outputs/figures/feature_distributions.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[EDA] Feature distributions saved.")


# %% [markdown]
# ## 5. Feature Distributions by Target


# %%
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()


for i, col in enumerate(num_cols):
    for label, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        subset = train_df[train_df["diagnosed_diabetes"] == label][col].dropna()
        axes[i].hist(subset, bins=40, alpha=0.6, color=color,
                     label=f"{'No Diabetes' if label == 0 else 'Diabetes'}")
    axes[i].set_title(col, fontsize=11)
    axes[i].legend(fontsize=8)


for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)


plt.suptitle("Feature Distributions by Target Class", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig("outputs/figures/feature_by_target.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[EDA] Features by target saved.")


# %% [markdown]
# ## 6. Correlation Heatmap


# %%
corr_matrix = train_df[num_cols + ["diagnosed_diabetes"]].corr()


fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.5,
    square=True, ax=ax, cbar_kws={"shrink": 0.8}
)
ax.set_title("Feature Correlation Heatmap", fontsize=14)
plt.tight_layout()
fig.savefig("outputs/figures/correlation_heatmap.png", dpi=150)
plt.close(fig)
print("[EDA] Correlation heatmap saved.")


# Print top correlations with target
print("\n[Top correlations with diabetes target]")
print(corr_matrix["diagnosed_diabetes"].drop("diagnosed_diabetes").abs().sort_values(ascending=False))


# %% [markdown]
# ## 7. Boxplots by Target


# %%
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()


for i, col in enumerate(num_cols):
    sns.boxplot(
        data=train_df, x="diagnosed_diabetes", y=col,
        hue="diagnosed_diabetes",
        palette={0.0: "#4C72B0", 1.0: "#DD8452"},
        legend=False, ax=axes[i]
    )
    axes[i].set_xticklabels(["No Diabetes", "Diabetes"])
    axes[i].set_title(col, fontsize=11)


for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)


plt.suptitle("Boxplots: Feature vs Target", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig("outputs/figures/boxplots_by_target.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("[EDA] Boxplots saved.")


# %% [markdown]
# ## 8. Missing Value Heatmap


# %%
missing = train_df.isnull().sum()
missing = missing[missing > 0]


if len(missing) > 0:
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap(train_df[missing.index].isnull().T,
                cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Values Heatmap")
    plt.tight_layout()
    fig.savefig("outputs/figures/missing_values.png", dpi=150)
    plt.close(fig)
    print("[EDA] Missing values heatmap saved.")
else:
    print("[EDA] No missing values detected in training set.")


# %% [markdown]
# ## 9. Outlier Detection (IQR Method)


# %%
print("\n[Outlier summary (IQR method)]")
for col in num_cols:
    q1, q3 = train_df[col].quantile([0.25, 0.75])
    iqr     = q3 - q1
    lb, ub  = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_out   = ((train_df[col] < lb) | (train_df[col] > ub)).sum()
    pct     = 100 * n_out / len(train_df)
    print(f"  {col:<25}  {n_out:>5} outliers ({pct:.1f}%)")


# %% [markdown]
# ## 10. EDA Summary Report


# %%
print("\n" + "="*60)
print(" EDA SUMMARY REPORT")
print("="*60)
print(f" Train samples  : {len(train_df):,}")
print(f" Test  samples  : {len(test_df):,}")
print(f" Features       : {len(num_cols)}")
print(f" Missing values : {train_df.isnull().sum().sum()}")
class_balance = train_df["diagnosed_diabetes"].mean() * 100
print(f" Class balance  : {class_balance:.1f}% positive (diagnosed_diabetes=1)")
print(f" Imbalance      : {'Yes — use class_weight' if class_balance < 40 else 'Mild / balanced'}")
print("="*60)
print("\n[EDA] All figures saved to outputs/figures/")
print("[EDA] Ready for preprocessing pipeline.")
