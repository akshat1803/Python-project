import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("../data/StudentPerformanceFactors.csv")

# 1. Histogram of the target
plt.figure(figsize=(8, 5))
sns.histplot(df["Exam_Score"], kde=True, bins=30)
plt.title("Distribution of Exam Score")
plt.xlabel("Exam Score")
plt.savefig("hist_exam_score.png", dpi=120, bbox_inches="tight")
plt.close()

# 2. Scatter plots of key features vs score
key_features = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores"]
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, feature in zip(axes.flatten(), key_features):
    sns.scatterplot(data=df, x=feature, y="Exam_Score", alpha=0.3, ax=ax)
    ax.set_title(f"{feature} vs Exam Score")
plt.tight_layout()
plt.savefig("scatter_features_vs_score.png", dpi=120, bbox_inches="tight")
plt.close()

# 3. Correlation heatmap (numeric columns only)
numeric_df = df.select_dtypes(include="number")
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap (numeric features)")
plt.savefig("correlation_heatmap.png", dpi=120, bbox_inches="tight")
plt.close()

print("Saved: hist_exam_score.png, scatter_features_vs_score.png, correlation_heatmap.png")
print("\nCorrelation with Exam_Score:\n", numeric_df.corr()["Exam_Score"].sort_values(ascending=False))
