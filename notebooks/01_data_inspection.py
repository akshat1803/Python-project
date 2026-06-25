import pandas as pd

df = pd.read_csv("../data/StudentPerformanceFactors.csv")

print("Shape:", df.shape)
print("\nDtypes:\n", df.dtypes)
print("\nMissing values:\n", df.isnull().sum()[df.isnull().sum() > 0])
print("\nSummary stats (numeric):\n", df.describe())
print("\nHead:\n", df.head())
