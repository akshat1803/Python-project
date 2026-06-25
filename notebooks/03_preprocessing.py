import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/StudentPerformanceFactors.csv")

FEATURES = ["Hours_Studied", "Attendance", "Sleep_Hours", "Previous_Scores"]
TARGET = "Exam_Score"

X = df[FEATURES].copy()
y = df[TARGET].copy()

# No missing values in our 4 chosen numeric features (the missing data
# lives only in categorical columns we aren't using), but we guard anyway.
print("Missing values in selected features:\n", X.isnull().sum())
X = X.fillna(X.mean(numeric_only=True))

# No categorical columns among our 4 features, so no one-hot encoding needed
# here. (If you later add columns like Parental_Involvement, encode with
# pd.get_dummies(X, columns=[...], drop_first=True) before splitting.)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape)

X_train.to_csv("../data/X_train.csv", index=False)
X_test.to_csv("../data/X_test.csv", index=False)
y_train.to_csv("../data/y_train.csv", index=False)
y_test.to_csv("../data/y_test.csv", index=False)
print("Saved train/test splits to data/")
