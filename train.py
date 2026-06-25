import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv").squeeze()
y_test = pd.read_csv("data/y_test.csv").squeeze()

FEATURES = list(X_train.columns)


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


results = []

# Baseline
mean_score = y_train.mean()
y_pred_baseline = np.full(shape=len(y_test), fill_value=mean_score)
results.append(evaluate("Baseline (mean)", y_test, y_pred_baseline))

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)
results.append(evaluate("Linear Regression", y_test, y_pred_lin))

print("\nLinear Regression coefficients:")
for feature, coef in zip(FEATURES, lin_reg.coef_):
    print(f"  {feature}: {coef:.4f}")
print(f"  Intercept: {lin_reg.intercept_:.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
results.append(evaluate("Random Forest", y_test, y_pred_rf))

print("\nRandom Forest feature importances:")
for feature, importance in sorted(
    zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1]
):
    print(f"  {feature}: {importance:.4f}")

# Comparison table
results_df = pd.DataFrame(results)
print("\n=== Model Comparison ===")
print(results_df.to_string(index=False))

# Save the best model (by R2)
best_row = results_df.loc[results_df["R2"].idxmax()]
best_name = best_row["Model"]
best_model = {"Linear Regression": lin_reg, "Random Forest": rf}.get(best_name, lin_reg)

joblib.dump(best_model, "model.pkl")
joblib.dump(FEATURES, "features.pkl")
print(f"\nSaved best model ({best_name}) to model.pkl")
