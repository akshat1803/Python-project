# Report Outline: Student Exam Score Prediction

1. **Problem**
   - Goal: predict final exam score from study/behavior features.
   - Why it matters (early intervention, study habit insights).

2. **Data**
   - Source: Kaggle "Student Performance Factors" (mirrored on GitHub).
   - Size: 6,607 rows, 20 columns; 4 features used (Hours_Studied, Attendance, Sleep_Hours, Previous_Scores).
   - Missing data: only in unused categorical columns.

3. **EDA**
   - Target distribution (near-normal, slight right tail of outliers).
   - Feature-target relationships (Attendance and Hours_Studied strong; Sleep_Hours negligible).
   - Correlation heatmap (no multicollinearity among features).

4. **Models**
   - Baseline: mean predictor.
   - Linear Regression: coefficients and interpretation.
   - Decision Tree: feature importances.
   - Random Forest: feature importances.
   - XGBoost: feature importances.
   - Why Linear Regression won here (near-linear relationships, small feature set, tree-based models overfit/underfit on so few features).

5. **Results table**
   - MAE / RMSE / R² for all five models (see README).

6. **Feature importance**
   - Consistent ranking across all tree-based models: Attendance > Hours_Studied > Previous_Scores > Sleep_Hours.

7. **Limitations**
   - Single 80/20 split, no cross-validation.
   - Only 4 of 20 available columns used.
   - Dataset may not generalize beyond its source population.

8. **Future work**
   - Cross-validation, hyperparameter tuning, richer feature set.
