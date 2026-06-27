# Concrete Compressive Strength Prediction

Predicting concrete compressive strength from mix composition and age using regression models, with exploratory data analysis, model comparison, feature importance analysis, and hyperparameter tuning.

## Dataset

[Concrete_Data.xls](Concrete_Data.xls) — the UCI [Concrete Compressive Strength dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength), 1030 samples with 8 input features:

| Feature | Description |
|---|---|
| Cement | kg in a m³ mixture |
| BlastFurnaceSlag | kg in a m³ mixture |
| FlyAsh | kg in a m³ mixture |
| Water | kg in a m³ mixture |
| Superplasticizer | kg in a m³ mixture |
| CoarseAggregate | kg in a m³ mixture |
| FineAggregate | kg in a m³ mixture |
| Age | days (1–365) |

**Target:** `CompressiveStrength` (MPa)

## Approach

1. **EDA** — summary stats, missing-value check, correlation heatmap, distribution and scatter plots of strength against key features (cement, water, age, fly ash, superplasticizer).
2. **Preprocessing** — train/test split (80/20) and feature standardization with `StandardScaler`.
3. **Modeling** — compared five regressors: Decision Tree, Random Forest, Linear Regression, Lasso, and Ridge, scored on RMSE, MSE, MAE, and R².
4. **Feature importance** — built-in importances from Decision Tree and Random Forest, cross-checked with permutation importance.
5. **Tuning** — `RandomizedSearchCV` for Random Forest and Decision Tree hyperparameters, plus a Gradient Boosting comparison with k-fold cross-validation.

## Repo contents

| File | Description |
|---|---|
| `concrete_strength_final.ipynb` | Main analysis and modeling notebook |
| `Concrete_Data.xls` | Raw dataset |
| `model.pkl` | Final trained model (Gradient Boosting Regressor), pickled |
| `requirements.txt` | Python dependencies |


## Results

The final model is a **Gradient Boosting Regressor** (default parameters), achieving an **R² of 0.906** on the test set. It's saved as `model.pkl`:

> Note: inputs must be scaled the same way as during training (`StandardScaler`) before calling `predict`.
