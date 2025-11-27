# California Housing Progress Log

## Project Overview
the goal is to predict California district median house values and compare linear vs non-linear approaches, with Random Forest as the main production-ready model. The pipeline covers:
- **Problem framing:** tabular features (location, housing age, rooms/bedrooms, population, households, median income, ocean proximity) → forecast `MedHouseVal`.
- **Motivation:** better guidance for buyers/sellers, real-estate agents, policy makers, and a learning exercise in interpreting linear vs ensemble models.
- **Core scope:** EDA with multicollinearity checks, StandardScaler preprocessing, Linear Regression baseline, Random Forest model, residual plots, and RMSE/R² comparison.
- **Stretch goals:** Lasso for feature selection, geographic visualizations, and Random Forest hyperparameter tuning.
- **Success criteria:** notebook runs start-to-finish, multiple visualizations, Random Forest test R² > 0.75, insights on top price drivers. 

## Day 1: EDA & Visualization
- Read in the classic California housing dataset and ran the usual dataset info / describe routines.
- Checked for missing values and printed a small summary table.
- Built quick plots to understand the target: histogram + box plot for `MedHouseVal`, correlation heatmap, and a longitude/latitude scatter colored by price.
- Saved the visual outputs into the `plots/` folder for later reference.

## Day 2: Preprocessing & Baseline Models
- Split features/target, followed by a train/test split (`test_size=0.2`, `random_state=42`).
- Applied `StandardScaler` to the numeric (non-target) columns and logged the before/after stats to confirm everything was centered and scaled.
- Wrote a small `evaluate_model` helper that returns MAE, RMSE, and R² for any predictions.
- Trained three baseline models (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`) and printed train/test metrics using the helper.

## Day 3: Log Features & Hyperparameter Tuning
- Identified skewed continuous columns (`MedInc`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`) and added log-transformed counterparts (`log_<feature>`) via `np.log1p`.
- Generated comparison plots to visualize the effect of the log transform and stored them alongside the other figures in `plots/`.
- Ran additional hyperparameter tuning experiments (starting from the baseline models) to see how the new features and tuned parameters impact accuracy.

## Day 4: (pending)
- 

## Day 5: Full Dataset + Categorical Encoding
- Swapped in the complete Kaggle housing dataset (now includes `OceanProximity`) and renamed columns to keep the existing code readable.
- Patched the lone missing column (`TotalBedrooms`) via median imputation so no rows were dropped.
- Updated the log-transform list for the new feature names, producing fresh `log_` columns for `MedInc`, `TotalRooms`, `TotalBedrooms`, `Population`, and `Households`.
- One-hot encoded `OceanProximity` with `pd.get_dummies(..., drop_first=True)` so every model gets access to coastal vs inland signals without multicollinearity.
