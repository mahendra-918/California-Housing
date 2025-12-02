# California Housing Price Prediction - Presentation Outline

## Slide 1: Title Slide
**Title:** California Housing Price Prediction Using Machine Learning
**Subtitle:** Comparing Linear vs Non-Linear Regression Models
**Your Name:** [Your Name]
**Date:** [Date]
**Course/Project:** [Course Name]

---

## Slide 2: Agenda
- Project Overview
- Problem Statement
- Dataset Description
- Exploratory Data Analysis
- Data Preprocessing & Feature Engineering
- Model Selection & Training
- Results & Evaluation
- Feature Importance Analysis
- Key Insights & Conclusions

---

## Slide 3: Project Overview
**Objective:**
- Predict median house values in California districts
- Compare linear (Linear Regression, Lasso) vs non-linear (Random Forest) models
- Identify key features driving house prices

**Dataset:**
- 20,640 housing districts
- 9 features + 1 target variable
- Source: California Housing Dataset (Kaggle)

**Approach:**
- Complete ML pipeline from EDA to model evaluation
- Multiple regression models with hyperparameter tuning
- Comprehensive residual analysis and feature importance

---

## Slide 4: Problem Statement
**Why This Problem Matters:**
- Real estate is a major economic indicator
- Accurate price prediction helps:
  - Buyers/Sellers make informed decisions
  - Real estate agents price properties competitively
  - Policy makers understand housing affordability

**Challenge:**
- Multiple factors influence house prices
- Non-linear relationships between features
- Need to balance accuracy vs interpretability

---

## Slide 5: Dataset Description
**Features:**
- **Location:** Longitude, Latitude, Ocean Proximity
- **Housing Characteristics:** House Age, Total Rooms, Total Bedrooms
- **Demographics:** Population, Households
- **Economic:** Median Income

**Target Variable:**
- Median House Value (in $100,000s)

**Dataset Statistics:**
- 20,640 samples
- 1 missing value column (TotalBedrooms: 207 missing)
- Mix of numeric and categorical features

---

## Slide 6: Exploratory Data Analysis - Key Findings
**Distribution Analysis:**
- House prices show right-skewed distribution
- Most houses in lower price range ($100k-$300k)
- Few expensive outliers ($500k+)

**Geographic Patterns:**
- Coastal areas (longitude ~-122) have higher prices
- Clear spatial clustering of price ranges
- Location is a strong predictor

**Correlation Insights:**
- Median Income has strongest correlation with price
- Latitude/Longitude show geographic price patterns
- Some multicollinearity between related features

---

## Slide 7: Data Preprocessing - Missing Values
**Problem:**
- `TotalBedrooms` had 207 missing values (1% of data)

**Solution:**
- **Median Imputation** (not mean)
- Why median? Robust to outliers, better for skewed data
- Preserved all 20,640 samples

**Result:**
- No data loss
- Maintained data distribution characteristics

---

## Slide 8: Feature Engineering - Log Transformation
**Why Log Transform?**
- Features like MedInc, TotalRooms, Population showed heavy right-skewness
- Long tail of extreme values affecting model performance

**Features Transformed:**
- `log_MedInc`, `log_TotalRooms`, `log_TotalBedrooms`
- `log_Population`, `log_Households`

**Benefits:**
- Compressed outliers
- Made distributions more bell-shaped
- Improved model performance for both linear and tree models

---

## Slide 9: Feature Engineering - Categorical Encoding
**Problem:**
- `OceanProximity` was categorical text (e.g., "NEAR BAY", "INLAND")
- Models need numeric input

**Solution:**
- **One-Hot Encoding** using `pd.get_dummies()`
- Created binary columns: `OceanProximity_INLAND`, `OceanProximity_NEAR BAY`, etc.
- Used `drop_first=True` to avoid multicollinearity

**Result:**
- Converted categorical to numeric format
- Models can now use location information

---

## Slide 10: Feature Scaling
**Method:** StandardScaler
- Transforms features to mean=0, std=1
- Formula: (x - mean) / std

**Why Important:**
- Linear models (Linear Regression, Lasso) require scaling
- Gradient descent converges faster
- Prevents features with larger scales from dominating

**Applied to:**
- All numeric features (original + log-transformed)
- Both training and test sets

---

## Slide 11: Model Selection - Three Approaches
**1. Linear Regression**
- Simple baseline model
- Assumes linear relationships
- Highly interpretable

**2. Lasso Regression**
- Linear model with L1 regularization
- Automatic feature selection (sets coefficients to zero)
- More interpretable than standard linear regression

**3. Random Forest**
- Ensemble of decision trees
- Captures non-linear relationships
- Handles feature interactions automatically
- Provides feature importance scores

---

## Slide 12: Hyperparameter Tuning
**Method:** RandomizedSearchCV
- Faster than exhaustive grid search
- Samples random combinations from parameter space

**Random Forest Parameters Tuned:**
- `n_estimators`: 100-1000 (number of trees)
- `max_depth`: 5-30, None (tree depth)
- `min_samples_split`: 2-20 (minimum samples to split)
- `max_features`: 5-8, 'auto' (features per split)

**Cross-Validation:**
- 3-fold CV for reliable performance estimates
- Evaluated 10 random combinations

---

## Slide 13: Evaluation Metrics
**RMSE (Root Mean Squared Error)**
- Average prediction error in same units ($100,000s)
- Penalizes large errors more heavily
- Formula: √(Σ(y_true - y_pred)² / n)

**MAE (Mean Absolute Error)**
- Average absolute difference
- Easier to interpret
- Formula: Σ|y_true - y_pred| / n

**R² Score (Coefficient of Determination)**
- Proportion of variance explained
- Range: 0 to 1 (higher is better)
- Formula: 1 - (SS_res / SS_tot)

---

## Slide 14: Model Performance Results
**Best Model: Random Forest**
- Test R²: [Your actual value, e.g., 0.75+]
- Test RMSE: [Your actual value]
- Test MAE: [Your actual value]

**Comparison:**
- Random Forest > Lasso > Linear Regression
- Random Forest captures non-linear patterns better
- Lasso provides good interpretability with feature selection

**Key Observation:**
- All models benefited from feature engineering
- Log transforms and encoding improved performance

---

## Slide 15: Residual Analysis
**Predicted vs Actual Plot:**
- Random Forest: Points tightly clustered around diagonal
- Linear Regression: More spread, showing limitations

**Residuals vs Predicted Plot:**
- Random Forest: Random scatter around zero (good)
- Linear Regression: Slight patterns (heteroscedasticity)

**Insights:**
- Random Forest captures non-linear relationships well
- Linear models struggle with complex patterns
- Residual plots confirm model assumptions

---

## Slide 16: Feature Importance Analysis
**Top 5 Most Important Features:**
1. **Median Income** - Strongest predictor
2. **Latitude** - Geographic location matters
3. **Longitude** - Coastal vs inland pricing
4. **Ocean Proximity** - Location premium
5. **Total Rooms** - Housing density indicator

**Key Finding:**
- Economic factors (income) + Location dominate
- Housing characteristics (age, bedrooms) less important
- Confirms real-world intuition about real estate

---

## Slide 17: Key Insights
**1. Model Performance:**
- Random Forest outperforms linear models
- Non-linear relationships are important for price prediction

**2. Feature Importance:**
- Income and location are strongest predictors
- Geographic features (lat/long/ocean proximity) crucial

**3. Preprocessing Impact:**
- Log transforms significantly improved all models
- Proper encoding and scaling essential for good results

**4. Trade-offs:**
- Random Forest: Better accuracy, less interpretable
- Lasso: Good balance of accuracy and interpretability
- Linear Regression: Most interpretable, lower accuracy

---

## Slide 18: Challenges Faced
**1. Missing Values**
- Solution: Median imputation (robust to outliers)

**2. Skewed Distributions**
- Solution: Log transformation

**3. Categorical Data**
- Solution: One-hot encoding

**4. Model Selection**
- Solution: Compare multiple models with same preprocessing

**5. Hyperparameter Tuning**
- Solution: RandomizedSearchCV for efficient search

---

## Slide 19: Future Improvements
**Model Enhancements:**
- Try Gradient Boosting (XGBoost, LightGBM)
- Experiment with Neural Networks
- Ensemble multiple models

**Feature Engineering:**
- Create ratio features (rooms per household)
- Add interaction features (income × location)
- Derive distance features (distance to coast)

**Deployment:**
- Create web app (Streamlit/Flask)
- Save models for inference
- Build prediction API

---

## Slide 20: Conclusion
**Project Achievements:**
- Successfully predicted house prices with good accuracy
- Compared multiple ML approaches
- Identified key price drivers
- Demonstrated complete ML pipeline

**Key Learnings:**
- Data preprocessing is crucial
- Non-linear models outperform linear for this problem
- Feature engineering significantly impacts performance
- Proper evaluation (residuals, metrics) essential

**Real-World Application:**
- Can help buyers/sellers make informed decisions
- Useful for real estate market analysis
- Demonstrates practical ML application

---

## Slide 21: Thank You
**Questions?**

**Contact:**
- [Your Email]
- [GitHub/Portfolio Link]

**Resources:**
- Notebook: `notebooks/housing-analysis.ipynb`
- Code: `housing-analysis.py`
- Dataset: California Housing (Kaggle)

---

## Presentation Tips:

1. **Visuals to Include:**
   - Screenshot of correlation heatmap
   - Geographic scatter plot
   - Residual plots (Predicted vs Actual, Residuals vs Predicted)
   - Feature importance bar chart
   - Model comparison table

2. **Keep Slides Simple:**
   - Max 5-6 bullet points per slide
   - Use large, readable fonts
   - Include visuals where possible

3. **Practice Points:**
   - Explain why you chose median over mean
   - Explain why log transform helps tree models
   - Be ready to explain Random Forest hyperparameters
   - Know your model performance numbers

4. **Time Management:**
   - 15-20 slides for 10-15 minute presentation
   - Spend more time on results and insights
   - Keep EDA and preprocessing concise

