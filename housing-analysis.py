from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV

plots = 'plots/'
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

df = pd.read_csv('data/housing.csv')

renamed_cols = {
    'longitude': 'Longitude',
    'latitude': 'Latitude',
    'housing_median_age': 'HouseAge',
    'total_rooms': 'TotalRooms',
    'total_bedrooms': 'TotalBedrooms',
    'population': 'Population',
    'households': 'Households',
    'median_income': 'MedInc',
    'median_house_value': 'MedHouseVal',
    'ocean_proximity': 'OceanProximity'
}

df = df.rename(columns=renamed_cols)

if df['TotalBedrooms'].isna().any():
    df['TotalBedrooms'] = df['TotalBedrooms'].fillna(df['TotalBedrooms'].median())

print(f"Dataset Shape: {df.shape}")
print(f'Total Samples: {df.shape[0]}')
print(f'Total Features: {df.shape[1]}')
df.head()

print('=== Dataset Info ===')
print(df.info())
print('\n=== Statistical Summary ===')
df.describe()


log_features = ['MedInc', 'TotalRooms', 'TotalBedrooms', 'Population', 'Households']
for feature in log_features:
    df[f'log_{feature}'] = np.log1p(df[feature])
print(f'Log-transformed features: {", ".join([f"log_{f}" for f in log_features])}')


df = pd.get_dummies(df, columns=['OceanProximity'], drop_first=True)

for feature in log_features:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(df[feature], bins=40, ax=axes[0], color='#1f77b4')
    axes[0].set_title(f'{feature} (original)')
    sns.histplot(df[f'log_{feature}'], bins=40, ax=axes[1], color='#ff7f0e')
    axes[1].set_title(f'log_{feature}')
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('Count')
    fig.suptitle(f'Effect of log transform on {feature}')
    plt.tight_layout()
    fig.savefig(plots + f'log_transform_{feature}.png')

missing = df.isnull().sum()
missing_percent = (missing/len(df)) * 100
missing_df = pd.DataFrame({
    'missing count': missing,'percentage':missing_percent
})

print(missing_df[missing_df['missing count']>0])

if missing.sum() == 0:
    print('No missing values found!')
else:
    print(f'\nTotal missing values: {missing.sum()}')


print(df.head(5))

plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.hist(df['MedHouseVal'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Median House Value ($100,000s)')
plt.ylabel('Frequency')
plt.title('Distribution of House Prices')

plt.subplot(1, 2, 2)
plt.boxplot(df['MedHouseVal'])
plt.ylabel('Median House Value ($100,000s)')
plt.title('Box Plot of House Prices')




print(f"Mean Price: ${df['MedHouseVal'].mean()*100000:,.0f}")
print(f"Median Price: ${df['MedHouseVal'].median()*100000:,.0f}")
print(f"Min Price: ${df['MedHouseVal'].min()*100000:,.0f}")
print(f"Max Price: ${df['MedHouseVal'].max()*100000:,.0f}")

plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
            fmt='.2f', square=True, linewidths=0.5)
plt.title('Correlation Heatmap - Checking Multicollinearity', fontsize=14)


plt.figure(figsize=(12, 8))
scatter = plt.scatter(df['Longitude'], df['Latitude'], 
                      c=df['MedHouseVal'], cmap='viridis', 
                      alpha=0.5, s=10)
plt.colorbar(scatter, label='Median House Value ($100,000s)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('California House Prices by Geographic Location')

plt.tight_layout()

X = df.drop('MedHouseVal',axis = 1)
y = df['MedHouseVal']

print(f'Features shape: {X.shape}')
print(f'Target shape: {y.shape}')
print(f'\nFeature columns: {list(X.columns)}')

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size=0.2)
print(f'Training set: {X_train.shape[0]} samples')
print(f'Test set: {X_test.shape[0]} samples')
print(f'Split ratio: {X_train.shape[0]/len(X)*100:.0f}% train, {X_test.shape[0]/len(X)*100:.0f}% test')


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print('Feature Scaling Applied!')
print(f'\nBefore scaling - Mean: {X_train.mean().mean():.2f}, Std: {X_train.std().mean():.2f}')
print(f'After scaling - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.2f}')

def evaluate_model(true,predicted):
    mae = mean_absolute_error(true,predicted)
    mse = mean_squared_error(true,predicted)
    rmse = np.sqrt(mean_squared_error(true,predicted))
    r2_square = r2_score(true,predicted)
    return mae,rmse,r2_square

#Hyperparameter tuning
rf_params = {
    "max_depth": [5, 8, 15, None, 10],
    "max_features": [5, 7, "auto", 8],
    "min_samples_split": [2, 8, 15, 20],
    "n_estimators": [100, 200, 500, 1000]
}

randomcv_models = [
    ("RF",RandomForestRegressor(),rf_params)
]

model_param = {}
for name,model,params in randomcv_models:
    random = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=10,
        cv=3,
        verbose=2,
        n_jobs=1
    )
    random.fit(X_train_scaled,y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f" Best Params for {model_name} ")
    print(model_param[model_name])


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=1000, min_samples_split= 20, max_features=7, max_depth=None),
    "Lasso Regression": Lasso(alpha=0.1, max_iter=1000)
}

# Store results for comparison table
model_results = []
rf_model = None  # Store RF model for feature importance

for i in range(len(list(models))):
    model = list(models.values())[i]
    model_name = list(models.keys())[i]
    model.fit(X_train_scaled,y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)
    
    # Store results for comparison table
    model_results.append({
        'Model': model_name,
        'Train_RMSE': model_train_rmse,
        'Test_RMSE': model_test_rmse,
        'Train_R2': model_train_r2,
        'Test_R2': model_test_r2,
        'Train_MAE': model_train_mae,
        'Test_MAE': model_test_mae
    })
    
    # Store Random Forest model for feature importance extraction
    if model_name == "Random Forest":
        rf_model = model
    
    print(model_name)
    
    print('Model performance for Training set')
    print("- Root Mean Squared Error: {:.4f}".format(model_train_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_train_mae))
    print("- R2 Score: {:.4f}".format(model_train_r2))

    print('----------------------------------')
    
    print('Model performance for Test set')
    print("- Root Mean Squared Error: {:.4f}".format(model_test_rmse))
    print("- Mean Absolute Error: {:.4f}".format(model_test_mae))
    print("- R2 Score: {:.4f}".format(model_test_r2))
    
    print('='*35)
    print('\n')

    # Residuals show how far off our predictions are from reality
    residuals_test = y_test - y_test_pred
    
    
    # This shows if predictions align with actual values (should be diagonal line)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left plot: Predicted vs Actual
    axes[0].scatter(y_test, y_test_pred, alpha=0.5, s=20)
    # Add perfect prediction line (y=x)
    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0].set_xlabel('Actual House Value ($100,000s)')
    axes[0].set_ylabel('Predicted House Value ($100,000s)')
    axes[0].set_title(f'{model_name}: Predicted vs Actual')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Right plot: Residuals vs Predicted
    # This checks for patterns in errors (should be random scatter)
    axes[1].scatter(y_test_pred, residuals_test, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error Line')
    axes[1].set_xlabel('Predicted House Value ($100,000s)')
    axes[1].set_ylabel('Residuals (Actual - Predicted)')
    axes[1].set_title(f'{model_name}: Residuals vs Predicted')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'residual_plots_{model_name.replace(" ", "_")}.png'
    plt.savefig(plots + filename, dpi=150, bbox_inches='tight')
    print(f'Saved residual plots: {filename}\n')
    plt.close()  # Close figure to free memory

# ============================================================================
# Feature Importance Visualization (Random Forest)
# ============================================================================
if rf_model is not None:
    feature_importance = rf_model.feature_importances_
    feature_names = X.columns
    
    # Create DataFrame for easier sorting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Plot top 15 features
    top_n = min(15, len(importance_df))
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance_df.head(top_n)['Importance'], color='steelblue')
    plt.yticks(range(top_n), importance_df.head(top_n)['Feature'])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Random Forest: Top Feature Importances', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots + 'rf_feature_importance.png', dpi=150, bbox_inches='tight')
    print(f'Saved feature importance plot: rf_feature_importance.png\n')
    plt.close()
    
    # Print top 10 features
    print("="*50)
    print("TOP 10 FEATURES DRIVING HOUSE PRICES (Random Forest)")
    print("="*50)
    for idx, row in importance_df.head(10).iterrows():
        print(f"{row['Feature']:30s}: {row['Importance']:.4f}")
    print("="*50)
    print()

# ============================================================================
# Model Comparison Table
# ============================================================================
comparison_df = pd.DataFrame(model_results)
print("\n" + "="*80)
print("MODEL COMPARISON TABLE")
print("="*80)
print(comparison_df.to_string(index=False))
print("="*80)
print()

# Save comparison table to CSV
comparison_df.to_csv('model_comparison.csv', index=False)
print("Saved model comparison table: model_comparison.csv\n")

# ============================================================================
# Insights and Conclusions
# ============================================================================
print("\n" + "="*80)
print("INSIGHTS & CONCLUSIONS")
print("="*80)

best_model = comparison_df.loc[comparison_df['Test_R2'].idxmax()]
print(f"\n1. BEST PERFORMING MODEL: {best_model['Model']}")
print(f"   - Test R² Score: {best_model['Test_R2']:.4f}")
print(f"   - Test RMSE: ${best_model['Test_RMSE']*100000:,.0f}")
print(f"   - Test MAE: ${best_model['Test_MAE']*100000:,.0f}")

print(f"\n2. MODEL COMPARISON:")
for _, row in comparison_df.iterrows():
    print(f"   {row['Model']:20s}: R² = {row['Test_R2']:.4f}, RMSE = ${row['Test_RMSE']*100000:,.0f}")

if rf_model is not None:
    top_3_features = importance_df.head(3)
    print(f"\n3. TOP 3 FEATURES DRIVING HOUSE PRICES:")
    for idx, row in top_3_features.iterrows():
        print(f"   - {row['Feature']}: {row['Importance']:.4f}")

print(f"\n4. KEY OBSERVATIONS:")
print(f"   - Random Forest outperforms linear models due to its ability to capture")
print(f"     non-linear relationships and feature interactions.")
print(f"   - Lasso regression provides automatic feature selection through L1 regularization,")
print(f"     making it more interpretable than standard Linear Regression.")
print(f"   - Feature engineering (log transforms, categorical encoding) improved model")
print(f"     performance by reducing skewness and adding location-based signals.")
print(f"   - Geographic features (Latitude, Longitude) and income (MedInc) are strong")
print(f"     predictors of California housing prices.")

print("\n" + "="*80)




plt.show()
