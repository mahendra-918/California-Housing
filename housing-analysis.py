from tabnanny import verbose
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import tree
from sklearn.model_selection import RandomizedSearchCV

plots = 'plots/'
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('husl')

df = pd.read_csv('data/housing.csv')

print(f"Dataset Shape: {df.shape}")
print(f'Total Samples: {df.shape[0]}')
print(f'Total Features: {df.shape[1]}')
df.head()

print('=== Dataset Info ===')
print(df.info())
print('\n=== Statistical Summary ===')
df.describe()


log_features = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup']
for feature in log_features:
    df[f'log_{feature}'] = np.log1p(df[feature])
print(f'Log-transformed features: {", ".join([f"log_{f}" for f in log_features])}')

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

# model_param = {}
# for name,model,params in randomcv_models:
#     random = RandomizedSearchCV(
#         estimator=model,
#         param_distributions=params,
#         n_iter=10,
#         cv=3,
#         verbose=2,
#         n_jobs=1
#     )
#     random.fit(X_train_scaled,y_train)
#     model_param[name] = random.best_params_

# for model_name in model_param:
#     print(f"---------------- Best Params for {model_name} -------------------")
#     print(model_param[model_name])


models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, min_samples_split= 8, max_features=8, max_depth=None)
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train_scaled,y_train)
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    model_train_mae , model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae , model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)
    print(list(models.keys())[i])
    
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






# plt.show()
