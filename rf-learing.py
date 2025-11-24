import numpy as np
import pandas as pd
import random
from sklearn.datasets import make_classification

X,y = make_classification(n_features=5,n_redundant=0,n_informative=5,n_clusters_per_class=1)

df = pd.DataFrame(X,columns=['col1','col2','col3','col4','col5'])
df['target'] = y

# function for row sampling
def samples_row(df,percent):
    return df.sample(int(percent*df.shape[0]),replace=True)


def samples_features(df,percent):
    cols = random.sample(df.columns.tolist()[:-1],int(percent*df.shape[1]-1))
    new_df = df[cols]
    new_df['target'] = df['target']
    return new_df

def combined_sampling(df,row_percent,col_percent):
    new_df = samples_row(df,row_percent)
    return samples_features(new_df,col_percent)

df1 = samples_row(df,0.2)
df2 = samples_row(df,0.2)
df3 = samples_row(df,0.2)

from sklearn.tree import DecisionTreeClassifier

cl1 = DecisionTreeClassifier()
cl2 = DecisionTreeClassifier()
cl3 = DecisionTreeClassifier()

cl1.fit(df1.iloc[:,0:5],df1.iloc[:,-1])
cl2.fit(df2.iloc[:,0:5],df2.iloc[:,-1])
cl3.fit(df3.iloc[:,0:5],df3.iloc[:,-1])



from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(10,8))
plot_tree(cl1, filled=True, feature_names=['col1', 'col2', 'col3', 'col4', 'col5'], class_names=['0', '1'])
plt.title('Decision Tree 1')
# plt.show()

# Predict with all 5 features
prediction = cl1.predict(np.array([[-0.943985, -1.727474, -1.727474, 0.5, 0.3]]).reshape(1,5))
print(f"Prediction: {prediction}")
print(f"Prediction class: {prediction[0]}")