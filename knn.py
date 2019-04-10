#%%
import numpy as np
import pandas as pd

import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import roc_curve, roc_auc_score, auc, mean_squared_error, confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, RandomizedSearchCV, train_test_split
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, VotingClassifier

def check_acc(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    print(accuracy_score(y_test, y_pred)*100)


#%%
filename = 'data/data3_M1.csv'
df = pd.read_csv(filename, parse_dates=['Datetime'], index_col='Datetime')

#%%
y = df.Target.values
X = df.drop('Target', axis=1).values

# split = int(len(df) * 0.60)
# X_train = X[:split]
# X_test = X[split:]
# y_train = y[:split]
# y_test = y[split:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, random_state=42)


#%%
poly = PolynomialFeatures()
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)


#%%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



#%%
knn = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn.fit(X_train, y_train)
# train 75%
# test 51%

#%%
knn_scaled = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
knn_scaled.fit(X_train_scaled, y_train)
# 
# 

#%%
params = {
    'n_neighbors': np.arange(1, 10),
    'n_jobs': [-1]
}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, params, cv=3, n_jobs=-1, verbose=True)
knn_cv.fit(X_train, y_train)
