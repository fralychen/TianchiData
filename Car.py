# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
import pydot
import pydotplus

import time


# %%
# open The csv file
Train_data = pd.read_csv('./data/car_info_train.csv')
Train_data.info()


# %%
# include all num columns
numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
print(numerical_cols)


# %%
# separlit data
Train_data['CUST_AGE'] = pd.qcut(Train_data.CUST_AGE, 3, labels=["1", "2", "3"])
Train_data['CAR_AGE'] = pd.qcut(Train_data.CAR_AGE, 5, labels=["1", "2", "3", "4", "5"])
Train_data['CAR_PRICE'] = pd.qcut(Train_data.CAR_PRICE, 4, labels=["1", "2", "3", "4"])
Train_data['LOAN_AMOUNT'] = pd.qcut(Train_data.LOAN_AMOUNT, 4, labels=["1", "2", "3", "4"])


# %%
# out 5 Rows
Train_data.head()


# %%
# check null
Train_data.isnull().any()


# %%
# select cols exclude 'IS_LOST' as X_data, IS_LOST cloums as Y_data
feature_cols = [col for col in numerical_cols if col != 'IS_LOST']
X_data = Train_data[feature_cols]
Y_data = Train_data['IS_LOST']
X_data


# %%

dtc = DTC(criterion='entropy')
dtc.fit(X_data, Y_data)
print('准确率:', dtc.score(X_data, Y_data))
with open('./tree.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=X_data.columns, out_file=f)
