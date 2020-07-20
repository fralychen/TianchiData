# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DTC, export_graphviz
import pydot
import pydotplus

import time

from IPython.display import Image


# %%
Train_data = pd.read_csv('./datalab/car_info_train.csv')
Train_data.info()


# %%
numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
print(numerical_cols)


# %%
Train_data['CUST_AGE'] = pd.qcut(Train_data.CUST_AGE, 3, labels=["1", "2", "3"])
Train_data['CAR_AGE'] = pd.qcut(Train_data.CAR_AGE, 5, labels=["1", "2", "3", "4", "5"])
Train_data['CAR_PRICE'] = pd.qcut(Train_data.CAR_PRICE, 4, labels=["1", "2", "3", "4"])
Train_data['LOAN_AMOUNT'] = pd.qcut(Train_data.LOAN_AMOUNT, 4, labels=["1", "2", "3", "4"])


# %%
Train_data.head()


# %%
Train_data.isnull().any()


# %%
feature_cols = [col for col in numerical_cols if col != 'IS_LOST']
data = Train_data[feature_cols]
target = Train_data['IS_LOST']
data


# %%
dtc = DTC(criterion='entropy', max_depth=4, max_leaf_nodes=5)
dtc.fit(data, target)
print('准确率:', dtc.score(data, target))


# %%
dot_data = tree.export_graphviz(dtc, out_file=None, 
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 


# %%
import pydot
import pydotplus
with open('./tree.dot', 'w') as f:
    f = export_graphviz(dtc, feature_names=data.columns, out_file=f)
graph = pydotplus.graph_from_dot_file('./tree.dot')
graph.write_pdf("iris.pdf")


# %%


