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
import re

import time

from IPython.display import Image


# %%
Train_data = pd.read_csv('./datalab/car_info_train.csv')
Train_data.info()


# %%
Train_data['CUST_AGE'] = Train_data.CUST_AGE.fillna(Train_data.CUST_AGE.mean()) #均值填充

# 分类
Train_data['CUST_AGE'] = Train_data.CUST_AGE.apply(lambda x: 1 if (x<=35 and x >=16) else 2 if (x >= 36 and x <= 60) else 3)
Train_data['CAR_AGE'] = Train_data.CAR_AGE.apply(lambda x: 1 if (x <=730) else 2 if (x >=731 and x <= 1460) else 3 if (x >= 1461 and x <= 2190) else 4 if (x >= 2191 and x <= 3650) else 5)
Train_data['CAR_PRICE'] = Train_data.CAR_PRICE.apply(lambda x: 1 if (x >=50000 and x <= 90000) else 2 if (x >= 90001 and x<= 150000) else 3 if (x >= 150001 and x <= 300000) else 4)
Train_data['LOAN_AMOUNT'] = Train_data.LOAN_AMOUNT.apply(lambda x: 1 if (x <= 50000) else 2 if(x >=50001 and x <= 200000) else 3 if(x >= 200001 and x <= 500000) else 4)

Train_data['CAR_MODEL'] = Train_data.CAR_MODEL.apply(lambda x: re.findall('\d', x)[0]).astype(int) #转换 int

# %%
Train_data.head()

# %%
# 获取数字特征列
numerical_cols = Train_data.select_dtypes(exclude = 'object').columns
print(numerical_cols)



# %%
# 空值填充（0）
Train_data.fillna(value=0,inplace=True)


# %%
# 判断空值
Train_data.isnull().any()


# %%
feature_cols = [col for col in numerical_cols if col != 'IS_LOST']
data = Train_data[feature_cols].fillna(value=0)
target = Train_data['IS_LOST']
data.head()


# %%
befordtc = DTC(criterion='entropy')
befordtc.fit(data, target)
print('准确率:', befordtc.score(data, target))
dtc = DTC(criterion='entropy',max_depth=4, max_leaf_nodes=5)
dtc.fit(data, target)
print('准确率:', dtc.score(data, target))


# %%
dot_data = tree.export_graphviz(dtc, out_file=None, 
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png()) 


# %%
with open('./tree.dot', 'w') as f:
    f = export_graphviz(befordtc, feature_names=data.columns, out_file=f)
graph = pydotplus.graph_from_dot_file('./tree.dot')
#graph.write_pdf("iris.pdf")



# %%
