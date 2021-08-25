#!/usr/bin/env python
# coding: utf-8

# # DIABETES PREDICTION

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


dataset = pd.read_csv("Diabities-210331-154610.csv")
dataset.head(10)


# In[3]:


dataset.isnull().sum()


# In[4]:


dataset.shape


# In[5]:


X = dataset.iloc[:767,1:9].values
y = dataset.iloc[:767,9].values
print(X)


# In[6]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 25,random_state = 0)


# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,r2_score,classification_report
logreg = LogisticRegression(solver='lbfgs',max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc_logreg1 = round(accuracy_score(y_pred, y_test) , 2)*100
print("Accuracy : ",acc_logreg1)
