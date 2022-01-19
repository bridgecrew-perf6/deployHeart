#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score

import pickle


# In[3]:


data=pd.read_csv("heart-disease.csv")


# In[4]:


X=data.drop("target",axis=1)
y=data["target"]

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2)


# In[5]:


np.random.seed(54)
model_clf=RandomForestClassifier(max_depth = 5, max_features = 'auto', min_samples_leaf = 3, min_samples_split = 8, n_estimators = 30)
model_clf.fit(X_train,y_train)
model_clf.score(X_test,y_test)


# In[6]:


model_cv = cross_val_score(model_clf, X, y, cv=5, scoring='accuracy')
model_cv.mean()


# In[7]:


pickle.dump(model_clf, open('model.pkl', 'wb'))


# In[10]:


y_preds = model_clf.predict(X_test)
y_preds


# In[18]:


arr = model_clf.predict_proba(X_test)
arr[:1,1:][0][0]*100


# In[6]:





# In[5]:





# In[3]:





# In[ ]:




