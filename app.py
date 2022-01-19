#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
from flask import Flask, jsonify, render_template, request
import pickle


# In[9]:


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))


# In[10]:


@app.route('/')
def home():
    return render_template('index.html')


# In[11]:


@app.route('/predict', methods=['POST'])
def predict():
    in_features = [float(x) for x in request.form.values()]
    final_features = [np.array(in_features)]
    prediction = model.predict(final_features)
    res = ''
    if(prediction==0):
        res = 'NO'
    else:
        res = 'YES'
    prob = model.predict_proba(final_features)
    probable_percent = prob[:1 , 1:][0][0]*100
    
    report = {"prediction of heart disease : " : res,
              "probaility of truth : " : probable_percent }
    return render_template('index.html', prediction_text = report) 

if __name__ == "__main__":
    app.run(debug=True)


# In[12]:


get_ipython().run_line_magic('tb', '')


# In[ ]:




