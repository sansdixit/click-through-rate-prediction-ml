#!/usr/bin/env python
# coding: utf-8

# In[6]:


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
pio.templates.default = "plotly_white"

data = pd.read_csv("ad_10000records.csv")
print(data.head())


# In[2]:


data.describe()


# In[2]:


data.head()


# In[3]:


data.tail()


# In[4]:


data["Clicked on Ad"] = data["Clicked on Ad"].map({0: "No", 1: "Yes"})


# In[5]:


data["Clicked on Ad"].value_counts()


# In[10]:


click_through_rate = 4915 / 10000 * 100
print(click_through_rate)


# In[7]:


data["Gender"] = data["Gender"].map({"Male": 1, 
                               "Female": 0})

x=data.iloc[:,0:7]
x=x.drop(['Ad Topic Line','City'],axis=1)
y=data.iloc[:,9]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=4)


# In[8]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x, y)
y_pred = model.predict(xtest)


# In[9]:


from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,y_pred))


# In[10]:


print("Ads Click Through Rate Prediction : ")
a = float(input("Daily Time Spent on Site: "))
b = float(input("Age: "))
c = float(input("Area Income: "))
d = float(input("Daily Internet Usage: "))
e = input("Gender (Male = 1, Female = 0) : ")

features = np.array([[a, b, c, d, e]])
print("Will the user click on ad = ", model.predict(features))


# In[2]:


data.hist(figsize=(10,11))
plt.show()


# In[ ]:





# In[ ]:




