#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[65]:


data=pd.read_csv('clg.csv')


# In[66]:


data.head()


# In[67]:


data.isnull().sum()


# In[68]:


data.duplicated().sum()


# In[69]:


x=data.iloc[:,1:8]
x
y=data.iloc[:,-1]
y


# In[70]:


from sklearn.model_selection import train_test_split


# In[71]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)


# In[72]:


from sklearn.preprocessing import StandardScaler


# In[73]:


sc=StandardScaler()


# In[74]:


xtrain=sc.fit_transform(xtrain)
xtrain.shape


# In[75]:


xtest=sc.transform(xtest)


# In[98]:


import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.layers import Dense,Flatten


# In[99]:


model=models.Sequential()


# In[100]:


model.add(Dense(64,activation='relu',input_dim=7))


# In[101]:


model.add(Dense(32,activation='relu'))


# In[102]:


model.add(Dense(1,activation='linear'))


# In[103]:


model.summary()


# In[104]:


model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])


# In[105]:


model_history=model.fit(xtrain,ytrain,epochs=50,validation_split=0.3)


# In[109]:


plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.show()


# In[107]:


pred=model.predict(xtest)


# In[108]:


from sklearn.metrics import r2_score
r2_score(pred,ytest)


# In[120]:


model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


# In[121]:


model_history2=model.fit(xtrain,ytrain,epochs=50,batch_size=10,validation_split=0.3)


# In[122]:


# summarize history for loss
plt.plot(model_history2.history['loss'])
plt.plot(model_history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[123]:


pred2=model.predict(xtest)


# In[124]:


r2_score(pred2,ytest)


# In[125]:


model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])


# In[126]:


model_history2=model.fit(xtrain,ytrain,epochs=50,validation_split=0.3)


# In[127]:


# summarize history for loss
plt.plot(model_history2.history['loss'])
plt.plot(model_history2.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.show()


# In[128]:


pred3=model.predict(xtest)


# In[129]:


r2_score(pred3,ytest)


# In[ ]:




