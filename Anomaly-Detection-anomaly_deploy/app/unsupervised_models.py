#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import StandardScaler 

from datetime import datetime

#import tensorflow as tf 

#from tensorflow import keras 
import matplotlib.dates 
import datetime

#from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed, Input
#from keras.models import Sequential
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
#from statsmodels.tsa.seasonal import seasonal_decompose
#from keras import Model

from sklearn.neighbors import LocalOutlierFactor


# In[2]:


def local_outlier_factor(df, vals = 0.00):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=vals)
    pred = lof.fit_predict(df)
    df['anomaly']=pred
    outliers=df.loc[df['anomaly']==-1]
    outlier_index=list(outliers.index)
    df['contamination'] = vals
        
    return df


# In[3]:


def isolation_forest(df, vals = 0.00):
    model =IsolationForest(n_estimators=100, max_samples=256, contamination= vals,random_state=42, verbose=0)
    model.fit(df)
    pred = model.predict(df)
    score = model.decision_function(df)
    df['scores'] = score
    df['anomaly']=pred
    outliers=df.loc[df['anomaly']==-1]
    outlier_index=list(outliers.index)
    df['contamination'] = vals
    
    return df 


# 
# 

# In[5]:


def autoencoder(X):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(25, activation='relu',input_shape = (X.shape[1],X.shape[2]),
                                return_sequences=True))
    model.add(keras.layers.LSTM(50, activation='relu', return_sequences=False))
    model.add(keras.layers.RepeatVector(X.shape[1]))
    model.add(keras.layers.LSTM(50,activation='relu', return_sequences=True))
    model.add(keras.layers.LSTM(25,activation='relu', return_sequences = True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(X.shape[2])))
    return model


# In[ ]:




