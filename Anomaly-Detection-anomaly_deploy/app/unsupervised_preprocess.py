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

# import tensorflow as tf 

# from tensorflow import keras 
import matplotlib.dates 
import datetime

#from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input
#from keras.models import Sequential
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
#from statsmodels.tsa.seasonal import seasonal_decompose
#from keras import Model

from sklearn.neighbors import LocalOutlierFactor


# In[14]:


def load_df(df):
     if df =='iot':
        path = "iot_telemetry.csv"
        return pd.read_csv(path)


# In[11]:




#Cleaning IOT sensor data set
def clean(df):
    df['timestamp'] = pd.to_datetime(df['ts'], unit='s')
    df = df.drop(['ts'], axis = 1)
    df.replace(['00:0f:00:70:91:0a','1c:bf:ce:15:ec:4d','b8:27:eb:bf:9d:51',],[1,2,3], inplace = True)
    model_df=pd.pivot_table(df,values='co',index= df.index,columns= 'device')
    model_df.reset_index(inplace=True)
    model_df.fillna(0,inplace=True)
    vals = model_df.columns[1:4]
    model_df = model_df[vals]
    
    return model_df 
    


# In[ ]:




