#!/usr/bin/env python
# coding: utf-8

# In[25]:


#Standard imports 
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

#from keras.layers import LSTM, Dense,  RepeatVector, TimeDistributed, Input
#from keras.models import Sequential
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
#from statsmodels.tsa.seasonal import seasonal_decompose
#from keras import Model

from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import confusion_matrix


# In[ ]:





# In[11]:




# In[ ]:


def model_loss(history):
    fig, ax = plt.subplots(figsize = (14,6), dpi = 80)
    ax.plot(history['loss'], 'b', label = 'Train', linewidth = 2)
    ax.plot(history['val_loss'], 'r', label = "Validation", linewidth = 2)

    ax.set_title('Model Loss', fontsize = 16)
    ax.set_ylabel('Loss (mae)')
    ax.set_xlabel('Epoch')
    ax.legend()
    plt.show()


# In[12]:


def loss_mae_dist(loss_mae):
    plt.figure(figsize=(16,9), dpi = 80)
    sns.displot(loss_mae, kde = True)


# In[13]:



def plot_threshold(scored):
    scored.plot(logy=True,ylim=[1e-2,1e2])


# In[28]:


def plot_confusion_matrix(prediction, actual):
    cf_matrix = confusion_matrix(actual, prediction)
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in
                cf_matrix.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in
                     cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# In[ ]:




