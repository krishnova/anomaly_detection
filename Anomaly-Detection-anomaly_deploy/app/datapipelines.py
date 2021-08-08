#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from numpy.core.numerictypes import sctype2char
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.utils.sparsefuncs import inplace_column_scale
#from progressbar import ProgressBar


# In[1]:




def load_data(data = "iot_telemetry"):
    """load data into a variable.

    Args:
        data (str, optional): name of the dataset used . Defaults to 'pump sensor'.

    Returns:
        Pandas Dataframe: a dataframe of the data loaded using pandas.
    """
    if data=='pump sensor':
        filepath='sensor.csv'
        
    elif data == 'iot_telemetry':
        filepath='./Mareana/Anomaly Detection 2021/data sets/iot_telemetry.csv'
    return pd.read_csv(filepath)

def pump_remove_nans(df):
    """Deal with the nans in the dataset. This is taken from Roma's EDA pipeline.

    Args:
        df (pandas.Dataframe): The pandas dataframe of the data.
    Returns:
        df (Pandas Dataframe): dataframe with no nans.
    """
    df.drop('sensor_15', axis = 1, inplace = True)
    df.drop('Unnamed: 0', axis = 1, inplace = True)

    # here I am interpolating instead of filling with 0 like Roma    
    machine_status=df['machine_status']
    timestamp=df['timestamp']
    tmpdf=df.iloc[:,1:df.shape[1]-2]

    tmpdf.interpolate(axis=1,inplace=True,limit_direction='forward')
    tmpdf['machine_status']=machine_status
    df=tmpdf
    tmpdf['timestamp']  = pd.to_datetime(timestamp)
    tmpdf = tmpdf.reset_index()

    return tmpdf

def add_prebroken(df,status="BROKEN", npoints=50):
    """Add prebroken status to the dataset, since there is very little timepoints of anomaly.
        The motivation is to add points to increase the number of points in our classification
        to improve the balance of classes.
    Args:
        df (pandas.DataFrame): input dataframe.
        status (str, optional): the status we are looking for to replicate. Defaults to "BROKEN".
        npoints (int, optional): number of points to add ahead of status point. Defaults to 50.

    Returns:
        df [pandas.DataFrame]: modified Dataframe
    """
    for i,val in enumerate(df['machine_status']):
        if val==status:
            df.loc[np.linspace(i-npoints+1,i,npoints),'machine_status']='prebroken'
    return df

def normalize_data(df):
    """return data x- mean(x)/(std(x))

    Args:
        df (pandas.Dataframe): input dataframe, with numerical data only.

    Returns:
        df (pandas.Dataframe): output dataframe or matrix
    """
    return StandardScaler().fit_transform(df)

def combine_anomalies(df, dataset="pump"):
    """Changes anything that isn't normal to anomaly class.

    Args:
        df (pandas.Dataframe): input dataset 
        dataset (str, optional): string that describes the data source. Defaults to "pump".

    Returns:
        df (pandas.Dataframe): corrected labels dataset.
    """
    if dataset=='pump':
        df['machine_status']=df['machine_status'].apply(lambda x: 'NORMAL' if x=='NORMAL'            else 'ANOMALY')
    return df
    
def pca_data(data,ncomps=2):
    """compute the principal components of the data, and return it.

    Args:
        data (pandas.Dataframe or np.array): input data of the features
        ncomps (int, optional): number of components to output. Defaults to 2.

    Returns:
        principalComponents (df, np.array): component transformation of the data.
    """
    pca = PCA(n_components=ncomps)
    principalComponents = pca.fit_transform(data)
    return principalComponents

def extend_n_timepoints(data,ntp=50):
    """ Add older timepoint data horizontally to the data array.
    example:
    a            a(t)   a(t-1)
    1    ===>     2        1
    2

    Args:
        data (np.array): input features array.
        ntp (int, optional): number of past time points inserted. Defaults to 50.

    Returns:
        newarray [np.array]: array with older time points inserted horizontally.
    """
    pbar=ProgressBar()
    ndp=50
    newarray=np.empty((data.shape[0]-ntp,data.shape[1]*ntp))
    ind=0
    for n in pbar(range(ntp,data.shape[0])):
        tmpdata=data[n-50:n,:]
        newarray[ind,:]=tmpdata.reshape((1,tmpdata.shape[0]*tmpdata.shape[1]))
        ind+=1        
    return newarray 


# In[ ]:




