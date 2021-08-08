# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 00:06:36 2021

@author: paranr
"""

import pickle
from flask import Flask, request, render_template, send_file
import os
from plotly.graph_objs import Scatter
from app.unsupervised_preprocess import *
from app.unsupervised_models import *
import pandas as pd
import numpy as np
import plotly
import json
import io
from app import app
from app.forms import upload_form
#app = Flask(__name__)
#global processed
@app.route('/',methods=['POST','GET'])
def index():
    form=upload_form()
    if form.validate_on_submit():
        print("form is submitted")
        # print(form.json())
        # myFile = secure_filename(form.fileName.file.filename)
        csvdata=request.files[form.uploadFile.name].read()
        data=pd.read_csv(io.BytesIO(csvdata))
        print(data.head())
        col=form.column.data
        #form.uploadFile.file.save('temp.csv')
        # uploadfile=form.uploadFile()
        # data=pd.read_csv('temp.csv')
        outliers=[]
        # data=data[col].to_numpy()
        datalof,outlierslof=lof_pipeline(data,col)
        dataif,outliersif=if_pipeline(data,col)

        print("LOF N_anomalies=",datalof[datalof['anomaly']==-1].shape[0])
        print("IF N_anomalies=",dataif[dataif['anomaly']==-1].shape[0])
        
        idslof,graphJSONlof=generate_plot(datalof,outlierslof,col,"LOF")
        idsif,graphJSONif=generate_plot(dataif,outliersif,col,"IF")
        return render_template("predict.html",graphJSONlof=graphJSONlof,
        graphJSONif=graphJSONif,idslof=idslof,idsif=idsif)
    return render_template("index.html", form=form)




@app.route('/predict',methods=['POST','GET'])
def predict():
    #user input value
    return render_template('predict.html')

    render_template("predict.html")

def if_pipeline(data,col):
    print(col)
    print(data.columns)
    print(data[col].to_numpy())
    ifdata=data.copy()
    model =IsolationForest(n_estimators=100, max_samples=256, contamination= 0.1,random_state=42, verbose=0)
    model.fit(ifdata[col].to_numpy().reshape(-1,1))
    pred = model.predict(ifdata[col].to_numpy().reshape(-1,1))
    score = model.decision_function(ifdata[col].to_numpy().reshape(-1,1))
    ifdata['scores'] = score
    ifdata['anomaly']=pred
    outliers= ifdata.loc[ifdata['anomaly']==-1]
    outlier_index=list(outliers.index)
    ifdata['contamination'] = 0.1
    return ifdata,outliers

def lof_pipeline(data,col):
    lof = LocalOutlierFactor(n_neighbors=int(np.round(data.shape[0]*0.001)), contamination=0.1)
    print(col)
    print(data.columns)
    print(data[col].to_numpy())
    lofdata=data.copy()
    pred = lof.fit_predict(lofdata[col].to_numpy().reshape(-1,1))
    lofdata['anomaly']=pred
    outliers=data.loc[lofdata['anomaly']==-1]
    outlier_index=list(outliers.index)
    lofdata['contamination'] = 0.1

    return lofdata, outliers

def generate_plot(model_df,outliers,col,title):
    outliers_only=model_df[model_df['anomaly']!=1]
    print(model_df.head())
    print(model_df.columns)
    fig=[
        # Plot the raw data
        {       "data":[
                {"type":"line",
                "x":model_df.index,
                "y":model_df[col],
                "name":col
                
            },
            Scatter(x=outliers_only.index,y=outliers_only[col],mode='markers',
            marker=dict(color='rgb(255, 0, 0)',size=5 ),name='Anomaly')

            ],
            "layout":{"title":{"text":title},
                    "xaxis":{"title":{"text":"sample #"}},
                    "yaxis":{"title":{"text":col}}
                    
                    }
        }#,
        # {
        #     "data":[{
        #         "type":"scatter",
        #         "size":5,
        #         "color":[1,0,0],
        #         "x":outliers.index,
        #         "y":outliers[outliers.columns[1]]
        #     }]
        # }
    ]
    ids = ["graph-{}-{}".format(i,title) for i, _ in enumerate(fig)]
    print(ids)
    return ids,json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

#if __name__ == "__main__":
    #app.run(debug=True, use_reloader=False)
    
    
