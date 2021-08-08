# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 18:43:55 2021

@author: paranr
"""
from flask import Flask
app = Flask(__name__, static_folder="templates", template_folder="templates")
app.config['SECRET_KEY']='thisisahappysecretkey'
from app import routes,forms