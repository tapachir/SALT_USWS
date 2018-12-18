# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:11:21 2018
@author: max
"""

import pprint
import pandas as pd
import numpy as np
from sklearn import linear_model
import pickle
from flask import Flask
from flask import request
import geopy.distance # pip install geopy

df = pd.read_csv(r'C:\Users\max\Downloads\test3.csv')

df1= df.drop(['price','name','review_count'], axis=1)#useless columns
df1.columns = ['latitude', 'longitude','rating']#rename
df1 = df1.dropna()#drop NaN rows

feature_df = df1.drop('rating', axis=1)
label_df= df1.drop(['longitude','latitude'], axis=1)

feature_df['neighbours'] = 0 # new column with 0


#print(df1.isnull().any())
#pprint.pprint(feature_df)
#Funktion soll Anzahl der Restaurants in der Nähe (<500m) zurückgeben
def neighbours(latitude,longitude):
    count =0
    coordinates1 = (latitude,longitude)
    for y in range(len(feature_df)):
        coordinates2_la = feature_df.at[y,'latitude']
        coordinates2_lo = feature_df.at[y,'longitude']
        coordinates2 = (coordinates2_la,coordinates2_lo)
        if (geopy.distance.distance(coordinates1,coordinates2).km < 0.5):
            count+=1
    print(count)        
    return count

# doppelte for-schleife ... womöglich überflüßig
#for x in range(len(feature_df)):
#    coordinates1_la = feature_df.at[x,'latitude']
#    coordinates1_lo = feature_df.at[x,'longitude']
#    coordinates1 = (coordinates1_la,coordinates1_lo)
#    for y in range(len(feature_df)):
#        coordinates2_la = feature_df.at[y,'latitude']
#        coordinates2_lo = feature_df.at[y,'longitude']
#        coordinates2 = (coordinates2_la,coordinates2_lo)
#        if (geopy.distance.distance(coordinates1,coordinates2).km < 0.5):
#            count+=1
#    feature_df.at[x,'neighbours']=count 
    
    
regr = linear_model.LinearRegression()
regr.fit(feature_df, label_df)

print(regr.predict([[52.4321,13.3210]],neighbours(52.4321,13.3210)).tolist())# test mit ausgedachter Breite und Länge

# kein Plan was der Code hier unten macht

#pickle.dump(regr, open("model.pk", "wb"))
#
## loading a model from file
#model = pickle.load(open("model.pkl", "r"))
#
# #################################
#
# test3['score'] = 0

##code which helps initialize our server
#app = flask.Flask(__name__)

##defining a /hello route for only post requests
#@app.route('/hello', methods=['POST'])
#def index():
#    #grabs the data tagged as 'name'
#    name = request.get_json()['name']
#    
#    #sending a hello back to the requester
#    return "Hello " + name