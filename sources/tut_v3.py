# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:11:21 2018
@author: max
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
from flask import request
import geopy.distance # pip install geopy

# "all.csv" wird benötigt, da nur diese "categories" enthält, dafür aber die Koordinaten nur in einer Spalte (nicht zwei)
alle = pd.read_csv(r'C:\Users\max\Downloads\all.csv')
#
df = pd.read_csv(r'C:\Users\max\Downloads\test3.csv')
df= df.drop(['price','name','review_count'], axis=1)#useless columns
df.columns = ['latitude', 'longitude','rating']#rename

feature_df = df.drop('rating', axis=1)
label_df= df.drop(['longitude','latitude'], axis=1)

feature_df['neighbours'] = 0 # new column with 0
feature_df['categories'] = alle['categories'].copy()
feature_df.dropna()
feature_df['koreanneigbours'] = 0
feature_df['italianneigbhours'] = 0
feature_df['vietnameseneigbhours'] = 0
feature_df['japaneseneigbhours'] = 0
#feature_df1 nur für LinearRegression

feature_df = feature_df[0:1000]

label_df = label_df[0:1000]

#Setzt den Radius für die Suche nach Nachbarn fest
radius = 750

#Bereitet das dataframe fürs machinelearning for. füllt die features aus
for x in range(len(feature_df)):
    if(x % 10 == 0):
            print(x + 'from' + str(len(feature_df)))
    coordinates1 = feature_df.at[x, 'latitude'], feature_df.at[x, 'longitude']
    for y in range(len(feature_df)):
        coordinates2 = feature_df.at[y, 'latitude'], feature_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius and x != y):
            if("Korean" in str(feature_df.at[y, 'categories'])):
                         feature_df.at[x, 'koreanneigbours']+=1
            if("Italian" in str(feature_df.at[y, 'categories'])):
                         feature_df.at[x, 'italianneigbhours']+=1
            if("Vietnamese" in str(feature_df.at[y, 'categories'])):
                         feature_df.at[x, 'vietnameseneigbhours']+=1
            if("Japanese" in str(feature_df.at[y, 'categories'])):
                         feature_df.at[x, 'japaneseneigbhours']+=1
            else:
                feature_df.at[x, 'neighbours']+=1
print(feature_df)

#exportiert das feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
feature_df.to_csv('feature_df.csv', sep='\t', encoding='utf-8')


#mainmethode
#Testewert: prediction(52.501143,13.318144)
def prediction(latitude,longitude):
    feature_df1 =feature_df.drop('categories', axis=1)
    near_df = gettingneighbours(latitude,longitude)
    koreans =0
    italians =0
    vietnameses=0
    japaneses=0
    others = 0
    coordinates1 = latitude, longitude
    for y in range(len(near_df)):
        coordinates2 = near_df.at[y, 'latitude'], near_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius):
            if("Korean" in str(near_df.at[y, 'categories'])):
                         koreans+=1
            if("Italian" in str(near_df.at[y, 'categories'])):
                         italians+=1
            if("Vietnamese" in str(near_df.at[y, 'categories'])):
                         vietnameses+=1
            if("Japanese" in str(near_df.at[y, 'categories'])):
                         japaneses+=1
            else:
                others+=1
    print(others,koreans,italians,vietnameses,japaneses)
    regr = linear_model.LinearRegression()
    regr.fit(feature_df1, label_df)
    print(regr.predict([[latitude,longitude,others,koreans,italians,vietnameses,japaneses]]))

#Gibt eie Dataframe mit nur "Nachbarn" zurück -> soll "prediction" optimieren
# gettingneighbours(52.4589,13.323,750)
def gettingneighbours(latitude,longitude):
    near_df = pd.DataFrame(columns=['latitude','longitude','categories'])
    coordinates1 = latitude, longitude
    for x in range(len(feature_df)):
        coordinates2 = feature_df.at[x, 'latitude'], feature_df.at[x, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius):
            lat = feature_df.at[x,'latitude']
            lon = feature_df.at[x,'longitude']
            cat = feature_df.at[x,'categories']
            print(x)
            near_df = near_df.append({'latitude' : lat , 'longitude' : lon , 'categories' : cat}, ignore_index = True)
    return(near_df)
    print(near_df)
    
    
#for x in range(len(feature_df)):
#    umkreis = 750
#    count = 0
#    coordinates1 = feature_df.at[x, 'latitude'], feature_df.at[x, 'longitude']
#    for y in range(len(feature_df1)):
#        if(x != y):
#            coordinates2 = feature_df.at[y, 'latitude'], feature_df.at[y, 'longitude']
#            if (geopy.distance.geodesic(coordinates1,coordinates2).m < umkreis):
#                count+=1
#                if("korean" in str(feature_df.at[y, 'categories'])):
#                     feature_df.at[x, 'neighbours']+=1
#                if("Italian" in str(feature_df.at[y, 'categories'])):
#                     feature_df.at[x, 'italianneigbhours']+=1
#    feature_df.at[x, 'neighbours'] = count-1
#    print(count)
#print(feature_df)

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
#print(regr.predict([[52.4321,13.3210]],neighbours(52.4321,13.3210)).tolist())

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