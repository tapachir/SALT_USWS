
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
import geopy.distance  # pip install geopy

#######################
# 
# Funktion schreiben, die nur die Koordinaten als eingabe erhät (zum predicten)
# Alle Kategorien finden
#######################

#
alle = pd.read_csv(r'/home/tahir/Downloads/all(1).csv')
alle = alle.dropna()
alle = alle.drop(
    ['name', '_id', 'phone', 'display_phone', 'alias', 'is_closed', 'url', 'image_url', 'transactions', 'name.1',
     'review_count'], axis=1)
label_alle = alle.drop(['categories', 'coordinates'], axis=1)
alle = alle.drop(['rating'], axis=1)
# alle.columns= ['categories','latitude', 'longitude']
#
df = pd.read_csv(r'/home/tahir/Downloads/test3(1).csv')
df1 = df.drop(['price', 'name', 'review_count'], axis=1)  # useless columns
df1.columns = ['latitude', 'longitude', 'rating']  # rename
feature_df = df1.drop('rating', axis=1)
label_df = df1.drop(['longitude', 'latitude'], axis=1)

feature_df['neighbours'] = 0  # new column with 0
feature_df['categories'] = alle['categories'].copy()
feature_df.dropna()
feature_df['koreanneigbours'] = 0
feature_df['italianneigbhours'] = 0
feature_df['vietnameseneigbhours'] = 0
feature_df['japaneseneigbhours'] = 0

feature_df1 = feature_df[0:100]


# print(df1.isnull().any())
# pprint.pprint(feature_df)

# regr = linear_model.LinearRegression()
# regr.fit(feature_df, label_df)


# print(regr.predict([[52.4321,13.3210]]).tolist())# test mit ausgedachter Breite und Länge
def prediction(latitude, longitude):
    koreans = 0
    italians = 0
    vietnameses = 0
    japaneses = 0
    for x in range(len(feature_df1)):
        coordinates1 = latitude, longitude
        for y in range(len(feature_df1)):
            coordinates2 = feature_df1.at[y, 'latitude'], feature_df1.at[y, 'longitude']
            dist = geopy.distance.geodesic(coordinates1, coordinates2).m
            if (dist < 750 and x != y):
                if ("Korean" in str(feature_df1.at[y, 'categories'])):
                    koreans += 1
                if ("Italian" in str(feature_df1.at[y, 'categories'])):
                    italians += 1
                if ("Vietnamese" in str(feature_df1.at[y, 'categories'])):
                    vietnameses += 1
                if ("Japanese" in str(feature_df1.at[y, 'categories'])):
                    japaneses += 1
                else:
                    feature_df1.at[x, 'neighbours'] += 1
    regr = linear_model.LinearRegression()
    regr.fit(feature_df, label_df)
    print(regr.predict([[52.4321, 13.3210]]).tolist())


#
# meine testfunktion
for x in range(len(feature_df1)):
    coordinates1 = feature_df1.at[x, 'latitude'], feature_df1.at[x, 'longitude']
    for y in range(len(feature_df1)):
        coordinates2 = feature_df1.at[y, 'latitude'], feature_df1.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1, coordinates2).m
        if (dist < 750 and x != y):
            if ("Korean" in str(feature_df1.at[y, 'categories'])):
                feature_df1.at[x, 'koreanneigbours'] += 1
            if ("Italian" in str(feature_df1.at[y, 'categories'])):
                feature_df1.at[x, 'italianneigbhours'] += 1
            if ("Vietnamese" in str(feature_df1.at[y, 'categories'])):
                feature_df1.at[x, 'vietnameseneigbhours'] += 1
            if ("Japanese" in str(feature_df1.at[y, 'categories'])):
                feature_df1.at[x, 'japaneseneigbhours'] += 1
            else:
                feature_df1.at[x, 'neighbours'] += 1
print(feature_df1)
#
# Diese Funktion schreibt die Anzahl (count) aller nahen (umkreis) Restaurants in die Spalte ('neighbours')
# und in die entsprechenden Spalten nach Kategorie
for x in range(len(feature_df)):
    umkreis = 750
    count = 0
    coordinates1 = feature_df.at[x, 'latitude'], feature_df.at[x, 'longitude']
    for y in range(len(feature_df1)):
        if (x != y):
            coordinates2 = feature_df.at[y, 'latitude'], feature_df.at[y, 'longitude']
            if (geopy.distance.geodesic(coordinates1, coordinates2).m < umkreis):
                count += 1
                if ("korean" in str(feature_df.at[y, 'categories'])):
                    feature_df.at[x, 'neighbours'] += 1
                if ("Italian" in str(feature_df.at[y, 'categories'])):
                    feature_df.at[x, 'italianneigbhours'] += 1
    feature_df.at[x, 'neighbours'] = count - 1
    print(count)
print(feature_df)

# doppelte for-schleife ... womöglich überflüßig
# for x in range(len(feature_df)):
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
# print(regr.predict([[52.4321,13.3210]],neighbours(52.4321,13.3210)).tolist())

# kein Plan was der Code hier unten macht

# pickle.dump(regr, open("model.pk", "wb"))
#
## loading a model from file
# model = pickle.load(open("model.pkl", "r"))
#
# #################################
#
# test3['score'] = 0

##code which helps initialize our server
# app = flask.Flask(__name__)

##defining a /hello route for only post requests
# @app.route('/hello', methods=['POST'])
# def index():
#    #grabs the data tagged as 'name'
#    name = request.get_json()['name']
#
#    #sending a hello back to the requester
#    return "Hello " + name