# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:11:21 2018
@author: max
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
import geopy.distance # pip install geopy

# "all.csv" wird benötigt, da nur diese "categories" enthält, dafür aber die Koordinaten nur in einer Spalte (nicht zwei)
alle = pd.read_csv(r'C:\Users\max\Downloads\all.csv')
#
df = pd.read_csv(r'C:\Users\max\Downloads\test3.csv')
df= df.drop(['name'], axis=1)#useless columns
df.columns = ['latitude', 'longitude','rating','review_count','price']#rename
#revwiew_count aufräumen / kategorisieren
#am wenigsten
df['review_count'] = df['review_count'].replace([range(1,3)], 1)
df['review_count'] = df['review_count'].replace([range(3, 6)], 2)
df['review_count'] = df['review_count'].replace([range(6, 11)], 3)
df['review_count'] = df['review_count'].replace([range(11,51)], 4)
df['review_count'] = df['review_count'].replace([range(51,10000)], 5)

#price feature aufräumen
df['price'] = df['price'].replace('€', 1)
df['price'] = df['price'].replace('€€', 2)
df['price'] = df['price'].replace('$$', 2)
df['price'] = df['price'].replace('€€€', 3)
df['price'] = df['price'].replace('€€€€', 4)
df['price'] = df['price'].fillna(2)

#label_df soll nur score (rating+review_count) haben
label_df= df.drop(['longitude','latitude','price','price'], axis=1)
label_df['score'] = label_df['rating'] + label_df['review_count']
label_df = label_df.drop(['rating','review_count'],axis=1)

#feature_df
feature_df = df.drop(['rating','review_count'], axis=1)
feature_df['categories'] = alle['categories'].copy()
feature_df['others'] = 0 # new column with 0
feature_df['koreanneigbours'] = 0
feature_df['italianneigbhours'] = 0
feature_df['vietnameseneigbhours'] = 0
feature_df['japaneseneigbhours'] = 0



#feature_df1 nur für LinearRegression
feature_df1 = feature_df.drop(['categories'],axis=1)

#fürs testen sind die Dataframes kleiner
feature_df = feature_df[0:100]
label_df = label_df[0:100]

#Setzt den Radius für die Suche nach Nachbarn fest
radius = 750

#Bereitet das dataframe fürs machinelearning for. füllt die features aus
for x in range(len(feature_df)):
    if(x % 10 == 0):
            print(str(x) + 'from' + str(len(feature_df)))
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
                feature_df.at[x, 'others']+=1
print(feature_df)

#exportiert das feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
feature_df.to_csv('feature_df.csv', sep='\t', encoding='utf-8')


#mainmethode
#Testewert: prediction(52.501143,13.318144,2) -> 8,9176477
def prediction(latitude,longitude,price):
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
    print(regr.predict([[latitude,longitude,price,others,koreans,italians,vietnameses,japaneses]]))

#Gibt eie Dataframe mit nur "Nachbarn" zurück -> soll "prediction" optimieren
# gettingneighbours(52.4589,13.323,750)
#Hier muss die Funktion noch an das veränderte feature_df angepasst werden
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
    