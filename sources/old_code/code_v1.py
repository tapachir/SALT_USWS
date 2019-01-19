# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:11:21 2018
@author: max
"""

import pandas as pd
import numpy as np
from sklearn import linear_model
import geopy.distance
import sklearn


# "all.csv" wird benötigt, da nur diese "categories" enthält, dafür aber die Koordinaten nur in einer Spalte (nicht zwei)
alle = pd.read_csv(r'C:\Users\max\Downloads\all.csv')
#
df = pd.read_csv(r'C:\Users\max\Downloads\test3.csv')
df= df.drop(['name'], axis=1)#useless columns
df.columns = ['latitude', 'longitude','rating','review_count','price']#rename
#revwiew_count aufräumen / kategorisieren
df['review_count'] = df['review_count'].replace([range(1,3)], 1.0)
df['review_count'] = df['review_count'].replace([range(3, 6)], 1.25)
df['review_count'] = df['review_count'].replace([range(6, 11)], 1.5)
df['review_count'] = df['review_count'].replace([range(11,51)], 1.75)
df['review_count'] = df['review_count'].replace([range(51,10000)], 2.0)

#price feature aufräumen
df['price'] = df['price'].replace('€', 1)
df['price'] = df['price'].replace('€€', 2)
df['price'] = df['price'].replace('$$', 2)
df['price'] = df['price'].replace('€€€', 3)
df['price'] = df['price'].replace('€€€€', 4)
df['price'] = df['price'].fillna(2)


#df splitten
train_df, test_df = sklearn.model_selection.train_test_split(df, test_size= 0.2)
train_df = train_df.sort_index()#vielleicht nicht nötig
test_df = test_df.sort_index()

#score = rating(1-5 Sterne) * review_count(1.0,1.25,1.5,1.75,2.0 je nach Anzahl an Reviews)
#train_label_df soll nur score (rating*review_count) haben
train_label_df= train_df.drop(['longitude','latitude','price','price'], axis=1)
train_label_df['score'] = train_label_df['rating'] * train_label_df['review_count']
train_label_df = train_label_df.drop(['rating','review_count'],axis=1)

#test_label_df
test_label_df= test_df.drop(['longitude','latitude','price','price'], axis=1)
test_label_df['score'] = test_label_df['rating'] * test_label_df['review_count']
test_label_df = test_label_df.drop(['rating','review_count'],axis=1)

#train_feature_df
train_feature_df = train_df.drop(['rating','review_count'], axis=1)
train_feature_df['categories'] = alle['categories'].copy()
train_feature_df['others'] = 0 # new column with 0
train_feature_df['koreanneigbours'] = 0
train_feature_df['italianneigbhours'] = 0
train_feature_df['vietnameseneigbhours'] = 0
train_feature_df['japaneseneigbhours'] = 0

#test_feature_df
test_feature_df = test_df.drop(['rating','review_count'], axis=1)
test_feature_df['categories'] = alle['categories'].copy()
test_feature_df['others'] = 0 # new column with 0
test_feature_df['koreanneigbours'] = 0
test_feature_df['italianneigbhours'] = 0
test_feature_df['vietnameseneigbhours'] = 0
test_feature_df['japaneseneigbhours'] = 0

#train_feature_df1 nur für LinearRegression
train_feature_df1 = train_feature_df.drop(['categories'],axis=1)

#dropping NAN coordinates
train_feature_df = train_feature_df.dropna(subset=['latitude','longitude'])
test_feature_df = test_feature_df.dropna(subset=['latitude','longitude'])
train_feature_df1 = train_feature_df1.dropna(subset=['latitude','longitude'])
#fürs testen sind die Dataframes kleiner
#train_feature_df = train_feature_df[0:100]
#train_label_df = train_label_df[0:100]
#test_label_df = test_label_df[0:100]
#test_feature_df = test_feature_df[0:100]
#
#TODO: Indizes für alle Test+Train DFs anpassen
#

#Setzt den Radius für die Suche nach Nachbarn fest
radius = 750

#Bereitet das train_dataframe fürs machinelearning for. füllt die features aus (anzahl der nachabrn)
#funktioniert nicht, weil problem mit leeren zeilen
train_index_df = 0
newindex1 = list(range(0,len(train_feature_df)))
train_feature_df['newindex'] = newindex1
train_feature_df = train_feature_df.set_index('newindex')
print('fun1')
for x in range(len(train_feature_df)):
    if(x % 100 == 0):
        print(str(x) + ' ' + 'out of' + ' ' + str(len(train_feature_df)))
    coordinates1 = train_feature_df.at[x,'latitude'], train_feature_df.at[x,'longitude']
    for y in range(len(train_feature_df)):
        coordinates2 = train_feature_df.at[y, 'latitude'], train_feature_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius and x != y):
            if("Korean" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'koreanneigbours']+=1
            if("Italian" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'italianneigbhours']+=1
            if("Vietnamese" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'vietnameseneigbhours']+=1
            if("Japanese" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'japaneseneigbhours']+=1
            else:
                train_feature_df.at[x, 'others']+=1
print(train_feature_df)
           

#prepare test features
#funktioniert nicht, weil problem mit leeren zeilen
newindex2 = list(range(0,len(test_feature_df)))
test_feature_df['newindex'] = newindex2
test_feature_df = test_feature_df.set_index('newindex')
print('fun2')
for x in range(len(test_feature_df)):
    if(x % 100 == 0):
        print(str(x) + ' ' + 'out of' + ' ' + str(len(test_feature_df)))
    coordinates1 = test_feature_df.at[x, 'latitude'], test_feature_df.at[x, 'longitude']
    for y in range(len(test_feature_df)):
        coordinates2 = test_feature_df.at[y, 'latitude'], test_feature_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius and x != y):
            if("Korean" in str(test_feature_df.at[y, 'categories'])):
                         test_feature_df.at[x, 'koreanneigbours']+=1
            if("Italian" in str(test_feature_df.at[y, 'categories'])):
                         test_feature_df.at[x, 'italianneigbhours']+=1
            if("Vietnamese" in str(test_feature_df.at[y, 'categories'])):
                         test_feature_df.at[x, 'vietnameseneigbhours']+=1
            if("Japanese" in str(test_feature_df.at[y, 'categories'])):
                         test_feature_df.at[x, 'japaneseneigbhours']+=1
            else:
                test_feature_df.at[x, 'others']+=1
print(test_feature_df)

#exportiert das train_feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
train_feature_df.to_csv('train_feature_df.csv', sep='\t', encoding='utf-8')
#exportiert das test_feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
test_feature_df.to_csv('test_feature_df.csv', sep='\t', encoding='utf-8')


#mainmethode
#Testewert: prediction(52.501143,13.318144,2) -> 8,9176477
def prediction(latitude,longitude,price):
    train_feature_df1 =train_feature_df.drop('categories', axis=1)
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
    regr.fit(train_feature_df1, train_label_df)
    predscore= regr.predict([[latitude,longitude,price,others,koreans,italians,vietnameses,japaneses]])
    print(str(predscore))
    return(predscore)

#Gibt eie Dataframe mit nur "Nachbarn" zurück -> soll "prediction" optimieren
# gettingneighbours(52.4589,13.323,750)
#Hier muss die Funktion noch an das veränderte train_feature_df angepasst werden
def gettingneighbours(latitude,longitude):
    near_df = pd.DataFrame(columns=['latitude','longitude','categories'])
    coordinates1 = latitude, longitude
    for x in range(len(train_feature_df)):
        coordinates2 = train_feature_df.at[x, 'latitude'], train_feature_df.at[x, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius):
            lat = train_feature_df.at[x,'latitude']
            lon = train_feature_df.at[x,'longitude']
            cat = train_feature_df.at[x,'categories']
            print(x)
            near_df = near_df.append({'latitude' : lat , 'longitude' : lon , 'categories' : cat}, ignore_index = True)
    return(near_df)
    print(near_df)
        
#testing
    
test_index_df = 0
newindex3 = list(range(0,len(test_df)))
test_df['newindex'] = newindex3
test_df = test_df.set_index('newindex')
test_df['dif'] = 0
test_df['score'] = test_label_df['score']
def testing(test_df):
    abw=0
    for x in range(len(test_df)):
        dif=0
        a=test_df.at[x,'latitude']
        b=test_df.at[x,'longitude']
        c=test_df.at[x,'price']
        pred = prediction(a,b,c)
        test_df.at[x,'test-rating'] = pred
        dif =abs(pred-test_label_df.at[x, 'score'])
        test_df.at[x,'dif'] = dif
        abw = dif / len(test_df)
        test_df.at[x,'abw'] = abw
print('Durschnitt der Differenz:' +' '+ str(np.mean(test_df['dif'])))
print('Durchschnitt Score:' + ' ' + str(np.mean(test_df['score'])))
print('Durchschnitt Prediction:' + ' ' + str(np.mean(test_df['pred'])))
print(test_df)
