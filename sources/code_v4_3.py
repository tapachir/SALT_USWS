# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 21:10:26 2019

@author: max
"""
import pandas as pd
import numpy as np
from sklearn import linear_model
import geopy.distance
import datetime

df = pd.read_csv(r'C:\Users\max\Documents\Dev\Prog2\Abgabe4\SALT_USWS\sources\allWithCategory.csv')
#renaming columns

#Setzt den Radius für die Suche nach Nachbarn fest
coordinatesMitte = (52.524436,13.409616)

radius = 750

#Alle wichtigen Dataframes werden importiert
test_feature_df = pd.read_csv(r'C:\Users\max\Documents\Dev\Prog2\Abgabe4\SALT_USWS\sources\CSVFiles\test_feature_df.csv')
test_label_df =pd.read_csv(r'C:\Users\max\Documents\Dev\Prog2\Abgabe4\SALT_USWS\sources\CSVFiles\test_label_df.csv')
test_df =pd.read_csv(r'C:\Users\max\Documents\Dev\Prog2\Abgabe4\SALT_USWS\sources\CSVFiles\test_df.csv')
train_feature_df =pd.read_csv(r'C:\Users\max\Documents\Dev\Prog2\Abgabe4\SALT_USWS\sources\CSVFiles\train_feature_df.csv')
train_label_df =pd.read_csv(r'C:\Users\max\Documents\Dev\Prog2\Abgabe4\SALT_USWS\sources\CSVFiles\train_label_df.csv')

#Unwichtige Spalte wird gelöscht
test_feature_df= test_feature_df.drop(['Unnamed: 0'],axis=1)
test_label_df= test_label_df.drop(['Unnamed: 0'],axis=1)
test_df= test_df.drop(['Unnamed: 0'],axis=1)
train_feature_df= train_feature_df.drop(['Unnamed: 0'],axis=1)
train_label_df = train_label_df.drop(['Unnamed: 0'],axis=1)

def prediction(latitude,longitude,price,category):
    own_category = category
    price = price
    train_feature_df_1 = train_feature_df.drop(['latitude','longitude','categories'],axis=1)
    near_df = gettingneighbours(latitude,longitude)
    vietnamese=0
    sushi=0
    pubs=0
    pizza=0
    kebab=0
    italian=0
    icecream=0
    hotdogs=0
    german=0
    divebars=0
    cocktailbars=0
    cafes=0
    burgers=0
    bars=0
    bakeries=0
    own= 0
    others=0
    distanceToMitte=0
    coordinates1 = latitude, longitude
    for predictiony in range(len(near_df)):
        coordinates2 = near_df.at[predictiony, 'latitude'], near_df.at[predictiony, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius):
            if("vietnamese" in str(near_df.at[predictiony, 'categories'])):
               vietnamese+=1
               if("vietnamese" == own_category):
                   own+=1
            if("sushi" in str(near_df.at[predictiony, 'categories'])):
                sushi+=1
                if("sushi" == own_category):
                   own+=1
            if("pubs" in str(near_df.at[predictiony, 'categories'])):
                pubs+=1
                if("pubs" == own_category):
                   own+=1
            if("pizza" in str(near_df.at[predictiony, 'categories'])):
                pizza+=1
                if("pizza" == own_category):
                   own+=1
            if("kebab" in str(near_df.at[predictiony, 'categories'])):
                kebab+=1
                if("kebab" == own_category):
                   own+=1
            if("italian" in str(near_df.at[predictiony, 'categories'])):
                italian+=1
                if("italian" == own_category):
                   own+=1
            if("icecream" in str(near_df.at[predictiony, 'categories'])):
                icecream+=1
                if("icecream" == own_category):
                   own+=1
            if("hotdogs" or "hotdog" in str(near_df.at[predictiony, 'categories'])):
                hotdogs+=1
                if("hotdogs" == own_category):
                   own+=1
            if("german" in str(near_df.at[predictiony, 'categories'])):
                german+=1
                if("german" == own_category):
                   own+=1
            if("divebars" in str(near_df.at[predictiony, 'categories'])):
                divebars+=1
                if("divebars" == own_category):
                   own+=1
            if("cocktailbars" in str(near_df.at[predictiony, 'categories'])):
                cocktailbars+=1
                if("cocktailbars" == own_category):
                   own+=1
            if("cafes" or "coffee" in str(near_df.at[predictiony, 'categories'])):
                cafes+=1
                if("cafes" == own_category):
                   own+=1
            if("burgers" in str(near_df.at[predictiony, 'categories'])):
                burgers+=1
                if("burgers" == own_category):
                   own+=1
            if("bars" in str(near_df.at[predictiony, 'categories'])):
                bars+=1
                if("bars" == own_category):
                   own+=1
            if("bakeries" in str(near_df.at[predictiony, 'categories'])):
                bakeries+=1
                if("bakeries" == own_category):
                   own+=1
            else:
                others+=1
                if(str(near_df.at[predictiony, 'categories']) == own_category):
                   own+=1
    regr = linear_model.LinearRegression()
    regr.fit(train_feature_df_1, train_label_df)
    predscore = regr.predict([[price,others,own,vietnamese,sushi,pubs,pizza,kebab,italian,
                               icecream,hotdogs,german,divebars,cocktailbars,cafes,burgers,bars,bakeries,distanceToMitte]])
    return(predscore)
    
#Gibt eie Dataframe mit nur "Nachbarn" zurück -> soll "prediction" optimieren
def gettingneighbours(latitude,longitude):
    near_df = pd.DataFrame(columns=['latitude','longitude','categories'])
    coordinates1 = latitude, longitude
    for gettingneighboursx in range(len(train_feature_df)):
        coordinates2 = train_feature_df.at[gettingneighboursx, 'latitude'], train_feature_df.at[gettingneighboursx, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius):
            lat = train_feature_df.at[gettingneighboursx,'latitude']
            lon = train_feature_df.at[gettingneighboursx,'longitude']
            cat = train_feature_df.at[gettingneighboursx,'categories']
            near_df = near_df.append({'latitude' : lat , 'longitude' : lon , 'categories' : cat}, ignore_index = True)
    return(near_df)
     
#testing
def test(test_df):
    ergebnis_df = pd.DataFrame(columns=['score','predictedscore','dif'])
    ergebnis_df.at[0,'score'] = test_label_df.at[0,'score']
    if(len(test_df)>=500):
        modop_test = 5
    if(len(test_df)<500):
        modop_test = 10
    for test in range(len(test_df)):
        if((test /len(test_df)*100) % modop_test ==0):
            print(datetime.datetime.now())
            print(str(test/len(test_df)*100)+'%')
        dif=0
        
        lat=test_df.at[test,'latitude']
        lon=test_df.at[test,'longitude']
        pri=test_df.at[test,'price']
        cat=test_df.at[test,'categories']
        pred = prediction(lat,lon,pri,cat)
        #print(pred)
        pred= float(pred)
        
        ergebnis_df.at[test,'score'] = test_label_df.at[test,'score']
        ergebnis_df.at[test,'predictedscore'] = pred
        dif = abs(ergebnis_df.at[test,'score']-pred)
        ergebnis_df.at[test,'dif'] = dif
    print(ergebnis_df)
    print('Score:')
    print('Maximaler Score: ' + str(np.max(ergebnis_df['score'])))
    print('Minimaler Score: ' + str(np.min(ergebnis_df['score'])))
    print('Dif von Score:' + str(np.max(ergebnis_df['score'])-np.min(ergebnis_df['score'])))
    print('PredictedScore:')
    print('Maximaler PredictedScore: ' + str(np.max(ergebnis_df['predictedscore'])))
    print('Minimaler PredictedScore: ' + str(np.min(ergebnis_df['predictedscore'])))
    print('Dif:')
    print('Dif von PredictedScore:' + str(np.max(ergebnis_df['predictedscore'])-np.min(ergebnis_df['predictedscore'])))
    print('Mean:' + str(np.mean(ergebnis_df['dif'])))
    print('minimum: '+ str(np.min(ergebnis_df['dif'])))
    print('maximum: ' + str(np.max(ergebnis_df['dif'])))
    print('Varianz: ' + str(np.var(ergebnis_df['dif'])))
    print('Spannweite Prediction: ' + str((np.min(ergebnis_df['predictedscore'] - np.max(ergebnis_df['predictedscore'])))))
