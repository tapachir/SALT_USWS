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
import datetime

df = pd.read_csv(r'C:\Users\max\Documents\Dev\Prog2\Abgabe4\SALT_USWS\sources\allWithCategory.csv')
#renaming columns
df.columns = ['latitude', 'longitude','rating','review_count','price','categories']

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

df = df.dropna()

df = df[0:2000]

#df splitten
train_df, test_df = sklearn.model_selection.train_test_split(df, test_size= 0.2)
train_df = train_df.reset_index()#vielleicht nicht nötig
train_df = train_df.drop(['index'],axis=1)
test_df = test_df.reset_index()
test_df = test_df.drop(['index'],axis=1)


#train_label_df soll nur score (rating+review_count) haben
train_label_df= train_df.drop(['longitude','latitude','price','price'], axis=1)
train_label_df['score'] = train_label_df['rating'] * train_label_df['review_count']
train_label_df = train_label_df.drop(['rating','review_count','categories'],axis=1)

#test_label_df
test_label_df= test_df.drop(['longitude','latitude','price','price'], axis=1)
test_label_df['score'] = test_label_df['rating'] * test_label_df['review_count']
test_label_df = test_label_df.drop(['rating','review_count','categories'],axis=1)

#train_feature_df
train_feature_df = train_df.drop(['rating','review_count'], axis=1)
train_feature_df['otherneigbours'] = 0
train_feature_df['own_category'] = 0
train_feature_df['vietnameseneigbours'] = 0
train_feature_df['sushineigbours'] = 0
train_feature_df['pubsneigbours'] = 0
train_feature_df['pizzaneigbours'] = 0
train_feature_df['kebabneigbours'] = 0
train_feature_df['italianneigbours'] = 0
train_feature_df['icecreamneigbours'] = 0
train_feature_df['hotdogsneigbours'] = 0
train_feature_df['germanneigbours'] = 0
train_feature_df['divebarsneigbours'] = 0
train_feature_df['cocktailbarsneigbours'] = 0
train_feature_df['cafesneigbours'] = 0
train_feature_df['burgersneigbours'] = 0
train_feature_df['barsneigbours'] = 0
train_feature_df['bakeriesneigbours'] = 0
train_feature_df['distanceToMitte'] =0

#test_feature_df
test_feature_df = test_df.drop(['rating','review_count'], axis=1)
test_feature_df['otherneigbours'] = 0
test_feature_df['own_category'] = 0
test_feature_df['vietnameseneigbours'] = 0
test_feature_df['sushineigbours'] = 0
test_feature_df['pubsneigbours'] = 0
test_feature_df['pizzaneigbours'] = 0
test_feature_df['kebabneigbours'] = 0
test_feature_df['italianneigbours'] = 0
test_feature_df['icecreamneigbours'] = 0
test_feature_df['hotdogsneigbours'] = 0
test_feature_df['germanneigbours'] = 0
test_feature_df['divebarsneigbours'] = 0
test_feature_df['cocktailbarsneigbours'] = 0
test_feature_df['cafesneigbours'] = 0
test_feature_df['burgersneigbours'] = 0
test_feature_df['barsneigbours'] = 0
test_feature_df['bakeriesneigbours'] = 0
test_feature_df['distanceToMitte'] =0

#train_feature_df1 nur für LinearRegression
train_feature_df1 = train_feature_df.drop(['categories'],axis=1)

#Setzt den Radius für die Suche nach Nachbarn fest
coordinatesMitte = (52.524436,13.409616)

radius = 750
#Bereitet das train_dataframe fürs machinelearning for. füllt die features aus (anzahl der nachabrn)
print(datetime.datetime.now())
for x in range(len(train_feature_df)):
    if(len(train_feature_df)>=100):
        modop_train = 2
    if(len(train_feature_df)>=500):
        modop_train = 5
    if(len(train_feature_df)<500):
        modop_train = 10
    if((x /len(train_feature_df)*100) % modop_train ==0):
            print(str(x/len(train_feature_df)*100)+'%')
    own_category = train_feature_df.at[x,'categories']
    coordinates1 = train_feature_df.at[x, 'latitude'], train_feature_df.at[x, 'longitude']
    train_feature_df.at[x,'distanceToMitte'] = geopy.distance.geodesic(coordinates1,coordinatesMitte).m
    for y in range(len(train_feature_df)):
        coordinates2 = train_feature_df.at[y, 'latitude'], train_feature_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius and x != y):
            if("vietnamese" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'vietnameseneigbours']+=1
            if("sushi" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'sushineigbours']+=1
            if("pubs" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'pubsneigbours']+=1
            if("pizza" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'pizzaneigbours']+=1
            if("kebab" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'kebabneigbours']+=1
            if("italian" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'italianneigbours']+=1
            if("icecream" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'icecreamneigbours']+=1
            if("hotdogs" or "hotdog" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'hotdogsneigbours']+=1
            if("german" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'germanneigbours']+=1
            if("divebars" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'divebarsneigbours']+=1
            if("cocktailbars" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'cocktailbarsneigbours']+=1
            if("cafes" or "coffee" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'cafesneigbours']+=1
            if("burgers" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'burgersneigbours']+=1
            if("bars" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'barsneigbours']+=1
            if("bakeries" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'bakeriesneigbours']+=1
            if(own_category in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x,'own_category']+=1
            else:
                train_feature_df.at[x, 'otherneigbours']+=1
                
#prepare test features
print(datetime.datetime.now())
for x in range(len(test_feature_df)):
    if(len(train_feature_df)>=500):
        modop_test = 2
    if(len(train_feature_df)>=250):
        modop_test = 5
    if(len(train_feature_df)<250):
        modop_test = 10
    if((x /len(test_feature_df)*100) % modop_test ==0):
            print(str(x/len(test_feature_df)*100)+'%')
    own_category = test_feature_df.at[x,'categories']
    coordinates1 = test_feature_df.at[x, 'latitude'], test_feature_df.at[x, 'longitude']
    train_feature_df.at[x,'distanceToMitte'] = geopy.distance.geodesic(coordinates1,coordinatesMitte).m
    for y in range(len(test_feature_df)):
        coordinates2 = test_feature_df.at[y, 'latitude'], test_feature_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius and x != y):
            if("vietnamese" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'vietnameseneigbours']+=1
            if("sushi" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'sushineigbours']+=1
            if("pubs" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'pubsneigbours']+=1
            if("pizza" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'pizzaneigbours']+=1
            if("kebab" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'kebabneigbours']+=1
            if("italian" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'italianneigbours']+=1
            if("icecream" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'icecreamneigbours']+=1
            if("hotdogs" or "hotdog" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'hotdogsneigbours']+=1
            if("german" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'germanneigbours']+=1
            if("divebars" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'divebarsneigbours']+=1
            if("cocktailbars" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'cocktailbarsneigbours']+=1
            if("cafes" or "coffee" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'cafesneigbours']+=1
            if("burgers" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'burgersneigbours']+=1
            if("bars" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'barsneigbours']+=1
            if("bakeries" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'bakeriesneigbours']+=1
            if(own_category in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x,'own_category']+=1
            else:
                test_feature_df.at[x, 'otherneigbours']+=1

##exportiert das train_feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
train_feature_df.to_csv(r'CSVFiles/train_feature_df.csv', sep=',', encoding='utf-8')
##exportiert das test_feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
test_feature_df.to_csv(r'CSVFiles/test_feature_df.csv', sep=',', encoding='utf-8')
##exportiert das test_df als .csv Datei in den Ordner, in dem diese Datei liegt
test_df.to_csv(r'CSVFiles/test_df.csv', sep=',', encoding='utf-8')
train_label_df.to_csv(r'CSVFiles/train_label_df.csv', sep=',', encoding='utf-8')
test_label_df.to_csv(r'CSVFiles/test_label_df.csv', sep=',', encoding='utf-8')


#
##mainmethode
#def prediction(latitude,longitude,price,category):
#    own_category = category
#    price = price
#    train_feature_df_1 = train_feature_df.drop(['latitude','longitude','categories'],axis=1)
#    near_df = gettingneighbours(latitude,longitude)
#    vietnamese=0
#    sushi=0
#    pubs=0
#    pizza=0
#    kebab=0
#    italian=0
#    icecream=0
#    hotdogs=0
#    german=0
#    divebars=0
#    cocktailbars=0
#    cafes=0
#    burgers=0
#    bars=0
#    bakeries=0
#    own= 0
#    others=0
#    distanceToMitte=0
#    coordinates1 = latitude, longitude
#    for predictiony in range(len(near_df)):
#        coordinates2 = near_df.at[predictiony, 'latitude'], near_df.at[predictiony, 'longitude']
#        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
#        if(dist < radius):
#            if("vietnamese" in str(near_df.at[predictiony, 'categories'])):
#               vietnamese+=1
#            if("sushi" in str(near_df.at[predictiony, 'categories'])):
#                sushi+=1
#            if("pubs" in str(near_df.at[predictiony, 'categories'])):
#                pubs+=1
#            if("pizza" in str(near_df.at[predictiony, 'categories'])):
#                pizza+=1
#            if("kebab" in str(near_df.at[predictiony, 'categories'])):
#                kebab+=1
#            if("italian" in str(near_df.at[predictiony, 'categories'])):
#                italian+=1
#            if("icecream" in str(near_df.at[predictiony, 'categories'])):
#                icecream+=1
#            if("hotdogs" or "hotdog" in str(near_df.at[predictiony, 'categories'])):
#                hotdogs+=1
#            if("german" in str(near_df.at[predictiony, 'categories'])):
#                german+=1
#            if("divebars" in str(near_df.at[predictiony, 'categories'])):
#                divebars+=1
#            if("cocktailbars" in str(near_df.at[predictiony, 'categories'])):
#                cocktailbars+=1
#            if("cafes" or "coffee" in str(near_df.at[predictiony, 'categories'])):
#                cafes+=1
#            if("burgers" in str(near_df.at[predictiony, 'categories'])):
#                burgers+=1
#            if("bars" in str(near_df.at[predictiony, 'categories'])):
#                bars+=1
#            if("bakeries" in str(near_df.at[predictiony, 'categories'])):
#                bakeries+=1
#            if(own_category in str(near_df.at[predictiony, 'categories'])):
#                own+=1
#            else:
#                others+=1
#    regr = linear_model.LinearRegression()
#    regr.fit(train_feature_df_1, train_label_df)
#    predscore = regr.predict([[price,others,own,vietnamese,sushi,pubs,pizza,kebab,italian,
#                               icecream,hotdogs,german,divebars,cocktailbars,cafes,burgers,bars,bakeries,distanceToMitte]])
#    return(predscore)
#    
##Gibt eie Dataframe mit nur "Nachbarn" zurück -> soll "prediction" optimieren
#def gettingneighbours(latitude,longitude):
#    near_df = pd.DataFrame(columns=['latitude','longitude','categories'])
#    coordinates1 = latitude, longitude
#    for gettingneighboursx in range(len(train_feature_df)):
#        coordinates2 = train_feature_df.at[gettingneighboursx, 'latitude'], train_feature_df.at[gettingneighboursx, 'longitude']
#        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
#        if(dist < radius):
#            lat = train_feature_df.at[gettingneighboursx,'latitude']
#            lon = train_feature_df.at[gettingneighboursx,'longitude']
#            cat = train_feature_df.at[gettingneighboursx,'categories']
#            near_df = near_df.append({'latitude' : lat , 'longitude' : lon , 'categories' : cat}, ignore_index = True)
#    return(near_df)
#        
##testing
#def test(test_df):
#    ergebnis_df = pd.DataFrame(columns=['score','predictedscore','dif'])
#    ergebnis_df.at[0,'score'] = test_label_df.at[0,'score']
#    if(len(test_df)>=500):
#        modop_test = 2
#    if(len(test_df)>=250):
#        modop_test = 5
#    if(len(test_df)<250):
#        modop_test = 10
#    for test in range(len(test_df)):
#        if((test /len(test_df)*100) % modop_test ==0):
#            print(str(test/len(test_df)*100)+'%')
#        dif=0
#        
#        lat=test_df.at[test,'latitude']
#        lon=test_df.at[test,'longitude']
#        pri=test_df.at[test,'price']
#        cat=test_df.at[test,'categories']
#        pred = prediction(lat,lon,pri,cat)
#        #print(pred)
#        pred= float(pred)
#        
#        ergebnis_df.at[test,'score'] = test_label_df.at[test,'score']
#        ergebnis_df.at[test,'predictedscore'] = pred
#        dif = abs(ergebnis_df.at[test,'score']-pred)
#        ergebnis_df.at[test,'dif'] = dif
#    print(ergebnis_df)
#    print('Score:')
#    print('Maximaler Score: ' + str(np.max(ergebnis_df['score'])))
#    print('Minimaler Score: ' + str(np.min(ergebnis_df['score'])))
#    print('Dif von Score:' + str(np.max(ergebnis_df['score'])-np.min(ergebnis_df['score'])))
#    print('PredictedScore:')
#    print('Maximaler PredictedScore: ' + str(np.max(ergebnis_df['predictedscore'])))
#    print('Minimaler PredictedScore: ' + str(np.min(ergebnis_df['predictedscore'])))
#    print('Dif:')
#    print('Dif von PredictedScore:' + str(np.max(ergebnis_df['predictedscore'])-np.min(ergebnis_df['predictedscore'])))
#    print('Mean:' + str(np.mean(ergebnis_df['dif'])))
#    print('minimum: '+ str(np.min(ergebnis_df['dif'])))
#    print('maximum: ' + str(np.max(ergebnis_df['dif'])))
#    print('Varianz: ' + str(np.var(ergebnis_df['dif'])))
#    print('Spannweite Prediction: ' + str((np.min(ergebnis_df['predictedscore'] - np.max(ergebnis_df['predictedscore'])))))
#    print('Spannweite echter Score: ' + str((np.min(ergebnis_df['score'] - np.max(ergebnis_df['score'])))))
#    
#plot.scatter(train_feature_df,train_label_df,color='red')
#plot.show()