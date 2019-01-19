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

df = pd.read_csv(r'/home/tahir/Documents/code/tahir/SALT_USWS/sources/allWithCategory.csv')
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

df = df[0:1000]

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
train_feature_df['turskishneigbours'] = 0
train_feature_df['sushineigbours'] = 0
train_feature_df['pubsneigbours'] = 0
train_feature_df['pizzaneigbours'] = 0
train_feature_df['museumsneigbours'] = 0
train_feature_df['landmarksneigbours'] = 0
train_feature_df['kebabneigbours'] = 0
train_feature_df['italianneigbours'] = 0
train_feature_df['icecreamneigbours'] = 0
train_feature_df['hotdogsneigbours'] = 0
train_feature_df['germanneigbours'] = 0
train_feature_df['divebarsneigbours'] = 0
train_feature_df['cocktailbarsneigbours'] = 0
train_feature_df['coffeesneigbours'] = 0
train_feature_df['cafesneigbours'] = 0
train_feature_df['burgersneigbours'] = 0
train_feature_df['barsneigbours'] = 0
train_feature_df['bakeriesneigbours'] = 0

#test_feature_df
test_feature_df = test_df.drop(['rating','review_count'], axis=1)
test_feature_df['otherneigbours'] = 0
test_feature_df['own_category'] = 0
test_feature_df['vietnameseneigbours'] = 0
test_feature_df['turskishneigbours'] = 0
test_feature_df['sushineigbours'] = 0
test_feature_df['pubsneigbours'] = 0
test_feature_df['pizzaneigbours'] = 0
test_feature_df['museumsneigbours'] = 0
test_feature_df['landmarksneigbours'] = 0
test_feature_df['kebabneigbours'] = 0
test_feature_df['italianneigbours'] = 0
test_feature_df['icecreamneigbours'] = 0
test_feature_df['hotdogsneigbours'] = 0
test_feature_df['germanneigbours'] = 0
test_feature_df['divebarsneigbours'] = 0
test_feature_df['cocktailbarsneigbours'] = 0
test_feature_df['coffeesneigbours'] = 0
test_feature_df['cafesneigbours'] = 0
test_feature_df['burgersneigbours'] = 0
test_feature_df['barsneigbours'] = 0
test_feature_df['bakeriesneigbours'] = 0


#train_feature_df1 nur für LinearRegression
train_feature_df1 = train_feature_df.drop(['categories'],axis=1)

#Setzt den Radius für die Suche nach Nachbarn fest
radius = 750
#Bereitet das train_dataframe fürs machinelearning for. füllt die features aus (anzahl der n    achabrn)
for x in range(len(train_feature_df)):

    dist = geopy.distance.geodesic(coordinates1, coordinates2).m
    dist = int(dist)
    if(len(train_feature_df)>=1000):
        modop_train = 5
    if(len(train_feature_df)<1000):
        modop_train = 10
    if((x /len(train_feature_df)*100) % modop_train ==0):
            print(str(x/len(train_feature_df)*100)+'%')
    own_category = train_feature_df.at[x,'categories']
    coordinates1 = train_feature_df.at[x, 'latitude'], train_feature_df.at[x, 'longitude']
    for y in range(len(train_feature_df)):
        coordinates2 = train_feature_df.at[y, 'latitude'], train_feature_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius and x != y):
            if("vietnamese" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'vietnameseneigbours']+=1
            if("turskish" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'turskishneigbours']+=1
            if("sushi" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'sushineigbours']+=1
            if("pubs" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'pubsneigbours']+=1
            if("pizza" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'pizzaneigbours']+=1
            if("museums" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'museumsneigbours']+=1
            if("landmarks" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'landmarksneigbours']+=1
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
            if("cafes" in str(train_feature_df.at[y, 'categories'])):
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
for x in range(len(test_feature_df)):
    if(len(train_feature_df)>=500):
        modop_test = 5
    if(len(train_feature_df)<500):
        modop_test = 10
    if((x /len(test_feature_df)*100) % modop_test ==0):
            print(str(x/len(test_feature_df)*100)+'%')
    own_category = test_feature_df.at[x,'categories']
    coordinates1 = test_feature_df.at[x, 'latitude'], test_feature_df.at[x, 'longitude']
    for y in range(len(test_feature_df)):
        coordinates2 = test_feature_df.at[y, 'latitude'], test_feature_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius and x != y):
            if("vietnamese" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'vietnameseneigbours']+=1
            if("turskish" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'turskishneigbours']+=1
            if("sushi" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'sushineigbours']+=1
            if("pubs" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'pubsneigbours']+=1
            if("pizza" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'pizzaneigbours']+=1
            if("museums" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'museumsneigbours']+=1
            if("landmarks" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'landmarksneigbours']+=1
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
            if("coffee" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'coffeesneigbours']+=1
            if("cafes" in str(test_feature_df.at[y, 'categories'])):
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
train_feature_df.to_csv('train_feature_df.csv', sep='\t', encoding='utf-8')
##exportiert das test_feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
test_feature_df.to_csv('test_feature_df.csv', sep='\t', encoding='utf-8')
##exportiert das test_df als .csv Datei in den Ordner, in dem diese Datei liegt
test_df.to_csv('test_feature.csv', sep='\t', encoding='utf-8')

#mainmethode
def prediction(latitude,longitude,price,category):
    own_category = category
    price = price
    train_feature_df_1 = train_feature_df.drop(['latitude','longitude','categories'],axis=1)
    near_df = gettingneighbours(latitude,longitude)
    vietnamese=0
    turskish=0
    sushi=0
    pubs=0
    pizza=0
    museums=0
    landmarks=0
    kebab=0
    italian=0
    icecream=0
    hotdogs=0
    german=0
    divebars=0
    coffee=0
    cocktailbars=0
    cafes=0
    burgers=0
    bars=0
    bakeries=0
    own= 0
    others=0
    coordinates1 = latitude, longitude
    for y in range(len(near_df)):
        coordinates2 = near_df.at[y, 'latitude'], near_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1,coordinates2).m
        if(dist < radius):
            if("vietnamese" in str(test_feature_df.at[y, 'categories'])):
               vietnamese+=1
            if("turskish" in str(test_feature_df.at[y, 'categories'])):
                turskish+=1
            if("sushi" in str(test_feature_df.at[y, 'categories'])):
                sushi+=1
            if("pubs" in str(test_feature_df.at[y, 'categories'])):
                pubs+=1
            if("pizza" in str(test_feature_df.at[y, 'categories'])):
                pizza+=1
            if("museums" in str(test_feature_df.at[y, 'categories'])):
                museums+=1
            if("landmarks" in str(test_feature_df.at[y, 'categories'])):
                landmarks+=1
            if("kebab" in str(test_feature_df.at[y, 'categories'])):
                kebab+=1
            if("italian" in str(test_feature_df.at[y, 'categories'])):
                italian+=1
            if("icecream" in str(test_feature_df.at[y, 'categories'])):
                icecream+=1
            if("hotdogs" or "hotdog" in str(test_feature_df.at[y, 'categories'])):
                hotdogs+=1
            if("german" in str(test_feature_df.at[y, 'categories'])):
                german+=1
            if("divebars" in str(test_feature_df.at[y, 'categories'])):
                divebars+=1
            if("cocktailbars" in str(test_feature_df.at[y, 'categories'])):
                cocktailbars+=1
            if("coffee" in str(test_feature_df.at[y, 'categories'])):
                coffee+=1
            if("cafes" in str(test_feature_df.at[y, 'categories'])):
                cafes+=1
            if("burgers" in str(test_feature_df.at[y, 'categories'])):
                burgers+=1
            if("bars" in str(test_feature_df.at[y, 'categories'])):
                bars+=1
            if("bakeries" in str(test_feature_df.at[y, 'categories'])):
                bakeries+=1
            if(own_category in str(test_feature_df.at[y, 'categories'])):
                own+=1
            else:
                others+=1
    regr = linear_model.LinearRegression()
    regr.fit(train_feature_df_1, train_label_df)
    predscore = regr.predict([[price,others,own,vietnamese,turskish,sushi,pubs,pizza,museums,landmarks,kebab,italian,
                               icecream,hotdogs,german,divebars,cocktailbars,coffee,cafes,burgers,bars,bakeries]])
    predscore = str(round(predscore[0][0], 2))
    return(predscore)

#Gibt eie Dataframe mit nur "Nachbarn" zurück -> soll "prediction" optimieren
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
    for x in range(len(test_df)):
        if((x /len(test_df)*100) % modop_test ==0):
            print(str(x/len(test_df)*100)+'%')
        dif=0
        
        lat=test_df.at[x,'latitude']
        lon=test_df.at[x,'longitude']
        pri=test_df.at[x,'price']
        cat=test_df.at[x,'categories']
        pred = prediction(lat,lon,pri,cat)
        print(pred)
        pred= float(pred)
        
        ergebnis_df.at[x,'score'] = test_label_df.at[x,'score']
        ergebnis_df.at[x,'predictedscore'] = pred
        dif = abs(ergebnis_df.at[x,'score']-pred)
        ergebnis_df.at[x,'dif'] = dif
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

#plot.scatter(train_feature_df,train_label_df,color='red')
#plot.show()