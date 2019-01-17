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

df = df[0:300]

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
train_feature_df['others'] = 0 # new column with 0
train_feature_df['koreanneigbours'] = 0
train_feature_df['italianneigbhours'] = 0
train_feature_df['vietnameseneigbhours'] = 0
train_feature_df['japaneseneigbhours'] = 0
train_feature_df['own_category'] = 0

#test_feature_df
test_feature_df = test_df.drop(['rating','review_count'], axis=1)
test_feature_df['others'] = 0 # new column with 0
test_feature_df['koreanneigbours'] = 0
test_feature_df['italianneigbhours'] = 0
test_feature_df['vietnameseneigbhours'] = 0
test_feature_df['japaneseneigbhours'] = 0
test_feature_df['own_category'] = 0

#train_feature_df1 nur für LinearRegression
train_feature_df1 = train_feature_df.drop(['categories'],axis=1)

#Setzt den Radius für die Suche nach Nachbarn fest
radius = 500
#Bereitet das train_dataframe fürs machinelearning for. füllt die features aus (anzahl der nachabrn)
for x in range(len(train_feature_df)):
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
            if("Korean" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'koreanneigbours']+=1
            if("Italian" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'italianneigbhours']+=1
            if("Vietnamese" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'vietnameseneigbhours']+=1
            if("Japanese" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'japaneseneigbhours']+=1
            if(own_category in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x,'own_category']+=1
            else:
                train_feature_df.at[x, 'others']+=1

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
            if("Korean" in str(test_feature_df.at[y, 'categories'])):
                         test_feature_df.at[x, 'koreanneigbours']+=1
            if("Italian" in str(test_feature_df.at[y, 'categories'])):
                         test_feature_df.at[x, 'italianneigbhours']+=1
            if("Vietnamese" in str(test_feature_df.at[y, 'categories'])):
                         test_feature_df.at[x, 'vietnameseneigbhours']+=1
            if("Japanese" in str(test_feature_df.at[y, 'categories'])):
                         test_feature_df.at[x, 'japaneseneigbhours']+=1
            if(own_category in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x,'own_category']+=1
            else:
                test_feature_df.at[x, 'others']+=1

##exportiert das train_feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
#train_feature_df.to_csv('train_feature_df.csv', sep='\t', encoding='utf-8')
##exportiert das test_feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
#test_feature_df.to_csv('test_feature_df.csv', sep='\t', encoding='utf-8')

#mainmethode
def prediction(latitude,longitude,price,category):
    price = price
    train_feature_df_1 = train_feature_df.drop(['latitude','longitude','categories'],axis=1)
    near_df = gettingneighbours(latitude,longitude)
    koreans =0
    italians =0
    vietnameses=0
    japaneses=0
    others = 0
    owncategory=0
    own_category = category
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
            if(own_category in str(near_df.at[y, 'categories'])):
                         owncategory+=1
            else:
                others+=1
    regr = linear_model.LinearRegression()
    regr.fit(train_feature_df_1, train_label_df)
    predscore = regr.predict([[price,others,koreans,italians,vietnameses,japaneses,owncategory]])
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
        pred= float(pred)
        
        ergebnis_df.at[x,'score'] = test_label_df.at[x,'score']
        ergebnis_df.at[x,'predictedscore'] = pred
        dif = abs(ergebnis_df.at[x,'score']-pred)
        ergebnis_df.at[x,'dif'] = dif
    print(ergebnis_df)
    print('Maximaler PredictedScore: ' + str(np.max(ergebnis_df['predictedscore'])))
    print('Minimaler PredictedScore: ' + str(np.min(ergebnis_df['predictedscore'])))
    print('Dif von PredictedScore:' + str(np.max(ergebnis_df['predictedscore'])-np.min(ergebnis_df['predictedscore'])))
    print('Mean:' + str(np.mean(ergebnis_df['dif'])))
    print('minimum: '+ str(np.min(ergebnis_df['dif'])))
    print('maximum: ' + str(np.max(ergebnis_df['dif'])))
    print('Varianz: ' + str(np.var(ergebnis_df['dif'])))

#plot.scatter(train_feature_df,train_label_df,color='red')
#plot.show()