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
# renaming columns
df.columns = ['latitude', 'longitude', 'rating', 'review_count', 'price', 'categories']

# revwiew_count aufräumen / kategorisieren
df['review_count'] = df['review_count'].replace([range(1, 3)], 1.0)
df['review_count'] = df['review_count'].replace([range(3, 6)], 1.25)
df['review_count'] = df['review_count'].replace([range(6, 11)], 1.5)
df['review_count'] = df['review_count'].replace([range(11, 51)], 1.75)
df['review_count'] = df['review_count'].replace([range(51, 10000)], 2.0)

# price feature aufräumen
df['price'] = df['price'].replace('€', 1)
df['price'] = df['price'].replace('€€', 2)
df['price'] = df['price'].replace('$$', 2)
df['price'] = df['price'].replace('€€€', 3)
df['price'] = df['price'].replace('€€€€', 4)
# df['price'] = df['price'].fillna(2)

df = df.dropna()

df = df[0:2000]

# df splitten
train_df, test_df = sklearn.model_selection.train_test_split(df, test_size=0.2)
train_df = train_df.reset_index()  # vielleicht nicht nötig
train_df = train_df.drop(['index'], axis=1)
test_df = test_df.reset_index()
test_df = test_df.drop(['index'], axis=1)

# train_label_df soll nur score (rating+review_count) haben
train_label_df = train_df.drop(['longitude', 'latitude', 'price', 'price'], axis=1)
train_label_df['score'] = train_label_df['rating'] * train_label_df['review_count']
train_label_df = train_label_df.drop(['rating', 'review_count', 'categories'], axis=1)

# test_label_df
test_label_df = test_df.drop(['longitude', 'latitude', 'price', 'price'], axis=1)
test_label_df['score'] = test_label_df['rating'] * test_label_df['review_count']
test_label_df = test_label_df.drop(['rating', 'review_count', 'categories'], axis=1)

# train_feature_df
train_feature_df = train_df.drop(['rating', 'review_count'], axis=1)
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
train_feature_df['distanceToMitte'] = 0

# test_feature_df
test_feature_df = test_df.drop(['rating', 'review_count'], axis=1)
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
test_feature_df['distanceToMitte'] = 0

# train_feature_df1 nur für LinearRegression
train_feature_df1 = train_feature_df.drop(['categories'], axis=1)

# Setzt den Radius für die Suche nach Nachbarn fest
coordinatesMitte = (52.524436, 13.409616)

radius = 750
# Bereitet das train_dataframe fürs machinelearning for. füllt die features aus (anzahl der nachabrn)
print(datetime.datetime.now())
for x in range(len(train_feature_df)):
    print('X:' + ' ' + str(x))
    print('Aktuelle Zeit:' + ' ' + str(datetime.datetime.now()))
    if (len(train_feature_df) >= 100):
        modop_train = 1
    if (len(train_feature_df) >= 500):
        modop_train = 5
    if (len(train_feature_df) < 500):
        modop_train = 10
    if ((x / len(train_feature_df) * 100) % modop_train == 0):
        print(str(x / len(train_feature_df) * 100) + '%')
    own_category = train_feature_df.at[x, 'categories']
    coordinates1 = train_feature_df.at[x, 'latitude'], train_feature_df.at[x, 'longitude']
    train_feature_df.at[x, 'distanceToMitte'] = geopy.distance.geodesic(coordinates1, coordinatesMitte).m
    for y in range(len(train_feature_df)):
        coordinates2 = train_feature_df.at[y, 'latitude'], train_feature_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1, coordinates2).m
        if (dist < radius and x != y):
            if ("vietnamese" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'vietnameseneigbours'] += 1
                if ("vietnamese" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("sushi" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'sushineigbours'] += 1
                if ("sushi" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("pubs" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'pubsneigbours'] += 1
                if ("pubs" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("pizza" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'pizzaneigbours'] += 1
                if ("pizza" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("kebab" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'kebabneigbours'] += 1
                if ("kebab" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("italian" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'italianneigbours'] += 1
                if ("italian" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("icecream" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'icecreamneigbours'] += 1
                if ("icecream" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("hotdogs" or "hotdog" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'hotdogsneigbours'] += 1
                if ("hotdogs" or "hotdog" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("german" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'germanneigbours'] += 1
                if ("german" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("divebars" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'divebarsneigbours'] += 1
                if ("divebars" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("cocktailbars" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'cocktailbarsneigbours'] += 1
                if ("cocktailbars" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("cafes" or "coffee" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'cafesneigbours'] += 1
                if ("cafes" or "coffee" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("burgers" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'burgersneigbours'] += 1
                if ("burgers" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("bars" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'barsneigbours'] += 1
                if ("bars" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            if ("bakeries" in str(train_feature_df.at[y, 'categories'])):
                train_feature_df.at[x, 'bakeriesneigbours'] += 1
                if ("bakeries" == own_category):
                    train_feature_df.at[x, 'own_category'] += 1
            else:
                train_feature_df.at[x, 'otherneigbours'] += 1
                if (str(train_feature_df.at[y, 'categories']) == own_category):
                    train_feature_df.at[x, 'own_category'] += 1

# prepare test features
print(datetime.datetime.now())
for x in range(len(test_feature_df)):
    if (len(train_feature_df) >= 500):
        modop_test = 1
    if (len(train_feature_df) >= 250):
        modop_test = 5
    if (len(train_feature_df) < 250):
        modop_test = 10
    if ((x / len(test_feature_df) * 100) % modop_test == 0):
        print(str(x / len(test_feature_df) * 100) + '%')
    own_category = test_feature_df.at[x, 'categories']
    coordinates1 = test_feature_df.at[x, 'latitude'], test_feature_df.at[x, 'longitude']
    test_feature_df.at[x, 'distanceToMitte'] = geopy.distance.geodesic(coordinates1, coordinatesMitte).m
    for y in range(len(test_feature_df)):
        coordinates2 = test_feature_df.at[y, 'latitude'], test_feature_df.at[y, 'longitude']
        dist = geopy.distance.geodesic(coordinates1, coordinates2).m
        if (dist < radius and x != y):
            if ("vietnamese" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'vietnameseneigbours'] += 1
                if ("vietnamese" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("sushi" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'sushineigbours'] += 1
                if ("sushi" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("pubs" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'pubsneigbours'] += 1
                if ("pubs" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("pizza" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'pizzaneigbours'] += 1
                if ("pizza" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("kebab" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'kebabneigbours'] += 1
                if ("kebab" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("italian" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'italianneigbours'] += 1
                if ("italian" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("icecream" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'icecreamneigbours'] += 1
                if ("icecream" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("hotdogs" or "hotdog" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'hotdogsneigbours'] += 1
                if ("hotdogs" or "hotdog" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("german" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'germanneigbours'] += 1
                if ("german" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("divebars" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'divebarsneigbours'] += 1
                if ("divebars" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("cocktailbars" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'cocktailbarsneigbours'] += 1
                if ("cocktailbars" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("cafes" or "coffee" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'cafesneigbours'] += 1
                if ("cafes" or "coffee" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("burgers" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'burgersneigbours'] += 1
                if ("burgers" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("bars" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'barsneigbours'] += 1
                if ("bars" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            if ("bakeries" in str(test_feature_df.at[y, 'categories'])):
                test_feature_df.at[x, 'bakeriesneigbours'] += 1
                if ("bakeries" == own_category):
                    test_feature_df.at[x, 'own_category'] += 1
            else:
                test_feature_df.at[x, 'otherneigbours'] += 1
                if (str(test_feature_df.at[y, 'categories']) == own_category):
                    test_feature_df.at[x, 'own_category'] += 1

# Exportiert ALLE wichtigen gefüllten Dataframes
train_feature_df.to_csv(r'CSVFiles/train_feature_df.csv', sep=',', encoding='utf-8')
test_feature_df.to_csv(r'CSVFiles/test_feature_df.csv', sep=',', encoding='utf-8')
test_df.to_csv(r'CSVFiles/test_df.csv', sep=',', encoding='utf-8')
train_label_df.to_csv(r'CSVFiles/train_label_df.csv', sep=',', encoding='utf-8')
test_label_df.to_csv(r'CSVFiles/test_label_df.csv', sep=',', encoding='utf-8')

