# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 08:11:21 2018
@author: max
"""
import csv

import pandas as pd
import numpy as np
from sklearn import linear_model
import geopy.distance
import sklearn


def compare(lat,lon):
    under1km = 0
    under500m = 0
    under250m = 0
    coordinates1 = float(lat),float(lon)

    with open('/home/tahir/Documents/code/tahir/SALT_USWS/sources/cleanedtops.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)


        for row in csv_reader:
            coordinates2 = float(row[4]), float(row[5])

            dist = geopy.distance.geodesic(coordinates1, coordinates2).m
            dist = int(dist)

            if dist <250:
                under250m +=1
            if dist <500:
                under500m +=1
            if dist <1000:
                under1km +=1

        print("under1000m" , under1km)
        print("under500m", under500m)
        print("under250m", under250m)

        result =  (under250m*6) + (under500m*3) + (under1km*1)
        return result
def addReachability(dataframe):

    reachabilities = []
    for x in range(len(dataframe)):
         lat = dataframe.at[x, "latitude"]
         lon = dataframe.at[x, "longitude"]
         resultReachability = compare(lat, lon)
         reachabilities.append(resultReachability)
         #dataframe["reachability"] = pd.Series(resultReachability, index=x)
         #dataframe.append({"reachability": resultReachability}, ignore_index=True)
         #print(dataframe.at[x,"reachability"])
    dataframe["reachability"] = reachabilities
    return dataframe
# "all.csv" wird benötigt, da nur diese "categories" enthält, dafür aber die Koordinaten nur in einer Spalte (nicht zwei)
alle = pd.read_csv(r'/home/tahir/Downloads/all(1).csv')
#
df = pd.read_csv(r'/home/tahir/Downloads/test3(1).csv')
df = df.drop(['name'], axis=1)  # useless columns
df.columns = ['latitude', 'longitude', 'rating', 'review_count', 'price']  # rename
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
df['price'] = df['price'].fillna(2)

df = df.dropna()
df = df[0:10]
df = addReachability(df)
print(df)
#df = df[0:100]

# # df splitten
# train_df, test_df = sklearn.model_selection.train_test_split(df, test_size=0.2)
# train_df = train_df.reset_index()  # vielleicht nicht nötig
# train_df = train_df.drop(['index'], axis=1)
# test_df = test_df.reset_index()
# test_df = test_df.drop(['index'], axis=1)
#
# # train_label_df soll nur score (rating+review_count) haben
# train_label_df = train_df.drop(['longitude', 'latitude', 'price', 'price'], axis=1)
# train_label_df['score'] = train_label_df['rating'] * train_label_df['review_count']
# train_label_df = train_label_df.drop(['rating', 'review_count'], axis=1)
#
# # test_label_df
# test_label_df = test_df.drop(['longitude', 'latitude', 'price', 'price'], axis=1)
# test_label_df['score'] = test_label_df['rating'] * test_label_df['review_count']
# test_label_df = test_label_df.drop(['rating', 'review_count'], axis=1)
#
# # train_feature_df
# train_feature_df = train_df.drop(['rating', 'review_count'], axis=1)
# train_feature_df['categories'] = alle['categories'].copy()
# train_feature_df['others'] = 0  # new column with 0
# train_feature_df['koreanneigbours'] = 0
# train_feature_df['italianneigbhours'] = 0
# train_feature_df['vietnameseneigbhours'] = 0
# train_feature_df['japaneseneigbhours'] = 0
#
# # test_feature_df
# test_feature_df = test_df.drop(['rating', 'review_count'], axis=1)
# test_feature_df['categories'] = alle['categories'].copy()
# test_feature_df['others'] = 0  # new column with 0
# test_feature_df['koreanneigbours'] = 0
# test_feature_df['italianneigbhours'] = 0
# test_feature_df['vietnameseneigbhours'] = 0
# test_feature_df['japaneseneigbhours'] = 0
#
# # train_feature_df1 nur für LinearRegression
# train_feature_df1 = train_feature_df.drop(['categories'], axis=1)
#
# # Setzt den Radius für die Suche nach Nachbarn fest
# radius = 750
#
# # Bereitet das train_dataframe fürs machinelearning for. füllt die features aus (anzahl der nachabrn)
# for x in range(len(train_feature_df)):
#     if (x % 10 == 0):
#         print(str(x) + 'from' + str(len(train_feature_df)))
#         coordinates1 = train_feature_df.at[x, 'latitude'], train_feature_df.at[x, 'longitude']
#         for y in range(len(train_feature_df)):
#             coordinates2 = train_feature_df.at[y, 'latitude'], train_feature_df.at[y, 'longitude']
#             dist = geopy.distance.geodesic(coordinates1, coordinates2).m
#             if (dist < radius and x != y):
#                 if ("Korean" in str(train_feature_df.at[y, 'categories'])):
#                     train_feature_df.at[x, 'koreanneigbours'] += 1
#                 if ("Italian" in str(train_feature_df.at[y, 'categories'])):
#                     train_feature_df.at[x, 'italianneigbhours'] += 1
#                 if ("Vietnamese" in str(train_feature_df.at[y, 'categories'])):
#                     train_feature_df.at[x, 'vietnameseneigbhours'] += 1
#                 if ("Japanese" in str(train_feature_df.at[y, 'categories'])):
#                     train_feature_df.at[x, 'japaneseneigbhours'] += 1
#                 else:
#                     train_feature_df.at[x, 'others'] += 1
#
# # prepare test features
# for x in range(len(test_feature_df)):
#     if (x % 10 == 0):
#         print(str(x) + 'from' + str(len(test_feature_df)))
#     coordinates1 = test_feature_df.at[x, 'latitude'], test_feature_df.at[x, 'longitude']
#     for y in range(len(test_feature_df)):
#         coordinates2 = test_feature_df.at[y, 'latitude'], test_feature_df.at[y, 'longitude']
#         dist = geopy.distance.geodesic(coordinates1, coordinates2).m
#         if (dist < radius and x != y):
#             if ("Korean" in str(test_feature_df.at[y, 'categories'])):
#                 test_feature_df.at[x, 'koreanneigbours'] += 1
#             if ("Italian" in str(test_feature_df.at[y, 'categories'])):
#                 test_feature_df.at[x, 'italianneigbhours'] += 1
#             if ("Vietnamese" in str(test_feature_df.at[y, 'categories'])):
#                 test_feature_df.at[x, 'vietnameseneigbhours'] += 1
#             if ("Japanese" in str(test_feature_df.at[y, 'categories'])):
#                 test_feature_df.at[x, 'japaneseneigbhours'] += 1
#             else:
#                 test_feature_df.at[x, 'others'] += 1
#
#
# ##exportiert das train_feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
# # train_feature_df.to_csv('train_feature_df.csv', sep='\t', encoding='utf-8')
# ##exportiert das test_feature_df als .csv Datei in den Ordner, in dem diese Datei liegt
# # test_feature_df.to_csv('test_feature_df.csv', sep='\t', encoding='utf-8')
#
# # mainmethode
# def prediction(latitude, longitude, price):
#     price = price
#     train_feature_df1 = train_feature_df.drop('categories', axis=1)
#     near_df = gettingneighbours(latitude, longitude)
#     koreans = 0
#     italians = 0
#     vietnameses = 0
#     japaneses = 0
#     others = 0
#     coordinates1 = latitude, longitude
#     for y in range(len(near_df)):
#         coordinates2 = near_df.at[y, 'latitude'], near_df.at[y, 'longitude']
#         dist = geopy.distance.geodesic(coordinates1, coordinates2).m
#         if (dist < radius):
#             if ("Korean" in str(near_df.at[y, 'categories'])):
#                 koreans += 1
#             if ("Italian" in str(near_df.at[y, 'categories'])):
#                 italians += 1
#             if ("Vietnamese" in str(near_df.at[y, 'categories'])):
#                 vietnameses += 1
#             if ("Japanese" in str(near_df.at[y, 'categories'])):
#                 japaneses += 1
#             else:
#                 others += 1
#     regr = linear_model.LinearRegression()
#     regr.fit(train_feature_df1, train_label_df)
#     predscore = regr.predict([[latitude, longitude, price, others, koreans, italians, vietnameses, japaneses]])
#
#     print('------ Lineare Regression -----')
#     print('Funktion via sklearn: y = %.3f * x + %.3f' % (regr.coef_[0], regr.intercept_))
#     print("Alpha: {}".format(regr.intercept_))
#     print("Beta: {}".format(regr.coef_[0]))
#     print("Training Set R² Score: {:.2f}".format(regr.score(train_feature_df1, train_label_df)))
#     #print("Test Set R² Score: {:.2f}".format(regr.score(X_test, y_test)))
#     print("\n")
#     return (predscore)
#
#
# # Gibt eie Dataframe mit nur "Nachbarn" zurück -> soll "prediction" optimieren
# def gettingneighbours(latitude, longitude):
#     near_df = pd.DataFrame(columns=['latitude', 'longitude', 'categories'])
#     coordinates1 = latitude, longitude
#     for x in range(len(train_feature_df)):
#         coordinates2 = train_feature_df.at[x, 'latitude'], train_feature_df.at[x, 'longitude']
#         dist = geopy.distance.geodesic(coordinates1, coordinates2).m
#         if (dist < radius):
#             lat = train_feature_df.at[x, 'latitude']
#             lon = train_feature_df.at[x, 'longitude']
#             cat = train_feature_df.at[x, 'categories']
#             near_df = near_df.append({'latitude': lat, 'longitude': lon, 'categories': cat}, ignore_index=True)
#     return (near_df)
#
#
# # testing
# ergebnis_df = pd.DataFrame(columns=['score', 'predictedscore', 'dif'])
# ergebnis_df.at[0, 'score'] = test_label_df.at[0, 'score']
# for x in range(len(test_df)):
#     dif = 0
#
#     lat = test_df.at[x, 'latitude']
#     lon = test_df.at[x, 'longitude']
#     pri = test_df.at[x, 'price']
#     pred = prediction(lat, lon, pri)
#     pred = float(pred)
#
#     ergebnis_df.at[x, 'score'] = test_label_df.at[x, 'score']
#     ergebnis_df.at[x, 'predictedscore'] = pred
#     dif = abs(ergebnis_df.at[x, 'score'] - pred)
#     ergebnis_df.at[x, 'dif'] = dif
#
#
# print('Mean:' + str(np.mean(ergebnis_df['dif'])))
# print('minimum: ' + str(np.min(ergebnis_df['dif'])))
# print('maximum: ' + str(np.max(ergebnis_df['dif'])))