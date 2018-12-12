
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

# test2 = pd.read_csv(r'C:\Users\max\Downloads\test2.csv')
# \\homedrive.login.htw-berlin.de\s0565546\WI-Profile\Desktop\Downloads
# test2 = pd.read_csv(r'\\homedrive.login.htw-berlin.de\s0565546\WI-Profile\Desktop\Downloads\test2.csv')
# test3 = pd.read_csv(r'C:\Users\max\Downloads\test3.csv')
# \\homedrive.login.htw-berlin.de\s0565546\WI-Profile\Desktop\Downloads
test3 = pd.read_csv(r'/home/tahir/test3.csv')

df = pd.read_csv(r'/home/tahir/test3.csv')
df1 = df.drop(['price','name','review_count','rating'], axis=1)


df2= df.drop('price', axis=1)
df2.drop('name', axis=1)
df2.drop('review_count', axis=1)
df2.drop('coordinates.longitude',axis=1)
df2.drop('coordinates.latitude',axis=1)


print(df.isnull().any())
pprint.pprint(df1)
df1.columns = ['latitude', 'longitude']
#
# # for i in range(len(features)):
# #     if (features.latitude[i].isnull()):
# #         features.drop(features.index[i])
#
NoValues = df1[df1.latitude.isnull()]# or features.coordinates.longitude ==0]
print(len(df1))
for i in range (len(NoValues)):
        temp = (NoValues.index[i])
        print(df1.index[NoValues.index[i]])
        df1.drop(df1.index[NoValues.index[i]])

print(len(df1))
# NoValues.drop(['longitude','latitude'],axis=1)
# print(NoValues)

# rating = df1['rating']
# features = df1.drop('rating', axis=1)
#
# regr = linear_model.LinearRegression()
# regr.fit(features, rating)

# label = df['quality']
# features = df.drop('quality', axis=1)

# regr = linear_model.LinearRegression()
# regr.fit(features, label)
#
# # 1print regr.predict([[7.4,0.66,0,1.8,0.075,13,40,0.9978,3.51,0.56,9.4]]).tolist()
#
# pickle.dump(regr, open("model.pk", "wb"))
#
# # loading a model from file
# model = pickle.load(open("model.pkl", "r"))
#
# #################################
#
# test3['score'] = 0
#
