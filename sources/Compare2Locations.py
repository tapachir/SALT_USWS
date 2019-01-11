

# -*- coding: utf-8 -*-

from __future__ import print_function

from pymongo import MongoClient
import argparse
import json
import pprint
import requests
import sys
import urllib
import csv
import mpu

from urllib.error import HTTPError
from urllib.parse import quote
from urllib.parse import urlencode



client = MongoClient('localhost', 27017)
db = client.test
yelp_collection= db.test


# Point one 52.542026, 13.413098
lat1 = 52.542026
lon1 = 13.413098
x = 0
category_of_Object = "sushi"

same_category= []
same_category_and_close = []

#for record in yelp_collection.find({"categories.0.alias": category_of_Object}):
for record in yelp_collection.find({}):
     same_category.append(record)


for i in range(len(same_category)):
    try:
         lat2 = same_category[i]["coordinates"]["latitude"]
         lon2 = same_category[i]["coordinates"]["longitude"]
         print(lat2,lon2)
         dist = mpu.haversine_distance((lat1, lon1), (lat2, lon2))
         print(dist)
         if dist < 0.5:
              same_category_and_close.append(same_category[i])
         x = x +1
         print(x)
    except:
        pass

print("total amount same category", x)
print("restaurants close (under 0,5km):",(len(same_category_and_close)))



with open("test.csv", "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')

        for values in range(len(same_category_and_close)):
            all =[]
            rating = same_category_and_close[values]["rating"]
            review_count = same_category_and_close[values]["review_count"]
            price= same_category_and_close[values]["price"]
            all.append(rating)
            all.append(review_count)
            all.append(price)
            writer.writerow(all)


# one_star_rating = []
# two_star_rating = []
# three_star_rating = []
# four_star_rating = []
# five_star_rating = []
#
# for y in range(len(same_category_and_close)):
#       rating = same_category_and_close[y]["rating"]
#
#       if rating < 1 and rating <2:
#            one_star_rating.append(same_category_and_close[y])
#       if rating < 2 and rating >1:
#            two_star_rating.append(same_category_and_close[y])
#       if rating < 3 and rating >2:
#            three_star_rating.append(same_category_and_close[y])
#       if rating < 4 and rating > 3 :
#            four_star_rating.append(same_category_and_close[y])
#       if (rating == 4) or rating < 5 and rating > 4:
#            five_star_rating.append(same_category_and_close[y])
#
# print("one stars : ", len(one_star_rating))
# print("two stars : ", len(two_star_rating))
# print("three stars : ", len(three_star_rating))
# print("four stars : ", len(four_star_rating))
# print("five stars : ", len(five_star_rating))
#
#
# review_count_under10 =[]
# review_count_under30 =[]
# review_count_under50 =[]
# review_count_under100 =[]
# review_count_under200 =[]
#
#
# for y in range(len(same_category_and_close)):
#       review_count = same_category_and_close[y]["review_count"]
#
#       if review_count < 10 :
#            review_count_under10.append(same_category_and_close[y])
#       if review_count < 30 and review_count > 10 :
#            review_count_under30.append(same_category_and_close[y])
#       if review_count < 50 and review_count >30 :
#            review_count_under50.append(same_category_and_close[y])
#       if review_count < 100 and review_count > 50 :
#            review_count_under100.append(same_category_and_close[y])
#       if review_count < 200 and review_count >100:
#            review_count_under200.append(same_category_and_close[y])
#
# print("Review Count under 10 : ", len(review_count_under10))
# print("Review Count under 30 : ", len(review_count_under30))
# print("Review Count under 50 : ", len(review_count_under50))
# print("Review Count under 100 : ", len(review_count_under100))
# print("Review Count under 200 : ", len(review_count_under200))
#
#


categories ={}

for w in range(len(same_category_and_close)):
    category= same_category_and_close[w]["categories"][0]["alias"]


    if not category in categories:
        categories[category] =1
    else:
        categories[category] += 1


pprint.pprint(categories)