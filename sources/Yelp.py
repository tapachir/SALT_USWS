# -*- coding: utf-8 -*-

from __future__ import print_function

from pymongo import MongoClient
import argparse
import json
import pprint
import requests
import sys
import urllib


from urllib.error import HTTPError
from urllib.parse import quote
from urllib.parse import urlencode




client = MongoClient('localhost', 27017)
db = client.test
yelp_collection= db.test
API_KEY= "G0j0MBJssiO-61bjpu-JYtmT2PnFO-lWL4TMh8QnG0Dg2laj2-Kx438OOjchucmAdF1DasKzVji73NPtmIShKIX5Sxirz7QaATe0Rl9FhXKMksvGuDmkEbIh02s7W3Yx"


# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.




# Defaults for our simple example.

DEFAULT_LONGITUDE = 13.3904
DEFAULT_LATITUDE = 52.5074
DEFAULT_RADIUS = 7000
SEARCH_LIMIT = 50
offset = 0
wholeLoopBoolean = True
counter = 0
index_for_terms = 0


def shift(latitude,longitude, shift_in_meters, lat_direction, long_direction):
    lat_shift = 0.000009 * shift_in_meters
    long_shift = 0.000015 * shift_in_meters
    return (
        latitude + lat_shift * lat_direction,
        longitude + long_shift * long_direction
    )


def refine_area_and_search_again(maps_type, latitude, longitude, radius,depth):
    print('Refining for {},{} - {}'.format(latitude,longitude, radius))
    new_radius = int(0.5858 * radius)
    center1 = shift(latitude,longitude, 0.4142 * radius, lat_direction=-1, long_direction=-1)
    center2 = shift(latitude,longitude, 0.4142 * radius, lat_direction=+1, long_direction=-1)
    center3 = shift(latitude,longitude, 0.4142 * radius, lat_direction=-1, long_direction=+1)
    center4 = shift(latitude,longitude, 0.4142 * radius, lat_direction=+1, long_direction=+1)

    query_api(maps_type, center1[0],center1[1], new_radius ,depth + '0')
    query_api(maps_type, center2[0],center2[1], new_radius, depth + '1')
    query_api(maps_type, center3[0],center3[1], new_radius, depth + '2')
    query_api(maps_type, center4[0],center4[1], new_radius, depth + '3')


def request(host, path, api_key, url_params=None):

    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {
        'Authorization': 'Bearer %s' % api_key,
    }

    print(u'Querying {0} ...'.format(url))

    response = requests.request('GET', url, headers=headers, params=url_params)
    #pprint.pprint(response.json())
    return response.json()


#def search(api_key, term, location):
def search(api_key, categorie_list, latitude, longitude, radius, offset):

    #print (offset)

    url_params = {
        'categories' : categorie_list,
        #'term': term.replace(' ', '+'),
        #'location': location.replace(' ', '+'),
        'latitude': latitude,
        'longitude': longitude,
        'radius': radius,
        'limit': SEARCH_LIMIT,
        'offset': offset

    }
    return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)


def get_business(api_key, business_id):

    business_path = BUSINESS_PATH + business_id

    return request(API_HOST, business_path, api_key)


def query_api(categories, latitude,longitude,  radius,depth):
    offset = 0
    response = search(API_KEY, categories, latitude, longitude, radius, offset)
    total = response.get('total')
    print(total, "This is the total and this is the depth", depth)




    if int(total) > 1000:
        print("ACHTUNG NEW CIRCLE")
        refine_area_and_search_again(categories,latitude,longitude,radius, depth)
    else:
        counter_circles = 0

        while True:

            businesses = response.get('businesses')
            if businesses is None or len(businesses) == 0:
                #global offset
                #print(u'No businesses for {0} in {1}, {2} with the Radius of {3} found.'.format(term,  latitude, longitude, radius))
                #offset = 0

                print("This is the circle counter", counter_circles)
                return
            counter_circles += len(businesses)



            #print(u'{0} businesses found, querying business info '.format(len(businesses)))


            for x in range(len(businesses)):
                global counter
                counter += 1
                #print("COUNTER :" + str(counter))
                #pprint.pprint(businesses[x])
                business_id = businesses[x]['id']
                print("\tAdding: {}".format(businesses[x]['alias']))
                #businesses[x]['term'] = term
                yelp_collection.insert(businesses[x])
                #dbhandler.add_record(businesses[x], datasource, "")
                #print(type(businesses))
                total = total - len(businesses)

                #response = get_business(API_KEY, business_id)
                #print(u'Result for business "{0}" found:'.format(business_id))
                #pprint.pprint(response, indent=2)
                print('---------------------------------------------------')

            offset = offset + 50
            response = search(API_KEY, categories, latitude, longitude, radius, offset)


Terms= ["Bars", "Restaurants"]
# categorie_list =("auto", "beautysvc","arts", "localflavor", "food", "nightlife", "hotelstravel", "religiousorgs", "restaurants", "shopping", "active", "eventservices")

point = (52.5074,13.3904)




#global offset
#global wholeLoopBoolean
print("Querying ",point)
wholeLoopBoolean = True
depth = "R"
query_api(Terms,point[0],point[1], DEFAULT_RADIUS,depth)





