from math import radians, cos, sin, asin, sqrt
import csv

import geopy.distance


with open('stops.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader, None)

    for row in csv_reader:
        center_point = [{'lat': 52.524436, 'lng': 13.409616}]

def compare():
    delete_list = []
    lat = 52.524436
    lon = 13.409616
    under1km = 0
    under500m = 0
    under250m = 0
    coordinates1 = float(lat),float(lon)

    with open('stops.txt') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        next(csv_reader, None)

        x = 0
        with open("newStops.txt", mode='w') as newStops:
            for row in csv_reader:
                coordinates2 = float(row[4]), float(row[5])

                dist = geopy.distance.geodesic(coordinates1, coordinates2).m
                dist = int(dist)

                if dist < 20000:



                    stops_writer = csv.writer(newStops, delimiter= ",",)
                    quotes = '"'
                    whole_row = row[0]+','+quotes+row[1]+quotes+','+quotes+row[2]+quotes+','+quotes+row[3]+quotes+','+quotes+row[4]+quotes+','+quotes+row[5]+quotes+','+row[6]
                    x += 1
                    print(whole_row)
                    #print('Filename:',)
                    newStops.write(whole_row )






compare()

