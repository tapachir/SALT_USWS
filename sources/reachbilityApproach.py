import csv

import geopy.distance


def compare(lat,lon):
    under1km = 0
    under500m = 0
    under250m = 0
    coordinates1 = float(lat),float(lon)

    with open('cleanedStops.txt') as csv_file:
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

        print(result)


compare(52.45887, 13.32299)
