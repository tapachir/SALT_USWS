#for x in range(len(feature_df)):
#    umkreis = 750
#    count = 0
#    coordinates1 = feature_df.at[x, 'latitude'], feature_df.at[x, 'longitude']
#    for y in range(len(feature_df1)):
#        if(x != y):
#            coordinates2 = feature_df.at[y, 'latitude'], feature_df.at[y, 'longitude']
#            if (geopy.distance.geodesic(coordinates1,coordinates2).m < umkreis):
#                count+=1
#                if("korean" in str(feature_df.at[y, 'categories'])):
#                     feature_df.at[x, 'neighbours']+=1
#                if("Italian" in str(feature_df.at[y, 'categories'])):
#                     feature_df.at[x, 'italianneigbhours']+=1
#    feature_df.at[x, 'neighbours'] = count-1
#    print(count)
#print(feature_df)

# doppelte for-schleife ... womöglich überflüßig
#for x in range(len(feature_df)):
#    coordinates1_la = feature_df.at[x,'latitude']
#    coordinates1_lo = feature_df.at[x,'longitude']
#    coordinates1 = (coordinates1_la,coordinates1_lo)
#    for y in range(len(feature_df)):
#        coordinates2_la = feature_df.at[y,'latitude']
#        coordinates2_lo = feature_df.at[y,'longitude']
#        coordinates2 = (coordinates2_la,coordinates2_lo)
#        if (geopy.distance.distance(coordinates1,coordinates2).km < 0.5):
#            count+=1
#    feature_df.at[x,'neighbours']=count 
#print(regr.predict([[52.4321,13.3210]],neighbours(52.4321,13.3210)).tolist())


# kein Plan was der Code hier unten macht

#pickle.dump(regr, open("model.pk", "wb"))
#
## loading a model from file
#model = pickle.load(open("model.pkl", "r"))
#
# #################################
#
# test3['score'] = 0

##code which helps initialize our server
#app = flask.Flask(__name__)

##defining a /hello route for only post requests
#@app.route('/hello', methods=['POST'])
#def index():
#    #grabs the data tagged as 'name'
#    name = request.get_json()['name']
#    
#    #sending a hello back to the requester
#    return "Hello " + name