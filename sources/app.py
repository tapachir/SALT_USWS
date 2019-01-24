from flask import Flask, render_template, request
import requests
import code_v4_3

app = Flask(__name__)

@app.route('/')
def student():
   return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   #GET DATA INPUT FROM WEBSITE/GUI
   if request.method == 'POST':
      result = request.form
      street = request.form['Street']
      number = request.form['Number']
      zipcode = request.form['Zipcode']
      price = int(request.form['price'])
      category = request.form['category']

      #MAKE USE OF "HERE API" TO CONVERT THE INSERTED ADDRESS TO COORDINATES
      link = 'https://geocoder.api.here.com/6.2/geocode.json?app_id=lAdOIrvbmOYqIwl0IyeI&app_code=pDseBnEcDVidvTZ3IjVZ-A&searchtext={}+{}+{}+Berlin'.format(street,number,zipcode)
      r = requests.get(link)
      data = r.json()
      lat = float(data["Response"]["View"][0]["Result"][0]["Location"]["DisplayPosition"]["Latitude"])
      lon = float(data["Response"]["View"][0]["Result"][0]["Location"]["DisplayPosition"]["Longitude"])
      print(lat, lon)


      #MAKE USE OF OUR PREDICIOTN FUNCTION WITH PASSED VARIABLES
      finalResult = code_v4_3.prediction(lat,lon,price,category)

      #CALCULATE REACHABILITY WITH OUR REACHABILITY FUNCTION
      reachability = code_v4_3.calc_reachability(lat,lon)

      #RETURN THE DATA TO THE NEW RESULT PAGE OF THE WEBSITE/GUI
      return render_template("result.html",result = finalResult, reachability = reachability)

if __name__ == '__main__':
   app.run(debug = True)
