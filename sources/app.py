from flask import Flask, render_template, request
import requests
from prediction import prediction


app = Flask(__name__)





@app.route('/')
def student():
   return render_template('input.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      street = request.form['Street']
      number = request.form['Number']
      zipcode = request.form['Zipcode']
      price = int(request.form['price'])

      link = 'https://geocoder.api.here.com/6.2/geocode.json?app_id=lAdOIrvbmOYqIwl0IyeI&app_code=pDseBnEcDVidvTZ3IjVZ-A&searchtext={}+{}+{}+Berlin'.format(street,number,zipcode)
      r = requests.get(link)
      data = r.json()
      lat = float(data["Response"]["View"][0]["Result"][0]["Location"]["DisplayPosition"]["Latitude"])
      lon = float(data["Response"]["View"][0]["Result"][0]["Location"]["DisplayPosition"]["Longitude"])
      print(lat, lon)
      finalResult = prediction(lat,lon,price)
      return render_template("result.html",result = finalResult)

if __name__ == '__main__':
   app.run(debug = True)
