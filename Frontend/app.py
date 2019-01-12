from flask import Flask, render_template, request
app = Flask(__name__)


def calc(latitude,longitude):
    z = latitude + longitude
    return z


@app.route('/')
def student():
   return render_template('input.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
   if request.method == 'POST':
      result = request.form
      lat = request.form['Latitude']
      lon = request.form['Longitude']
      print(lat,lon)
      finalResult = calc(lat,lon)
      return render_template("result.html",result = finalResult)

if __name__ == '__main__':
   app.run(debug = True)
