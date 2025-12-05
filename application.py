import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regression model and standard scaler
ridge_model=pickle.load(open("models/ridge_model.pkl","rb"))
standard_scaler=pickle.load(open("models/scaler.pkl","rb"))




@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():

    if request.method == "POST":

        data = [
            float(request.form.get("Temperature")),
            float(request.form.get("RH")),
            float(request.form.get("Ws")),
            float(request.form.get("Rain")),
            float(request.form.get("FFMC")),
            float(request.form.get("DMC")),
            float(request.form.get("ISI")),
            float(request.form.get("Classes")),
            float(request.form.get("Region")),
        ]

        scaled_data = standard_scaler.transform([data])

        prediction = ridge_model.predict(scaled_data)[0]   # ✅ Correct model variable

        return render_template("home.html", result=prediction)

    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5001,debug=True)
