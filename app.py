# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Project:- Predict the weight of people by Machine learning.
"""
# Import essential libraries
# Make a flask app
import numpy as np
import joblib
from flask import Flask, request, render_template
from flask_cors import CORS, cross_origin
app = Flask(__name__)
model = joblib.load(open('height_weight_prediction.pkl','rb'))
@app.route('/')
@cross_origin()
def home():
    return render_template('index.html')
@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():
    data1 = request.form['a']
    arr = np.array([[data1]])
    pred = model.predict(arr)
    output = round(pred[0],2)
    return render_template('index.html',prediction_text ="Your predicted weight in pounds is: {}".format(output))
if __name__=='__main__':
    app.run()
