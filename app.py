import numpy as np
import pandas as pd
import pickle 
import os
from flask import Flask, request, app, jsonify, redirect, render_template



app = Flask(__name__)

# Loading the model
reg_model = pickle.load(open('regmodel.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output=reg_model.predict(final_input)[0]
    return render_template("home.html",prediction_text="The House price prediction is {}".format(output))

    return output

if __name__ == '__main__':
    app.run()

