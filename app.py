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

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = pd.DataFrame(request.files.get())


    #print(data)
    #print(np.array(list(data.values())).reshape((1,-1)))
    #x = np.array(list(data.values())).reshape((1,-1))
    data_std = scaler.transform(data.values)
    output = reg_model.predict(data_std)

    return output

if __name__ == '__main__':
    app.run()

