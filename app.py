from flask import Flask, render_template, request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np

import sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, BatchNormalization, Flatten, InputLayer
from tensorflow.keras.optimizers import Adam

def lstm_model(input_shape):
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(LSTM(128, return_sequences=True))

    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=False))

    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))

    # Output Layer
    model.add(Dense(1,activation='linear'))

    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='mean_absolute_error')

    return model

def create_data_lstm(data,days=60):
    '''
        data : 2D array
        train_size : float -> value in interval (0,1]
        days : int -> banyaknya data hari yang diinginkan untuk memprediksi 1 hari berikutnya
    '''
    x_test, y_test = [],[]
    days = int(days)

    for i in range(days,len(data)):
        x_test.append(data[i-days:i,0])
        y_test.append(data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    return x_test, y_test

def evaluate(y_test,y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    return mae,mse,rmse

model = lstm_model(input_shape=(10,1))
model.load_weights("lstm_pred_tomorrow_weights.h5")
scaler = pickle.load(open('scaler.pkl','rb'))

    
app = Flask(__name__)


@app.route('/')
def index():
    return render_template('base.html')

@app.route('/uploads', methods = ['POST'])
def uploads():
    if 'filecsv' not in request.files:
        return "No file part"
    
    file = request.files['filecsv']
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'])
    data_test = df.filter(['Close'])
    data_test = data_test.values
    data_test = scaler.transform(data_test)
    look_back_days = 10
    x_test, y_test = create_data_lstm(data_test,days=look_back_days)
    prediction = model.predict(x_test)
    mae,mse,rmse = evaluate(y_test,prediction)
    prediction_inverse = scaler.inverse_transform(prediction)
    prediction_inverse = prediction_inverse.reshape(-1)

    plt.figure(figsize=(15, 5))
    plt.plot(df["Date"],df["Close"],label = 'Data actual')
    plt.plot(df.loc[look_back_days:,"Date"],prediction_inverse,label = 'Data Prediction')
    plt.legend()
    img_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),"static/img.png")
    plt.savefig(img_path, format='png')

    # data_test = data_test.reshape(-1, 1)
    # data_test = scaler.transform(data_test)
    # data_test = data_test.reshape(1,-1)
    # prediction = model.predict(data_test)
    # prediction = scaler.inverse_transform(prediction)

    return render_template('base.html',
                           klik_button=1,
                           mae=mae,
                           mse=mse,
                           rmse=rmse)
    
    
if __name__ == "__main__":
    app.run(debug=True)