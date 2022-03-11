## Import the Libraries
import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request
import joblib
from utils import process_one, process_batch   ## the function I craeted to process the data in utils.py



## Intialize the Flask APP
app = Flask(__name__)

## Loading the Model
model = joblib.load('xgboost_model.pkl')


## Route for Home page
@app.route('/')
def home():
    return render_template('index.html')


## Route for predict only one Instances
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        fc = int(request.form['fc'])
        lc = int(request.form['lc'])
        tc = int(request.form['tc'])
        rut = int(request.form['rut'])

        X_new = np.array([[age, fc, lc, tc, rut]])
        X_new = process_one(X_new)
        outs = np.exp(model.predict(X_new)) - 1
        outs = '{:.4f}'.format(outs[0])

        return render_template('predict.html', pred_value=outs)

    else:
        return render_template('predict.html')


## Route for predicting batch of instances exsists in xlsx file.
@app.route('/predict_batch', methods=['GET', 'POST'])
def predict_batch():
    if request.method == 'POST':
        file_path = request.form['upload_file']
        if file_path:
            df_batch = pd.read_excel(file_path, sheet_name='Sheet1')
            age = np.array([*df_batch.to_dict()['AGE'].values()])
            fc = np.array([*df_batch.to_dict()['FC%'].values()])
            lc = np.array([*df_batch.to_dict()['LC(ft/mile)'].values()])
            tc = np.array([*df_batch.to_dict()['TC(ft/mile)'].values()])
            rut = np.array([*df_batch.to_dict()['RUT(in)'].values()])
            X_new = np.column_stack((age, fc, lc, tc, rut))

            ## processing 
            X_new = process_batch(X_new)

            ## predicting
            y_pred = model.predict(X_new)
            y_pred = np.exp(y_pred) - 1
            df_pred = pd.DataFrame(y_pred, columns=['Predictions'])
            return render_template('predict_batch.html', tables=[df_pred.to_html(classes='data')], titles=df_pred.columns.values)
        else:
            df_pred = pd.DataFrame(columns=['Predictions'])
            return render_template('predict_batch.html', tables=[df_pred.to_html(classes='data')], titles=df_pred.columns.values)


    else:
        df_pred = pd.DataFrame(columns=['Predictions'])
        return render_template('predict_batch.html', tables=[df_pred.to_html(classes='data')], titles=df_pred.columns.values)

## Run the App
if __name__ == '__main__':
    app.run(debug=True)
