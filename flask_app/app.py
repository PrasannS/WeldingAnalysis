import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

models  = ['RandomForest','XGBoost', 'LightGBM', 'CatBoost', 'Gaussian Processes']
props  = ['Rm',
 'A5',
 'Diffusible Hydrogen',
 'Moisture',
 'Yield strength',
 'Tensile',
 'Hardness',
 'Ferrite (Fn)',
 'Category',
 'Elongation',
 'Charpy',
 'DIN_W',
 'HT_Temp',
 'IE-20',
 'IE-40',
 'IE-60',
 'Redry_Time',
 'Rp0']

elem_names = ['V', 'C', 'Cr', 'Mn', 'Mo', 'Ni', 'P ', 'S ', 'Si', 'Cb', 'Cu', 'P',
       'S', 'Fe']
@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template('index.html', names=elem_names, models = models, props=props)

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Sales should be $ {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)