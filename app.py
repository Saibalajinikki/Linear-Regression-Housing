import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np

app = Flask(__name__)
regmodel = pickle.load(open('lr.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict.api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        print(np.array(list(data.values())).reshape(1, -1))
        new_data = np.array(list(data.values())).reshape(1, -1)
        output = regmodel.predict(new_data)
        print(output[0])
        return jsonify({'prediction': output[0]})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
