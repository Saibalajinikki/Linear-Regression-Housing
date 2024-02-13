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
        data = request.json
        if data is not None and 'data' in data:
            data = data['data']
            app.logger.info("Raw data from JSON: %s", data)

            input_array = np.array(list(data.values())).reshape(1, -1)
            app.logger.info("Input array for prediction: %s", input_array)

            output = regmodel.predict(input_array)
            app.logger.info("Prediction output: %s", output[0])

            return jsonify({'prediction': output[0]})
        else:
            return jsonify({'error': 'Invalid JSON format. Expected {"data": {...}}'})
    except Exception as e:
        app.logger.error("Error during prediction: %s", str(e))
        return jsonify({'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        final_input = np.array(data).reshape(1, -1)
        app.logger.info("Input array for prediction: %s", final_input)

        output = regmodel.predict(final_input)[0]
        app.logger.info("Prediction output: %s", output)

        return render_template('home.html', prediction_test='The predicted House value is {}'.format(output))
    except Exception as e:
        app.logger.error("Error during prediction: %s", str(e))
        return render_template('home.html', prediction_test='Error during prediction: {}'.format(str(e)))

if __name__ == '__main__':
    app.run(debug=True)
