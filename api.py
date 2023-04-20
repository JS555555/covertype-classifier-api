import joblib
import numpy as np
from flask import Flask, jsonify, request
from covertype_class import CovertypeClassifier

app = Flask(__name__)
model = CovertypeClassifier()

@app.route('/predict', methods=['POST'])
def predict():
    content = request.json
    input_data = content['input_data']
    model = content['model_name']

    if model == 'lr':
        lr = joblib.load('lr_model.joblib')
        pred = (lr.predict(input_data)).tolist()

    elif model == 'dtc':
        dtc = joblib.load('dtc_model.joblib')
        pred = (dtc.predict(input_data)).tolist()

    elif model == 'ann':
        ann = joblib.load('ann_model.joblib')
        pred = (np.argmax(ann.predict(input_data), axis=1)).tolist()

    elif model == 'heuritic':
        clf = CovertypeClassifier()
        pred = (clf.simple_heuristic_classification(input_data)).tolist()


    return jsonify({'predictions': pred})


if __name__ == '__main__':
    app.run()