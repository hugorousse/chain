from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
import os

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

model = load_model('token_price_predictor_lstm.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if 'day' not in data:
        return jsonify({'error': 'Invalid input, "day" is required'}), 400
    try:
        day = np.array([[data['day']]])
        scaled_day = scaler.transform(day)
        prediction = model.predict(scaled_day)
        prediction = scaler.inverse_transform(prediction)
        
        return jsonify({'prediction': float(prediction[0][0]), 'day': data['day'], 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
