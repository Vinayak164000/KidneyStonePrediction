from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Ensure that the features are provided
    required_features = ['gravity', 'urea', 'ph', 'cond', 'calc']
    for feature in required_features:
        if feature not in data:
            return jsonify({'error': f'Missing {feature} in input data'}), 400

    # Extract features from the JSON data
    features = [data[feature] for feature in required_features]

    # Make a prediction
    prediction = model.predict([features])[0]

    # Return the prediction as JSON
    return jsonify({'prediction': int(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
