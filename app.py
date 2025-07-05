# app.py
import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables
model = None
scaler = None
features = ['population', 'poverty_rate', 'unemployment_rate', 'median_income', 'education_level']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    try:
        # Check if training data was uploaded
        if 'data_file' not in request.files:
            return jsonify({'error': 'No data file uploaded'}), 400

        file = request.files['data_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save the uploaded file
        data_path = os.path.join('data', 'crime_data.csv')
        file.save(data_path)
        
        # Load the data
        data = pd.read_csv(data_path)
        
        # Validate that the data has the required columns
        if not all(col in data.columns for col in features + ['crime_rate']):
            return jsonify({'error': 'Data file missing required columns'}), 400
        
        # Preprocess data
        X = data[features]
        y = data['crime_rate']
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Feature scaling
        global scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train the model
        global model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Save the model and scaler
        joblib.dump(model, os.path.join('models', 'crime_model.pkl'))
        joblib.dump(scaler, os.path.join('models', 'scaler.pkl'))
        
        return jsonify({
            'message': 'Model trained successfully',
            'train_score': train_score,
            'test_score': test_score
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the model if it's not already loaded
        global model, scaler
        if model is None:
            model_path = os.path.join('models', 'crime_model.pkl')
            scaler_path = os.path.join('models', 'scaler.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        
        # Get input data
        data = request.get_json()
        
        # Validate input data
        if not all(key in data for key in features):
            return jsonify({'error': f'Missing required features: {features}'}), 400
        
        # Prepare input data
        input_data = np.array([[data[feature] for feature in features]])
        
        # Scale the input data
        input_data_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_data_scaled)[0]
        
        return jsonify({
            'prediction': prediction,
            'features_importance': dict(zip(features, model.feature_importances_))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        # Check if prediction data was uploaded
        if 'data_file' not in request.files:
            return jsonify({'error': 'No data file uploaded'}), 400
        
        file = request.files['data_file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Load the model if it's not already loaded
        global model, scaler
        if model is None:
            model_path = os.path.join('models', 'crime_model.pkl')
            scaler_path = os.path.join('models', 'scaler.pkl')
            
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return jsonify({'error': 'Model not trained yet. Please train the model first.'}), 400
            
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        
        # Save the uploaded file
        data_path = os.path.join('data', 'prediction_data.csv')
        file.save(data_path)
        
        # Load the data
        data = pd.read_csv(data_path)
        
        # Validate that the data has the required columns
        if not all(col in data.columns for col in features):
            return jsonify({'error': 'Data file missing required columns'}), 400
        
        # Preprocess data
        X = data[features]
        
        # Scale the input data
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        
        # Save predictions to CSV
        output_data = data.copy()
        output_data['predicted_crime_rate'] = predictions
        output_path = os.path.join('data', 'predictions.csv')
        output_data.to_csv(output_path, index=False)
        
        return jsonify({
            'message': 'Batch predictions completed',
            'output_file': output_path
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)