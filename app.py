from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Correct path for Project 4 requirements
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model', 'titanic_survival_model.pkl')

# Load the model
try:
    model = joblib.load(model_path)
    print("✓ Titanic Model loaded successfully")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not found on server'}), 500
    
    try:
        data = request.get_json()
        
        # Mapping inputs to numerical values for the model
        pclass = int(data['Pclass'])
        sex = 1 if data['Sex'].lower() == 'female' else 0
        age = float(data['Age'])
        sibsp = int(data['SibSp'])
        fare = float(data['Fare'])
        
        # Create feature array (Must match training order)
        features = np.array([[pclass, sex, age, sibsp, fare]])
        
        # Make prediction (0 = Died, 1 = Survived)
        prediction = model.predict(features)[0]
        
        result_text = "Survived" if prediction == 1 else "Did Not Survive"
        
        return jsonify({
            'prediction': result_text,
            'status': int(prediction)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)