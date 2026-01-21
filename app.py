from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained Knowledge (the .h5 model)
model = load_model('titanic_model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Capture Percepts
    pclass = int(request.form['pclass'])
    sex = int(request.form['sex'])
    age = float(request.form['age'])
    sibsp = int(request.form['sibsp'])
    
    # 2. Process percepts into a vector
    features = np.array([[pclass, sex, age, sibsp]])
    
    # 3. Decision making (Inference)
    prediction = model.predict(features)
    
    # Logic: If output > 0.5, the agent predicts 'Survived'
    result = "Survived" if prediction[0][0] > 0.5 else "Did Not Survive"
    
    return f"The model predicts this passenger would have: <b>{result}</b>"

if __name__ == "__main__":
    app.run(debug=True)