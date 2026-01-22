from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
scaler = joblib.load("model/scaler.pkl")
model = joblib.load("model/breast_cancer_model.pkl")


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get feature values from form
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    
    # Scale features
    final_features_scaled = scaler.transform(final_features)
    
    # Make prediction
    prediction = model.predict(final_features_scaled)[0]
    result = 'Benign' if prediction == 1 else 'Malignant'
    
    return render_template('index.html', prediction_text=f'Tumor is {result}')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

