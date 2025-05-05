from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define a function to segment customers (mapping numeric predictions to segment names)
def segment_customer(recency, frequency, monetary):
    data = np.array([[recency, frequency, monetary]])
    scaled_data = scaler.transform(data)
    prediction = rf_model.predict(scaled_data)
    
    # Map numeric predictions to customer segment names
    if prediction == 'Low-value Customer':
        return 'Low-value Customer'
    elif prediction == 'High-value Customer':
        return 'High-value Customer'
    else:
        return 'Unknown'

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle form submission and prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from the form
        recency = float(request.form['recency'])
        frequency = float(request.form['frequency'])
        monetary = float(request.form['monetary'])

        # Get customer segment prediction
        segment = segment_customer(recency, frequency, monetary)

        # Display the result on the page
        return render_template('index.html', segment=segment)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
