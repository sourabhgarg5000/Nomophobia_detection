from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('nomophobia_model.pkl')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form  # Collect form data
    input_data = [int(data[f'q{i}']) for i in range(1, 21)]  # Extract answers for 20 questions
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    # Get raw prediction (numerical value)
    raw_prediction = model.predict(input_data_as_numpy_array)
    # Get rounded stage
    rounded_stage = int(round(raw_prediction[0]))

    # Return both raw prediction and rounded stage
    return jsonify({
        'numerical_value': float(raw_prediction[0]),  # Exact numerical value
        'stage': rounded_stage  # Rounded stage value 
    })

# Home route for the frontend
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
