"""

Link to the render web-application:

https://flask-render-integration-plqu.onrender.com

"""

from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and encoders
model = joblib.load('models/xgb_model.joblib')

# Load all encoders for categorical features
try:
    gender_encoder = joblib.load('models/gender_encoder.joblib')
    international_encoder = joblib.load('models/international_encoder.joblib')
    major_encoder = joblib.load('models/major_encoder.joblib')
    race_encoder = joblib.load('models/race_encoder.joblib')
    work_industry_encoder = joblib.load('models/work_industry_encoder.joblib')
    admission_encoder = joblib.load('models/admission_encoder.joblib')
except FileNotFoundError as e:
    print(f"Error loading encoder: {e}")

# Define the homepage route to render the HTML form
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for handling form submissions
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    
    # Check if any unseen labels exist and map them to known values
    if data['major'] not in major_encoder.classes_:
        data['major'] = 'Engineering'  # Default to a known major

    # Encode the categorical data using the loaded encoders
    encoded_data = [
        gender_encoder.transform([data['gender']])[0],
        international_encoder.transform([data['international']])[0],
        major_encoder.transform([data['major']])[0],
        race_encoder.transform([data['race']])[0],
        work_industry_encoder.transform([data['work_industry']])[0],
        float(data['gpa']),  
        int(data['gmat']),   
        int(data['work_exp'])
    ]

    # Convert the input into a numpy array and reshape for prediction
    encoded_data = np.array(encoded_data).reshape(1, -1)

    # Make a prediction using the trained model
    prediction = model.predict(encoded_data)

    # Convert the prediction back to the original labels
    prediction_label = admission_encoder.inverse_transform(prediction)[0]

    # Render the result.html template with the prediction result
    return render_template('result.html', prediction_label=prediction_label)

# Run the app
if __name__ == '__main__':
    app.run()