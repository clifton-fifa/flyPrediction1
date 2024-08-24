from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

# Loading the pre-trained models
family_model = joblib.load('family_model.pkl')
species_model = joblib.load('species_model.pkl')

logging.debug(f"Family model features: {family_model.feature_names_in_}")
logging.debug(f"Species model features: {species_model.feature_names_in_}")

# Feature columns used in the models
shape_columns = ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)', 'e(6-7)', 'f(7-10)', 
                 'g(9-10)', 'h(9-15)', 'i(15-16)', 'j(14-15)', 'k(13-14)', 
                 'l(13-17)', 'm(17-18)', 'n(1-18)', 'o(2-13)', 'p(3-12)', 
                 'q(12-13)', 'r(5-12)', 's(11-14)', 't(8-11)', 'u(7-8)', 
                 'v(8-9)', 'w(11-12)']

categorical_columns = ['Gena color', 'Body color']

# Rescaling function for points a(1-2), b(2-3), c(3-4), d(4-5)
def rescale_abcd(row):
    cols = [col for col in ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)'] if col in row.index]
    total = sum(row[cols])
    if total != 0:
        factor = 5000 / total
        for col in cols:
            row[col] *= factor
    return row

# Preparing input data for the model
def prepare_input_data(data):
    df = pd.DataFrame([data])
    
    df = df.apply(rescale_abcd, axis=1)
    
    # One-hot encoding for categorical columns
    if 'Gena color' in df.columns and 'Body color' in df.columns:
        df = pd.get_dummies(df, columns=categorical_columns)
    
    return df

# Prediction function
def predict(data, model):
    df = prepare_input_data(data)
    
    # Add missing columns with 0
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    
    df = df[model.feature_names_in_]
    prediction = model.predict(df)[0]
    
    # Get prediction probabilities
    probabilities = model.predict_proba(df)[0]
    max_probability = np.max(probabilities)
    
    return prediction, max_probability

# Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict_species_family():
    try:
        logging.debug(f"Received form data: {request.form}")
        
        mode = request.form.get('mode', 'full')
        measurements = {}
        
        if mode == 'minimal':
            required_columns = shape_columns[:5] + categorical_columns
        else:
            required_columns = shape_columns + categorical_columns
        
        for key in required_columns:
            if key in shape_columns:
                try:
                    measurements[key] = float(request.form.get(key, 0))
                except ValueError:
                    measurements[key] = 0
            elif key in categorical_columns:
                measurements[key] = request.form.get(key, '')
        
        # Check for missing required data
        missing_required = [col for col in required_columns if not measurements.get(col)]
        if missing_required:
            error_message = f"Missing required input data: {', '.join(missing_required)}"
            logging.error(error_message)
            return render_template('index.html', prediction_result={'success': False, 'error_message': error_message})
        
        # Make predictions
        family, family_probability = predict(measurements, family_model)
        species, species_probability = predict(measurements, species_model)
        
        logging.debug(f"Prediction result: Family={family}, Species={species}")
        logging.debug(f"Probabilities: Family={family_probability}, Species={species_probability}")
        
        # Calculate overall accuracy as the average of family and species probabilities
        accuracy = ((family_probability + species_probability) / 2) * 100
        
        result = {
            'success': True,
            'family': family,
            'species': species,
            'accuracy': round(accuracy, 2)
        }
        
        logging.debug(f"Sending result to template: {result}")
        return render_template('index.html', prediction_result=result)
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        return render_template('index.html', prediction_result={'success': False, 'error_message': error_message})
# Running the app
if __name__ == "__main__":
    app.run(debug=True)