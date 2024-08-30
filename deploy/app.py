from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import joblib
import logging

logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained models
try:
    family_model = joblib.load('family_model.pkl')
    species_model = joblib.load('species_model.pkl')
    logging.debug(f"Family model features: {family_model.feature_names_in_}")
    logging.debug(f"Species model features: {species_model.feature_names_in_}")
except Exception as e:
    logging.error(f"Error loading models: {str(e)}")
    raise e

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
    logging.debug(f"Initial DataFrame: \n{df}")
    
    # Rescale specific columns
    df = df.apply(rescale_abcd, axis=1)
    logging.debug(f"DataFrame after rescaling: \n{df}")
    
    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, columns=categorical_columns)
    logging.debug(f"DataFrame after one-hot encoding: \n{df}")
    
    # List of all possible values for 'Gena color' and 'Body color'
    all_possible_values = ['orange', 'white', 'metallic_green', 'metallic_blue', 'cupreous', 'grey']

    # Ensure all possible one-hot encoded columns are present
    for col in categorical_columns:
        for value in all_possible_values:
            col_name = f"{col}_{value}"
            if col_name not in df.columns:
                df[col_name] = 0  # Add missing color columns with value 0
        # Ensure all expected shape columns are present, fill missing with zeros
        for col in shape_columns:
            if col not in df.columns:
                df[col] = 0
    
    logging.debug(f"Final DataFrame ready for prediction: \n{df}")
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
    return render_template("index2.html")

@app.route("/predict", methods=['POST'])
def predict_species_family():
    try:
        logging.debug(f"Incoming form data: {request.form}")

        measurements = {}

        # Determine the mode from the form data
        mode = request.form.get('mode')

        if mode == 'minimal':
            required_fields = ['a(1-2)', 'b(2-3)', 'e(6-7)', 'g(9-10)', 'Gena color', 'Body color']
        else:  # Assuming 'full' mode by default
            required_fields = shape_columns + categorical_columns

        # Collect and validate measurements
        for key in required_fields:
            if key in shape_columns:
                try:
                    measurements[key] = float(request.form.get(key, 0))
                except ValueError:
                    measurements[key] = 0
            elif key in categorical_columns:
                measurements[key] = request.form.get(key, '')

        # Check if any required fields are missing
        missing_fields = [key for key in required_fields if key not in measurements or not measurements[key]]
        if missing_fields:
            return jsonify({'success': False, 'error_message': f'Missing required field(s): {", ".join(missing_fields)}'})

        # Make predictions
        species, species_probability = predict(measurements, species_model)

        logging.debug(f"Prediction result: Species={species}")
        logging.debug(f"Probability: Species={species_probability}")

        result = {
            'success': True,
            'species': species,
            'probability': float(species_probability)
        }

        logging.debug(f"Sending result to template: {result}")
        return jsonify(result)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        logging.error(error_message)
        return jsonify({'success': False, 'error_message': error_message})



# Running the app
if __name__ == "__main__":
    app.run(debug=True)