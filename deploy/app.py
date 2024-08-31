import numpy as np
import pandas as pd
import joblib
import logging
from flask import Flask, request, render_template, jsonify

# Setup logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load the pre-trained model
try:
    species_model = joblib.load('species_model.pkl')
    logging.debug(f"Species model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {str(e)}")
    raise e

# Define the feature columns for both modes
minimal_columns = ['a(1-2)', 'b(2-3)', 'e(6-7)', 'g(9-10)', 'Gena color', 'Body color']
full_columns = ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)', 'e(6-7)', 'f(7-10)', 
                'g(9-10)', 'h(9-15)', 'i(15-16)', 'j(14-15)', 'k(13-14)', 
                'l(13-17)', 'm(17-18)', 'n(1-18)', 'o(2-13)', 'p(3-12)', 
                'q(12-13)', 'r(5-12)', 's(11-14)', 't(8-11)', 'u(7-8)', 
                'v(8-9)', 'w(11-12)']

def normalize_full_mode(data):
    a, b, c, d = data['a(1-2)'], data['b(2-3)'], data['c(3-4)'], data['d(4-5)']
    total = a + b + c + d
    factor = 5000 / total
    for key in ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)']:
        data[key] *= factor
    return data

def prepare_input_data(data, mode):
    df = pd.DataFrame([data])
    
    if mode == 'full':
        df = df.apply(normalize_full_mode, axis=1)
    elif mode == 'minimal':
        df = pd.get_dummies(df, columns=['Gena color', 'Body color'])
        
        # Ensure all possible one-hot encoded columns are present
        possible_values = {'Gena color': ['Orange', 'White'], 
                           'Body color': ['Metallic green', 'Metallic blue', 'Cupreous', 'Grey']}
        for col, values in possible_values.items():
            for value in values:
                col_name = f"{col}_{value}"
                if col_name not in df.columns:
                    df[col_name] = 0
        
        # Ensure all minimal mode columns are present in the DataFrame
        expected_columns = minimal_columns + [f'Gena color_{v}' for v in possible_values['Gena color']] + [f'Body color_{v}' for v in possible_values['Body color']]
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default value 0
    
    # Ensure all expected columns are present, fill missing with zeros
    missing_cols = [col for col in species_model.feature_names_in_ if col not in df.columns]
    for col in missing_cols:
        df[col] = 0

    df = df[species_model.feature_names_in_]
    
    return df


def predict(data, mode):
    df = prepare_input_data(data, mode)
    logging.debug(f"DataFrame used for prediction:\n{df.to_string()}")
    
    prediction = species_model.predict(df)[0]
    probabilities = species_model.predict_proba(df)[0]
    max_probability = np.max(probabilities)
    
    logging.debug(f"Prediction: {prediction}, Probability: {max_probability}")
    
    return prediction, max_probability

@app.route('/')
def index():
    return render_template("index2.html")

@app.route("/predict", methods=['POST'])
def predict_species():
    try:
        mode = request.form.get('mode', 'minimal')
        required_fields = minimal_columns if mode == 'minimal' else full_columns
        
        measurements = {}
        missing_fields = []
        for field in required_fields:
            value = request.form.get(field)
            if value is None or value.strip() == '':
                missing_fields.append(field)
            else:
                if field not in ['Gena color', 'Body color']:
                    try:
                        measurements[field] = float(value)
                    except ValueError:
                        return jsonify({'success': False, 'error_message': f"Invalid input for {field}."})
                else:
                    measurements[field] = value

        logging.debug(f"Received measurements: {measurements}")

        if missing_fields:
            error_message = f"Missing required fields: {', '.join(missing_fields)}"
            logging.error(error_message)
            return jsonify({'success': False, 'error_message': error_message})

        species, probability = predict(measurements, mode)

        result = {
            'success': True,
            'species': species,
            'probability': float(probability)
        }

        logging.debug(f"Prediction result: {result}")
        return jsonify(result)

    except Exception as e:
        error_message = f"An error occurred during prediction: {str(e)}"
        logging.error(error_message)
        return jsonify({'success': False, 'error_message': error_message})
    
if __name__ == "__main__":
    app.run(debug=True)
