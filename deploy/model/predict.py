import pandas as pd
import joblib

# Load pre-trained models
family_model = joblib.load('family_model.pkl')
species_model = joblib.load('species_model.pkl')

# Feature columns
shape_columns = ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)', 'e(6-7)', 'f(7-10)', 
                 'g(9-10)', 'h(9-15)', 'i(15-16)', 'j(14-15)', 'k(13-14)', 
                 'l(13-17)', 'm(17-18)', 'n(1-18)', 'o(2-13)', 'p(3-12)', 
                 'q(12-13)', 'r(5-12)', 's(11-14)', 't(8-11)', 'u(7-8)', 
                 'v(8-9)', 'w(11-12)']

categorical_columns = ['Gena color', 'Body color']

def rescale_abcd(row):
    cols = ['a(1-2)', 'b(2-3)', 'e(6-7)', 'g(9-10)']
    total = sum(row[cols])
    if total != 0:
        factor = 5000 / total
        for col in cols:
            row[col] *= factor
    return row

# Prepare input data
def prepare_input_data(data):
    df = pd.DataFrame([data])
    
    # Rescale specific columns
    df = df.apply(rescale_abcd, axis=1)
    
    # One-hot encoding for categorical columns
    df = pd.get_dummies(df, columns=categorical_columns)
    
    # Ensure all expected columns are present, fill missing with zeros
    for col in shape_columns + [f'{cat}_{val}' for cat in categorical_columns for val in ['orange', 'white', 'metallic', 'cupreous', 'grey']]:
        if col not in df.columns:
            df[col] = 0
            
    return df

# Prediction for family
def predict_family(data):
    df = prepare_input_data(data)
    df = df[family_model.feature_names_in_]
    prediction = family_model.predict(df)[0]
    return prediction

# Prediction for species
def predict_species(data):
    df = prepare_input_data(data)
    df = df[species_model.feature_names_in_]
    prediction = species_model.predict(df)[0]
    return prediction
