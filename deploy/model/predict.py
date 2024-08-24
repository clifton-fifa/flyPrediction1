import pandas as pd
import joblib
import numpy as np

shape_columns = ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)', 'e(6-7)', 'f(7-10)', 
                 'g(9-10)', 'h(9-15)', 'i(15-16)', 'j(14-15)', 'k(13-14)', 
                 'l(13-17)', 'm(17-18)', 'n(1-18)', 'o(2-13)', 'p(3-12)', 
                 'q(12-13)', 'r(5-12)', 's(11-14)', 't(8-11)', 'u(7-8)', 
                 'v(8-9)', 'w(11-12)']

minimal_columns = ['a(1-2)', 'b(2-3)', 'g(9-10)', 'e(6-7)']
categorical_columns = ['Gena color', 'Body color']

def rescale_abcd(row):
    cols = [col for col in ['a(1-2)', 'b(2-3)', 'c(3-4)', 'd(4-5)'] if col in row.index]
    total = sum(row[cols])
    if total != 0:
        factor = 5000 / total
        for col in cols:
            row[col] *= factor
    return row

def prepare_input_data(data):
    df = pd.DataFrame([data])
    df = df.apply(rescale_abcd, axis=1)
    
    if 'Gena color' in df.columns and 'Body color' in df.columns:
        df = pd.get_dummies(df, columns=categorical_columns)
    
    return df

def predict(data, model_path):
    model = joblib.load(model_path)
    df = prepare_input_data(data)
    
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    
    df = df[model.feature_names_in_]
    return model.predict(df)[0]

def predict_family(data):
    return predict(data, 'family_model.pkl')

def predict_species(data):
    return predict(data, 'species_model.pkl')

def predict_from_measurements(measurements):
    try:
        # แปลงข้อมูลเป็น float
        data = {key: float(value) if key not in categorical_columns else value 
                for key, value in measurements.items()}
        
        family = predict_family(data)
        species = predict_species(data)
        
        return {
            "family": family,
            "species": species
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # ตัวอย่างการใช้งานแบบที่ 1: ใช้แค่ตัวเลขทั้งหมด
    example_data_1 = {col: np.random.rand() * 1000 for col in shape_columns}
    
    result_1 = predict_from_measurements(example_data_1)
    print("Result for all numerical features:")
    print(f"Predicted Family: {result_1['family']}")
    print(f"Predicted Species: {result_1['species']}")

    # ตัวอย่างการใช้งานแบบที่ 2: ใช้ตัวเลขแค่ 5 ตัว และมี Gena color, Body color
    example_data_2 = {
        'a(1-2)': 691.063555,
        'b(2-3)': 598.1629275,
        'g(9-10)': 195.0163107,
        'e(6-7)': 693.2883964,
        'Gena color': 'White',
        'Body color': 'Metallic green'
    }
    
    result_2 = predict_from_measurements(example_data_2)
    print("\nResult for minimal features with categorical data:")
    print(f"Predicted Family: {result_2['family']}")
    print(f"Predicted Species: {result_2['species']}")