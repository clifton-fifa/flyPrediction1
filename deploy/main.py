from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# โหลดโมเดลที่ฝึกไว้
family_model = joblib.load('family_model.pkl')
species_model = joblib.load('species_model.pkl')

# สร้างแอปพลิเคชัน FastAPI
app = FastAPI()

# สร้าง class สำหรับรับอินพุต
class PredictionInput(BaseModel):
    features: dict

# สร้าง endpoint สำหรับการทำนาย Family
@app.post("/predict_family")
def predict_family(input_data: PredictionInput):
    try:
        # แปลงอินพุตให้เป็น DataFrame
        input_df = pd.DataFrame([input_data.features])

        # ทำนาย
        prediction = family_model.predict(input_df)
        
        # ส่งผลลัพธ์
        return {"predicted_family": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# สร้าง endpoint สำหรับการทำนาย Species
@app.post("/predict_species")
def predict_species(input_data: PredictionInput):
    try:
        # แปลงอินพุตให้เป็น DataFrame
        input_df = pd.DataFrame([input_data.features])

        # ทำนาย
        prediction = species_model.predict(input_df)
        
        # ส่งผลลัพธ์
        return {"predicted_species": prediction[0]}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# สร้าง root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Fly Prediction API!"}
