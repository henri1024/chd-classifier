
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import numpy as np
import tensorflow as tf
import joblib
import os

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

model = tf.keras.models.load_model('model/ann_model/')
scaler_filename = "model/scaler.save"
sc = joblib.load(scaler_filename)

# id,age,sex,is_smoking,cigsPerDay,BPMeds,prevalentStroke,prevalentHyp,diabetes,totChol,sysBP,diaBP,BMI,heartRate,glucose,TenYearCHD
class Record(BaseModel):
    age: float
    sex: float
    is_smoking: float
    cigs_per_day: float
    blood_pressure_med_consumption: float
    prevalent_stroke: float
    prevalent_hypertension: float
    prevalent_diabetes: float
    cholesterol_levels: float
    systolic_blood_pressure: float
    diastolic_blood_pressure: float
    body_mass_index: float
    heart_rate: float
    glucose_levels: float


@app.get('/')
def index():
    return {'message': 'This is Coronary Heart Diseases Detector API!'}


@app.post('/predict')
def predict_chd(data: Record):
    """ FastAPI 
    Args:
        data (Record): json file 
    Returns:
        prediction: classification CHD potential
    """

    try:
        arr = _preprocessing(data)

        result = _predict(arr)

        if result[0][0] > 0.5:
            return {
                'prediction': 1,
                'probability': str(result[0][0])
            }
        else:
            return{
                'prediction': 0,
                'probability': str(result[0][0])
            }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    


def _preprocessing(data: Record):
    try:
        return sc.transform(np.array([data.age, data.sex, data.is_smoking, data.cigs_per_day, 
        data.blood_pressure_med_consumption, data.prevalent_stroke, data.prevalent_hypertension, 
        data.prevalent_diabetes, data.cholesterol_levels, data.systolic_blood_pressure, 
        data.diastolic_blood_pressure, data.body_mass_index, data.heart_rate, data.glucose_levels, ]).reshape(1, -1))
    except Exception as e:
        raise ValueError('failed to normalize data')

def _predict(arr):
    try:
        return model.predict(arr)
    except Exception as e:
        raise ValueError('failed to fed model')


if __name__ == '__main__':

    host = os.getenv('APP_HOST') if os.getenv('APP_HOST') else '0.0.0.0'
    port = int(os.getenv('APP_PORT')) if os.getenv('APP_PORT') else '9999'

    uvicorn.run(app, host=host, port=port)