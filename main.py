
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import numpy as np
import tensorflow as tf
import joblib
import os

import pandas as pd
from fastapi import FastAPI, File, UploadFile

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

model = tf.keras.models.load_model('model/ann_model/')
scaler_filename = "model/scaler.save"
sc = joblib.load(scaler_filename)

def make_model(x_train):

    metrics=[
        keras.metrics.BinaryAccuracy(name='accuracy'),
    ] 

    model = keras.Sequential([
        keras.layers.Dense(
            32, activation='relu',
            input_shape=(x_train.shape[-1],)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(
            24, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(
            16, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', 
    verbose=1,
    patience=32,
    mode='max',
    restore_best_weights=True)

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

class RetrainVariables(BaseModel):
    split_rate: float
    random_state: int
    batch_size: int
    epochs: int


@app.get('/')
def index():
    return {'message': 'This is Coronary Heart Diseases Detector API!'}

@app.post("/tmp/upload")
async def upload(file: UploadFile = File(...), ):

    if os.getenv('ENVIRONMENT') == 'production':
        return {'message': 'This environment dont support retraining!'}

    extension = file.filename.split(".")[-1] in ("csv")
    if not extension:
        return "file must be in csv extension!"

    file.filename = 'temp_data.csv'

    # save file to local
    file_path = "tmp/" + file.filename
    with open(file_path, "wb") as f:
        f.write(file.file.read())


@app.post('/tmp/train')
async def training(data: RetrainVariables):

    if os.getenv('ENVIRONMENT') == 'production':
        return {'message': 'This environment dont support retraining!'}

    file_path = 'tmp/temp_data.csv'

    if data.epochs > 30:
        return {'message': 'reach maximum epochs, max retraining epochs is 30'}

    result = _training(file_path, data)
    return {'message': result}

@app.post('/tmp/predict')
def predict_chd_temp_model(data: Record):

    if os.getenv('ENVIRONMENT') == 'production':
        return {'message': 'This environment dont support retraining!'}

    try:
        arr = _preprocessing(data)

        result = _predict(arr, True)

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

def _predict(arr, is_temp=False):
    try:
        if is_temp:
            tmp_model = tf.keras.models.load_model('tmp/model/tmp_model/')
            return tmp_model.predict(arr)

        return model.predict(arr)
    except Exception as e:
        raise ValueError('failed to fed model')


def dropColomn(df, col):
    try:
        df = df.drop(col,1)
    except:
        print('colomn {0} not found'.format(col))
    return df

def split_feature_label(df):
    return df.iloc[:, :-1].values, df.iloc[:, -1].values

def _training(file, data):
    try:
        # import dataset
        dataset = pd.read_csv(file)
        # drop colomn id
        dataset = dropColomn(dataset, 'id')

        # split features and labels
        x,y = split_feature_label(dataset)

        # split to training and validation
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = data.split_rate, random_state = data.random_state)

        x_train = sc.transform(x_train)
        x_val = sc.transform(x_val)

        model = make_model(x_train)

        model.fit(
            x_train,
            y_train,
            batch_size=data.batch_size,
            epochs=data.epochs,
            validation_data=(x_val, y_val),
            callbacks=early_stopping)


        model.save('tmp/model/tmp_model/')

        y_pred = model.predict(x_val)
        cm = confusion_matrix(y_val, y_pred >= 0.5)

        return {"confusion_matrix": cm.tolist(), "accuracy": accuracy_score(y_val, y_pred >= 0.5)}

    except Exception as e:
        raise ValueError('failed to train model')


if __name__ == '__main__':

    host = os.getenv('APP_HOST') if os.getenv('APP_HOST') else '0.0.0.0'
    port = int(os.getenv('APP_PORT')) if os.getenv('APP_PORT') else '9999'

    uvicorn.run(app, host=host, port=port)