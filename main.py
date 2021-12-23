
from io import StringIO
import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import numpy as np
import tensorflow as tf
import joblib
import os

import pandas as pd
from fastapi import FastAPI, File, UploadFile

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras

from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

model = tf.keras.models.load_model('model/ann_model/')
scaler_filename = "model/scaler.save"
sc = joblib.load(scaler_filename)


def make_model(x_train, hidden_layers):

    reqLayers = [int(x) for x in hidden_layers.split(',')]

    print(reqLayers)

    metrics = [
        keras.metrics.BinaryAccuracy(name='val_recall'),
    ]

    model = keras.Sequential()
    model.add(keras.layers.Dense(32, activation='relu',
                                 input_shape=(x_train.shape[-1],)))

    for layer in reqLayers:
        model.add(keras.layers.Dense(layer, activation='relu'))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_recall',
    verbose=1,
    patience=10,
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
    split_rate_training_testing: float
    split_rate_training_val: float
    random_state: int
    batch_size: int
    hidden_layer: str
    learning_rate: float


@app.get('/')
def index():
    return {'message': 'This is Coronary Heart Diseases Detector API!'}


@app.post("/tmp/upload")
async def upload(file: UploadFile = File(...), ):

    if os.getenv('ENVIRONMENT') == 'production':
        return HTTPException(status_code=400, detail=str({'message': 'This environment dont support retraining!'}))

    extension = file.filename.split(".")[-1] in ("csv")
    if not extension:
        return HTTPException(status_code=400, detail=str({'message': 'file must be in csv extension!'}))

    file.filename = 'temp_data.csv'

    # save file to local
    file_path = "tmp/" + file.filename
    with open(file_path, "wb") as f:
        f.write(file.file.read())

    return {'message': 'file uploaded successfully'}


@app.post('/tmp/train')
async def training(data: RetrainVariables):

    if os.getenv('ENVIRONMENT') == 'production':
        return HTTPException(status_code=400, detail=str({'message': 'This environment dont support retraining!'}))

    file_path = 'datasets/sampled_dataset.csv'

    result = _training(file_path, data)
    return {'message': result}


@app.post('/tmp/predict')
async def predict_chd_temp_model(data: Record):

    if os.getenv('ENVIRONMENT') == 'production':
        return HTTPException(status_code=400, detail=str({'message': 'This environment dont support retraining!'}))

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
        return sc.transform(np.array([data.sex, data.age, data.is_smoking, data.cigs_per_day,
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
        df = df.drop(col, 1)
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
        x, y = split_feature_label(dataset)

        # split to training and testing
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=data.split_rate_training_testing, random_state=data.random_state)

        # split to training and validation
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=data.split_rate_training_val, random_state=data.random_state)

        x_train = sc.transform(x_train)
        x_val = sc.transform(x_val)
        x_test = sc.transform(x_test)

        model = make_model(x_train, data.hidden_layer)

        model.summary()

        model.fit(
            x_train,
            y_train,
            batch_size=data.batch_size,
            epochs=128,
            validation_data=(x_val, y_val),
            callbacks=early_stopping)

        model.save('tmp/model/tmp_model/')

        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred >= 0.5)

        cm = cm.tolist()

        return {"true_positive": cm[0][0], 'false_positive': cm[0][1], 'false_negative': cm[1][0], 'true_negative': cm[1][1], "accuracy": accuracy_score(y_test, y_pred >= 0.5), "recall": recall_score(y_test, y_pred >= 0.5), "precision": precision_score(y_test, y_pred >= 0.5)}
    except Exception as e:
        raise ValueError('failed to train model')


if __name__ == '__main__':

    host = os.getenv('APP_HOST') if os.getenv('APP_HOST') else '0.0.0.0'
    port = int(os.getenv('APP_PORT')) if os.getenv('APP_PORT') else '9999'

    uvicorn.run(app, host=host, port=port)
