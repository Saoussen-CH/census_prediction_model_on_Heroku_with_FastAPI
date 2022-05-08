# Put the code for your API here.
import os
import yaml
import pandas as pd
import uvicorn
import pickle

from fastapi import FastAPI
from pydantic import BaseModel, Field


from starter.ml.model import *
from starter.ml.data import *

# FastAPI instance
app = FastAPI()


class Input(BaseModel):
    age : int = 23
    workclass : str = 'Self-emp-inc'
    fnlgt : int = 76516
    education : str = 'Bachelors'
    education_num : int = 13
    marital_status : str = Field(..., example="Married-civ-spouse", alias="marital-status")
    occupation : str = 'Exec-managerial'
    relationship : str = 'Husband'
    race : str = 'White'
    sex : str = 'Male'
    capital_gain : int = 0
    capital_loss : int = 0
    hours_per_week : int = 40
    native_country : str = Field(..., example="United-States", alias="native-country")
    
    class Config:
        allow_population_by_field_name = True

class Output(BaseModel):
    prediction:str

@app.get("/")
def welcome():
    return "Hi, Welcome to Census API"


model = pickle.load(open("model/model.pkl", "rb"))
encoder = pickle.load(open("model/oneHotEnoder.pkl", "rb"))
binarizer = pickle.load(open("model/LabelBinarizer.pkl", "rb"))

@app.post("/predict")
async def get_prediction(payload: Input):


    # Categorical features for transform model

    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
    

    # load predict_data
    request_data = pd.DataFrame(payload.dict(by_alias=True), index=[0])
    #request_dict = data.dict(by_alias=True)
    #request_data = pd.DataFrame(request_dict, index=[0])
    
    X, _, _, _ = process_data(
                request_data,
                categorical_features=cat_features,
                training=False,
                encoder=encoder,
                lb=binarizer)

    prediction = model.predict(X[:, :103])
    

    label = binarizer.inverse_transform(prediction)[0]

    response = {"prediction": {"salary": label}}
    return response
