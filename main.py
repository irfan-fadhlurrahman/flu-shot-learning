from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import train
from train import impute

class PersonalInformation(BaseModel):
      age_group: str
      sex: str
      h1n1_concern: int
      h1n1_knowledge: int

def load_pipeline(folder="saved-pipeline/", filename="saved_pipeline_3.bin"):
    PATH = "/mnt/c/Project/flu-shot-learning/"
    with open(PATH + folder + filename, "rb") as file_out:
         pipeline_dict = pickle.load(file_out)
    return pipeline_dict

# define an app
app = FastAPI()

# load the saved pipeline
pipeline_dict = load_pipeline()

@app.get("/")
def homepage():
    return {"messages": "this is homepage"}

@app.post("/predictions")
def get_probabilities(data: PersonalInformation):
    # convert to python dictionary
    received = data.dict()

    # load the pipeline
    feature_names = list(pipeline_dict['features'].keys())
    imputer = pipeline_dict['imputer']
    preprocessor = pipeline_dict['preprocessor']
    model = pipeline_dict['model']

    # make a dataframe
    X = np.zeros((1, 35), int)
    X = pd.DataFrame(X, columns=feature_names)

    # add the input features
    input_features = list(received.keys())
    for col in input_features:
        X.loc[0, col] = received[col]

    # impute missing values if any
    if X.isnull().sum().sum() > 0:
        X = imputer.transform(X)

    # one-hot encode
    X = X.to_dict(orient='records')
    X = preprocessor.transform(X)

    # predict
    y_pred = model.predict_proba(X)
    y_pred_h1n1 = y_pred[0][:, 1][0]
    y_pred_seasonal = y_pred[1][:, 1][0]

    return {
        "Getting H1N1 vaccine probabilities": float(y_pred_h1n1),
        "Getting Seasonal vaccine probabilities": float(y_pred_seasonal)
    }













# end of code
