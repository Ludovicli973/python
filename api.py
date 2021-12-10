import joblib
import re
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy


app = FastAPI()
model = joblib.load('ntm.joblib')

def make_row_dataframe(APM,SelectByHotkeys,AssignToHotkeys,NumberOfPACs,GapBetweenPACs,ActionLatency):
    data = {'APM':[],'SelectByHotkeys': [],'AssignToHotkeys': [],'NumberOfPACs': [],'GapBetweenPACs': [],'ActionLatency':[]}
    df= pd.DataFrame(data)
    new_row = {'APM':APM,'SelectByHotkeys':SelectByHotkeys,'AssignToHotkeys':AssignToHotkeys, 'NumberOfPACs':NumberOfPACs,'GapBetweenPACs':GapBetweenPACs,'ActionLatency':ActionLatency}
    df = df.append(new_row, ignore_index=True)
    return df
def classify_gamer(model,APM,SelectByHotkeys,AssignToHotkeys,NumberOfPACs,GapBetweenPACs,ActionLatency):
    data_frame=make_row_dataframe(APM,SelectByHotkeys,AssignToHotkeys,NumberOfPACs,GapBetweenPACs,ActionLatency)
    r=model.predict(data_frame[:1])[0]
    return {'league':int(r)}


@app.get("/")
async def hello_world():
      return {"message":"Bienvenu dans l'api de star craft"}


@app.get('/data/')
async def detect_spam_query(APM: float , SelectByHotkeys: float,AssignToHotkeys:float,NumberOfPACs:float,GapBetweenPACs:float,ActionLatency:float):
	return classify_gamer(model,APM,SelectByHotkeys,AssignToHotkeys,NumberOfPACs,GapBetweenPACs,ActionLatency)

@app.get('/data/{message}')
async def detect_spam_path(APM: float , SelectByHotkeys: float,AssignToHotkeys:float,NumberOfPACs:float,GapBetweenPACs:float,ActionLatency:float):
	return classify_gamer(model,APM,SelectByHotkeys,AssignToHotkeys,NumberOfPACs,GapBetweenPACs,ActionLatency)