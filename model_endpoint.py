from fastapi import FastAPI
from pydantic import BaseModel
from price_pred.pipelines import TestPreprocessingPipeline
from sklearn.pipeline import Pipeline
from pickle import load
import pandas as pd

app = FastAPI()

class PredictionRequest(BaseModel):
    shop_id : int
    item_id : int

def preprocess_test_data(request):
    etl_eda_pipeline = load(open("pipelines/etl_eda_pipeline.pkl", "rb"))
    
    preprocessed_train = pd.read_csv("data/preprocessed_train_airflow.csv")
    sales_train = pd.read_csv("data/data/sales_train.csv")
    
    test = pd.DataFrame(request, index=[0])
    preprocessed_test = TestPreprocessingPipeline(sales_train, preprocessed_train, etl_eda_pipeline).pipeline.fit_transform(test)
    
    return preprocessed_test
    
@app.post("/predict")
def predict(data:PredictionRequest):
    test_data = preprocess_test_data(dict(data))
    model = load(open("models/rfr_model.pkl", "rb"))
    if test_data.empty:
        print("There is no historical data about this item in this shop, so prediction is undefined")
        return {"prediction" : None}
    prediction = model.predict(test_data)
    return({"prediction": float(prediction)})
    
    