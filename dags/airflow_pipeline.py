from airflow import DAG
from airflow.operators.python import PythonOperator

from price_pred.pipelines import TrainPreprocessingPipeline, TestPreprocessingPipeline, ETLPipeline, EDAPipeline
from price_pred.transformers import *

import os
from datetime import datetime, timedelta
import logging
from pickle import load, dump
import pandas as pd
from sklearn.pipeline import Pipeline


default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

### Default params for pipeline
feature_map = {"date" : "%Y-%m-%d",
            "date_block_num" : "int",
            "shop_id" : "O",
            "item_id" : "O",
            "item_price" : "float",
            "item_cnt_day" : "float",
            "shop_name" : "O",
            "item_name" : "O",
            "item_category_name" : "O", 
            "item_category_id" : "O"}


project_path = "/home/almasrimilad21/airflow/"


def pipeline_preparations():
    if os.path.exists(project_path):
        
        shops = pd.read_csv(f"{project_path}source_data/shops.csv")
        items = pd.read_csv(f"{project_path}source_data/items.csv")
        item_categories = pd.read_csv(f"{project_path}source_data/item_categories.csv")

        preprocessed_train = pd.read_csv(f"{project_path}source_data/preprocessed_train_airflow.csv")

        merge_list = [(shops, "shop_id"), (items, "item_id"), (item_categories ,"item_category_id")]
    
        etl_pipeline = ETLPipeline(merge_list, feature_map=feature_map) 
        eda_pipeline = EDAPipeline()

        etl_eda_pipeline = Pipeline([
            ("etl", etl_pipeline), 
            ("eda", eda_pipeline)
        ])
    
        etl_eda_pipeline.fit(preprocessed_train)
        dump(etl_eda_pipeline, open(f"{project_path}source_data/etl_eda_pipeline.pkl", "wb"))

    
def preprocess_train_data():
    
    shops = pd.read_csv(f"{project_path}source_data/shops.csv")
    items = pd.read_csv(f"{project_path}source_data/items.csv")
    item_categories = pd.read_csv(f"{project_path}source_data/item_categories.csv")
    sales_train = pd.read_csv(f"{project_path}source_data/sales_train.csv")

    merge_list = [(shops, "shop_id"), (items, "item_id"), (item_categories ,"item_category_id")]
    
    preprocessed_train = TrainPreprocessingPipeline(merge_list).pipeline.fit_transform(sales_train)
    preprocessed_train.to_csv(f"{project_path}source_data/preprocessed_train_airflow.csv")
    
    
def preprocess_test_data():
    
    etl_eda_pipeline = load(open(f"{project_path}source_data/etl_eda_pipeline.pkl", "rb"))
    
    preprocessed_train = pd.read_csv(f"{project_path}source_data/preprocessed_train_airflow.csv")
    sales_train = pd.read_csv(f"{project_path}source_data/sales_train.csv")
    test = pd.read_csv(f"{project_path}source_data/test.csv")
    
    preprocessed_test = TestPreprocessingPipeline(sales_train, preprocessed_train, etl_eda_pipeline).pipeline.fit_transform(test)
    logging.info("Preprocessing finished!")
    preprocessed_test.to_csv(f"{project_path}source_data/preprocessed_test_airflow.csv")
    
def make_predictions():
    model = load(open(f"{project_path}models/rfr_model.pkl", "rb"))
    test = pd.read_csv(f"{project_path}source_data/test.csv")
    test_data = pd.read_csv(f"{project_path}source_data/preprocessed_test_airflow.csv", index_col=0)
    pd.Series(model.predict(test_data).reindex(test.index, fill_value=0)).to_csv(f"{project_path}results.csv")
    

with DAG(
    'prediction_pipeline',
    default_args=default_args,
    description='From raw data to predictions.',
    schedule='@daily',
    start_date=datetime(2024, 11, 7),
    catchup=False,
) as dag:
    

    creating_pipelines_task = PythonOperator(
        task_id="creating_pipeline",
        python_callable=pipeline_preparations,
        execution_timeout=timedelta(minutes=20)
    )

    train_preprocess_task = PythonOperator(
        task_id='preprocess_data_train_data',
       python_callable=preprocess_train_data,
    )
    
    test_preprocess_task = PythonOperator(
        task_id='preprocess_test_data',
        python_callable=preprocess_test_data,
        execution_timeout=timedelta(minutes=20),
    )

    predict_task = PythonOperator(
        task_id='load_model_and_predict',
        python_callable=make_predictions,
        execution_timeout=timedelta(minutes=20)
    )

    train_preprocess_task >> creating_pipelines_task >> test_preprocess_task  >> predict_task