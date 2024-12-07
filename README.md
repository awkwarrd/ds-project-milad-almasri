# ds-project-milad-al-masri

## 1. Overview

This project is my internship project, that was created in order to show my machine learning and data science skills on close-to-real project. In this project I was creating model for sales volume predictions based on historical data and other features. Read more about problem: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/overview

## 2. Motivation

The motivation behind this project is to create accurate model, which will helps business to:
1. Predict volumes of sales 
3. Understand, what are the reasons behind this prediction


## 3. Success metrics
This solution will help business to correct their marketing efforts and reduce unnecessary product costs.


## 4. Requirements & Constraints
Functional Requirements: Reduce unnecessary product costs by __>=10%__

Technical Requirements: Model should run in the cloud, have an explanation layer and __RMSE <=2__

## 4.1 What's in-scope & out-of-scope?
All problems, which were defined, were successfully solved.

## 5. Methodology
### 5.1. Problem statement
This problem could be framed as a __Supervised Regression Problem based on Time Series Data__

### 5.2. Data
For training process, model requires:
1. Historical data about sales (date, shop id, item id, date block number, item price and sales per day)
2. Info about Shops(shops info corresponding to their ids)
3. Items (items info corresponding to their ids)
4. Items categories (item category info corresponding to the item)

For test process, model requires:
1. What item in what shop is going to be predicted
2. Data for training process

This raw data is going to be preprocessed with the help of different pipelines and passed to the model (read more in [5.3](#53-techniques))


### 5.3. Techniques
For data preprocessing I have used different techniques. DQC and EDA processes, which had defined all preprocessing, are explained in `dqc.ipynb` and `eda.ipynb` files. Also there are transformers and pipelines for train/test preprocessing based on this two files, which you can find in `scripts/price_pred` folder or on [PyPI page](https://pypi.org/project/price-predictions/) and install it. This package contains all necessary transformers and pipelines for data preprocessing and documentation to it.


### 5.4. Experimentation & Validation
>How will you validate your approach offline? What offline evaluation metrics will you use?

#### 5.4.1 Model Validation

As in this project we've worked with Time Series Data, For validation I've chosen `Expanding Window` approach for validation. Method for model validation is also included in `validation.py` file, which you can find in `scripts/price_pred` folder or on [PyPI page](https://pypi.org/project/price-predictions/) and install it. 

#### 5.4.2 Feature Selection
For Feature Selection process in this project, I've used combination of different statistical methods like `Pearson Correlation`, `ANOVA` and others with `Boruta Algorithm`

In first step, statistical methods vote for most promising features, and choose that features, which have at least 50% of votes. Then, based on this features, Boruta chooses final features, that are going to be fed into model. Feature Selection function could be find in `feature_selection.py` file in `scripts/price_pred` folder or on [PyPI page](https://pypi.org/project/price-predictions/).


### 5.5. Human-in-the-loop
User interacts with the system, deployed in the cloud with the help of API. It could ask model to predict sales by request and get results.

## 6. Deployment Details

Model was deployed with the help of `Google Cloud Platform`. All processes for data preprocessing, model training and predicting were implemented with the help of `Apache Airflow` pipelines (in `dags` folder).  All this pipelines run on Virtual Machine created on GCP. Also in this project I've used DVC for storing and getting data. All data for this project is stored in bucket on GCP. Different files, like scripts for running Airflow Server, file with requirements and other, you can find in `airflow_scripts` folder. This files should be transferred on VM with `dags` folder for correct work.

## 7. Future Work
The following can be done to further work on this project

1. Implement more airflow pipelines to enhance automatization of different process.
2. Monitor and control model performance in order to detect data drift/model degradation




