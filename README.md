# ds-project-milad-al-masri

## 1. Overview

This project is my internship project, created to demonstrate my machine learning and data science skills on close-to-real-word problem. In this project I developed model for sales volume predictions based on historical data and other features. Learn more about problem: https://www.kaggle.com/competitions/competitive-data-science-predict-future-sales/overview

## 2. Motivation

The motivation behind this project is to create an accurate model, which will help business to:
1. Predict volumes of sales 
3. Understand, what is the model's motivation behind these predictions.


## 3. Success metrics
This solution will help businesses optimize their marketing efforts and reduce unnecessary product costs.


## 4. Requirements & Constraints
Functional Requirements: Reduce unnecessary product costs by __>=10%__

Technical Requirements: The model should run in the cloud, include an explanation layer and achieve __RMSE <=2__ 

## 4.1 Scopes and achievements
All problems defined at the beginning of the project were successfully solved.

During the work on this project, following results were achieved:

- Low error (__1.8 RMSE__ on test set),
- Transparent explanation layer, that clearly explains predictions
- Automatization of prediction process, using Apache Airflow and cloud technologies
- Model deployment, that provide an easy way to get predictions

## 5. Methodology
### 5.1. Problem statement
This problem can be framed as a __Supervised Regression Problem based on Time Series Data__

### 5.2. Data
For the training process, the model requires:
1. Historical sales data (date, shop ID, item ID, date block number, item price and sales per day)
2. Shop information, corresponding to their IDs
3. Items information, corresponding to their IDs
4. Items categories information, corresponding to the item

For test process, model requires:
1. Shop and item IDs
2. Data used during the training process

This raw data will be preprocessed using different pipelines and passed to the model (read more in [5.3](#53-techniques))


### 5.3. Techniques
For data preprocessing, I employed various techniques. The DQC (Data Quality Check) and EDA (Exploratory Data Analysis) processes, which defined the preprocessing strategy, are detailed in the `dqc.ipynb` and `eda.ipynb` files. Also there are transformers and pipelines for train/test preprocessing based on these two files, which you can be found in `scripts/price_pred` folder or on the [PyPI page](https://pypi.org/project/price-predictions/) and install it. This package includes all necessary transformers and pipelines for data preprocessing and documentation to it.


### 5.4. Experimentation & Validation

#### 5.4.1 Model Validation

Since this project deals with Time Series Data, `Expanding Window` approach for validation was chosen. Method for model validation is also implemented in `validation.py` file, which  can be found in the `scripts/price_pred` folder or on the [PyPI page](https://pypi.org/project/price-predictions/) and install it. 

#### 5.4.2 Feature Selection
For feature selection process in this project, I used combination of different statistical methods like `Pearson Correlation`, `ANOVA` and others with `Boruta Algorithm`

1. In the first step, statistical methods identify promising features by voting. Features with at least 50% of votes are selected.
2. In the second step, Boruta determines the final set of features to be fed into the model.

Feature Selection function could be found in the `feature_selection.py` file in the `scripts/price_pred` folder or on the [PyPI page](https://pypi.org/project/price-predictions/).


### 5.5. Human-in-the-loop
The user interacts with the system, deployed in the cloud, via API requests to the endpoints. Users can request sales predictions and receive results in response.

## 6. Deployment Details

The model was deployed using the `Google Cloud Platform (GCP)`. All processes for data preprocessing, model training, and prediction were implemented using `Apache Airflow` pipelines (located in the dags folder). These pipelines run on a `Virtual Machine` hosted on `GCP`. Additionally, I used `DVC` for data storage and retrieval. All project data is stored in a `GCP Bucket`.

Scripts for setting up the Airflow server, requirements, and other utilities are included in the airflow_scripts folder. These files, along with the dags folder, must be transferred to the VM for proper operation.

## 7. Future Work
The following improvements can be made to further enhance the project:

1. Implement additional Airflow pipelines to automate more processes.
2. Monitor and control model performance to detect data drift or model degradation.



