from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from category_encoders import TargetEncoder

from datetime import timedelta
import pandas as pd
import numpy as np

class TrainTransformer(BaseEstimator, TransformerMixin):
    """
    Keep only that records, which occure in test dataset.
    
    Parameters:
        test (_DataFrame_): Test dataset
    """
    
    def __init__ (self, test):
        self.test = test
        
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        train = X.copy()
        test_pairs = self.test.groupby(["shop_id", "item_id"]).count().reset_index().loc[:, ["shop_id", "item_id"]]
        
        transformed_train = train.merge(test_pairs, on=["shop_id", "item_id"])
        return transformed_train        
            
class UniquenessTransformer(BaseEstimator, TransformerMixin):
    """
    Delete repeated records in DataFrame
    
    Parameters:
        features (_list_): Features, which define uniqueness. Records, which have same values for this features, are defined as repeated.
    """
    def __init__(self, features):
        self.features = features
  
    def fit(self, X, y=None):
        return self
    
    def transform(self, X:pd.DataFrame, y=None):
        X_copy = X.copy()
        X_copy["index"] = X_copy.index
        
        X_group = X_copy.groupby(self.features).count().reset_index()
        count_feature = X_group.columns[len(self.features)]
        non_unique = X_group[X_group[count_feature] > 1]
        X_copy.drop(X_copy.merge(non_unique, on=self.features)["index_x"].values, axis="rows", inplace=True)
        return X_copy.drop(["index"], axis="columns")

class MergeTransformer(BaseEstimator, TransformerMixin):
    """ 
    Merge data from different dataframes into single one.
    
    Parameters:
        merge_list (_list_): List of tuples `(DataFrame, column)`. Each DataFrame joins `X` on corresponding column
    """
    
    def __init__ (self, merge_list):
        self.merge_list = merge_list
            
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        
        mergeX = X.copy()
        
        for key, value in self.merge_list:
            mergeX = mergeX.join(key, on=value, lsuffix="_to_delete")
            
            
        to_delete = []   
        for column in mergeX.columns:
            if column.find("_to_delete") != -1:
                to_delete.append(column)
            
        
        return mergeX.drop(to_delete, axis=1)
    
    
class NegativeValueTransformer(BaseEstimator, TransformerMixin):
    """
    Delete negative values from DataFrame
    
    Parameters:
        feature (_str_): Name of feature. All records, where this feature value is negative will be deleted
    """
    
    def __init__ (self, feature):
        self.feature = feature
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[X[self.feature] > 0]
    

class OutliersTransformer(BaseEstimator, TransformerMixin):
    """ 
    Delete all records, where item price is more than 1000 
    """
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[X["item_cnt_day"] < 1000]
    
    
    
class DtypesTransformer(BaseEstimator, TransformerMixin):
    """
    Assign correct data types for columns
    
    Parameters:
        feature_map (_dict_) : Defines, which data types to assign to corresponding features.
    """
    
    def __init__ (self, feature_map):
        self.feature_map = feature_map
        
    
    def fit(self, X, y=None):
        return self
    
    
    def transform(self, X):
        for key, value in self.feature_map.items():
            if key == "date":
                X[key] = pd.to_datetime(X[key], format=value)
            else : X[key] = X[key].astype(value)
            
            
        return X            

class SeasonalityTransformer(BaseEstimator, TransformerMixin):
    """ 
    Create new features, based on dates. New features are:
    - `weekday`
    - `month`
    - `year` 
    
    Parameters:
        date_column (_str_): Name of column, which contains date information
    """
    
    def __init__ (self, date_column):
        self.date_column = date_column
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed["weekday"] = X[self.date_column].apply(lambda x : x.weekday())
        X_transformed["month"] = X[self.date_column].apply(lambda x : x.month)
        X_transformed["year"] = X[self.date_column].apply(lambda x : x.year)
        
        return X_transformed
    
    
class EventsTransformer(BaseEstimator, TransformerMixin):
    """ 
    Create new features for different events. New features are:
    - `is_NewYear` - **True**, if date is close to New Year (from 20.12 till 31.12), else **False**
    - `is_OctoberSales` - **True**, if date is close to October Sales dates (from 01.10 till 10.10), else **False**
    
    Parameters:
        date_column (_str_): Name of column, which contains date information
    """
    def __init__(self, date_column):
        self.date_column = date_column
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = X.copy()
        X_transformed["is_NewYear"] = X[self.date_column].apply(lambda x : 1 if (x.month == 12 and x.day > 20) else 0)
        X_transformed["is_OctoberSales"] = X[self.date_column].apply(lambda x : 1 if (x.month == 10 and x.day < 10) else 0)
        
        return X_transformed
    
    
class PriceClusterTransform(BaseEstimator, TransformerMixin):
    """
    Create new features for different price clusters. Features are `One Hot Encoded` and each feature shows if record belongs to corresponding price cluster
    
    Parameters:
        price_column (_str_) : Name of column, which contains price information
        n_cluster (_str_) : Amount of clusters we want to create 
    """
    
    def __init__ (self, price_column, n_clusters):
        self.price_column = price_column
        self.model = KMeans(n_clusters=n_clusters, random_state=52)
        
    def fit(self, X, y=None):
        self.model.fit(X[self.price_column].apply(lambda x: np.log(x)).values.reshape(-1, 1))
        return self
    
    def transform(self, X, y=None):
        price_modes = X.loc[:, ["item_id", self.price_column]].groupby("item_id").agg({self.price_column: lambda x : x.mode()[0]}).reset_index()
        price_modes["price_cluster"] = self.model.predict(price_modes[self.price_column].apply(lambda x : np.log(x)).values.reshape(-1, 1))
        price_modes = price_modes.set_index("item_id")
        price_cluster_map = price_modes["price_cluster"].to_dict()        
        
        X_transformed = X.copy()
        X_transformed["price_category"] = X_transformed["item_id"].apply(lambda x : price_cluster_map[x])
        encoder = OneHotEncoder(sparse_output=False)
        X_transformed = pd.concat([X_transformed, pd.DataFrame(encoder.fit_transform(X_transformed[["price_category"]]), columns=encoder.get_feature_names_out(), index=X_transformed.index)], axis="columns")
        return X_transformed


def get_city_name(x:str):
		if x[0] == "!":
			x = x[1:]
		return x.split()[0]
    
def get_shop_type(x:str):
    words = x.split()
    if x == "Цифровой склад 1С-Онлайн" or x == "Интернет-магазин ЧС":
        return "Digital"
    if x == "Выездная Торговля" or x == 'Москва "Распродажа"':
        return "Event"
    for word in words:
        if word.upper() == word and word.isalpha():
            return word
    return "Other"    

def get_group(x:str):
	if x.find("-") == -1:
		return x
	return x[ : x.find("-")  - 1]
    
class NewCategoriesTransformer(BaseEstimator, TransformerMixin): 
    """ 
    Create new features, based on shop names and item category names
    
    Shop and category names contain information about city, where shop is situated, type of shop and group of products, which corresponds to item.
    This transformer extracts this information and creates new features. Features are `One Hot Encoded` and each feature shows if record belongs to feature or not. 
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X["shop_type"] = X["shop_name"].apply(get_shop_type)
        X["city_name"] = X["shop_name"].apply(get_city_name)
        X["group"] = X["item_category_name"].apply(get_group)
        
        return X
	
    
    
class CategoryOneHotEncoder(BaseEstimator, TransformerMixin):
    """ 
    Encode categorical features using `sklearn.preprocessing.OneHotEncoder`
    
    Parameters:
        columns (_list_): List of columns, which are going to be encoded
    """
    
    
    def __init__ (self, columns):
        self.columns = columns
        self.encoder = OneHotEncoder(sparse_output=False)
        
    def fit(self, X, y=None):
        self.encoder.fit(X.loc[:, self.columns])
        return self
        
    def transform(self, X, y=None):
        encoded = pd.DataFrame(self.encoder.transform(X.loc[:, self.columns]), columns=self.encoder.get_feature_names_out(), index=X.index)
        X_transformed = pd.concat([X, encoded], axis="columns")
        return X_transformed
    
    
class NewProductsTransformer(BaseEstimator, TransformerMixin):
    """
    Preprocess early item records
    
    Often in our data, when new items appear in shops, their sales are increased because of their novelty.
    This transformer replaces data in first days (number of days is defined by `delta` parameter) with mode in order to generalize sales.
    
    Parameters:
        delta (_int_) : Number of days in which we consider the product to be new
    """
    
    def __init__(self, delta):
        self.delta = delta
    
    def repeat_dates_(self, row):
        new_dates = [row["date"] + timedelta(days=i) for i in range(self.delta)]
        repeated_rows = pd.DataFrame(
			{
				"item_id" : [row["item_id"]] * self.delta,
				"date" : new_dates,
				"item_cnt_day" : [row["item_cnt_day"]] * self.delta
			}
		)
        
        return repeated_rows
    
    
    def fit(self, X, y=None):   
        self.first_mentions = X.loc[:, ["date", "item_id", "item_cnt_day"]].groupby("item_id").agg({"item_cnt_day" : lambda x : x.mode()[0], "date": "min"}).reset_index()
        self.first_mentions = pd.concat(self.first_mentions.apply(self.repeat_dates_, axis="columns").to_list(), ignore_index=True)
        return self
    
    
    def transform(self, X, y=None):
        X_tf = X.copy()
        merged = X_tf.merge(self.first_mentions, on=["item_id", "date"], how="left", suffixes=("", "_merged"))
        X_tf["item_cnt_day"] = merged["item_cnt_day_merged"].combine_first(X_tf["item_cnt_day"])
        
        return X_tf
    
    
class IsOpenTransformer(BaseEstimator, TransformerMixin):
    """ 
    Create features which indicate if shop is still open
    
    Parameters:
        delta (_int_): Number of month in which we consider the shop to be closed if we have no records about sales in it
    """
    
    def __init__(self, delta):
        self.delta = delta
    
    def fit(self, X, y=None):
        self.shops_info = X.loc[:, ["shop_id", "date_block_num"]].groupby("shop_id").max()
        return self
    
    def transform(self, X, y=None):
        X_tf = X.copy()
        X_tf["still_opened"] = X_tf["date_block_num"].apply(lambda x : 1 if 33 - x < self.delta else 0)
        return X_tf

class OutliersCleaningTransformer(BaseEstimator, TransformerMixin):
    """ 
    Delete outliers according to outliers map
    
    Parameters:
        outliers_map (_dict_) : Outliers Map. Each value shows what is the max value for key (feature) 
    """
    
    def __init__(self, outliers_map):
        self.outliers_map = outliers_map
	
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_tf = X.copy()
        for column, value in self.outliers_map.items():
            X_tf = X_tf[X_tf[column] <= value]
        
        return X_tf    
    
class LagsEncoder(BaseEstimator, TransformerMixin):
    """ 
    Create lags for **price** and **sales**. Number of lags is 4, which is defined by EDA stage.
    """
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        lag_train = X.loc[:, ["date_block_num", "shop_id", "item_id", "item_cnt_day", "item_price"]]
        print(lag_train.shape)	     
        lag_train_agg = lag_train.groupby(["date_block_num", "shop_id", "item_id"]).agg({"item_price" : lambda x: x.mode()[0], "item_cnt_day" : "sum"})
        for lag in range(1, 4):
            lag_train_agg[f"item_price_lag_{lag}"] = lag_train_agg.groupby(["shop_id", "item_id"])["item_price"].shift(lag)
            lag_train_agg[f"item_cnt_day_lag_{lag}"] = lag_train_agg.groupby(["shop_id", "item_id"])["item_cnt_day"].shift(lag)

        lag_train_agg = lag_train_agg.reset_index()
        print(lag_train_agg.shape)
        fill_price_map = lag_train_agg.groupby(["shop_id", "item_id"])["item_price"].agg("first").to_dict()
        fill_item_cnt_map = lag_train_agg.groupby(["shop_id", "item_id"])["item_cnt_day"].agg("first").to_dict()
        
        for lag in range(1, 4):
            lag_train_agg[f"item_price_lag_{lag}"] = lag_train_agg.apply(
        		lambda x: x[f"item_price_lag_{lag}"] if not np.isnan(x[f"item_price_lag_{lag}"]) else fill_price_map[(x["shop_id"], x["item_id"])], axis=1
        )
            lag_train_agg[f"item_cnt_day_lag_{lag}"] = lag_train_agg.apply(
        		lambda x: x[f"item_cnt_day_lag_{lag}"] if not np.isnan(x[f"item_cnt_day_lag_{lag}"]) else fill_item_cnt_map[(x["shop_id"], x["item_id"])], axis=1
        )
        print(lag_train_agg.shape)
        lag_train_agg = lag_train_agg.set_index(X.index)
        return pd.concat([X, lag_train_agg.drop(["date_block_num", "shop_id", "item_id", "item_cnt_day", "item_price"], axis="columns")], axis="columns")
    
    
class CategoryTargetEncoder(BaseEstimator, TransformerMixin):
    """ 
    Encode categorical features using `categorical_encoders.TargetEncoder`
    
    Parameters:
        columns (_list_): List of columns, which are going to be encoded
    """
    
    def __init__ (self, columns):
        self.columns = columns
        self.encoder = TargetEncoder()
        
    def fit(self, X, y=None):
        self.encoder.fit(X.loc[:, self.columns], X.item_cnt_day)
        return self
        
    def transform(self, X, y=None):
        encoded = pd.DataFrame(self.encoder.transform(X.loc[:, self.columns]), columns=self.encoder.get_feature_names_out(), index=X.index)
        X_transformed = pd.concat([X.drop(self.columns, axis="columns"), encoded], axis="columns")
        return X_transformed
    
    
class ColumnDropper(BaseEstimator, TransformerMixin):
    """
    Save particular columns from DataFrame
    
    Parameters:
        columns_to_save (_list_): List of columns to save from DataFrame 
    """
    def __init__(self):
        self.columns_to_save = list()
    
    def fit(self, X, y=None):
        for feature in X.columns:
            print(feature)
            if X[feature].dtype != np.dtype("object") and X[feature].dtype != np.dtype("datetime64[ns]") :
                self.columns_to_save.append(feature)
            print(feature)
        return self
                
    def transform(self, X, y=None):
        return X.loc[:, self.columns_to_save]    
    

class AggregationTransformer(BaseEstimator, TransformerMixin):
    """
    Aggregate DataFrame by months
    
    Parameters:
        start_date (_str_): Start date. This date defines from which date we will start to count date blocks 
        periods (_int_): Number of months in DataFrame
    """

    def __init__(self, start_date, periods):    
        self.start_date = start_date
        self.periods = periods
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        date_range = pd.date_range(start=self.start_date, periods=self.periods, freq="MS")
        date_blocks = [i for i in range(0, self.periods)]
        dates_map = dict(zip(date_blocks, date_range))
        
        aggregated_X = X.drop(["date"], axis="columns")
        aggregated_X = aggregated_X.groupby(["date_block_num", "shop_id", "item_id"]).agg({"item_price" : lambda x : x.mode()[0], "item_cnt_day": "sum"}).reset_index()
        aggregated_X["date"] = aggregated_X["date_block_num"].apply(lambda x : dates_map[x])
        return aggregated_X
    
class FeatureSelectionTransformer(BaseEstimator, TransformerMixin):
    """
    Select particular features from DataFrame
    
    Parameters:
        features (_list_) : List of features to keep 
    """    

    
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return X.loc[:, self.features]
    
    
class TestPreprocessTransformer(BaseEstimator, TransformerMixin):
    """ 
    Preprocess test DataFrame. For test data we only know item and shop, so we need to preprocess DataFrame to be simmilar to train set
    
    Parameters:
        raw_train (_DataFrame_) : Raw train data. We will use it to fill missing values in test set.
        start_date (_str_): Start date. This date will fill `date` column in order to make test and train shapes same.
        period (_int_): Number of month of test data.
    """
    
    def __init__(self, raw_train, start_date, period):
        self.train = raw_train
        self.start_date = start_date
        self.period = period
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X["date_block_num"] = self.period
        X["date"] = pd.to_datetime(self.start_date, dayfirst=True)

        item_price_map = self.train.loc[:, ["item_id", "shop_id", "item_price"]].groupby(["item_id", "shop_id"]).agg(lambda x : x.mode()[0]).to_dict()["item_price"]

        X["item_price"] = X.apply(lambda x : item_price_map[(x["item_id"], x["shop_id"])] if (x["item_id"], x["shop_id"]) in item_price_map.keys() else -1, axis=1)
        X["item_cnt_day"] = 0
        return X[X["item_price"] != -1]
    

class TestTrainMergeTransformer(BaseEstimator, TransformerMixin):
    """
    Merge Train and Test DataFrame
    
    Parameters:
        agg_train (_DataFrame_): Aggregated train DataFrame that will be merged with test.
    """
    
    def __init__ (self, agg_train):
        self.train = agg_train
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        return pd.concat([self.train, X])
    

class TestSetExtractionTransformer(BaseEstimator, TransformerMixin):
    """ 
    Extract Test dataset from merged train-test
    """
    
    def fit(self, X, y=None):
        self.test_month = X["date_block_num"].max()
        return self
    
    def transform(self, X, y=None):
        return X[X["date_block_num"] == self.test_month]