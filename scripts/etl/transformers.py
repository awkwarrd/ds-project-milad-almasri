from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

from datetime import timedelta
import pandas as pd
import numpy as np

class TrainTransformer(BaseEstimator, TransformerMixin):
    
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
    
    def __init__ (self, feature):
        self.feature = feature
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[X[self.feature] > 0]
    

class OutliersTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[X["item_cnt_day"] < 1000]
    
    
    
class DtypesTransformer(BaseEstimator, TransformerMixin):
    
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
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X["shop_type"] = X["shop_name"].apply(get_shop_type)
        X["city_name"] = X["shop_name"].apply(get_city_name)
        X["group"] = X["item_category_name"].apply(get_group)
        
        return X
	
    
    
class CategoryOneHotEncoder(BaseEstimator, TransformerMixin):
    
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
    
    def __init__(self, outliers_map):
        self.outliers_map = outliers_map
	
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_tf = X.copy()
        for column, value in self.outliers_map.items():
            X_tf = X_tf[X_tf[column] <= value]
        
        return X_tf    