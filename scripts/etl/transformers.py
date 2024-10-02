from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

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