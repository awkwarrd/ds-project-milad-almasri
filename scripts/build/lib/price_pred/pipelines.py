from price_pred.transformers import *
from sklearn.pipeline import Pipeline

import pickle

class BasePipeline:
    """Base Pipeline Class. Contains basic pipelines methods like *fit*, *transform*, *fit_transfrom*, *save_pipeline* and *load_pipeline*. Used in building different pipelines
    """
    def fit(self, X, y=None):
        """Fits the pipeline to the data."""
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        """Transforms the data using the pipeline."""
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """Fits and transforms the data."""
        return self.pipeline.fit_transform(X, y)

    def save_pipeline(self, file_path):
        """Serializes the pipeline to a file."""
        with open(file_path, 'wb') as f:
            pickle.dump(self.pipeline, f)
    
    @staticmethod
    def load_pipeline(file_path):
        """Loads a serialized pipeline from a file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    


class ETLPipeline(BasePipeline):
    """Base pipelines for ETL process """
    def __init__(self, unique_features, merge_list, feature_map):
        self.unique_features = unique_features
        self.merge_list = merge_list
        self.feature_map = feature_map
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self):
        """Creates the ETL pipeline with the specified transformers."""
        return Pipeline([
            ("uniqueness_check", UniquenessTransformer(self.unique_features)),
            ("merge_dataframe", MergeTransformer(self.merge_list)),
            ("negative_values", NegativeValueTransformer("item_price")),
            ("selected_outliers", OutliersTransformer()),
            ("dtypes", DtypesTransformer(self.feature_map))
        ])


class EDAPipeline(BasePipeline):
    """Base pipeline for EDA process"""
    def __init__(self, outliers_map, date_column, price_columns, n_clusers, columns_target_encoding, new_product_delta):
        self.outliers_map = outliers_map
        self.date_column = date_column
        self.price_columns = price_columns
        self.n_clusters = n_clusers
        self.columns_target_encoding = columns_target_encoding
        self.new_product_delta = new_product_delta
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self):
        """Creates the EDA pipeline with the specified transformers."""
        return Pipeline([
	            ("ouliers_cleaning", OutliersCleaningTransformer(self.outliers_map)), 
	            ("lags", LagsEncoder()),
                ("seasonality", SeasonalityTransformer(self.date_column)),
                ("events", EventsTransformer(self.date_column)),
                ("price_clusters", PriceClusterTransform(self.price_column, self.n_clusters)),
                ("new_categories", NewCategoriesTransformer()),
                ("label_category_encoding", CategoryTargetEncoder(self.columns_one_hot)),
                ("new_products", NewProductsTransformer(self.new_product_delta)),
])
        
        
        
class AgregationPipeline(BasePipeline):
    """Pipeline for data aggregation. Aggregates sales data by month"""
    
    def __init__(self, start_date, periods):
        self.start_date = start_date
        self.periods = periods
        self.pipeline = self._create_pipeline()
        
    def _create_pipeline(self):
        """Creates the Aggregation pipeline with the specified transformers."""
        return Pipeline([("aggregation", AggregationTransformer(self.start_date, self.periods))])
    
    
class TestPreprocessingPipeline(BasePipeline):
    """Pipeline for test data preprocessing"""
    
    def __init__(self, raw_train, start_date, period, agg_train, features, etl_eda_pipeline):
        self.raw_train = raw_train
        self.start_date = start_date
        self.period = period
        self.agg_train = agg_train
        self.features = features
        self.etl_eda_pipeline = etl_eda_pipeline
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self):
        
        test_preprocessing_pipeline = Pipeline([
        	("etl", self.etl_eda_pipeline[0][1]),
	        ("dtypes", self.etl_eda_pipeline[0][-1]),
 	        ("eda", self.etl_eda_pipeline[1][1:-1])
        ])
        
        return Pipeline([
            ("test_preprocess", TestPreprocessTransformer(self.raw_train, self.start_date, self.period)),
            ("train_test_merge", TestTrainMergeTransformer(self.agg_train)),
            ("feature_extraction", test_preprocessing_pipeline),
            ("features_selection", FeatureSelectionTransformer(self.features)),
            ("test_set_extraction", TestSetExtractionTransformer())
        ])


class TrainPreprocessingPipeline(BasePipeline):
    
    def __init__(self, unique_features, merge_list, feature_map, start_date, periods):
        self.unique_features = unique_features
        self.merge_list = merge_list
        self.feature_map = feature_map
        self.start_date = start_date
        self.periods = periods
        self.pipeline = self._create_pipeline()
    
    
    def _create_pipeline(self):
        return Pipeline([
            ("etl", ETLPipeline(self.unique_features, self.merge_list, self.feature_map)), 
            ("agg", AggregationTransformer(self.start_date, self.periods))
        ])
    