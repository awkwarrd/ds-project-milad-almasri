

from price_pred.transformers import *
from sklearn.pipeline import Pipeline

import pickle

unique_features = ["date", "shop_id", "item_id"]
    
feature_map = {"date" : "%d.%m.%Y",
            "date_block_num" : "int",
            "shop_id" : "O",
            "item_id" : "O",
            "item_price" : "float",
            "item_cnt_day" : "float",
            "shop_name" : "O",
            "item_name" : "O",
            "item_category_name" : "O", 
            "item_category_id" : "O"}



features = ["date_block_num", "item_price", "item_price_lag_1", "item_cnt_day_lag_1", "item_price_lag_2",
                     "item_cnt_day_lag_2", "item_price_lag_3", "item_cnt_day_lag_3", "item_price_lag_4", "item_cnt_day_lag_4", "month", "is_NewYear", "group"]

outliers_map = {"item_cnt_day": 400, "item_price": 60000}
columns_one_hot = ["city_name", "group", "shop_type"]



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
    """
    Pipeline according to ETL scheme
    
    Preprocessing Steps:
    1. **UniquenessTransformer** : Deletes simmilar records from dataframe 
    2. **MergeTransformer**: As the data we work with is separated into few files, in this step we merge all data together
    3. **NegativeValueTransformer** : Deletes records with negative price_values
    4. **OutliersCleaningTransformer** : Deletes detected outliers
    5. **DtypesTransformer** : Assign corresponding data types for features
    
    All Transformers in this steps are defined in `price_pred.transformers` module
    """

    def __init__(self, merge_list, unique_features=unique_features, feature_map=feature_map, outliers_map=outliers_map):
        self.unique_features = unique_features
        self.merge_list = merge_list
        self.feature_map = feature_map
        self.outliers_map = outliers_map
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self):
        """Creates the ETL pipeline with the specified transformers."""
        return Pipeline([
            ("uniqueness_check", UniquenessTransformer(self.unique_features)),
            ("merge_dataframe", MergeTransformer(self.merge_list)),
            ("negative_values", NegativeValueTransformer("item_price")),
            ("selected_outliers", OutliersCleaningTransformer(self.outliers_map)),
            ("dtypes", DtypesTransformer(self.feature_map))
        ])


class EDAPipeline(BasePipeline):
    """
    Pipeline according to EDA scheme
    
    Preprocessing Steps:
    1. **OutliersCleaningTransformer**: Deletes Outliers according to outliers map
    2. **LagsEncoder**: Create Lags for price and sales features
    3. **SeasonalityTransformer**: Creates new features based on dates 
    4. **EventsTransformer**: Creates features, which correspond to different events 
    5. **PriceClusterTransform**: Create features for price clusters separation
    6. **NewCategoriesTransformer**: Create features based on item categories and shops
    7. **CategoryTargetEncoder**: Encodes categorical features using **TargetEncoding**
    8. **NewProductsTransformer**: Deletes first sales of items as they usually higher for brand new items
    
    All Transformers in this steps are defined in `price_pred.transformers` module
    """
    def __init__(self, outliers_map=outliers_map, date_column="date", price_columns="item_price", n_clusters=4, columns_target_encoding=columns_one_hot, new_product_delta=14):
        self.outliers_map = outliers_map
        self.date_column = date_column
        self.price_columns = price_columns
        self.n_clusters = n_clusters
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
                ("price_clusters", PriceClusterTransform(self.price_columns, self.n_clusters)),
                ("new_categories", NewCategoriesTransformer()),
                ("label_category_encoding", CategoryTargetEncoder(self.columns_target_encoding)),
                ("new_products", NewProductsTransformer(self.new_product_delta)),
])
        
        
        
class AgregationPipeline(BasePipeline):
    """Pipeline for data aggregation.
    
    Preprocessing Steps:
    1. **AggregationTransformer**: Aggregates sales data by month
    
    All Transformers in this steps are defined in `price_pred.transformers` module
    """
    
    def __init__(self, start_date="01.01.2013", periods=34):
        self.start_date = start_date
        self.periods = periods
        self.pipeline = self._create_pipeline()
        
    def _create_pipeline(self):
        """Creates the Aggregation pipeline with the specified transformers."""
        return Pipeline([("aggregation", AggregationTransformer(self.start_date, self.periods))])
    
    
class TestPreprocessingPipeline(BasePipeline):
    """
    Pipeline for test data preprocessing
    
    Preprocessing Steps:
    1. **TestPreprocessTransformer**: Transforms Test Data to the shape which corresponds to original Train Data
    2. **TestTrainMergeTransformer**: Concatenates Test and Train Datasets
    3. **Feature Extraction**: Extracts new features using a bit modified ETL-EDA pipeline.
    4. **FeatureSelectionTransformer**: Selects best features. **This implementation does not find them, but just chooses from list defined by user.** If you want to find them, use `price_pred.feature_selection.VotingSelector` 
    5. **TestSetExtractionTransformer**. As we've merged train and test set, in this step, we delete train set and save only preprocessed test data.
    
    All Transformers in this steps (except of **Feature Extraction** step) are defined in `price_pred.transformers` module
    """
    
    
    def __init__(self, raw_train, agg_train, etl_eda_pipeline, start_date="01.01.2013", period=34, features=features):
        self.raw_train = raw_train
        self.start_date = start_date
        self.period = period
        self.agg_train = agg_train
        self.features = features
        self.etl_eda_pipeline = etl_eda_pipeline
        self.pipeline = self._create_pipeline()
    
    def _create_pipeline(self):
        
        test_preprocessing_pipeline = Pipeline([
        	("etl", self.etl_eda_pipeline['etl'].pipeline[1]),
	        ("dtypes", self.etl_eda_pipeline["etl"].pipeline[-1]),
 	        ("eda", self.etl_eda_pipeline["eda"].pipeline[1:-1])
        ])
        
        return Pipeline([
            ("test_preprocess", TestPreprocessTransformer(self.raw_train, self.start_date, self.period)),
            ("train_test_merge", TestTrainMergeTransformer(self.agg_train)),
            ("feature_extraction", test_preprocessing_pipeline),
            ("features_selection", FeatureSelectionTransformer(self.features)),
            ("test_set_extraction", TestSetExtractionTransformer())
        ])


class TrainPreprocessingPipeline(BasePipeline):
    """
    Pipeline for train Preprocessing
    
    Preprocessing Steps:
    1. **ETLPipeline**: Transforms DataFrame according to ETL scheme. Look `price_pred.pipelines.ETLPipeline` for more details.
    2. **AggregationTransformer** : Aggregates data by months
    
    All Transformers in this steps (except of `ETLPipeline`) are defined in `price_pred.transformers` module

    """
    def __init__(self, merge_list, unique_features=unique_features, feature_map=feature_map, start_date="01.01.2013", periods=34, outliers=outliers_map):
        
        self.unique_features = unique_features
        self.merge_list = merge_list
        self.outliers_map = outliers
        self.feature_map = feature_map
        self.start_date = start_date
        self.periods = periods
        self.pipeline = self._create_pipeline()
    
    
    def _create_pipeline(self):
        return Pipeline([
            ("etl", ETLPipeline(self.merge_list, self.unique_features, self.feature_map, self.outliers_map)), 
            ("agg", AggregationTransformer(self.start_date, self.periods))
        ])
    