from sklearn.feature_selection import SelectKBest, r_regression, mutual_info_regression, f_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from itertools import compress
import pandas as pd
import numpy as np

class VoitingSelector():
    """
    Class for feature selection step.
    
    **VotingSelector** use 4 methods for feature selection:
    
    - Pearson Correlation
    - Mutual Information
    - Variance-Inflation Factor (VIF)
    - Analysis of variance (Anova)
    
    Each of this methods vote for best features according to their methods and statistics.
    Final subset of best fetures is defined by *voting_threshhold* parameter,
    which corresponds to percent of votes for feature to be added to best feature subset
    """    
    
    def __init__ (self):
        self.votes = None
        self.selectors = {
            "pearson" : self._select_pearson,
            "vif" : self._select_vif,
            "mi" : self._select_mi,
            "anova" : self._select_anova
		}
        
    @staticmethod
    def _select_pearson(X, y=None, **kwargs):
        selector = SelectKBest(r_regression, k=kwargs.get("n_features_to_select", 20)).fit(X, y)
        return selector.get_feature_names_out()


    @staticmethod
    def _select_mi(X, y=None, **kwargs):
        selector = SelectKBest(mutual_info_regression, k=kwargs.get("n_features_to_select", 20)).fit(X, y)
        return selector.get_feature_names_out()
        
    
    @staticmethod
    def _select_vif(X, y=None, **kwargs):
        return [
           X.columns[feature_index]
           for feature_index in range(len(X.columns))
           if variance_inflation_factor(X.values, feature_index) <= kwargs.get("vif_threshold", 5)
       ]
 
    @staticmethod
    def _select_anova(X, y=None, **kwargs):
        selector = SelectKBest(f_regression, k=kwargs.get("n_features_to_select", 20)).fit(X, y)
        return selector.get_feature_names_out()
    
    def select(self, X:pd.DataFrame, y:pd.Series | np.ndarray , voting_threshold=0.5, **kwargs) -> pd.DataFrame:
        """_summary_

        Args:
            X (pd.DataFrame): Source DataFrame, in which we are looking for best features to choose
            y (pd.Series | np.ndarray): Corresponding target values for X
            voting_threshold (float, optional): Defines a percents of votes, needed for feature to be selected_. Defaults to 0.5.

        Returns:
            pd.DataFrame: DataFrame with best features_
        """
        votes = []
        for selector_name, selector_method in self.selectors.items():
           features_to_keep = selector_method(X, y, **kwargs)
           votes.append(
               pd.DataFrame([int(feature in features_to_keep) for feature in X.columns]).T
           )
           print(f"{selector_name} calculation completed!")
        self.votes = pd.concat(votes)
        self.votes.columns = X.columns
        self.votes.index = self.selectors.keys()
        features_to_keep = list(compress(X.columns, self.votes.mean(axis=0) >= voting_threshold))
        return X[features_to_keep]
