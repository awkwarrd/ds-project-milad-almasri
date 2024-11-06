from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

class ModelValidation():
    
    def __init__(self, X, y, model, verbose=1):
        self.X = X
        self.y = y
        self.model = model
        self.verbose = verbose
        
        
    def validate(self, n_splits):
        self.scores = []
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        for i, (train_index, valid_index) in enumerate(tscv.split(self.X)):
            if self.verbose:
                print(f"Model: {i}")
             
 
            X_train = self.X.iloc[train_index]
            y_train = self.y.iloc[train_index]
            
            X_valid = self.X.iloc[valid_index]
            y_valid = self.y.iloc[valid_index]
            
            self.model.fit(X_train, y_train)
            predictions = self.model.predict(X_valid)
            self.scores.append(root_mean_squared_error(y_valid, predictions))
        if self.verbose:
            print("Validation Completed!")
         
        return self


    def score(self):
        return self.score.sum() / len(self.score)
    
     