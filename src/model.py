import numpy as np
import pandas as pd
import os
from colorama import Fore, Style, Back
from collections import OrderedDict
import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from typing import List, Tuple, Dict, Any, Optional
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error

from data.file_mapping import *
from .data_transformation import Tr

class Model(object):
    def __init__(self, data_transformer, **kwargs):
        '''
        Initialize the Model object.
        '''
        self.data_transformer = data_transformer
        self._default_model = kwargs.get("default_model", DummyRegressor(strategy="median"))
        self._best_model = kwargs.get("best_model", None)
        self.data_pipeline = data_transformer.make_pipeline()
        self._available_models = OrderedDict()
        self._mse_score = None
        self.fitted_model = None

    @property
    def get_default_model(self):
        '''
        Get the default model.
        '''
        return self._default_model

    @property
    def get_best_model(self):
        '''
        Get the best model.
        '''
        if self._best_model:
            return self._best_model
        raise Exception(Fore.RED + "Best model is None. Run the grid search or pass the best_model kwarg to the class." + Style.RESET_ALL)
    
    @property
    def get_mse(self):
        '''
        Get the MSE of the best model.
        '''
        print(Fore.GREEN + f"The MSE of the best model is {self._mse_score:.2f}" + Style.RESET_ALL)
        return self._mse_score

    @property
    def get_available_models(self):
        '''
        Get the available models.
        '''
        for i, (model_name, model_score) in enumerate(self._available_models.items()):
            print('-' * 20)
            print(Fore.GREEN + f"{i + 1}. {model_name} with {model_score:.2f} R2 score" + Style.RESET_ALL)

        return self._available_models

    def grid_search(
        self,
        models: Optional[List[Tuple[str, BaseEstimator]]] = None,
        param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
        save_models: bool = True
    ):
        '''
        A method to find the most suitable model amongst default models or models passed as inputs.
        Utilizes the validation dataset and stores models ranked by performance.
        If no models are passed, the default models and hyperparameter grids are used.
        If no models are passed and the models directory exists, the pretrained models are loaded and scored.
        Optionally, the models can be saved to the models directory, then existing models can be reloaded from there.

        Args:
            models (Optional): list of tuples, each containing a model name and a model object.
            param_grids (Optional): dict of parameter grids for each model.
            save_models (Optional): bool, if True, the models will be saved to the models directory.
        '''
        if models is None:  # Define default models if none are passed
            models = [
                ('K Neighbors Regression', KNeighborsRegressor()),
                ('SVR', SVR()),
                ('Gradient Boosting Regressor', GradientBoostingRegressor()),
                ('SGD Regressor', SGDRegressor())
            ]
            
            os.makedirs('models', exist_ok=True)
            
            print(Fore.YELLOW + 'No models were specified. Using default models.' + Style.RESET_ALL)
            print(Fore.GREEN + "The default models are:" + Style.RESET_ALL)
            for model in models:
                print(Fore.GREEN + f'-   {model[0]}' + Style.RESET_ALL)

        if param_grids is None: # Define default parameter grids if none are passed
            param_grids = {
                'K Neighbors Regression': {'regressor__n_neighbors': [3, 5, 7]},
                'SVR': {'regressor__C': [0.1, 1, 10], 'regressor__kernel': ['linear', 'rbf']},
                'Gradient Boosting Regressor': {'regressor__n_estimators': [50, 100], 'regressor__learning_rate': [0.1, 0.05]},
                'SGD Regressor': {'regressor__alpha': [0.0001, 0.001, 0.01]}
            }

        assert set(param_grids.keys()) == set([model[0] for model in models]), \
            Fore.RED + "Model names must match parameter grid keys for grid search." + Style.RESET_ALL

        best_score = -float('inf')
        best_model_name = None
        scoring = 'r2'  # Use R² as the primary scoring metric

        for model_name, model in models:
            
            model_path = os.path.join('models', f'{model_name.replace(" ", "_")}_{self.data_transformer.target_label}.joblib')
            
            if os.path.exists(model_path):
                print(Fore.GREEN + f"\n -Model {model_name} already exists. Skipping grid search for this model..." + Style.RESET_ALL)
                best_estimator = joblib.load(model_path)
                validation_score = best_estimator.score(self.data_transformer.X_val, self.data_transformer.y_val)
            else:
                model_pipe = Pipeline([
                    ('preprocessor', self.data_pipeline),
                    ('regressor', model)
                ])
                
                grid_search = GridSearchCV(
                    model_pipe, 
                    param_grid=param_grids.get(model_name, {}),
                    cv=3,
                    scoring=scoring,
                )

                grid_search.fit(self.data_transformer.X_train, self.data_transformer.y_train)
                validation_score = grid_search.score(self.data_transformer.X_val, self.data_transformer.y_val)
                best_estimator = grid_search.best_estimator_
                
                if save_models:
                    os.makedirs('models', exist_ok=True)
                    joblib.dump(best_estimator, model_path)
                    print(Fore.GREEN + f"\n - Model saved to" + Style.RESET_ALL + " " + model_path)

            self._available_models[model_name] = validation_score

            if validation_score > best_score:
                best_score = validation_score
                best_model_name = model_name
                self._best_model = best_estimator

            print(Fore.BLACK + Back.GREEN + f"\n -Validation score for {model_name}: {float(validation_score):.3f}" + Style.RESET_ALL)
            
        # Rank models by performance
        self._available_models = OrderedDict(
            sorted(self._available_models.items(), key=lambda x: x[1], reverse=True)
        )

        print("\n" + Fore.BLACK + Back.BLUE + f"-----Best model: {best_model_name} with validation score: {best_score:.2f}----- \n" + Style.RESET_ALL)

    def train(self, method: Optional[str] = None):
        '''
        Train the model using a labelled set of samples.

        Parameters:
            - method (Optional): string - from get_available_models
        '''
        if method == None:
            method = self.get_best_model if self.get_best_model != None else self.get_default_model
        else:
            try:
                method = self._available_models[method]
            except Exception:
                print(Fore.RED + 'Specified method is not available. Make sure you run the grid search to rank available models by performance and select a valid model.' + Style.RESET_ALL)
                print(Fore.YELLOW + 'Switching to using the default model since specified model is unavailable...' + Style.RESET_ALL)
                method = self.get_default_model

        model_pipe = Pipeline([ ('preprocessor',self.data_transformer.make_pipeline()),
                                ('regressor', method)])
        self.fitted_model = model_pipe.fit(self.data_transformer.X_data, self.data_transformer.y_data)

    def evaluate_model(self):
        '''
        Evaluate the model on the test dataset. 
        Uses MSE as the scoring metric.
        '''
        assert self.fitted_model is not None, Fore.RED + "Fit a model first to evaluate its performance on the test dataset" + Style.RESET_ALL

        y_pred = self.fitted_model.predict(self.data_transformer.X_test)
        self._mse_score = mean_squared_error(self.data_transformer.y_test, y_pred)
        self.get_mse

    def predict(self, X: pd.DataFrame, method: Optional[str] = None):
        '''
        Predict the target variable for a given set of features.

        Parameters:
            - X: pd.DataFrame, the features to predict on.
            - method: Optional[str], the method to use for prediction. If None, the best model is used.
        '''
        if self.fitted_model is None:
            if method is None and self.fitted_model is None:
                print(Fore.YELLOW + "No model specified. Using the best model to make predictions..." + Style.RESET_ALL)
                method = self.get_best_model if self.get_best_model != None else self.get_default_model
            else:
                try:
                    method = self._available_models[method]
                except Exception:
                    print(Fore.YELLOW + 'Specified method is not available. Make sure you run the grid search to rank available models by performance and select a valid model.' + Style.RESET_ALL)
                    print(Fore.YELLOW + 'Switching to using the default model since specified model is unavailable...' + Style.RESET_ALL)
                    method = self.get_default_model

        if not all(self.data_transformer.X_data.columns == X.columns):
            raise ValueError(Fore.RED + "The columns of X do not match the columns of the training data" + Style.RESET_ALL)
        
        if self.fitted_model is not None:
            method = self.fitted_model

        predictions_arr = np.array(method.predict(X))
        return pd.Series(
            predictions_arr,
            index=range(len(predictions_arr)),  # Create an index matching the length of predictions
            name=self.data_transformer.target_label  # Use name attribute instead of index for the column name
        )
    
    

    

