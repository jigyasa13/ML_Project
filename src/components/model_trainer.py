#code to train the model

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models



@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        # Automatically load the file path setup where our trained model will be saved
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data") #SPLITTING TRAIN AND TEST ARRAY WE GOT FROM DATA TRANSFORMATION FILE
            
            # 1. Un-glue the target column (math_score) from the input features using array slicing
            # [:, :-1] takes all columns EXCEPT the last one (these are our X features)
            # [:, -1] takes ONLY the very last column (this is our Y target answer key)
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1], #X_train
                train_array[:,-1], #Y_train
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            # 2. Define a list of different AI algorithms we want to test out
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            '''
            # 3. Hyperparameter tuning each model
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256] # Number of trees in the forest
                },
                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{}, # Standard linear regression doesn't need special tuning options
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }'''

            # 4. Run our custom evaluation function to train every model and get a report card of their scores
            #this function is taken from utils
            #isime fitting, prediction and scoring ho jayega
            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models)
            
            # 5. Look through the report card and find the highest score
            best_model_score = max(sorted(model_report.values()))

            # 6. Find the name of the model that achieved that highest score
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            #7.Setting a threshold value
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # 8. Save the winning model out to a file (.pkl) so we can run predictions on our website later
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # 9. Use our winning model to predict the math scores for our hidden test set (the final exam)
            predicted=best_model.predict(X_test)

            # 10. Calculate the final R-squared accuracy score (how close our predictions were to real scores)
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e,sys)