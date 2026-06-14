#contained code to transform the data

import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

#From src folder
from src.exception import CustomException
from src.logger import logging
import os

#importing from utils file
from src.utils import save_object

 

#For any type of inputs i will be requiring for data transformation
# This class just sets up the path where our final transformer file (.pkl) will be saved
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

# This is the main class that handles all the data cleaning and conversions
class DataTransformation:
    def __init__(self):
        # Automatically pull the file path config we created right above
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is for data transformation.
        '''
        try:
            # Group 1: Define the columns that contain numbers
            numerical_columns = ["writing_score", "reading_score"]
            
            # Group 2: Define the columns that contain text categories
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Fill missing spots with the middle value (median), then scale them evenly
            #works like fit_tranform on train dataset AND TRANSFORM on test dataset
            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")), #Handles the missing values
                ("scaler",StandardScaler())
                ]
            )
            '''
            A pipeline automates everything so it runs smoothly from start to finish.
            pipeline connects your files together so you can run your entire project automatically
            '''

            # Blueprint for Text: Fill missing spots with the most common value, turn text into 0s and 1s (OneHot), then scale
            cat_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")), #MODE
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False)) # with_mean=False keeps it memory-efficient
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both PIPELINES together into one preprocessor tool
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        '''
        This function actually takes the train and test data, applies the formulas,
        and saves out the finished transformer model.
        '''
        try:
            # 1. Load the CSV files we created in data ingestion
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # 2. Get the empty formula blueprint we made in get_data_transformer_object
            preprocessing_obj=self.get_data_transformer_object()

            # 3. Define the question we want the ML model to solve (predict math_score)
            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # 4. Separate Training Data: X (Input Features) and Y (The Math Score Answer Key)
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            # 5. Separate Testing Data: X (Input Features) and Y (The Math Score Answer Key)
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            # 6. Apply transformations: Fitting the data
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            # 7. Glue the transformed INPUT feature columns and target values(MATH SCORES) back together side-by-side using np.c_
            #So that the model can see , everything at one place and is able to learn the relationship.
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            # 8. Save the completed transformer as a file (.pkl) so a website or app can use it later
            #Function from utils
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            ) 

            # Return the finished arrays out to the pipeline so the Model Trainer can use them
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)