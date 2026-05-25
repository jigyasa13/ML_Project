#Will have all the methods required to read the data
import os
import sys

from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

'''
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
'''


# Define a configuration class to hold file paths automatically
@dataclass #you dont need to give init method for this class
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")  #to save the training data, in artifacts folder as train.csv file
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

    '''
    path: A specific submodule inside os that contains tools specifically meant for managing and manipulating file path names.
    .join(): A highly useful function that glues folders and file names together into a single path.
    '''



# Main class responsible for loading, saving, and splitting data
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig() #all three path will be AUTOMATICALLY saved in ingestion_config variable

    def initiate_data_ingestion(self):
        #keep writing the logs so that in case of exception , you can identify the code whre exception occured
        logging.info("Entered the data ingestion method or component")#log 1
        try:
            df = pd.read_csv('notebook/data/stud.csv') 
            logging.info("Read the dataset as dataframe")#log 2
            
            #Train data
            # 2. Create the 'artifacts' folder directory if it doesn't exist yet
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            '''
            os.path.dirname(...): Looks at the full path (e.g., artifacts/train.csv) and strips away the filename, leaving just the directory name: 'artifacts'.
            os.makedirs(...): Creates that folder on your computer.
            '''

            # 3. Save the completely untouched raw data into the artifacts folder
            logging.info("Raw data saved in artifacts folder")#log 3
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            # 4. Split the data
            logging.info("Train test split initiated")#log 4
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)

            # 5. Save the newly created train and test files into the artifacts folder
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")


            # Return the file paths of the train and test sets for the next pipeline steps
            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            

        except Exception as e:
            raise CustomException(e,sys)


# The execution entry point of your ML pipeline script
if __name__=="__main__":
    # STEP 1: Run data ingestion (load data, split it, and save it to artifacts folder)
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()
'''
    # STEP 2: Pass those saved data paths into data transformation (cleaning, encoding, scaling)
    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    # STEP 3: Pass the clean transformed arrays into the model trainer to run ML algorithms
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))'''