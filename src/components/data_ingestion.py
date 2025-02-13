import os
import sys
from src.exception import CustomException
from src.logger import logging as logger
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        '''
        This function reads the data from the csv file and splits it into train and test data.
        '''
        logger.info('Initiating Data Ingestion')
        try:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
            data_file = os.path.join(base_dir, "notebooks", "data", "student.csv")
            df = pd.read_csv(data_file)
            logger.info('Data Ingested Successfully as Dataframe')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logger.info("Initiating Train Test Split")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logger.info('Data Ingestion Completed Successfully')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    modelTrainer = ModelTrainer()
    print(modelTrainer.initiate_model_training(train_arr, test_arr))