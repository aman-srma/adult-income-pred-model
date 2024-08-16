import os, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.logger import logging
from  src.exception import CustomException
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation


@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join("artifacts/data_ingestion", "train.csv")
    test_data_path = os.path.join("artifacts/data_ingestion", "test.csv")
    original_data_path = os.path.join("artifacts/data_ingestion", "original.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Has Started")

        try:
            data = pd.read_csv(os.path.join("notebook/data", "income_cleandata.csv"))

            os.makedirs(os.path.dirname(self.ingestion_config.original_data_path), exist_ok=True)
            data.to_csv(self.ingestion_config.original_data_path, index=False)

            train_set, test_set = train_test_split(data, test_size=0.25, random_state=7)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            train_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion Has Been Completed")


            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logging.info("Error Has Occurred In Data Ingestion Stage")
            raise CustomException(e, sys)


if __name__ =="__main__":
    obj = DataIngestion()
    train_data_path, test_data_path = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    