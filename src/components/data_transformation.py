import os, sys
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.utils.common import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts/data_transformation", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self. data_transformation_config = DataTransformationConfig()


    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Has Begun")

            numerical_features = ['age', 'workclass', 'educationalNum', 'maritalStatus', 'occupation',
       'relationship', 'race', 'gender', 'capitalGain', 'capitalLoss',
       'hoursPerWeek']
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features)
            ]) 

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)

            IQR = Q3 - Q1

            upperLimit = Q3 + 1.5 * IQR
            lowerLimit = Q1 - 1.5 * IQR

            df.loc[(df[col] > upperLimit), col] = upperLimit
            df.loc[(df[col] < lowerLimit), col] = lowerLimit

            return df

        except Exception as e:
            logging.info("Outliers Has Been Handled")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            numerical_features = ['age', 'workclass', 'educationalNum', 'maritalStatus', 'occupation',
       'relationship', 'race', 'gender', 'capitalGain', 'capitalLoss',
       'hoursPerWeek']
            
            for col in numerical_features:
                self.remove_outliers_IQR(col=col, df=train_data)

            logging.info("Outliers Has Been Capped On Train Data")

            for col in numerical_features:
                self.remove_outliers_IQR(col=col, df=test_data)
            
            logging.info("Outliers Has Been Capped On Test Data")

            preprocessor_obj = self.get_data_transformation_obj()

            target_columns = "income"
            drop_columns = [target_columns]

            logging.info("Splitting Train Data Into Dependent And Independent Features")
            input_feature_train_data = train_data.drop(drop_columns, axis=1)
            target_feature_train_data = train_data[target_columns]

            logging.info("Splitting Test Data Into Dependent And Independent Features")
            input_feature_test_data = test_data.drop(drop_columns, axis=1)
            target_feature_test_data = test_data[target_columns]



            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocessor_obj.transform(input_feature_test_data)

            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)
            
            return (train_array,
                    test_array,
                    self.data_transformation_config.preprocessor_obj_file_path)


        except Exception as e:
            raise CustomException(e, sys)


