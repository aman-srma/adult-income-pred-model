import os, sys
import pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=true)

        with open (file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    


def evaluate_model():
    pass