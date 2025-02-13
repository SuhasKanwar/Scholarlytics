import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging as logger
from src.utils import save_object, evaluate_models

from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

@dataclass
class ModelTrainerConfig:
    model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    
    def initiate_model_training(self, train_arr, test_arr):
        try:
            logger.info('Splitting training and testing input data')

            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'Linear Regression': LinearRegression(),
                'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'XGBoost': XGBRegressor()
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            logger.info(f'Model Evaluation Report: {model_report}')

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No good model found", sys)
            
            logger.info(f'Best Model: {best_model_name}')

            save_object(self.model_trainer_config.model_file_path, best_model)

            prediction = best_model.predict(X_test)
            r2_score_value = r2_score(y_test, prediction)

            return r2_score_value

        except Exception as e:
            raise CustomException(e, sys)