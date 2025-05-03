from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_ingestion import DataIngestionConfig
from src.mlproject.components.data_transformation import DataTransformationConfig, DataTransformation
from src.mlproject.components.model_tranier import ModelTrainerConfig, ModelTrainer
import sys



if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        #data_ingestion_config=DataIngestionComfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path = data_ingestion.initiate_data_ingestion()

        #data_tansformation_config = DataTransformationConfig()
        data_tansformation = DataTransformation()
        train_arr,test_arr,_=data_tansformation.initiate_data_transformation(train_data_path,test_data_path)

        # Modeling Training
        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e,sys)

 