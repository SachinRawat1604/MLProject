## End to End Data Science Project
# ML Project

A simple machine learning project focused on predicting student performance using various features.

## Overview

This project aims to predict student performance in math exams based on various demographic and educational factors. It uses a dataset containing information such as gender, race/ethnicity, parental education level, lunch type, and test preparation course completion to predict math scores.

The implementation provides a complete end-to-end ML solution that includes:
- Data collection and validation
- Exploratory Data Analysis (EDA)
- Feature engineering and preprocessing
- Model training, evaluation, and selection
- A web interface for making predictions

The project is structured as a modular Python package that demonstrates best practices in ML project organization, making it easy to understand, modify, and extend.

## Features

- Complete data processing pipeline
- Multiple ML models (RandomForest, XGBoost, CatBoost)
- Model evaluation and selection based on r2 score
- Custom exception handling and logging

## Project Structure

```
MLProject/
│
├── artifacts/         # Storage for model outputs, processed data
├── notebooks/         # Jupyter notebooks for EDA
├── src/               # Main source code
│   ├── components/    # Key pipeline components
│   ├── exception.py   # Custom exception handling
│   ├── logger.py      # Logging functionality
│   ├── utils.py       # Utility functions
│   └── pipeline/      # Training and prediction pipelines
└── templates/         # Web application templates
```

## Installation

```bash
# Clone the repository
git clone https://github.com/SachinRawat1604/MLProject.git
cd MLProject

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Usage

### Training Pipeline

To train the model:

```python
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

# Data ingestion
data_ingestion = DataIngestion()
train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

# Data transformation
data_transformation = DataTransformation()
train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)

# Model training
model_trainer = ModelTrainer()
model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
print(f"Model r2 score: {model_score}")
```

### Prediction Pipeline

To make predictions:

```python
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Sample data
custom_data = CustomData(
    gender='female',
    race_ethnicity='group B',
    parental_level_of_education='bachelor\'s degree',
    lunch='standard',
    test_preparation_course='completed',
    reading_score=72,
    writing_score=74
)

# Get features as DataFrame
input_df = custom_data.get_data_as_data_frame()

# Make prediction
predict_pipeline = PredictPipeline()
prediction = predict_pipeline.predict(input_df)
print(f"Prediction: {prediction}")
```

## Web Application

A simple Flask web application is included to demonstrate the model functionality. To run it:

```bash
python app.py
```

Then access the application in your browser at http://localhost:5000

## Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- Flask
- xgboost
- catboost

## Contributors

- Sachin Rawat

## License

This project is licensed under the MIT License.
