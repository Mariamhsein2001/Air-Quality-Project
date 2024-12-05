# src/main.py
from air_pollution.config import load_config
from air_pollution.data_loader.factory import DataLoaderFactory
from air_pollution.data_pipeline.preprocessing import Preprocessor
import argparse
from air_pollution.model.factory import ModelFactory
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

parser = argparse.ArgumentParser(description="Run the ML data pipeline with specified configuration.")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to the configuration YAML file."
)

def main():
    # Load configuration
    args = parser.parse_args()
    config = load_config(args.config)
    print("Loaded Configuration:")
    print(config)

    # Load data
    data_loader = DataLoaderFactory.get_data_loader(config.data_loader.file_type)
    data = data_loader.load_data(config.data_loader.file_path)
    print("Loaded Data:")
    print(data)

    # Initialize the Preprocessor with the config
    target_column = "Air Quality"
    preprocessor = Preprocessor(config, target_column)

    # Preprocess the data and split into training and test sets
    X_train, X_test, y_train, y_test = preprocessor.preprocess(data, target_column)
    print("Transformed labels: ", y_train)
    print("Transformed X_train:")
    print(X_train)
    print("Transformed X_test:")
    print(X_test)

    # Train and evaluate the model
    model = ModelFactory.get_model(config.model.type)
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    print("Predictions:")
    print(predictions)

    # Generate and print the confusion matrix as a table
    cm = confusion_matrix(y_test, predictions)
    print("Confusion Matrix:")
    print(cm)

    # Generate and print the classification report
    report = classification_report(y_test, predictions)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    main()
