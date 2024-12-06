# src/ml_data_pipeline/config.py
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, field_validator


class DataLoaderConfig(BaseModel):
    """Configuration for the data loader.

    Attributes:
        file_path (str): The path to the data file.
        file_type (str): The type of the data file (csv or json).
    """

    file_path: str
    file_type: str

    @field_validator("file_type")
    def validate_file_type(cls, value: str) -> str:
        """Validates the file type.

        Args:
            value (str): The file type to validate.

        Returns:
            str: The validated file type.

        Raises:
            ValueError: If the file type is not 'csv' or 'json'.
        """
        if value not in {"csv", "json"}:
            raise ValueError("file_type must be 'csv' or 'json'")
        return value


class TransformationConfig(BaseModel):
    """Configuration for the data transformation.

    Attributes:
        normalize (bool): Whether to normalize the data.
        scaling_method (str): The method to use for scaling (standard or minmax).
    """

    normalize: bool
    scaling_method: str

    @field_validator("scaling_method")
    def validate_scaling_method(cls, value: str) -> str:
        """Validates the scaling method.

        Args:
            value (str): The scaling method to validate.

        Returns:
            str: The validated scaling method.

        Raises:
            ValueError: If the scaling method is not 'standard' or 'minmax'.
        """
        if value not in {"standard", "minmax"}:
            raise ValueError("scaling_method must be 'standard' or 'minmax'")
        return value


class ModelConfig(BaseModel):
    """Configuration for the model.

    Attributes:
        type (str): The type of the model (linear or tree).
    """

    type: str

    @field_validator("type")
    def validate_model_type(cls, value: str) -> str:
        """Validates the model type.

        Args:
            value (str): The model type to validate.

        Returns:
            str: The validated model type.

        Raises:
            ValueError: If the model type is not 'logistic' or 'decisiontree'.
        """
        if value not in {"decisiontree", "logistic"}:
            raise ValueError("model type must be 'logistic' or 'decisiontree'")
        return value


class SplittingConfig(BaseModel):
    """Configuration for data splitting.

    Attributes:
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): The seed used for random number generation.
    """

    test_size: float = Field(
        ..., ge=0.0, le=1.0, description="Proportion of test data."
    )

    @field_validator("test_size")
    def validate_test_size(cls, value: float) -> float:
        """Validates the test size.

        Args:
            value (float): The test size to validate.

        Returns:
            float: The validated test size.

        Raises:
            ValueError: If the test size is not between 0 and 1.
        """
        if not (0.0 < value < 1.0):
            raise ValueError("test_size must be between 0 and 1.")
        return value


class Config(BaseModel):
    """Overall configuration for the pipeline.

    Attributes:
        data_loader (DataLoaderConfig): Configuration for the data loader.
        transformation (TransformationConfig): Configuration for the data transformation.
        model (ModelConfig): Configuration for the model.
        splitting (SplittingConfig): Configuration for data splitting.
    """

    data_loader: DataLoaderConfig
    transformation: TransformationConfig
    model: ModelConfig
    splitting: SplittingConfig


def load_config(config_path: str) -> Config:
    """Loads the configuration from a file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        Config: The loaded configuration.
    """
    raw_config = OmegaConf.load(config_path)
    config_dict = OmegaConf.to_container(raw_config, resolve=True)
    return Config.model_validate(config_dict)
