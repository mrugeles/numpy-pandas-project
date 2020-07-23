import json
import pandas as pd
import numpy as np

from enum import Enum
from typing import List
from dataset_attributes import DataSetAtrributes

import warnings

warnings.filterwarnings('ignore')


class ImputerStrategy(Enum):
    MEAN = 'mean'
    MEDIAN = 'median'
    MODE = 'mode'
    CONSTANT = 'constant'
    REGRESSOR_MODEL = 'regressor_model'
    CLASSIFICATION_MODEL = 'clasification_model'


class DataBot:
    numeric_types: List[str] = ['int64', 'float64', 'datetime64']
    string_types: List[str] = ['object', 'category']

    def __init__(self,
                 dataset=None,
                 target_name=None,
                 null_threshold=0.3,
                 cardinal_threshold=0.3,
                 project_path=None):
        self.dataset = dataset
        self.target = target_name
        self.null_threshold = null_threshold
        self.cardinal_threshold = cardinal_threshold
        self.project_path = project_path
        self.categorical_columns = []
        self.numeric_columns = []

        self.datasetAttributes = DataSetAtrributes(self.project_path)

        if target_name is not None:
            self.target_name = target_name
            self.target = self.dataset[self.target_name]
            self.features = self.dataset.drop([self.target_name], axis=1)
        else:
            self.features = dataset

    # Lambda for a series object that fill nulls with the mean.
    fill_mean = None
    # Lambda for a series object that fill nulls with the mode.
    fill_mode = None

    def scale_range(self, x):
        return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

    def scale_log(self, x):
        return np.log(x + 1)

    impute_strategies = {
        ImputerStrategy.MEAN: fill_mean,
        ImputerStrategy.MODE: fill_mode
    }

    def impute(self, columns, impute_strategy):
        """Impute selected columns (pd.Series) from self.features with the given strategy.

        Parameters
        ----------
        :param columns: list of columns names to impute.
        :param impute_strategy: Selected ImputerStrategy
        """
        pass

    def one_hot_encode(self, col_name, categorical_values):
        """ Apply one hot encoding to the given column.

        :param col_name: Name of the column to one hot encode.
        :param categorical_values: Unique values from self.features[col_name]
        :return:
        """
        pass

    def normalize(self, columns):
        """Apply self.scale_range and self.scale_log to the given columns
        :param columns: list of columns names to normalize
        """
        self.features[columns] = None
        self.features[columns] = None

    def remove_null_columns(self):
        """Remove columns with a percentage of null values greater than the given threshold (self.null_threshold).
        
        """
        pass

    def remove_high_cardinality_columns(self):
        """Remove columns with a cardinality percentage greater than the given threshold (self.cardinal_threshold).

        """
        pass

    def pre_process(self):
        """Preprocess dataset features before being send to ML algorithm for training.
        """
        # Implement this method with the given indications in the given order

        # Remove columns with null values above the threshold

        # Remove columns with cardinality above the threshold

        # Create a python list with the names of columns with numeric values.
        # Numeric columns have one of the types stored in the list self.numeric_types
        self.numeric_columns = None

        # Create a python list with the names of columns with string values.
        # Categorical columns have one of the types stored in the list self.string_types
        self.categorical_columns = None

        # Create a python list with the names of numeric columns with at least one null value.
        numeric_nulls = None

        # Create a python list with the names of categorical columns with at least one null value.
        categorical_nulls = None

        # Impute numerical columns with at least one null value.

        # Impute categorical columns with at least one null value.

        # These two lines gather information from the dataset for further use.
        self.datasetAttributes.set_column_values(self.categorical_columns, self.features)
        self.datasetAttributes.set_number_values(self.numeric_columns, self.features)

        # Apply one hot encoding to all categorical columns.

        # Normalize all numeric columns

        # This line store relevant information from the processed dataset for further use.
        self.datasetAttributes.save()

    def pre_process_prediction(self, parameters):
        """Preprocess records from API calls before running predictions

        :param parameters: information from the processed dataset in the training stage.

        """
        self.features.drop(parameters['removed_columns'], axis=1, inplace=True)

        for column in parameters['categorical_columns'].keys():
            categorical_values = parameters['categorical_columns'][column]["values"]
            self.one_hot_encode(column, categorical_values)

        for column in parameters['numeric_columns'].keys():
            n_min = parameters['numeric_columns'][column]["min"]
            n_max = parameters['numeric_columns'][column]["max"]
            self.features[column] = self.features[column].apply(lambda x: (x - n_min) / (n_max - n_min))
            self.features[column] = self.features[column].apply(lambda x: np.log(x + 1))

    def get_dataset(self):
        """Returns a dataset with features and labels.

        :return: Dataset with features and labels.
        """
        self.dataset = self.features
        self.dataset[self.target_name] = self.target
        return self.dataset
