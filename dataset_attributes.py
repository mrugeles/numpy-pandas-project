import json


class DataSetAtrributes:

    def __init__(self, project_path):
        self.parameters = {
            'removed_columns': [],
            'numeric_columns': {},
            'categorical_columns': {}
        }
        self.project_path = project_path

    def set_column_values(self, categorical_columns, features):
        for column in categorical_columns:
            self.parameters['categorical_columns'][column] = {
                'values': list(features[column].unique())
            }

    def set_number_values(self, numerical_columns, features):
        for column in numerical_columns:
            self.parameters['numeric_columns'][column] = {
                'min': features[column].min(),
                'max': features[column].max()
            }


    def load(self):
        with open(f'{self.project_path}/dataset_attributes.json') as json_file:
            self.parameters = json.load(json_file)

    def save(self):
        with open(f'{self.project_path}/dataset_attributes.json', 'w') as file:
            json.dump(self.parameters, file)
