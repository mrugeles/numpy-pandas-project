import unittest
import pandas as pd
import numpy as np
from data_bot import ImputerStrategy
from data_bot import DataBot


class TestDataBot(unittest.TestCase):

    df = pd.read_csv('tests/titanic.csv')

    def test_impute(self):
        dataBot = DataBot(self.df, target_name='Survived')
        dataBot.impute(['Age'], ImputerStrategy.MEAN)
        expected_a = 21205.17

        self.assertTrue(np.allclose(expected_a, dataBot.features['Age'].sum(), 2))

    def test_one_hot_encode(self):
        dataBot = DataBot(self.df, target_name='Survived')

        dataBot.impute(['Embarked'], ImputerStrategy.MODE)
        dataBot.one_hot_encode('Embarked', self.df['Embarked'].values)
        encoded_cols = {'Embarked_S', 'Embarked_C', 'Embarked_Q'}
        self.assertTrue(encoded_cols.intersection(set(dataBot.features.columns)) == encoded_cols)

    def test_normalize(self):
        dataBot = DataBot(self.df, target_name='Survived')
        dataBot.impute(['Age'], ImputerStrategy.MEAN)
        dataBot.normalize(['Age'])
        self.assertTrue(np.allclose(272.84, dataBot.features['Age'].sum(), 2))

    def test_remove_null_columns(self):
        dataBot = DataBot(self.df, target_name='Survived')
        dataBot.remove_null_columns()

        self.assertTrue('B' not in dataBot.features.columns)
        self.assertTrue('C' not in dataBot.features.columns)

    def test_remove_high_cardinality_columns(self):
        dataBot = DataBot(self.df, target_name='Survived', cardinal_threshold=0.5)

        dataBot.remove_high_cardinality_columns()
        columns = ['PassengerId', 'Name', 'Cabin']
        self.assertTrue(columns not in list(dataBot.features.columns))

    def test_pre_process(self):
        import math

        dataBot = DataBot(
            self.df,
            target_name='Survived',
            project_path='./tests')

        expected = {
            'Pclass': 0.46,
            'Age': 0.30,
            'SibSp': 0.05,
            'Parch': 0.05,
            'Fare': 0.05,
            'Sex_male': 0.64,
            'Sex_female': 0.35,
            'Embarked_S': 0.72,
            'Embarked_C': 0.18,
            'Embarked_Q': 0.08
        }

        dataBot.pre_process()
        features_average = dataBot.features.describe().loc['mean'].to_dict()
        for key in features_average.keys():
            self.assertAlmostEqual(expected[key], features_average[key], delta=0.1)


