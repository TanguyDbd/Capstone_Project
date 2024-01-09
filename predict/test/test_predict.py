import unittest
from unittest.mock import MagicMock
import tempfile

import pandas as pd

from train.train import run
from preprocessing.preprocessing import utils
from predict.predict import run as predict_run


def load_dataset_mock():
    titles = [
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
        "Is it possible to execute the procedure of a function in the scope of the caller?",
        "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
    ]
    tags = ["php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails", "php", "ruby-on-rails",
            "php", "ruby-on-rails"]

    return pd.DataFrame({
        'title': titles,
        'tag_name': tags
    })


class TestPrediction(unittest.TestCase):

    # use the function defined above as a mock for utils.LocalTextCategorizationDataset.load_dataset
    utils.LocalTextCategorizationDataset.load_dataset = MagicMock(return_value=load_dataset_mock())

    def test_prediction(self):

        # Create a temporary directory to store artefacts
        with tempfile.TemporaryDirectory() as model_dir:

            params = {
                'batch_size': 2,
                'epochs': 1,
                'dense_dim': 64,
                'min_samples_per_label': 2,
                'verbose': 1
            }
            accuracy, arte_path = run.train(dataset_path="fake.csv", train_conf=params, model_path=model_dir,
                                            add_timestamp=True)

            # artefacts to load the model for prediction
            model = predict_run.TextPredictionModel.from_artefacts(arte_path)

            # Mocking a list of texts for prediction
            text_list = [
                "Is it possible to execute the procedure of a function in the scope of the caller?",
                "ruby on rails: how to change BG color of options in select list, ruby-on-rails",
            ]

            # predictions using the model
            predictions = model.predict(text_list)
            # print(predictions)

            # assertion
            self.assertEqual(len(predictions), len(text_list))
