"""
This module contains unit tests for src/housing_price/train.py.
"""
import os
import pickle

import housing_price.train as train
from housing_price.logger import configure_logger
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

args = train.parse_args()
logger = configure_logger()


def test_parse_args():
    """
    Tests parse_args function.
    """
    assert args.dataset == "data/processed/housing_train.csv"
    assert args.models == "artifacts/"
    assert args.log_level == "DEBUG"
    assert not args.no_console_log
    assert args.log_path == ""


def test_load_data():
    """
    Tests load_data function.
    """
    X, y = train.load_data(args.dataset)
    assert len(X) == len(y)
    assert "median_house_value" not in X.columns
    assert not X.isna().sum().sum()
    assert len(y.shape) == 1


def test_save():
    """
    Tests save_model function.
    """
    train.run(args, logger)
    assert os.path.isfile(f"{args.models}/LinearRegression.pkl")
    assert os.path.isfile(f"{args.models}/DecisionTreeRegressor.pkl")
    assert os.path.isfile(f"{args.models}/RandomForestRegressor.pkl")


def test_run():
    """
    Tests run function.
    """
    X, y = train.load_data(args.dataset)

    lr = LinearRegression()
    lr.fit(X, y)
    loaded_lr = pickle.load(open(f"{args.models}/LinearRegression.pkl", "rb"))
    assert lr.score(X, y) == loaded_lr.score(X, y)

    dtree = DecisionTreeRegressor(random_state=42)
    dtree.fit(X, y)
    loaded_dtree = pickle.load(
        open(f"{args.models}/DecisionTreeRegressor.pkl", "rb")
    )
    assert dtree.score(X, y) == loaded_dtree.score(X, y)
