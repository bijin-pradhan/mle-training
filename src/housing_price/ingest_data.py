import os
import tarfile
from argparse import ArgumentParser, Namespace
from logging import Logger

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit

from housing_price.logger import configure_logger


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "-r",
        "--raw",
        type=str,
        default="data/raw/",
        help="Path to raw dataset.",
    )
    parser.add_argument(
        "-p",
        "--processed",
        type=str,
        default="data/processed/",
        help="Path to processed dataset.",
    )
    parser.add_argument("--log-level", type=str, default="DEBUG")
    parser.add_argument("--no-console-log", action="store_true")
    parser.add_argument("--log-path", type=str, default="")
    return parser.parse_args()


def fetch_housing_data(housing_url: str, housing_path: str):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    os.remove(tgz_path)


def stratified_shuffle_split(base_df: pd.DataFrame):
    base_df["income_cat"] = pd.cut(
        base_df["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(base_df, base_df["income_cat"]):
        strat_train_set = base_df.loc[train_index]
        strat_test_set = base_df.loc[test_index]

    for set_ in (strat_test_set, strat_train_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set


def pre_process_data(df: pd.DataFrame, imputer: SimpleImputer = None):
    df = pd.get_dummies(df, columns=["ocean_proximity"])

    if imputer is None:
        imputer = SimpleImputer(strategy="median")
        imputer.fit(df)

    data = imputer.transform(df)
    df = pd.DataFrame(data, columns=df.columns, index=df.index)

    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["population_per_household"] = df["population"] / df["households"]

    return df, imputer


def run(args: Namespace, logger: Logger):
    DOWNLOAD_ROOT = (
        "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    )
    HOUSING_PATH = args.raw
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    logger.debug("Fetched housing data.")

    housing_df = pd.read_csv(os.path.join(args.raw, "housing.csv"))
    train_set, test_set = stratified_shuffle_split(housing_df)

    logger.debug("Preprocessing...")
    train_set, imputer = pre_process_data(train_set)
    test_set, _ = pre_process_data(test_set, imputer)
    logger.debug("Preprocessing finished.")

    train_path = os.path.join(args.processed, "housing_train.csv")
    train_set.to_csv(train_path)
    logger.debug(f"Preprocessed train datasets stored at {train_path}.")

    test_path = os.path.join(args.processed, "housing_test.csv")
    test_set.to_csv(test_path)
    logger.debug(f"Preprocessed test datasets stored at {test_path}.")


if __name__ == "__main__":
    args = parse_args()
    logger = configure_logger(
        log_level=args.log_level,
        log_file=args.log_path,
        console=not args.no_console_log,
    )

    run(args, logger)