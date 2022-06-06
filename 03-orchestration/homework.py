import pandas as pd

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from datetime import date, timedelta, datetime
import dateutil.relativedelta

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
from prefect import logging as prefect_logging

import pickle


@task
def read_data(path):
    df = pd.read_parquet(path)
    return df


@task
def prepare_features(df, categorical, train=True):
    logger = prefect_logging.get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df


@task
def train_model(df, categorical):
    logger = prefect_logging.get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv


@task
def run_model(df, categorical, dv, lr):
    logger = prefect_logging.get_run_logger()

    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return


'''
a. `date` should default to None
b. If `date` is None, use the current day. Use the data from 2 months back as the training data and the data from the previous month as validation data.
c. If a `date` value is supplied, get 2 months before the `date` as the training data, and the previous month as validation data.
d. As a concrete example, if the date passed is "2021-03-15", the training data should be "fhv_tripdata_2021-01.parquet" and the validation file will be "fhv_trip_data_2021-02.parquet"
'''
@task
def get_paths(date_string=None, months_back_for_training_range=(2, 3), months_back_for_valid_range=(1, 2)):
    if date_string is None:
        date_string = date.today()

    datem = date.fromisoformat(date_string)

    training_months_dt = [datem - dateutil.relativedelta.relativedelta(months=i) for i in months_back_for_training_range]
    validation_months_dt = [datem - dateutil.relativedelta.relativedelta(months=i) for i in months_back_for_valid_range]

    # pad with extra '0' for months
    training_months_paths = [f'./data/fhv_tripdata_{dt.year}-{dt.month:02}.parquet' for dt in training_months_dt]
    validation_months_paths = [f'./data/fhv_tripdata_{dt.year}-{dt.month:02}.parquet' for dt in validation_months_dt]

    # return single files for training, since main() expects it in this format
    # more files for training, validation are possible
    return training_months_paths[0], validation_months_paths[0]


# assert(get_paths("2021-03-15") == ('./data/fhv_tripdata_2021-01.parquet', './data/fhv_tripdata_2021-02.parquet'))


@flow(task_runner=SequentialTaskRunner())
def main(date):
    logger = prefect_logging.get_run_logger()

    train_path, val_path = get_paths(date).result()
    logger.info(f'Path for training: {train_path}, Path for validation: {val_path}')
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()

    # save the model and dv
    # In this example we use a DictVectorizer. That is needed to run future data through our model.
    with open(f'models/model-{date}.pkl', "wb") as f_out:
        pickle.dump(lr, f_out)
    with open(f'models/dv-{date}.pkl', "wb") as f_out:
        pickle.dump(dv, f_out)
    run_model(df_val_processed, categorical, dv, lr)


main(date="2021-08-15")
