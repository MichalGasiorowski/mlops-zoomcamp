#!/usr/bin/env python
# coding: utf-8

# In[2]:

import os
import sys

import pickle
import pandas as pd
from datetime import datetime

from dateutil.relativedelta import relativedelta


import mlflow

from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

def read_data(filename):
    categorical = ['PUlocationID', 'DOlocationID']
    df = pd.read_parquet(filename)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    df = preprocess_data(df)

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PUlocationID', 'DOlocationID']

    df[categorical] = df[categorical].astype(str)

    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    dicts = df[categorical].to_dict(orient='records')

    return dicts


def load_dv_model():
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)
    return dv, lr

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    df_result.to_parquet(output_file, index=False)


def predict(dicts, dv, lr):
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    return y_pred

def preprocess_data(df):
    df_copied = df.copy()
    df_copied["pickup_yyyy_mm"] = pd.to_datetime(df_copied['pickup_datetime']).dt.strftime('%Y/%m')
    # apply has to be used
    df_copied['ride_id'] = df_copied.apply(lambda row: row["pickup_yyyy_mm"] + "_" + str(row.name) , axis=1)

    return df_copied


@task
def apply_model(input_file, output_file):
    logger = get_run_logger()

    logger.info(f'reading the data from {input_file}...')
    df = read_data(input_file)
    logger.info(f'df columns: {df.columns} ')
    dicts = prepare_dictionaries(df)

    logger.info(f'loading the model')
    dv, lr = load_dv_model()

    logger.info(f'applying the model...')
    y_pred = predict(dicts, dv, lr)

    logger.info(f'Mean predicted duration = {y_pred.mean():.2f}')

    logger.info(f'saving the result to {output_file}...')

    save_results(df, y_pred, output_file)
    return output_file


def get_paths(run_date):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month
    # https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_2021-03.parquet
    input_file = f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet'
    #output_file = f's3://nyc-duration-prediction-alexey/taxi_type={taxi_type}/year={year:04d}/month={month:02d}/{run_id}.parquet'
    output_file = f'./fhv_tripdata_year={year:04d}_month={month:02d}.parquet'

    return input_file, output_file


def save_result(df_result):
    df_result.to_parquet(
        'df_result.parquet',
        engine='pyarrow',
        compression=None,
        index=False
    )

@flow
def ride_duration_prediction(
        run_date: datetime = None):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time

    #input_file, output_file = get_paths(run_date, taxi_type, run_id)
    input_file, output_file = get_paths(run_date)

    apply_model(
        input_file=input_file,
        output_file=output_file
    )


def run():
    year = int(sys.argv[1]) # 2021
    month = int(sys.argv[2]) # 3

    ride_duration_prediction(
        run_date=datetime(year=year, month=month, day=1)
    )


if __name__ == '__main__':
    run()