import imp
import os, sys
from pathlib import Path

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

BASE_DIR = Path.cwd().parent
DATA_DIR = BASE_DIR / 'data'
YELLOW_DIR = DATA_DIR / 'yellow_taxi_trip_records'
GREEN_DIR = DATA_DIR / 'green_taxi_trip_records'
FHV_DIR = DATA_DIR / 'for_hire_vehicle_trip_records'

# Loader and preprocessor for NYC rideshare trip records
def load_monthly_trip_records(start_month, end_month=None, dataset='yellow', **kwargs):
    """Load NYC rideshare trip records for specified month(s)

    Parameters
    ----------
    start_month : str
        The start month in 'YYYY-MM' format
    end_month : str, optional
        The end month in 'YYYY-MM' format. If not specified, only the start month will be loaded
    dataset: str, optional
        The dataset to load. Default is 'yellow'. Other options are 'green' and 'for_hire'
    **kwargs : dict
        columns : list of str, optional. 
        The columns to load from the dataset. Default is ['tpep_pickup_datetime','tpep_dropoff_datetime',
            'PULocationID','DOLocationID'].
    
    Returns
    -------
    list of pandas.DataFrame
        A list of DataFrames containing the trip data for each month
    """
    if end_month is None:
        end_month = start_month
    
    if dataset == 'yellow':
        dataset_path = YELLOW_DIR
    elif dataset == 'green':
        dataset_path = GREEN_DIR
    elif dataset == 'for_hire':
        dataset_path = FHV_DIR
    else:
        raise ValueError("Invalid dataset. Please specify 'yellow', 'green', or 'for_hire'")
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Directory for '{dataset}' is not found in {dataset_path}")
    
    columns = kwargs.get('columns', ['tpep_pickup_datetime','tpep_dropoff_datetime',
                                    'PULocationID','DOLocationID'])

    dates = pd.date_range(start_month, end_month, freq='MS')
    months = [date.strftime('%Y-%m') for date in dates]
    monthly_trip_records = [pd.read_parquet(dataset_path / f'yellow_tripdata_{month}.parquet',
                                            columns=columns)
                            for month in months]
    return monthly_trip_records


def detect_faulty_records_in_monthly_record(month_record, month, only_stats=False):
    """Detect records with faulty time-stamps in the monthly trip record
    
    Detect records with the following issues:
    1. Drop-off time earlier than or equal to the pick-up time
    2. Pick-up time earlier than the begining of the month
    3. Pick-up time later than end of the month

    Parameters:
    month_record : pandas.DataFrame
        The monthly trip record
    month : str
        The month in 'YYYY-MM' format
    """
    month_begin_ts = pd.Timestamp(month)
    next_month_ts = month_begin_ts + pd.offsets.MonthBegin()

    early_rows = month_record['tpep_pickup_datetime'] < month_begin_ts
    late_rows = month_record['tpep_pickup_datetime'] >= next_month_ts
    pu_lateroe_than_do = month_record['tpep_pickup_datetime'] >= month_record['tpep_dropoff_datetime']
    pul_nan = month_record['PULocationID'].isnull()
    do_nan = month_record['DOLocationID'].isnull()
    if only_stats:
        return sum(early_rows), sum(late_rows), sum(pu_lateroe_than_do), sum(pul_nan), sum(do_nan)
    else:
        return early_rows, late_rows, pu_lateroe_than_do, pul_nan, do_nan


def load_and_count_trip_records(start_month, end_month=None, freq='h', dataset='yellow', columns=None):
    """Parse the monthly trip record and count the number of records with faulty time-stamps
    
    Parameters:
    monthly_record : pandas.DataFrame
        The monthly trip record
    month : str
        The month in 'YYYY-MM' format
    freq : str, optional
        The period to aggregate the trip counts. Default is 'h' for hourly
    dataset : str, optional
        The dataset to load. Default is 'yellow'. Other options are 'green' and 'for_hire'
    columns : list of str, optional
        The columns to load from the trip records. Default is ['tpep_pickup_datetime',
        'tpep_dropoff_datetime', 'PULocationID', 'DOLocationID']

    Returns:
    pandas.DataFrame
        A DataFrame containing the parsed trip records
    """
    dates = pd.date_range(start_month, end_month, freq='MS')
    months = [date.strftime('%Y-%m') for date in dates]
    monthly_trip_records = load_monthly_trip_data(start_month, end_month, dataset, drop_other_columns)
    
    monthly_trip_counts = []
    for i, month in tqdm(enumerate(months)):
        early_rows, late_rows, pu_lateroe_than_do, pul_nan, do_nan = detect_faulty_records_in_monthly_record(monthly_trip_records[i], month)
        monthly_trip_records[i] = monthly_trip_records[i][~(early_rows | late_rows | pu_lateroe_than_do | pul_nan | do_nan)]

        monthly_trip_counts.append(monthly_trip_records[i].resample(freq, on='tpep_pickup_datetime').agg({'PULocationID': 'value_counts'}))
        
        monthly_trip_counts[i].rename(columns={'PULocationID': 'PU_count'}, inplace=True)
        monthly_trip_counts[i] = monthly_trip_counts[i].reorder_levels([1,0])

        mstart = pd.Timestamp(month)
        mend = mstart + pd.offsets.MonthBegin()
        datetime_index = pd.date_range(mstart, mend, freq=freq)[:-1]
        zones_index = range(1,266, 1)
        multi_index = pd.MultiIndex.from_product([zones_index, datetime_index], names=['PULocationID', 'datetime'])
        monthly_trip_counts[i] = monthly_trip_counts[i].reindex(multi_index, fill_value=0).sort_index()

    trip_counts = pd.concat(monthly_trip_counts, axis=0).sort_index()
    return trip_counts


# class NYC_Rideshare_Loader: