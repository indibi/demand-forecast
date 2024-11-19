from pathlib import Path

import numpy as np
import pandas as pd
# import geopandas as gpd
from tqdm import tqdm
import torch
# from torchvision import datasets
# from torchvision.transforms import ToTensor

from src.utils.nyc_taxi_zones import TaxiZones

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / 'data'
YELLOW_DIR = DATA_DIR / 'yellow_taxi_trip_records'
GREEN_DIR = DATA_DIR / 'green_taxi_trip_records'
FHV_DIR = DATA_DIR / 'for_hire_vehicle_trip_records'
TAXI_ZONES_DIR = DATA_DIR / 'taxi_zones'
PRECOMPUTED_GRAPH_DIR = TAXI_ZONES_DIR / 'precomputed_graphs'

FILL_NA_VALUES = {
'rain_1h': 0,
'rain_3h': 0,
'snow_1h': 0,
'snow_3h': 0,
}

class NYCTripData:
    """Wrapper class to load and preprocess the NYC taxi trip data"""
    def __init__(self, start_month, end_month=None, freq='h', dataset='yellow',
                taxi_zones=TaxiZones, **kwargs):
        """Load and preprocess the NYC taxi trip data
        
        It's stored in a tensor of shape (n_locations, n_times, n_features) in the attribute
        `trip_data`. 
        
        Parameters
        ----------
        start_month : str
            The start month in 'YYYY-MM' format
        end_month : str, optional
            The end month in 'YYYY-MM' format. Default is the same as `start_month`
        freq : str, optional
            The period to aggregate the trip counts. Default is 'h' for hourly
        dataset : str, optional
            The dataset to load. Default is 'yellow'. Other options are 'green' and 'for_hire'
        taxi_zones : TaxiZones or list of int, or TaxiZones instance. optional
            The taxi zones to load. Default is `TaxiZones` which loads all the zones.
            If a list of integers is provided, it loads only the specified zones.
            If an instance of `TaxiZones` is provided, it uses the instance.
        **kwargs : dict
            weather_features : list of str, optional
                The weather features to load. Default is ['feels_like',
                'wind_speed', 'wind_gust', 'weather_main']
            column_to_onehot_encode : list of str, optional
                The column to do one-hot encoding on. Default is ['weather_main']
            save : bool, optional
                Whether to save the trip data. Default is True
            dataset_name : str, optional
                The name of the dataset to save. Default is None
        """
        self.start_month = start_month
        self.end_month = end_month
        self.dataset = dataset
        self.freq = freq
        self.weather_features = kwargs.get('weather_features', 
                                           ['feels_like', 'wind_speed',  'wind_gust', 
                                            'weather_main'])
        self.column_to_onehot_encode = kwargs.get('column_to_onehot_encode', 
                                                  ['weather_main'])
        self.taxi_zones = None
        self.trip_data = None
        self.location_ids = None
        self.datetime_index = None
        self.feature_names = None
        self.tensor_dimensions = ['Zones', 'datetime', 'features']
        self._load_dataset(start_month, end_month, freq, dataset, taxi_zones, **kwargs)

        
    def _load_dataset(self, start_month, end_month, freq, dataset, taxi_zones, **kwargs):
        self._load_taxi_zones(taxi_zones)
        save = kwargs.get('save', True)
        if taxi_zones is TaxiZones:
            dataset_name = dataset+'_' + start_month + '_' + end_month + '_' + freq
        else:
            dataset_name = kwargs.get('dataset_name', None)
            if dataset_name is None and save:
                raise ValueError("Please provide a dataset name for the trip data to save it")
            elif dataset_name is None:
                dataset_name = 'xasdasldksadl'
        
        if (DATA_DIR / (dataset_name + '.parquet')).exists():
            self.trip_data = pd.read_parquet(DATA_DIR / (dataset_name + '.parquet'))
            # Add weather data
            start_date = self.trip_data.index.get_level_values('datetime').min()
            end_date = self.trip_data.index.get_level_values('datetime').max()
            weather_data = self._load_weather_data(start_date, end_date)
        else:
            trip_counts = load_and_count_trip_records(start_month, end_month, dataset=dataset, freq=freq)
            zones_to_discard = {i for i in range(1,266)}.difference(set(self.taxi_zones.zones['LocationID']))
            trip_counts.drop(index=zones_to_discard, level='PULocationID', inplace=True)

            # Add time encodings
            time_encodings = pd.DataFrame({   
                'PULocationID': trip_counts.index.get_level_values('PULocationID').values,
                'hour': trip_counts.index.get_level_values('datetime').hour.values,
                'dayofweek': trip_counts.index.get_level_values('datetime').dayofweek.values,
                'dayofyear': trip_counts.index.get_level_values('datetime').to_period('h').dayofyear.values,
                'month': trip_counts.index.get_level_values('datetime').month.values,
                'quarter': trip_counts.index.get_level_values('datetime').quarter.values,
                # 'PU_count': trip_counts.values.ravel()
            }, index=trip_counts.index)

            # Add position encoding to the dataframe
            pos_x = self.taxi_zones.zones['x'].values
            pos_y = self.taxi_zones.zones['y'].values
            number_of_locations = trip_counts.index.get_level_values('PULocationID').nunique()
            number_of_hours = len(trip_counts.index.get_level_values('datetime'))//number_of_locations
            position_encodings = np.array([pos_x, pos_y]).T.reshape((number_of_locations, 1, 2)) * np.ones(( number_of_locations, number_of_hours, 2))
            position_encodings = position_encodings.reshape(-1, 2)
            position_encodings = pd.DataFrame({
                'x': position_encodings[:,0]/TaxiZones.x_scale, # Scale the x and y coordinates
                'y': position_encodings[:,1]/TaxiZones.y_scale,
            }, index=trip_counts.index)
            trip_data = pd.concat([time_encodings, position_encodings, trip_counts],axis=1)

            # Add weather data
            start_date = trip_data.index.get_level_values('datetime').min()
            end_date = trip_data.index.get_level_values('datetime').max()
            weather_data = self._load_weather_data(start_date, end_date)
            trip_data = trip_data.join(weather_data, on='datetime')

            columns = list(trip_data.columns)
            columns.remove('PU_count')
            columns.append('PU_count')
            self.trip_data = trip_data.reindex(columns=columns)
            if save:
                print(f"Saving the trip data to {DATA_DIR / (dataset_name + '.parquet')}")
                self.trip_data.to_parquet(DATA_DIR / (dataset_name + '.parquet'))

        self.location_ids = self.taxi_zones.zones['LocationID'].values
        self.datetime_index = weather_data.index
        self.feature_names = self.trip_data.columns
        if len(self.location_ids) * len(self.datetime_index) != self.trip_data.shape[0]:
            raise ValueError("The number of locations and times in the trip data is not consistent")
        self.trip_data = multiindex_spatio_temporal_df_to_tensor(self.trip_data)


    def _load_weather_data(self, start_date, end_date):
        weather_data = pd.read_csv(DATA_DIR / 'nyc_weather_bulk_2000_2024.csv')
        weather_data['datetime'] = pd.to_datetime(weather_data['dt_iso'], format='%Y-%m-%d %H:%M:%S +0000 UTC', utc=True).dt.tz_convert('US/Eastern').dt.tz_localize(None)
        weather_data.drop(columns=['dt', 'dt_iso'], inplace=True)
        weather_data.set_index('datetime', inplace=True)
        
        weather_data = weather_data[self.weather_features]
        date_range = pd.date_range(start_date, end_date, freq='h')

        weather_filter = (weather_data.index >= start_date) & \
                        (weather_data.index <= end_date)
        weather_data = weather_data[weather_filter]
        duplicate_dates = weather_data.index.duplicated(keep='first')
        weather_data = weather_data[~duplicate_dates]
        weather_data = weather_data.reindex(date_range, method='ffill', copy=True)
        

        weather_data.fillna(FILL_NA_VALUES, inplace=True)
        weather_data.fillna(method='ffill', inplace=True)
        weather_data = pd.get_dummies(weather_data,
                                    prefix=self.column_to_onehot_encode)
        return weather_data


    def _load_taxi_zones(self, taxi_zones):
        if isinstance(taxi_zones, TaxiZones):
            self.taxi_zones = taxi_zones
        elif taxi_zones is TaxiZones:
            self.taxi_zones = TaxiZones
            if self.taxi_zones.G_nyc is None:
                self.taxi_zones._create_and_connect_graph()
        elif isinstance(taxi_zones[0], int):
            self.taxi_zones = TaxiZones(loc_ids=taxi_zones)
        else:
            raise ValueError(("Invalid taxi_zones. Please provide",
                              " a list of location IDs or an instance of TaxiZones"))





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
    
    if columns is None:
        columns =  ['tpep_pickup_datetime','tpep_dropoff_datetime', 'PULocationID','DOLocationID']

    dates = pd.date_range(start_month, end_month, freq='MS')
    months = [date.strftime('%Y-%m') for date in dates]
    monthly_trip_counts = []
    for i, month in tqdm(enumerate(months), desc='Processing monthly records'):
        monthly_trip_record = pd.read_parquet(dataset_path / f'yellow_tripdata_{month}.parquet',
                                            columns=columns)
        early_rows, late_rows, pu_lateroe_than_do, pul_nan, do_nan = detect_faulty_records_in_monthly_record(monthly_trip_record, month)
        monthly_trip_record = monthly_trip_record[~(early_rows | late_rows | pu_lateroe_than_do | pul_nan | do_nan)]

        monthly_trip_counts.append(monthly_trip_record.resample(freq, on='tpep_pickup_datetime').agg({'PULocationID': 'value_counts'}))

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

def multiindex_spatio_temporal_df_to_tensor(df):
    """Convert a multi-index spatio-temporal DataFrame to a tensor
    
    Parameters
    ----------
    df : pandas.DataFrame
        A multi-index spatio-temporal DataFrame. The first level of the 
        index should be the location and the second level should be the time.
    
    Returns
    -------
    numpy.ndarray
        A tensor of the DataFrame
    """
    n_locations = df.index.get_level_values(0).nunique()
    n_times = len(df.index.get_level_values(1))//n_locations
    tensor = df.values.reshape(n_locations, n_times, -1)
    return tensor

def spatio_temporal_tensor_to_multiindex_df(tensor, index, columns):
    """Convert a spatio-temporal tensor to a multi-index DataFrame
    
    Parameters
    ----------
    tensor : numpy.ndarray
        A spatio-temporal tensor, where the first dimension is the location, 
        the second dimension is the time, and the third dimension is the features
    index : pandas.MultiIndex
        The index of the DataFrame with the levels 'LocationID' and 'datetime'
    columns : list
        The columns of the DataFrame
    
    Returns
    -------
    pandas.DataFrame
        A multi-index DataFrame
    """
    n_locations = len(index.get_level_values(0).unique())
    n_times = len(index)//n_locations
    n_features = len(columns)
    df = pd.DataFrame(tensor.reshape(n_locations*n_times, n_features), index=index, columns=columns)
    return df