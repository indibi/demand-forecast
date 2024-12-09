import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pytz 
import openmeteo_requests
import requests_cache
from retry_requests import retry

# Preprocess the data
def preprocess_data(newdata, target_column='PU_count'):
    newdata = create_dataset(newdata)
    print(newdata.columns)
    newdata = newdata[['PU_count']] # Assuming 'PU_count' is the target column
    newdata = newdata.fillna(newdata.mean())

    # Normalize the data using MinMaxScaler
    scalar = MinMaxScaler()
    data = scalar.fit_transform(newdata)
    print(newdata)

    return data, scalar

# Prepare data for LSTM
def create_dataset(dataset, look_back=4):
    
    

    print(type(datetime))
    
    df = dataset.values.tolist()
    my_list = df[0]
    my_list.append(0)
    timestamp = pd.to_datetime(datetime.now()).round(freq='H')
    my_list= get_meteo_data(timestamp)
    timestamp = timestamp -pd.DateOffset(years=1)
    hour = timestamp.hour
    day_of_year = timestamp.timetuple().tm_yday
    
    #do the lookback
    zone = int(df[0][0])
    #day_of_year = int(df[0][3])
    #hour = int(df[0][1])
    print(zone, day_of_year, hour)
    
    test_path = "train_data.parquet"
    old_data = pd.read_parquet(test_path)
    
    
    old_data.sort_index(inplace=True)
    #timestamp = generate_timestamp(hour, day_of_year)
    
    matching_row = old_data.loc[(zone, timestamp),:].copy()
    matching_row.update(pd.Series(my_list))
    hour_list = []
    for i in range(0,4): 
        hour -=1 
        
        if hour < 3: 
            hour = 23
        hour_list.append(hour)
        
        
    filtered_df = old_data[
    (old_data['PULocationID'] == zone) &
    (old_data['hour'].isin(hour_list)) &
    (old_data['dayofyear'] == day_of_year)
    ]
    filtered_df.index = filtered_df.index.droplevel(0)
    filtered_df.reset_index(drop=False, inplace = True)
    filtered_df.set_index(['PULocationID', 'datetime'])
    return filtered_df
        

def get_meteo_data(timestamp): 
        # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 40.7143,
        "longitude": -74.006,
        "hourly": ["temperature_2m", "relative_humidity_2m", "rain", "snowfall", "weather_code", "surface_pressure", "cloud_cover", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
        "daily": ["temperature_2m_max", "temperature_2m_min"],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "GMT",
        "past_days": 1,
        "forecast_days": 2
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_rain = hourly.Variables(2).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(4).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(5).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(6).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(7).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(8).ValuesAsNumpy()
    hourly_wind_gusts_10m = hourly.Variables(9).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s"),
	    end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    
    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["rain"] = hourly_rain
    hourly_data["snowfall"] = hourly_snowfall
    hourly_data["weather_code"] = hourly_weather_code
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    matching_rows = hourly_dataframe[hourly_dataframe['date']== timestamp].copy()
    
    
    daily = response.Daily()
    daily_temperature_2m_max = daily.Variables(0).ValuesAsNumpy()
    daily_temperature_2m_min = daily.Variables(1).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["temperature_2m_max"] = daily_temperature_2m_max
    daily_data["temperature_2m_min"] = daily_temperature_2m_min

    daily_dataframe = pd.DataFrame(data = daily_data)
    matching_rows2 = daily_dataframe[daily_dataframe['date'].dt.date == timestamp.date()]
    series1 = matching_rows.iloc[0]
    series2 = matching_rows2.iloc[0]
    series2.drop('date')
    matching_rows = pd.concat([series1, series2])
    print(matching_rows)

    
timestamp = pd.to_datetime(datetime.now()).round(freq='H')
get_meteo_data(timestamp)