import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import datetime


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
    
def generate_timestamp(hour, day_of_year):
    """Generates a timestamp given the hour and day of the year in EST.

    Args:
        hour: The hour of the day (0-23).
        day_of_year: The day of the year (1-366).

    Returns:
        A datetime object representing the timestamp in EST.
        Returns None if input is invalid.
    """
    current_year = datetime.datetime.now().year

    # Create the date string
    date_str = f"{current_year}-{day_of_year}-{hour}"

    # Attempt to parse the date, handling potential errors gracefully.
    timestamp = pd.to_datetime(date_str, format='%Y-%j-%H') # Parse with UTC
    return timestamp
# Prepare data for LSTM
def create_dataset(dataset, look_back=4):
    
    
    
    
    df = dataset.values.tolist()
    my_list = df[0]
    my_list.append(0)
    
    #do the lookback
    zone = int(df[0][0])
    day_of_year = int(df[0][3])
    hour = int(df[0][1])
    print(zone, day_of_year, hour)
    
    test_path = "train_data.parquet"
    old_data = pd.read_parquet(test_path)
    
    
    old_data.sort_index(inplace=True)
    timestamp = generate_timestamp(hour, day_of_year)
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
        

    
    
    
    
    
