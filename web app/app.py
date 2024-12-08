import flask
import tensorflow as tf
import preprocess_data as ppd
import pandas as pd


app = flask.Flask(__name__, template_folder='templates')

model = tf.keras.models.load_model('model/my_model.keras')

    
@app.route('/', methods=['GET', 'POST'])

def main():

    if flask.request.method == 'GET':

        return(flask.render_template('main.html'))

    if flask.request.method == 'POST':

        PULocationID= flask.request.form['PULocationID']
        hour= flask.request.form['hour']
        dayofweek= flask.request.form['dayofweek']
        dayofyear= flask.request.form['dayofyear']
        month= flask.request.form['month']
        quarter= flask.request.form['quarter']
        x= flask.request.form['x']
        y= flask.request.form['y']
        temp= flask.request.form['temp']
        temp_min= flask.request.form['temp_min']
        temp_max= flask.request.form['temp_max']
        pressure= flask.request.form['pressure']
        humidity= flask.request.form['humidity']
        wind_speed= flask.request.form['wind_speed']
        wind_deg= flask.request.form['wind_deg']
        wind_gust= flask.request.form['wind_gust']
        clouds_all= flask.request.form['clouds_all']
        rain_1h= flask.request.form['rain_1h']
        rain_3h= flask.request.form['rain_3h']
        snow_1h= flask.request.form['snow_1h']
        snow_3h= flask.request.form['snow_3h']
        weather_main_Clear= flask.request.form['weather_main_Clear']
        weather_main_Clouds= flask.request.form['weather_main_Clouds']
        weather_main_Fog= flask.request.form['weather_main_Fog']
        weather_main_Haze= flask.request.form['weather_main_Haze']
        weather_main_Mist= flask.request.form['weather_main_Mist']
        weather_main_Rain= flask.request.form['weather_main_Rain']
        weather_main_Snow= flask.request.form['weather_main_Snow']
        weather_main_Thunderstorm= flask.request.form['weather_main_Thunderstorm']
        
        input_variables = pd.DataFrame(
                        [[PULocationID,hour,dayofweek,dayofyear,month,quarter,x,y,temp,temp_min,temp_max,pressure,humidity,wind_speed,wind_deg,wind_gust,clouds_all,rain_1h,rain_3h,snow_1h,snow_3h,weather_main_Clear,weather_main_Clouds,weather_main_Fog,weather_main_Haze,weather_main_Mist,weather_main_Rain,weather_main_Snow,weather_main_Thunderstorm]],
                        columns=['PULocationID','hour','dayofweek','dayofyear','month','quarter','x','y','temp','temp_min','temp_max','pressure','humidity','wind_speed','wind_deg','wind_gust','clouds_all','rain_1h','rain_3h','snow_1h','snow_3h','weather_main_Clear','weather_main_Clouds','weather_main_Fog','weather_main_Haze','weather_main_Mist','weather_main_Rain','weather_main_Snow','weather_main_Thunderstorm'],
                        dtype=float,
                        index=['input'])
        data,scalar = ppd.preprocess_data(input_variables)
        model = tf.keras.models.load_model("model/my_model.keras")
        prediction_unscaled = model.predict(data)
        pred_list = scalar.inverse_transform(prediction_unscaled).tolist()
        prediction= pred_list[-1][0]
        prediction = round(prediction)
        return flask.render_template('main.html',
                                    original_input={'PULocationID':PULocationID,
                                                     'Hour':hour,
                                                     'Dayofweek':dayofweek,
                                                     'Dayofyear':dayofyear,
                                                     'Month':month,
                                                     'Quarter':quarter,
                                                     'X':x,
                                                     'Y':y,
                                                     'Temp':temp,
                                                     'Temp_Min':temp_min,
                                                     'Temp_Max':temp_max,
                                                     'Pressure':pressure,
                                                     'Humidity':humidity,
                                                     'Wind_Speed':wind_speed,
                                                     'Wind_Deg':wind_deg,
                                                     'Wind_Gust':wind_gust,
                                                     'Clouds_All':clouds_all,
                                                     'Rain_1H':rain_1h,
                                                     'Rain_3H':rain_3h,
                                                     'Snow_1H':snow_1h,
                                                     'Snow_3H':snow_3h,
                                                     'Weather_Main_Clear':weather_main_Clear,
                                                     'Weather_Main_Clouds':weather_main_Clouds,
                                                     'Weather_Main_Fog':weather_main_Fog,
                                                     'Weather_Main_Haze':weather_main_Haze,
                                                     'Weather_Main_Mist':weather_main_Mist,
                                                     'Weather_Main_Rain':weather_main_Rain,
                                                     'Weather_Main_Snow':weather_main_Snow,
                                                     'Weather_Main_Thunderstorm':weather_main_Thunderstorm},
                                    result=prediction,)
if __name__ == '__main__':

    app.run()
