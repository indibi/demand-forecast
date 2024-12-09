import flask
import tensorflow as tf
import preprocess_data as ppd
import pandas as pd


app = flask.Flask(__name__, template_folder='templates')


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        zone = flask.request.form['zone']

        
        input_variables = pd.DataFrame([[zone]],
                                       columns=['zone'],
                                       dtype=float,
                                       index=['input'])

        data,scalar = ppd.preprocess_data(input_variables)
        model = tf.keras.models.load_model("web app/model/my_model.keras")
        prediction_unscaled = model.predict(data)
        pred_list = scalar.inverse_transform(prediction_unscaled).tolist()
        prediction= pred_list[-1][0]
        prediction = round(prediction)
        
        return flask.render_template('main.html',
                                     original_input={'Zone':zone},
                                     result=prediction,
                                     )

if __name__ == '__main__':
    app.run()