# Data Mining Course Project: NYC Rideshare Demand Forecasting

# Data
The data used in this project is the New York City Taxi and Limousine Commission (TLC) Trip Record Data which is publicly available at https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page

# Data Preprocessing
The data preprocessing steps are outlined in the jupyter notebooks `nyc_rideshare_preprocessing.ipynb` which can be found in the `protoboard` folder. The data preprocessing steps include:
- Filtering the data
- Feature engineering
- Exploratory data analysis

The final developed data loaders can be found in `src.utils` folder with names `nyc_trip_loader.py` and `nyc_taxi_zones.py` which are responsible for the pre-processing of the datasets.

# Models
The models used in this project include:
- GNN
- LSTM
- MLR

# Model implementation and evaluations
The model implementation and evaluation steps are outlined in the jupyter notebooks `gnn.ipynb`, `MLR.ipynb`, and 'LSTM_Models.ipynb' which can be found in the `protoboard` folder. The model evaluation steps include:
- Model training
- Model evaluation
- Model comparison
- 
# Web Application
The web application can be found under the web app folder. The folder should be cloned to the local machine and app.py should be run in order to use the application. 

# Results
The results of the project can be found in the `docs` folder in the final and intermediate reports.
<!-- # Conclusion -->
<!-- The results of the project show that the GNN model outperforms the other models in terms of forecasting accuracy. The GNN model is able to capture the complex patterns in the data and provide accurate forecasts.Additionally weather data provided a small but tangible benefit for the model prediction performance. -->

<!-- # Future Work -->
<!-- In the future, we plan to further improve the LSTM model and the GNN model by tuning the hyperparameters and adding more features to the model. We also plan to explore other deep learning models such as CNNs and RNNs for demand forecasting. -->

# License
This project is licensed under the MIT License - see the LICENSE file for details

# Acknowledgements
We would like to thank the New York City Taxi and Limousine Commission for providing the data for this project.

# Contributors
Uta Nishii, Keerthana Byreddy, Mert Indibi - October 2024-Present
