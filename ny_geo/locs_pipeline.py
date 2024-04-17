import numpy as np
import pandas
import pandas as pd
import geopandas as gpd
import re
import matplotlib.pyplot as plt
import folium
from folium.plugins import MarkerCluster
from IPython.display import IFrame
import seaborn as sns
from typing import List
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import ast
import pickle

par_dict_config = {'zip_col_name':'ZIP', 'float_col_name': 'weighted_avg_income'
                   ,'predictor_names' : ['Square footage', 'yearly price per square footage', 'year built',
                                      'distance to closest store', 'average income per zip code']
}  # dictionary containing configurations

class locs_suitability:
    def __init__(self, config):
        self._config = config
        self.geolocator = Nominatim(user_agent='my_geocoder')
        #self.loaded_models = None  # Initialize loaded_models attribute

    @staticmethod
    def load_model():
        """
        Load trained models from files.
        """
        loaded_models = {}
        model_names = ['logreg', 'rf', 'svm', 'gbm', 'mlp', 'knn']
        for model_name in model_names:
            with open(f'{model_name}.pkl', 'rb') as f:
                loaded_models[model_name] = pickle.load(f)
        return loaded_models

    def predict_with_models(self, data, row_df):
        '''Make predictions with loaded models'''

        predictor_names = self._config['predictor_names']
        indices_none_in_data = [index for index, item in enumerate(data) if item is None]
        if indices_none_in_data:
            missing_predictors = [predictor_names[i] for i in indices_none_in_data]
            return predictor_names[0] + ' input needed for this location: '+ str(dict(row_df))

        # Reshape data if there are no missing values
        data_reshaped = np.array(data).reshape(1, -1)

        loaded_models = self.load_model()
        predictions = {}
        for model_name, model_values in loaded_models.items():
            prediction = model_values.predict(data_reshaped)  # Reshape data for prediction
            predictions[model_name] = prediction
        return predictions

    @staticmethod
    def csv_to_pandas_df(path_to_file, separator):
        """addresses to verify"""
        return pd.read_csv(path_to_file, sep=separator)  # Use sep parameter to specify the separator

    @staticmethod
    def extract_zip(address):
        # Define a regex pattern to match ZIP codes
        pattern = r'\b\d{5}(?:-\d{4})?\b'  # Matches 5-digit ZIP codes, optionally followed by a hyphen and 4 more digits

        # Search for ZIP code pattern in the address
        match = re.search(pattern, address)

        if match:
            return match.group()  # Return the matched ZIP code
        else:
            return None  # Return None if no ZIP code is found


    def zip_matching_rows(self, df, zip_code_to_match):
        """ Return rows of a DataFrame where the 'ZIP_code' column matches the specified ZIP code """
        # Convert zip_code_to_match to np.int64
        zip_code_to_match = np.int64(zip_code_to_match)

        # Boolean indexing to filter rows where ZIP code matches
        matching_rows = df[df[self._config['zip_col_name']] == zip_code_to_match]

        # Check if any matching rows were found
        if len(matching_rows) == 0:
            #print("No matching ZIP code found.")
            return None
        else:
            return matching_rows


    def float_col_avg(self, df_col):
        if  df_col is None or df_col.empty:
            #print('info not in database')
            return [None]
        else:
            return np.mean(df_col[self._config['float_col_name']])

    def get_coords(self, address):
        location = self.geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None  # Return None if location not found or geocoding fails

    @staticmethod
    def read_txt_file(txt_file_path):
        list_tuples = []
        with open (txt_file_path, 'r') as file:
            for line in file:
                list_tuples.append(ast.literal_eval(line)[0])
        return  list_tuples[:-1]

    @staticmethod
    def min_distance(coord_tuple, list_of_coords):
        min_distance = min([geodesic(coord, coord_tuple).meters for coord in list_of_coords])
        return min_distance

    @staticmethod
    def one_hot_encode_column(income):
        if income == [None]:
            return [None]
        elif income < 60:
            return [0, 1, 0]
        elif income >120:
            return [1, 0, 0]
        else:
            return [0, 0, 1]




locs_config = locs_suitability(par_dict_config)
df_addresses = locs_config.csv_to_pandas_df(r'address_sample.csv', ';')
df_processed = locs_config.csv_to_pandas_df(r'processed_df.csv', ',')
print(df_addresses.columns)
for index, row in df_addresses.iterrows():
    address_zip = locs_config.extract_zip(row.address)
    matching_rows = locs_config.zip_matching_rows(df_processed, np.int64(address_zip))
    avg_income = locs_config.float_col_avg(matching_rows) # zip code avg weighted income
    coords = locs_config.get_coords(row.address)
    apple_coords_list = locs_config.read_txt_file('apple_store_locs_nyc.txt')
    distance_to_closest_store = locs_config.min_distance(coords, apple_coords_list)
    encoded_income = locs_suitability.one_hot_encode_column(avg_income)
    predictors = [row['SF'], row['price SF/YR'], row['year_built'], distance_to_closest_store] + encoded_income
    predictions = locs_config.predict_with_models(np.array(predictors), row)
    print('predictions=   ',predictions)



