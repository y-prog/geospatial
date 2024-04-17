import numpy as np
import pandas as pd
import geopandas as gpd
import re
from typing import List
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import ast
import pickle
import warnings
import os

# Filter out UserWarning categories
warnings.filterwarnings("ignore", category=UserWarning)

# Configuration dictionary
par_dict_config = {
    'zip_col_name': 'ZIP',
    'float_col_name': 'weighted_avg_income',
    'predictor_names': ['SF', 'price_SF/YR', 'year_built', 'distance_to_closest_store', 'avg_income']
}

class locs_suitability:
    def __init__(self, config):
        self._config = config
        self.geolocator = Nominatim(user_agent='my_geocoder')
        self.loaded_models = None  # Initialize loaded_models attribute with None

    def load_model(self):
        """
        Load trained models from files.
        """
        loaded_models = {}
        model_names = ['logreg', 'rf', 'svm', 'gbm', 'mlp', 'knn']
        for model_name in model_names:
            root_dir = os.path.dirname(os.path.dirname(__file__))  # Get root directory of project
            file_path = os.path.join(root_dir, 'pkl_files', model_name + '.pkl')  # Construct file path
            with open(file_path, 'rb') as f:
                model = pickle.load(f)
                loaded_models[model_name] = model
        return loaded_models

    def predict_with_models(self, data: np.array, row_df: pd.Series) -> tuple:
        """Make predictions with loaded models."""
        predictor_names = self._config['predictor_names']
        indices_none_in_data = [index for index, item in enumerate(data) if item is None]
        if indices_none_in_data:
            missing_predictors = [predictor_names[i] for i in indices_none_in_data]
            location_info = ', '.join([f"{key}: {value}" for key, value in row_df.items()])
            message = f"{missing_predictors} input needed for this location: {location_info}"
            return message

        # Reshape data if there are no missing values
        data_reshaped = np.array(data).reshape(1, 7)

        loaded_models = self.load_model()
        predictions = {}
        for model_name, model_values in loaded_models.items():
            prediction = model_values.predict(data_reshaped)  # Reshape data for prediction
            predictions[model_name] = prediction
        return predictions, dict(row_df)

    @staticmethod
    def csv_to_pandas_df(file_name, separator):
        """Read CSV file into a Pandas DataFrame."""
        root_dir = os.path.dirname(os.path.dirname(__file__))  # Get root directory of project
        file_path = os.path.join(root_dir, file_name)  # Construct file path
        return pd.read_csv(file_path, sep=separator)

    @staticmethod
    def extract_zip(address: str) -> str:
        """Extract ZIP code from address."""
        pattern = r'\b\d{5}(?:-\d{4})?\b'
        match = re.search(pattern, address)
        return match.group() if match else None

    def zip_matching_rows(self, df: pd.DataFrame, zip_code_to_match: int) -> pd.DataFrame:
        """Return rows of a DataFrame where the 'ZIP_code' column matches the specified ZIP code."""
        zip_code_to_match = np.int64(zip_code_to_match)
        matching_rows = df[df[self._config['zip_col_name']] == zip_code_to_match]
        return matching_rows if len(matching_rows) > 0 else None

    def float_col_avg(self, df_col: pd.Series) -> list:
        """Calculate the average of a float column."""
        return [None] if df_col is None or df_col.empty else np.mean(df_col[self._config['float_col_name']])

    def get_coords(self, address: str) -> tuple:
        """Get coordinates (latitude, longitude) for an address."""
        location = self.geolocator.geocode(address)
        return (location.latitude, location.longitude) if location else None

    @staticmethod
    def read_txt_file(txt_file_name: str) -> list:
        """Read data from a text file."""
        root_dir = os.path.dirname(os.path.dirname(__file__))  # Get root directory of project
        file_path = os.path.join(root_dir, 'text_files', txt_file_name)  # Construct file path
        list_tuples = []
        with open(file_path, 'r') as file:
            for line in file:
                list_tuples.append(ast.literal_eval(line)[0])
        return list_tuples[:-1]

    @staticmethod
    def min_distance(coord_tuple: tuple, list_of_coords: list) -> float:
        """Calculate the minimum distance between a coordinate and a list of coordinates."""
        return min([geodesic(coord, coord_tuple).meters for coord in list_of_coords])

    @staticmethod
    def one_hot_encode_column(income: float) -> list:
        """One-hot encode income into categories."""
        if income == [None]:
            return [None]
        elif income < 60:
            return [0, 1, 0]
        elif income > 120:
            return [1, 0, 0]
        else:
            return [0, 0, 1]

    @staticmethod
    def prediction_message(res_dict: dict, key: str) -> str:
        """Generate prediction message based on the result dictionary and the specified key."""
        res_dict = res_dict[0]
        if type(res_dict) != dict:
            return 'additional info is needed'
        else:
            res_key = res_dict[key][0]
            return 'location suitable' if res_key == 1 else 'location NOT suitable'