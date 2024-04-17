import numpy as np
from locs_pipeline import locs_suitability
import folium
import webbrowser
import yaml

# Load the configuration from the YAML file
with open('config.yaml', 'r') as yaml_file:
    par_dict_config = yaml.safe_load(yaml_file)


# Initialize locs_suitability object
locs_config = locs_suitability(par_dict_config)

# Read dataframes
df_addresses = locs_config.csv_to_pandas_df(r'excel_csv_files/address_sample.csv', ';')
df_processed = locs_config.csv_to_pandas_df(r'excel_csv_files/processed_df.csv', ',')

# Base map centered at New York City
base_map = folium.Map(location=[40.7128, -74.0060], zoom_start=11)

# Iterate over addresses
for index, row in df_addresses.iterrows():
    address_zip = locs_config.extract_zip(row.address)
    matching_rows = locs_config.zip_matching_rows(df_processed, np.int64(address_zip))
    avg_income = locs_config.float_col_avg(matching_rows)
    coords = locs_config.get_coords(row.address)
    apple_coords_list = locs_config.read_txt_file('apple_store_locs_nyc.txt')
    distance_to_closest_store = locs_config.min_distance(coords, apple_coords_list)
    encoded_income = locs_suitability.one_hot_encode_column(avg_income)
    predictors = [row['SF'], row['price SF/YR'], row['year_built'], distance_to_closest_store] + encoded_income
    predictions = locs_config.predict_with_models(np.array(predictors), row)
    pred_message = locs_config.prediction_message(predictions, 'gbm')
    message = f"prediction output = {predictions} ,/n prediction message= {pred_message}"
    print(message)
    if coords is not None:
        # Create a marker with address and prediction message
        marker_message = f"Address: {row.address}<br>Suitability: {pred_message}"
        marker = folium.Marker(location=[coords[0], coords[1]], popup=marker_message)
        marker.add_to(base_map)

# Save and open the map
base_map.save('map.html')
webbrowser.open('map.html')