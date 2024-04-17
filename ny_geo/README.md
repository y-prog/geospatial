# Geospatial Analysis of Potential Apple Store Locations in New York City

![apple store nyc](apple.png)

## Description
This project aims to identify suitable locations for opening new Apple Stores within the New York City area. It leverages geospatial coordinates and data from loopnet.com to analyze available retail rental spaces.


The following additional variables:
potential_location: Binary variable indicating suitability of a location.
nearest_distance: Distance to the nearest existing store.
weighted_avg_income: Weighted average income in the area.
yearly_price_per_SF: Yearly price per square foot.

have been derived by the following initial variables:

Address: Address of the potential location.
City: City where the potential location is located.
ZIP: ZIP code of the potential location.
Year Built: Year the building was constructed.
SF: Square footage of the potential location.
Price: Rental price of the potential location.

which have been gathered by the following sources:

-census.gov
-irs.gov
-loopnet.com
-apple.com


## Project Structure
the most important files needed for our analysis are the following
- nyc_apple_stores_proj.ipynb
    this file contains data processing and machine lerning methods needed to make predictions on potential locations  
- `geospatial_coords_generator.ipymb`
   this file is needed to generate geospatial coordinates (latitude, longitude) given the addresses extracted from loopnet.com
   and apple.com. However, this step is only needed if there is a need of new locations to be converted as the file 'nyc_apple_stores_proj.ipymb' will work on its own since it relies on the .txt files in this folder.

- 'locs_pipeline.py' this file contains methods that take an excel file as an input containing the variables needed to make new stores suitability predictions.
  in order to use this file and display all location with their suitability description on a web browser the file 'pipeline_usage.py' is needed.
  N.B. it is important to assure that the 'nyc_apple_stores_proj.ipymb' file is ran before the files in the pipeline folder since it is where all ML models are trained and imported into the pipeline

## Usage

### Setting up the Environment
1. Install required packages, create and activate the conda environment e.g. 'base'
   conda activate base.


2. Assure the 'base' interpreter is activated in your jupyter notebook.

3.  install the following packages
 - numpy, pandas, geopandas


## Getting Started
- Clone the repository:
   ```bash
   git clone https://github.com/y-prog/geospatial.git
   ```
  
- Navigate to the directory /geospatial.

- launch 'jupyter notebook' from your conda prompt in the 'base' environment
  
- open  and run 'nyc_apple_stores_proj.ipymb'.

- for experimenting with new locations navigate to the /pipeline_files folder and run 'pipeline_usage.py' and feel free to add/replace
  data in excel_csv_files/address_sample.csv
     
## Output Description
The script provided performs the following tasks:

- preprocesses data from various .xlsx and .txt files in order to obtain the aforementioned variables
- generates interactive map showing both prospective and unsuitable locations for new store openings.
- visually represents the selected predictors and their distribution with respect to the target variable
- Splits the preprocessed data into training and testing sets.
- runs the data through various ML algorithms and compares their performance.
- illustrates the ML performance via confusion matrices.
  
