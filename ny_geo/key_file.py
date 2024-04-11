import json

# Load API keys from a JSON file
with open('api_keys.json') as f:
    api_keys = json.load(f)

# Access the specific API key you need
census_api_key = api_keys['census']

# Now you can use the API key in your requests
