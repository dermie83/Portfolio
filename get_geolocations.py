import pandas as pd
import requests
import time
import os

# --- Configuration ---
# You can set your API key as an environment variable or enter it when prompted.
# It's recommended to use environment variables for security in production.
GOOGLE_API_KEY = os.environ.get("")
GEOCODING_API_URL = "https://maps.googleapis.com/maps/api/geocode/json"

# --- Helper Function to Get Geo-Coordinates ---
def get_coordinates(address, api_key, delay_seconds=0.1):
    """
    Fetches latitude and longitude for a given address using Google Geocoding API.

    Args:
        address (str): The place description or address to geocode.
        api_key (str): Your Google Geocoding API key.
        delay_seconds (float): Delay between API requests to avoid hitting rate limits.

    Returns:
        tuple: (latitude, longitude) if successful, otherwise (None, None).
    """
    if not address or not isinstance(address, str):
        print(f"Skipping invalid address: {address}")
        return None, None

    params = {
        'address': address,
        'key': api_key
    }
    try:
        response = requests.get(GEOCODING_API_URL, params=params)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        if data['status'] == 'OK' and data['results']:
            location = data['results'][0]['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']
            print(f"Geocoded '{address}': Lat={latitude}, Lng={longitude}")
            time.sleep(delay_seconds) # Be polite to the API
            return latitude, longitude
        elif data['status'] == 'ZERO_RESULTS':
            print(f"No results found for '{address}'. Status: {data['status']}")
        else:
            print(f"Error geocoding '{address}'. Status: {data['status']}. Error Message: {data.get('error_message', 'N/A')}")
        time.sleep(delay_seconds) # Still delay on errors to prevent rapid retries
        return None, None
    except requests.exceptions.RequestException as e:
        print(f"Network or API error for '{address}': {e}")
        time.sleep(delay_seconds * 2) # Longer delay on network errors
        return None, None
    except Exception as e:
        print(f"An unexpected error occurred for '{address}': {e}")
        time.sleep(delay_seconds)
        return None, None

# --- Main Script ---
def main():
    print("--- Google Geocoding Script ---")

    # Get CSV file path from user
    csv_file_path = input("Enter the path to your CSV file: ")
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found at '{csv_file_path}'")
        return

    # Get column name from user
    address_column_name = input("Enter the exact name of the column containing place descriptions (e.g., 'Address', 'Place', 'Description'): ")

    # Get API key
    api_key = GOOGLE_API_KEY
    if not api_key:
        api_key = input("Enter your Google Geocoding API Key (or set GOOGLE_GEOCODING_API_KEY environment variable): ")
        if not api_key:
            print("API Key is required. Exiting.")
            return

    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded '{csv_file_path}'. Shape: {df.shape}")

        if address_column_name not in df.columns:
            print(f"Error: Column '{address_column_name}' not found in the CSV file.")
            print(f"Available columns: {df.columns.tolist()}")
            return

        # Initialize new columns
        df['latitude'] = None
        df['longitude'] = None

        print(f"\nStarting geocoding for column '{address_column_name}'...")
        total_rows = len(df)
        for index, row in df.iterrows():
            address = row[address_column_name]
            print(f"Processing row {index + 1}/{total_rows}: '{address}'")
            lat, lng = get_coordinates(address, api_key)
            df.at[index, 'latitude'] = lat
            df.at[index, 'longitude'] = lng

        # Define output file path
        output_csv_path = os.path.splitext(csv_file_path)[0] + "_geocoded.csv"
        # df.to_csv(output_csv_path, index=False)
        print(f"\nGeocoding complete! Results saved to '{output_csv_path}'")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except KeyError:
        print(f"Error: The column '{address_column_name}' was not found in the CSV. Please check the column name.")
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file '{csv_file_path}' is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
