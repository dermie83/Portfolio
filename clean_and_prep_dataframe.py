
import os
import re
from typing import Counter
import pandas as pd # Import the regular expression module

# --- Predefined list of words to check for in the 'Specific Word Extraction' section ---
TARGET_EVENT_WORDS = ['stranded', 'struck', 'smack', 'wrecked', 'collided', 'rocks', 'cliffs', 'grounded', 'severe', 
                      'weather', 'extreme', 'wind','gale', 'seabed', 'ashore', 'washed', 'sink', 'sinking',
                      'breached', 'hit', 'sunk','sank', 'lost', 'disappeared' ,'storm', 'fog', 'mist' , 'heavy',
                       'shallow', 'tide', 'snow', 'broke', 'conditions', 'condition', 'violent', 'keeled', 'rogue', 
                       'filled', 'water', 'lost', 'rudder', 'control', 'sandbank','pierced', 'wreck', 'missing', 
                       'ran', 'aground','distress', 'squall', 'upturned', 'capsized', 
                      'wreckage', 'found', 'discovered', 'burnt', 'fire', 'shot', 'sunken', 'mast', 'dismasted', 'derelict',
                      'destroyed', 'destruction', 'burned', 'abandoned', 'drowned', 'loss', 'all', 'foundered', 'located', 
                      'lowwater', 'strong', 'tides', 'currents', 'river', 'en', 'route', 'parted', 'driven', 
                      'explosion', 'blew up', 'hurricane', 'sloops', 'anchored', 'torpedoed', 'uboat', 
                      'drifting','reef', 'broke', 'wrecks', 'broken', 'pieces', 'floating', 'waterlogged',
                      'bottom', 'upwards', 'logboat', 'shore']


# --- Helper Function for Text Cleaning ---
def clean_text_and_remove_duplicates(text):
    """
    Cleans a string by:
    1. Converting to lowercase.
    2. Removing special characters (keeping only alphanumeric and spaces).
    3. Removing duplicate words.

    Args:
        text (str): The input string to clean.

    Returns:
        str: The cleaned string with unique words.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string inputs (e.g., NaN)

    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove special characters (keep alphanumeric and spaces)
    # This regex keeps letters, numbers, and spaces.
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 3. Split into words, remove empty strings, and get unique words
    words = text.split()
    # Use a set to maintain order of first appearance while ensuring uniqueness
    seen = set()
    unique_words = []
    for word in words:
        if word not in seen:
            unique_words.append(word)
            seen.add(word)

    # Join unique words back into a string
    return ' '.join(unique_words)


# --- Helper Function for Extracting Specific Words (using TARGET_EVENT_WORDS) ---
def extract_specific_event_words(text, target_words_list):
    """
    Checks for the presence of specific words from a predefined list within a given text
    and returns a space-separated string of only those found words (unique per row).

    Args:
        text (str): The input string (expected to be already cleaned/lowercased).
        target_words_list (list): A list of words to look for.

    Returns:
        str: A space-separated string of found target words, or an empty string if none are found.
    """
    if not isinstance(text, str):
        return ""

    # Split the text into individual words
    words_in_text = text.split()

    unique_found_words_ordered = []
    seen_found_words = set() # Use a set to track words already added for this row

    for word in words_in_text:
        # Check if the word is in our target list and hasn't been added yet for this row
        if word in target_words_list and word not in seen_found_words:
            unique_found_words_ordered.append(word)
            seen_found_words.add(word)
    if not unique_found_words_ordered:
        return "no details yet"
    else:
        return ' '.join(unique_found_words_ordered)



def main():
    print("--- Google Geocoding, Text Cleaning, and Concatenation Script ---")

    # Get CSV file path from user
    csv_file_path = input("Enter the path to your CSV file: ")
    if not os.path.exists(csv_file_path):
        print(f"Error: File not found at '{csv_file_path}'")
        return

    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded '{csv_file_path}'. Shape: {df.shape}")

        # --- Text Cleaning Section ---
        perform_text_cleaning = input("Do you want to perform text cleaning on a column? (yes/no): ").lower()
        if perform_text_cleaning == 'yes':
            text_column_name = input("Enter the exact name of the column to clean: ")
            if text_column_name not in df.columns:
                print(f"Error: Column '{text_column_name}' not found for cleaning.")
                print(f"Available columns: {df.columns.tolist()}")
                return

            cleaned_column_name = input(f"Enter the name for the new column to store cleaned text (e.g., '{text_column_name}_cleaned'): ")
            print(f"Applying text cleaning to '{text_column_name}' and saving to '{cleaned_column_name}'...")
            df[cleaned_column_name] = df[text_column_name].apply(clean_text_and_remove_duplicates)
            print("Text cleaning complete.")
        else:
            print("Skipping text cleaning.")
            cleaned_column_name = None # Indicate that no new cleaned column was created


        # --- Specific Word Extraction Section (using TARGET_EVENT_WORDS) ---
        perform_specific_word_extraction = input(f"Do you want to extract specific event words (from {TARGET_EVENT_WORDS}) into a new column? (yes/no): ").lower()
        if perform_specific_word_extraction == 'yes':
            source_extract_column = input("Enter the exact name of the column to check for these specific event words (e.g., 'Description', 'Description_cleaned'): ")
            if source_extract_column not in df.columns:
                print(f"Error: Column '{source_extract_column}' not found for specific event word extraction.")
                print(f"Available columns: {df.columns.tolist()}")
            else:
                new_extract_column_name = input(f"Enter the name for the new column to store the found event words (e.g., '{source_extract_column}_events'): ")
                print(f"Extracting specific event words from '{source_extract_column}' and saving to '{new_extract_column_name}'...")
                df[new_extract_column_name] = df[source_extract_column].apply(lambda x: extract_specific_event_words(x, TARGET_EVENT_WORDS))
                df[new_extract_column_name].fillna("No Details Available")
                print("Specific event word extraction complete.")
        else:
            print("Skipping specific event word extraction.")


        df["DD_Lat"] = pd.to_numeric(df["DD_Lat"], errors='coerce')
        df["DD_Long"] = pd.to_numeric(df["DD_Long"], errors='coerce')

        zero_mask = (df["DD_Lat"] == 0)
        zero_mask_ = (df["DD_Lat"] == 0)

        # Replace values in 'column_to_modify' with values from 'source_column' where zero_mask is True
        df.loc[zero_mask, "DD_Lat"] = df.loc[zero_mask, "latitude"]

        # Replace values in 'column_to_modify' with values from 'source_column' where zero_mask is True
        df.loc[zero_mask_, "DD_Long"] = df.loc[zero_mask_, "longitude"]
            
        output_csv_path = os.path.splitext(csv_file_path)[0] + "_processed.csv"
        df.to_csv(output_csv_path, index=False)
        print(f"\nProcessing complete! Results saved to '{output_csv_path}'")


    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file '{csv_file_path}' is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()