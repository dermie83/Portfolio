
import os
import re
from typing import Counter
import pandas as pd # Import the regular expression module

# --- Predefined list of words to check for in the 'Specific Word Extraction' section ---
TARGET_ALL_WORDS = ['stranded', 'struck', 'smack', 'wrecked', 'collided', 'rocks', 'cliffs', 'grounded', 'severe', 
                      'weather', 'extreme', 'wind','gale', 'seabed', 'ashore', 'washed', 'sink', 'sinking',
                      'breached', 'hit', 'sunk', 'sank', 'lost', 'disappeared' ,'storm', 'fog', 'mist' , 'heavy', 'force',
                       'shallow', 'tide', 'snow', 'broke', 'broken', 'conditions', 'condition', 'violent', 'keeled', 'rogue', 
                       'filled','full', 'water', 'rudder', 'control', 'sandbank', 'pierced', 'wreck', 'missing', 
                       'ran', 'running', 'aground','distress', 'squall', 'upturned', 'capsized', 'collision',
                      'wreckage', 'found', 'discovered', 'burnt', 'fire', 'shot', 'sunken', 'mast', 'dismasted', 'derelict',
                      'destroyed', 'destruction', 'burned', 'burning', 'abandoned', 'drowned', 'loss', 'foundered', 
                      'located', 'lowwater', 'strong', 'tides', 'currents', 'river', 'parted', 'driven', 
                      'explosion', 'blew up', 'hurricane', 'sloops', 'anchored', 'torpedoed', 'uboat', 
                      'drifting','reef', 'wrecks', 'broken', 'pieces', 'floating', 'waterlogged',
                      'bottom', 'upwards', 'logboat', 'shore', 'perished','en', 'route','went', 'going', 'down',
                      'all', 'rescued', 'saved', 'no', 'crew', 'died', 'stricken', 'explosive', 'bodies', 'survived',
                      'alive', 'life', 'stormy', 'dead', 
                      'one', 'two','three', 'four', 'five', 'six', 'seven','eight', 'nine', 'ten', 'eleven', 'twelve',
                      'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty']

TARGET_COLLISION_WORDS = ['struck', 'smack', 'wrecked', 'collided', 'rocks', 'cliffs', 'grounded', 'seabed', 'ashore',
                        'washed', 'sink', 'sunk','sank', 'sinking', 'breached', 'hit', 'shallow', 
                        'tide', 'broke', 'keeled', 'filled', 'water', 'rudder', 'control', 'sandbank', 'pierced', 
                        'wreck', 'ran', 'running', 'aground', 'upturned', 'capsized', 'collision',
                        'wreckage', 'discovered', 'burnt', 'fire', 'shot', 'sunken', 'mast', 'dismasted', 'derelict',
                        'destroyed', 'destruction', 'burned', 'burning', 'abandoned', 'loss', 'foundered', 
                        'lowwater', 'tides', 'currents', 'river', 'parted', 'driven', 'explosion', 'blew up', 
                        'sloops', 'anchored', 'torpedoed', 'uboat', 'drifting','reef', 'wrecks', 'broken', 'waterlogged',
                        'bottom', 'upwards', 'shore', 'went', 'down', 'stricken', 'explosive'
                      
                      ]

# --- Predefined list of words to check for in the 'Specific Word Extraction' section ---
TARGET_CASUALTIES_WORDS = ['stranded', 'lost', 'disappeared' ,'full', 'missing', 
                        'distress', 'found',  'drowned', 'located', 'perished', 'going',
                        'all', 'rescued', 'saved', 'no', 'crew', 'died', 'bodies', 'survived', 'alive', 'life', 'dead', 
                        'one', 'two','three', 'four', 'five', 'six', 'seven','eight', 'nine', 'ten', 'eleven', 'twelve',
                        'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty'
                        ]

# --- Predefined list of words to check for in the 'Specific Word Extraction' section ---
TARGET_WEATHER_WORDS = ['severe', 'weather', 'extreme', 'wind', 'winds', 'force', 'gale', 'storm', 'stormy', 'fog', 'mist', 'heavy',
                       'snow', 'conditions', 'condition', 'violent', 'rogue', 'squall', 'strong', 'tides', 'currents', 'river'
                       ]


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


# --- Helper Function for Extracting Specific Words (using TARGET_WORDS) ---
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
        return "none"
    else:
        return ' '.join(unique_found_words_ordered)
    

# --- Helper Function for Finding Most Common Words (kept for potential future use) ---
def get_most_common_words(df, source_column_name, top_n=10):
    """
    Finds the most common words in a specified column across all rows.

    Args:
        df (pd.DataFrame): The input DataFrame.
        source_column_name (str): The name of the column containing text data.
        top_n (int): The number of most common words to retrieve.

    Returns:
        pd.DataFrame: A new DataFrame with 'Word' and 'Count' columns for the most common words,
                      or None if the column is not found or empty.
    """
    if source_column_name not in df.columns:
        print(f"Error: Source column '{source_column_name}' not found for common word analysis.")
        return None

    # Concatenate all text from the column into a single string
    # Ensure to handle non-string types by converting to string and dropping NaNs
    all_text = ' '.join(df[source_column_name].dropna().astype(str))

    # Split the combined text into words
    words = re.findall(r'\b[a-z0-9]+\b', all_text.lower())

    if not words:
        print(f"No words found in column '{source_column_name}' for common word analysis.")
        return None

    # Count word frequencies
    word_counts = Counter(words)

    # Get the top N most common words
    most_common = word_counts.most_common(top_n)

    # Create a new DataFrame from the most common words
    if most_common:
        common_words_df = pd.DataFrame(most_common, columns=['Word', 'Count'])
        print(f"Found top {len(common_words_df)} most common words from column '{source_column_name}'.")
        return common_words_df
    else:
        print(f"Could not find any common words in column '{source_column_name}'.")
        return None



def main():
    print("--- Google Geocoding, Text Cleaning, and Concatenation Script ---")

    # Get CSV file path from user
    # csv_file_path = input("Enter the path to your CSV file: ")
    # if not os.path.exists(csv_file_path):
    #     print(f"Error: File not found at '{csv_file_path}'")
    #     return
    csv_file_path = r'Wrecks20240620_table_geocoded.csv'

    try:
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        print(f"Successfully loaded '{csv_file_path}'. Shape: {df.shape}")

        # --- Filter rows that contain "none" in the 'Description' column ---
        if 'Description' in df.columns:
            text = "We regret that we are unable to supply descriptive details for this record at present."
            initial_rows = len(df)
            # Convert column to string, then filter out rows where 'Description' contains 'none' (case-insensitive)
            df = df[~df['Description'].astype(str).str.contains(text, case=False, na=False)]
            print(f"Filtered out rows where 'Description' contains {text}. New shape: {df.shape} ({initial_rows - len(df)} rows removed).")
        else:
            print("Warning: 'Description' column not found for filtering.")

        # --- Text Cleaning Section ---
        perform_text_cleaning = input("Do you want to perform text cleaning on a column? (yes/no): ").lower()
        if perform_text_cleaning == 'yes' or perform_text_cleaning == 'y':
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


        # --- Specific Word Extraction Section (using TARGET_ALL_WORDS) ---
        perform_specific_word_extraction = input(f"Do you want to extract specific ALL words (from {TARGET_ALL_WORDS}) into a new column? (yes/no): ").lower()
        if perform_specific_word_extraction == 'yes' or perform_specific_word_extraction == 'y':
            source_extract_column = input("Enter the exact name of the column to check for these specific ALL words (e.g., 'Description', 'Description_cleaned'): ")
            if source_extract_column not in df.columns:
                print(f"Error: Column '{source_extract_column}' not found for specific event word extraction.")
                print(f"Available columns: {df.columns.tolist()}")
            else:
                new_extract_column_name = input(f"Enter the name for the new column to store the found ALL words (e.g., '{source_extract_column}_all_events'): ")
                print(f"Extracting specific event words from '{source_extract_column}' and saving to '{new_extract_column_name}'...")
                df[new_extract_column_name] = df[source_extract_column].apply(lambda x: extract_specific_event_words(x, TARGET_ALL_WORDS))
                print("Specific event word extraction complete.")
        else:
            print("Skipping specific event word extraction.")
        
        # --- Specific Word Extraction Section (using TARGET_WEATHER_WORDS) ---
        perform_specific_word_extraction = input(f"Do you want to extract specific WEATHER words (from {TARGET_WEATHER_WORDS}) into a new column? (yes/no): ").lower()
        if perform_specific_word_extraction == 'yes' or perform_specific_word_extraction == 'y':
            source_extract_column = input("Enter the exact name of the column to check for these specific WEATHER words (e.g., 'Description', 'Description_cleaned'): ")
            if source_extract_column not in df.columns:
                print(f"Error: Column '{source_extract_column}' not found for specific event word extraction.")
                print(f"Available columns: {df.columns.tolist()}")
            else:
                new_extract_column_name = input(f"Enter the name for the new column to store the found WEATHER words (e.g., '{source_extract_column}_weather'): ")
                print(f"Extracting specific event words from '{source_extract_column}' and saving to '{new_extract_column_name}'...")
                df[new_extract_column_name] = df[source_extract_column].apply(lambda x: extract_specific_event_words(x, TARGET_WEATHER_WORDS))
                print("Specific event word extraction complete.")
        else:
            print("Skipping specific event word extraction.")
        
         # --- Specific Word Extraction Section (using TARGET_CASUALTIES_WORDS) ---
        perform_specific_word_extraction = input(f"Do you want to extract specific CASUALTIES words (from {TARGET_CASUALTIES_WORDS}) into a new column? (yes/no): ").lower()
        if perform_specific_word_extraction == 'yes' or perform_specific_word_extraction == 'y':
            source_extract_column = input("Enter the exact name of the column to check for these specific CASUALTIES words (e.g., 'Description', 'Description_cleaned'): ")
            if source_extract_column not in df.columns:
                print(f"Error: Column '{source_extract_column}' not found for specific event word extraction.")
                print(f"Available columns: {df.columns.tolist()}")
            else:
                new_extract_column_name = input(f"Enter the name for the new column to store the found CASUALTIES words (e.g., '{source_extract_column}_casualties'): ")
                print(f"Extracting specific event words from '{source_extract_column}' and saving to '{new_extract_column_name}'...")
                df[new_extract_column_name] = df[source_extract_column].apply(lambda x: extract_specific_event_words(x, TARGET_CASUALTIES_WORDS))
                print("Specific event word extraction complete.")
        else:
            print("Skipping specific event word extraction.")


        # --- Specific Word Extraction Section (using TARGET_COLLISION_WORDS) ---
        perform_specific_word_extraction = input(f"Do you want to extract specific COLLISION words (from {TARGET_COLLISION_WORDS}) into a new column? (yes/no): ").lower()
        if perform_specific_word_extraction == 'yes' or perform_specific_word_extraction == 'y':
            source_extract_column = input("Enter the exact name of the column to check for these specific COLLISION words (e.g., 'Description', 'Description_cleaned'): ")
            if source_extract_column not in df.columns:
                print(f"Error: Column '{source_extract_column}' not found for specific event word extraction.")
                print(f"Available columns: {df.columns.tolist()}")
            else:
                new_extract_column_name = input(f"Enter the name for the new column to store the found COLLISION words (e.g., '{source_extract_column}_collisions'): ")
                print(f"Extracting specific event words from '{source_extract_column}' and saving to '{new_extract_column_name}'...")
                df[new_extract_column_name] = df[source_extract_column].apply(lambda x: extract_specific_event_words(x, TARGET_COLLISION_WORDS))
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

        df = df.drop(['DD_Lat', 'DD_Long', 'Source of Co-ordinate', 'Description', 
                      'Record Source'], axis=1)

        output_csv_path = os.path.splitext(csv_file_path)[0] + "_processed.csv"
        df.to_csv(output_csv_path, index=False)
        print(f"\nProcessing complete! Results saved to '{output_csv_path}'")


        # --- Common Words Section ---
        perform_common_words = input("Do you want to find the most common words in a column? (yes/no): ").lower()
        if perform_common_words == 'yes' or perform_common_words == 'y':
            common_words_source_column = input("Enter the exact name of the column for common word analysis (e.g., 'Description', 'Cleaned_Text', 'Concatenated_Text'): ")
            if common_words_source_column not in df.columns:
                print(f"Error: Column '{common_words_source_column}' not found for common word analysis.")
                print(f"Available columns: {df.columns.tolist()}")
            else:
                try:
                    top_n_str = input("Enter the number of most common words to find (e.g., 10): ")
                    top_n = int(top_n_str)
                    if top_n <= 0:
                        print("Number of common words must be a positive integer. Using default (10).")
                        top_n = 10
                except ValueError:
                    print("Invalid input for number of common words. Using default (10).")
                    top_n = 10

                common_words_df = get_most_common_words(df, common_words_source_column, top_n)
                if common_words_df is not None:
                    common_words_output_path = os.path.splitext(csv_file_path)[0] + "_common_words.csv"
                    common_words_df.to_csv(common_words_output_path, index=False)
                    print(f"Most common words saved to '{common_words_output_path}'")
                else:
                    print("No common words DataFrame was generated.")
        else:
            print("Skipping common word analysis.")


    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file '{csv_file_path}' is empty.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()