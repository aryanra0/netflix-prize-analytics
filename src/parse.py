"""
Data Parser for Netflix Prize.

This script takes the messy raw text files and turns them into nice, clean Parquet files.
I handle the weird comma issues in movie titles and make sure timestamps play nice with PySpark.
"""

import os
import glob
import pandas as pd
import numpy as np
import argparse
from src import config  # Using our new central config!

def parse_movie_titles(input_path=None, output_path=None):
    """Reads movie_titles.csv and saves it as parquet."""
    input_path = input_path or (config.RAW_DATA_DIR / 'movie_titles.csv')
    output_path = output_path or config.MOVIES_FILE
    
    print(f"Parsing {input_path}...")
    
    # The raw file is ISO-8859-1 encoded and has some tricky commas in titles.
    # We parse it manually line-by-line to be safe.
    movies = []
    try:
        with open(input_path, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Split by comma, but max 2 splits (ID, Year, Title)
                parts = line.split(',', 2)
                if len(parts) < 3: continue
                
                movie_id = int(parts[0])
                year = parts[1]
                title = parts[2]
                
                # Some years are NULL, handle that gracefully
                if year == 'NULL':
                    year = np.nan
                else:
                    year = float(year)
                    
                movies.append({'movie_id': movie_id, 'year': year, 'title': title})
                
        df = pd.DataFrame(movies)
        df['movie_id'] = df['movie_id'].astype(np.int32)
        
        print(f"Saving {len(df)} movies to {output_path}...")
        df.to_parquet(output_path, index=False)
        
    except FileNotFoundError:
        print(f"Warning: {input_path} not found. Skipping movies parsing.")

def parse_probe(input_path=None, output_path=None):
    """Parses probe.txt (the validation set IDs)."""
    input_path = input_path or (config.RAW_DATA_DIR / 'probe.txt')
    output_path = output_path or config.PROBE_FILE
    
    print(f"Parsing {input_path}...")
    data = []
    current_movie_id = None
    
    try:
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                # Movie IDs end with a colon (e.g. "123:")
                if line.endswith(':'):
                    current_movie_id = int(line[:-1])
                else:
                    # Otherwise it's a user ID
                    user_id = int(line)
                    data.append({'movie_id': current_movie_id, 'user_id': user_id})
                    
        df = pd.DataFrame(data)
        df['movie_id'] = df['movie_id'].astype(np.int32)
        df['user_id'] = df['user_id'].astype(np.int32)
        
        print(f"Saving {len(df)} probe entries to {output_path}...")
        df.to_parquet(output_path, index=False)
        
    except FileNotFoundError:
        print(f"Warning: {input_path} not found. Skipping probe parsing.")

def parse_qualifying():
    """Parses qualifying.txt (the test set to predict)."""
    input_path = config.RAW_DATA_DIR / 'qualifying.txt'
    output_path = config.QUALIFYING_FILE
    
    print(f"Parsing {input_path}...")
    data = []
    current_movie_id = None
    
    try:
        with open(input_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                
                if line.endswith(':'):
                    current_movie_id = int(line[:-1])
                else:
                    parts = line.split(',')
                    user_id = int(parts[0])
                    date = parts[1]
                    data.append({'movie_id': current_movie_id, 'user_id': user_id, 'date': date})
    
        df = pd.DataFrame(data)
        # Important: Spark hates nanosecond timestamps, so we cast to microseconds [us]
        df['date'] = pd.to_datetime(df['date']).astype('datetime64[us]')
        df['movie_id'] = df['movie_id'].astype(np.int32)
        df['user_id'] = df['user_id'].astype(np.int32)
        
        print(f"Saving {len(df)} qualifying entries to {output_path}...")
        df.to_parquet(output_path, index=False)
        
    except FileNotFoundError:
        print(f"Warning: {input_path} not found. Skipping qualifying parsing.")

def parse_ratings_file(file_path):
    """Generator that yields chunks of data from a single combined_data file."""
    print(f"Processing {file_path}...")
    current_movie_id = None
    buffer = []
    CHUNK_SIZE = 1_000_000  # Keep memory usage sane
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            if line.endswith(':'):
                current_movie_id = int(line[:-1])
            else:
                parts = line.split(',')
                user_id = int(parts[0])
                rating = int(parts[1])
                date = parts[2]
                
                buffer.append({
                    'movie_id': current_movie_id,
                    'user_id': user_id,
                    'rating': rating,
                    'date': date
                })
                
                if len(buffer) >= CHUNK_SIZE:
                    yield pd.DataFrame(buffer)
                    buffer = []
    
    if buffer:
        yield pd.DataFrame(buffer)

def parse_all_ratings():
    """Parses all combined_data_*.txt files into one big Parquet file."""
    input_dir = config.RAW_DATA_DIR
    output_path = config.RATINGS_FILE
    
    files = sorted(glob.glob(os.path.join(input_dir, 'combined_data_*.txt')))
    if not files:
        print("No combined_data_*.txt files found! Skipping ratings parsing.")
        return

    all_chunks = []
    for file_path in files:
        for chunk in parse_ratings_file(file_path):
            # Optimize types to save space
            chunk['movie_id'] = chunk['movie_id'].astype(np.int16)
            chunk['user_id'] = chunk['user_id'].astype(np.int32)
            chunk['rating'] = chunk['rating'].astype(np.int8)
            # Spark compatibility fix: use microseconds
            chunk['date'] = pd.to_datetime(chunk['date']).astype('datetime64[us]')
            all_chunks.append(chunk)
            
    print("Concatenating all chunks...")
    full_df = pd.concat(all_chunks, ignore_index=True)
    
    print(f"Saving {len(full_df)} ratings to {output_path}...")
    full_df.to_parquet(output_path, index=False)

def main():
    # We don't really need args anymore since we have config.py, 
    # but keeping main() clean is good practice.
    
    # 1. Movies
    parse_movie_titles()
    
    # 2. Probe (Validation IDs)
    parse_probe()
    
    # 3. Qualifying (Test IDs)
    parse_qualifying()
    
    # 4. Ratings (The big one)
    parse_all_ratings()

if __name__ == "__main__":
    main()
