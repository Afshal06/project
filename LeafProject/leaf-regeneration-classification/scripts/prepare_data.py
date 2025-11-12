import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(raw_data_dir, processed_data_dir, annotations_file):
    # Create processed data directory if it doesn't exist
    os.makedirs(processed_data_dir, exist_ok=True)

    # Load annotations
    annotations = pd.read_csv(annotations_file)

    # Split data into training and validation sets
    train_df, val_df = train_test_split(annotations, test_size=0.2, random_state=42)

    # Process training data
    for _, row in train_df.iterrows():
        image_path = os.path.join(raw_data_dir, row['filename'])
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(processed_data_dir, 'train', row['filename']))

    # Process validation data
    for _, row in val_df.iterrows():
        image_path = os.path.join(raw_data_dir, row['filename'])
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(processed_data_dir, 'val', row['filename']))

if __name__ == "__main__":
    raw_data_directory = '../data/raw'
    processed_data_directory = '../data/processed'
    annotations_file_path = '../data/annotations/annotations.csv'
    
    prepare_data(raw_data_directory, processed_data_directory, annotations_file_path)