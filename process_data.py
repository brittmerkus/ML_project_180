import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Feature_extraction.color_feature import extract_color_features
from Feature_extraction.shape_feature import extract_shape_features
from Feature_extraction.gabor_filter import extract_gabor_features
from Feature_extraction.local_binary_pattern import extract_lbp_features
from concurrent.futures import ProcessPoolExecutor
import os


def create_train_test_splits_and_extract_features(input_csv, dataset_dir, test_size, subset_size=None,
                                                  random_state=None):
    """
    Creates balanced train and test splits from a given CSV file, optionally using a subset of the data,
    extracts features from the images in the splits using ProcessPoolExecutor, and saves them as NumPy arrays.

    Parameters:
    - input_csv (str): Path to the input CSV file.
    - dataset_dir (str): Directory containing the images (base path for file_name in CSV).
    - test_size (float or int): If float, between 0.0 and 1.0 (proportion of test split).
                               If int, absolute number of test samples.
    - subset_size (int or None): Number of samples to use from the dataset. If None, uses entire dataset.
    - random_state (int or None): Seed for reproducibility. If None, uses random seed.
    """
    # Step 1: Create train/test splits
    # Read the CSV file with the first column as index
    df = pd.read_csv(input_csv, index_col=0)

    # If subset_size is specified, sample that many rows
    if subset_size is not None:
        sample_size = min(subset_size, len(df))
        df = df.sample(n=sample_size, random_state=random_state)

    # Split the data into train and test sets with stratification
    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)

    # Save the train and test dataframes to CSV files
    train_df.to_csv('train.csv', index=True)
    test_df.to_csv('test.csv', index=True)

    # Step 2: Extract features for train and test splits
    def process_split(csv_file, output_npy_file):
        # Read the split CSV
        split_df = pd.read_csv(csv_file, index_col=0)

        # Get full image paths by joining dataset_dir with file_name
        image_paths = [os.path.join(dataset_dir, row['file_name']) for _, row in split_df.iterrows()]

        # Use ProcessPoolExecutor to extract features in parallel
        with ProcessPoolExecutor() as executor:
            features_list = list(executor.map(extract_manual_features, image_paths))

        # Convert to NumPy array and save
        features_array = np.array(features_list)
        np.save(output_npy_file, features_array)
        return features_array.shape  # Optional: return shape for verification

    # Process train and test splits
    print("Extracting features for training set...")
    train_shape = process_split('train.csv', 'train_features.npy')
    print(f"Train features shape: {train_shape}")

    print("Extracting features for test set...")
    test_shape = process_split('test.csv', 'test_features.npy')
    print(f"Test features shape: {test_shape}")


def extract_manual_features(image_path):
    color = extract_color_features(image_path)
    lbp = extract_lbp_features(image_path)
    gabor = extract_gabor_features(image_path)
    shape = extract_shape_features(image_path)
    return np.concatenate([color, lbp, gabor, shape])


if __name__ == '__main__':
    input_csv_path = os.path.join(os.getcwd(), "datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4/train.csv")

    dataset_dir = os.path.join(os.getcwd(),"datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4")
    test_size_value = 0.2  # % of the  data to use a test split
    subset_size_value = 1000  # How many total images to  use
    random_state_value = 42  # For reproducibility

    create_train_test_splits_and_extract_features(
        input_csv_path,
        dataset_dir,
        test_size_value,
        subset_size_value,
        random_state_value
    )