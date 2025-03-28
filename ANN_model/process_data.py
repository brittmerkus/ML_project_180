import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from Feature_extraction.color_feature import extract_color_features
from Feature_extraction.shape_feature import extract_shape_features
from Feature_extraction.gabor_filter import extract_gabor_features
from Feature_extraction.local_binary_pattern import extract_lbp_features
from concurrent.futures import ProcessPoolExecutor
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def create_train_test_splits_and_extract_features(input_csv, dataset_dir,  test_size, img_size=(128, 128),subset_size=None,
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
    train_df, temp_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)

    # Second split - Validation (10%) & Test (10%)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=random_state)

    # Save the train and test dataframes to CSV files
    train_df.to_csv(f'../train_data/train-{subset_size}.csv', index=True)
    test_df.to_csv(f'../test_data/test-{subset_size}.csv', index=True)
    val_df.to_csv(f'../val_data/val-{subset_size}.csv', index=True)

    # Step 2: Extract features for train and test splits
    def process_split_features(csv_file, output_npy_file):
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

    # Step 4: Function to process images and save as NumPy arrays
    def process_split_images(split_df, output_npy_file):
        images = []
        for _, row in split_df.iterrows():
            img_path = os.path.join(dataset_dir, row['file_name'])
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array)

        images_array = np.array(images)
        np.save(output_npy_file, images_array)
        return images_array.shape


    # Step 5: Extract and save images and features
    print("Extracting training images...")
    train_shape = process_split_images(train_df, f'../train_data/cnn_train_images-{subset_size}.npy')
    print(f"Train images shape: {train_shape}")

    print("Extracting features for training set...")
    train_shape = process_split_features(f'../train_data/train-{subset_size}.csv',
                                f'../train_data/nn_train_features-{subset_size}.npy')
    print(f"Train features shape: {train_shape}")


    print("Extracting test images...")
    test_shape = process_split_images(test_df, f'../test_data/cnn_test_images-{subset_size}.npy')
    print(f"Test images shape: {test_shape}")

    print("Extracting features for test set...")
    test_shape = process_split_features(f'../test_data/test-{subset_size}.csv',
                               f'../test_data/nn_test_features-{subset_size}.npy')
    print(f"Test features shape: {test_shape}")


    print("Extracting validation images...")
    val_shape = process_split_images(val_df, f'../val_data/cnn_val_images-{subset_size}.npy')
    print(f"Validation images shape: {val_shape}")

    print("Extracting features for val set...")
    test_shape = process_split_features(f'../val_data/val-{subset_size}.csv',
                               f'../val_data/nn_val_features-{subset_size}.npy')
    print(f"Test features shape: {test_shape}")




def extract_manual_features(image_path):
    color = extract_color_features(image_path)
    lbp = extract_lbp_features(image_path)
    gabor = extract_gabor_features(image_path)
    shape = extract_shape_features(image_path)
    return np.concatenate([color, lbp, gabor, shape])


if __name__ == '__main__':
    input_csv_path = os.path.join(os.getcwd(), "../datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4/train.csv")

    dataset_dir = os.path.join(os.getcwd(),"../datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4")
    test_size_value = 0.2  # % of the  data to use a test split
    subset_size_value = 1000  # How many total images to  use
    random_state_value = 42  # For reproducibility
    image_size=(128,128)
    create_train_test_splits_and_extract_features(
        input_csv_path,
        dataset_dir,
        test_size_value,
        image_size,
        subset_size_value,
        random_state_value
    )