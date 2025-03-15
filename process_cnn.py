import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

def create_train_test_splits_and_extract_images(input_csv, dataset_dir, img_size=(128, 128), test_size=0.2, subset_size=None, random_state=None):
    # Step 1: Create train/test splits
    df = pd.read_csv(input_csv, index_col=0)

    if subset_size is not None:
        sample_size = min(subset_size, len(df))
        df = df.sample(n=sample_size, random_state=random_state)

    train_df, test_df = train_test_split(df, test_size=test_size, stratify=df['label'], random_state=random_state)
    
    train_df.to_csv('train.csv', index=True)
    test_df.to_csv('test.csv', index=True)

    # Step 2: Load images and save as NumPy arrays
    def process_split(split_df, output_npy_file):
        images = []
        for _, row in split_df.iterrows():
            img_path = os.path.join(dataset_dir, row['file_name'])
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array)

        images_array = np.array(images)
        np.save(output_npy_file, images_array)
        return images_array.shape

    print("Extracting training images...")
    train_shape = process_split(train_df, 'train_images.npy')
    print(f"Train images shape: {train_shape}")

    print("Extracting test images...")
    test_shape = process_split(test_df, 'test_images.npy')
    print(f"Test images shape: {test_shape}")

if __name__ == '__main__':
    input_csv_path = "/Users/brittmerkus/Desktop/machine_learning/ML_project_180/datasets/train.csv"
    dataset_dir = "/Users/brittmerkus/Desktop/machine_learning/ML_project_180/datasets"
    
    create_train_test_splits_and_extract_images(
        input_csv_path,
        dataset_dir,
        img_size=(128, 128),
        test_size=0.2,
        subset_size=2000,
        random_state=42
    )
