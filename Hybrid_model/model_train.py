import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

def train_hybrid(subset_size, dir_path=''):
    # Step 1: Load the data
    def load_image_data(images_file, csv_file):
        """Load image data and labels."""
        images = np.load(images_file)
        df = pd.read_csv(csv_file, index_col=0)
        labels = df['label'].values
        print(f"Loaded {images_file}: {images.shape}, Labels: {labels.shape}")
        return images, labels

    def load_feature_data(features_file, csv_file):
        """Load tabular features and labels."""
        features = np.load(features_file)
        df = pd.read_csv(csv_file, index_col=0)
        labels = df['label'].values
        print(f"Loaded {features_file}: {features.shape}, Labels: {labels.shape}")
        return features, labels

    # Load image and feature data for train, validation, and test sets
    train_images, train_labels = load_image_data(f'{dir_path}train_data/cnn_train_images-{subset_size}.npy', f'{dir_path}train_data/train-{subset_size}.csv')
    val_images, val_labels = load_image_data(f'{dir_path}val_data/cnn_val_images-{subset_size}.npy', f'{dir_path}test_data/test-{subset_size}.csv')
    test_images, test_labels = load_image_data(f'{dir_path}test_data/cnn_test_images-{subset_size}.npy', f'{dir_path}val_data/val-{subset_size}.csv')

    train_features, _ = load_feature_data(f'{dir_path}train_data/nn_train_features-{subset_size}.npy', f'{dir_path}train_data/train-{subset_size}.csv')
    val_features, _ = load_feature_data(f'{dir_path}val_data/nn_val_features-{subset_size}.npy', f'{dir_path}val_data/val-{subset_size}.csv')
    test_features, _ = load_feature_data(f'{dir_path}test_data/nn_test_features-{subset_size}.npy', f'{dir_path}test_data/test-{subset_size}.csv')

    # Compute class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Verify data shapes
    assert train_images.shape[0] == train_features.shape[0] == len(train_labels), "Mismatch in training data"
    assert val_images.shape[0] == val_features.shape[0] == len(val_labels), "Mismatch in validation data"
    assert test_images.shape[0] == test_features.shape[0] == len(test_labels), "Mismatch in test data"
    assert train_features.shape[1] == 33, f"Expected 33 features, got {train_features.shape[1]}"

    # Step 2: Data Augmentation for Training Images
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    val_test_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
    val_generator = val_test_datagen.flow(val_images, val_labels, batch_size=32)
    test_generator = val_test_datagen.flow(test_images, test_labels, batch_size=32)

    # Step 3: Define the CNN branch
    image_input = Input(shape=(128, 128, 3), name='image_input')
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(image_input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    cnn_output = Dense(128, activation='relu')(x)

    # Step 4: Define the NN branch
    feature_input = Input(shape=(33,), name='feature_input')
    y = Dense(256, activation='relu')(feature_input)
    y = Dense(128, activation='relu')(y)
    y = Dense(64, activation='relu')(y)
    y = Dense(32, activation='relu')(y)
    nn_output = Dense(16, activation='relu')(y)

    # Step 5: Combine the branches
    combined = Concatenate()([cnn_output, nn_output])
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.3)(z)
    z = Dense(64, activation='relu')(z)
    z = Dropout(0.3)(z)
    output = Dense(1, activation='sigmoid')(z)

    # Step 6: Build and compile the model
    model = Model(inputs=[image_input, feature_input], outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.001, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Step 7: Train the model
    history = model.fit(
        [train_images, train_features], train_labels,
        validation_data=([val_images, val_features], val_labels),
        epochs=50,
        batch_size=32,
        callbacks=[early_stop],
        class_weight=class_weight_dict
    )

    # Step 8: Evaluate the model
    test_loss, test_accuracy = model.evaluate([test_images, test_features], test_labels, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Step 9: Save the model
    model.save(f'{dir_path}models/hybrid_model-{subset_size}.keras')
    print(f"Model saved as 'hybrid_model-{subset_size}.keras'")

# Example usage
if __name__ == '__main__':
    train_hybrid(1000, dir_path='../')  # subest_size as param