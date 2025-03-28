import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

def train_cnn(subset_size):

    # Step 1: Load the data
    def load_data(images_file, csv_file):
        """Load image data and labels."""
        images = np.load(images_file)
        df = pd.read_csv(csv_file, index_col=0)
        labels = df['label'].values
        print(f"Loaded {images_file}: {images.shape}, Labels: {labels.shape}")
        return images, labels

    # Load train, validation, and test sets
    train_images, train_labels = load_data(f'train_data/cnn_train_images-{subset_size}.npy', f'train_data/train-{subset_size}.csv')
    val_images, val_labels = load_data(f'val_data/cnn_val_images-{subset_size}.npy', f'test_data/test-{subset_size}.csv')
    test_images, test_labels = load_data(f'test_data/cnn_test_images-{subset_size}.npy', f'val_data/val-{subset_size}.csv')

    # Compute class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weight_dict = dict(enumerate(class_weights))

    # Verify data shapes
    assert train_images.shape[0] == len(train_labels), "Mismatch between train images and labels"
    assert val_images.shape[0] == len(val_labels), "Mismatch between validation images and labels"
    assert test_images.shape[0] == len(test_labels), "Mismatch between test images and labels"

    # Step 2: Data Augmentation for Training Images
    train_datagen = ImageDataGenerator(
        rotation_range=20,  # Random rotation
        width_shift_range=0.2,  # Horizontal shift
        height_shift_range=0.2,  # Vertical shift
        shear_range=0.2,  # Shearing
        zoom_range=0.2,  # Zoom
        horizontal_flip=True,  # Flip images horizontally
        fill_mode='nearest'  # Fill empty pixels
    )

    # No augmentation for validation/test images, only normalization
    val_test_datagen = ImageDataGenerator()

    # Apply transformations
    train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
    val_generator = val_test_datagen.flow(val_images, val_labels, batch_size=32)
    test_generator = val_test_datagen.flow(test_images, test_labels, batch_size=32)

    # Step 3: Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),  # Keep as is

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),  # Keep as is

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # Slight increase in deeper layers

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),  # Slight increase here

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),  # Increased from 0.2 -> 0.3

        Dense(64, activation='relu'),
        Dropout(0.3),  # Increased from 0.2 -> 0.3

        Dense(1, activation='sigmoid')  # Binary classification
    ])


    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001, decay=1e-6),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Step 4: Train the model
    history = model.fit(
        train_generator,  # Use augmented data
        epochs=50,
        callbacks=[early_stop],
        class_weight=class_weight_dict,
    )

    # Step 5: Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Step 6: Save the model
    model.save(f'models/cnn_model-{subset_size}.keras')
    print(f"Model saved as 'cnn_model-{subset_size}.keras'")

