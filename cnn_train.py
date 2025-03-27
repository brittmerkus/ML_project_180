import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Load the data
def load_data(images_file, csv_file):
    """Load image data and labels."""
    images = np.load(images_file)
    df = pd.read_csv(csv_file, index_col=0)
    labels = df['label'].values
    print(f"Loaded {images_file}: {images.shape}, Labels: {labels.shape}")
    return images, labels

# Load train and test data
train_images, train_labels = load_data('train_images.npy', 'train.csv')
test_images, test_labels = load_data('test_images.npy', 'test.csv')

# Verify data shapes
assert train_images.shape[0] == len(train_labels), "Mismatch between train images and labels"
assert test_images.shape[0] == len(test_labels), "Mismatch between test images and labels"

# Step 2: Build the CNN model
model = Sequential([
    Input(shape=(128, 128, 3)),  # Input layer for images
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),  # Reduced dropout after pooling

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),  # Reduced dropout

    Conv2D(128, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),
    Dropout(0.2),  # Reduced dropout

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.2),  # Reduced dropout before dense layers
    Dense(64, activation='relu'),
    Dropout(0.2),  # Reduced dropout
    Dense(1, activation='sigmoid')  # Binary classification
])


# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    epochs=50,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Step 4: Evaluate the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# Optional: Save the model
model.save('cnn_ai_vs_human_model.keras')
print("Model saved as 'cnn_ai_vs_human_model.keras'")
