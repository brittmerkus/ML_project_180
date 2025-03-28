import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
def train_nn(subset_size):
    # Step 1: Load the data
    def load_data(features_file, csv_file):
        """Load features and labels from files."""
        features = np.load(features_file)
        df = pd.read_csv(csv_file, index_col=0)
        labels = df['label'].values  # Extract labels (0 or 1)
        print(f"Loaded {features_file}: {features.shape}, Labels: {labels.shape}")
        return features, labels

    # Load train and test data
    train_features, train_labels = load_data(f'train_data/nn_train_features-{subset_size}.npy', f'train_data/train-{subset_size}.csv')
    test_features, test_labels = load_data(f'test_data/nn_test_features-{subset_size}.npy', f'test_data/test-{subset_size}.csv')
    val_features, val_labels = load_data(f'val_data/nn_val_features-{subset_size}.npy', f'val_data/val-{subset_size}.csv')
    # Verify data shapes
    assert train_features.shape[1] == 33, f"Expected 33 features, got {train_features.shape[1]}"
    assert train_features.shape[0] == len(train_labels), "Mismatch between train features and labels"
    assert test_features.shape[1] == 33, f"Expected 33 features, got {test_features.shape[1]}"
    assert test_features.shape[0] == len(test_labels), "Mismatch between test features and labels"

    # Step 2: Build the model
    model = Sequential([
        Input(shape=(33,)),  # Input layer for 33 features
        Dense(128, activation='relu'),  # Hidden layer with 64 units
        Dense(64, activation='relu'),  # Hidden layer with 64 units
        Dense(32, activation='relu'),  # Hidden layer with 32 units
        Dense(1, activation='sigmoid')  # Output layer for binary classification
    ])

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Summary of the model
    model.summary()

    # Step 3: Train the model
    history = model.fit(
        train_features, train_labels,
        validation_data=(test_features, test_labels),
        epochs=50,  # Adjust based on convergence
        batch_size=32,  # Small batch size since dataset is small
        verbose=1
    )

    # Step 4: Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_features, test_labels, verbose=0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")


    # Optional: Save the model
    model.save(f'models/nn_model-{subset_size}.h5')
    print(f"Model saved as 'nn_model-{subset_size}.h5'")

    return model, test_loss, test_accuracy