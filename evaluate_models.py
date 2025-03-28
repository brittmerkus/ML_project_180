import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.metrics import accuracy_score


def evaluate_models(subset_size, cnn_model_path, nn_model_path, hybrid_model_path):
    """
    Evaluate CNN, NN, and Hybrid models and return their performance metrics in specified format.
    Optimized to load only required data for each model to reduce memory usage.

    Parameters:
    subset_size (int): Size of the data subset to evaluate
    cnn_model_path (str): Path to CNN .keras model (ignored, using subset-specific path)
    nn_model_path (str): Path to NN .h5 model (ignored, using subset-specific path)
    hybrid_model_path (str): Path to Hybrid .keras model (ignored, using subset-specific path)

    Returns:
    dict: Dictionary containing lists of metrics for each model
    """

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

    # Initialize result lists
    models = ["CNN", "Neural Network", "Hybrid"]
    val_accuracy = []
    test_accuracy = []
    val_loss = []
    test_loss = []

    # Evaluate CNN model
    cnn_model = load_model(f'models/cnn_model-{subset_size}.keras')


    # Validation data
    val_images, val_labels = load_image_data(
        f'val_data/cnn_val_images-{subset_size}.npy',
        f'val_data/val-{subset_size}.csv'
    )
    cnn_val_eval = cnn_model.evaluate(val_images, val_labels, verbose=0, return_dict=True)
    cnn_val_acc = cnn_val_eval['accuracy'] if 'accuracy' in cnn_val_eval else cnn_val_eval.get('acc')
    cnn_val_loss = cnn_val_eval['loss']
    del val_images, val_labels

    # Test data
    test_images, test_labels = load_image_data(
        f'test_data/cnn_test_images-{subset_size}.npy',
        f'test_data/test-{subset_size}.csv'
    )
    cnn_test_eval = cnn_model.evaluate(test_images, test_labels, verbose=0, return_dict=True)
    cnn_test_acc = cnn_test_eval['accuracy'] if 'accuracy' in cnn_test_eval else cnn_test_eval.get('acc')
    cnn_test_loss = cnn_test_eval['loss']

    val_accuracy.append(cnn_val_acc)
    test_accuracy.append(cnn_test_acc)
    val_loss.append(cnn_val_loss)
    test_loss.append(cnn_test_loss)

    del test_images, test_labels, cnn_model

    # Evaluate NN model
    nn_model = load_model(f'models/nn_model-{subset_size}.h5')

    # Training data

    # Validation data
    val_features, val_labels = load_feature_data(
        f'val_data/nn_val_features-{subset_size}.npy',
        f'val_data/val-{subset_size}.csv'
    )
    nn_val_eval = nn_model.evaluate(val_features, val_labels, verbose=0, return_dict=True)
    nn_val_acc = nn_val_eval['accuracy'] if 'accuracy' in nn_val_eval else nn_val_eval.get('acc')
    nn_val_loss = nn_val_eval['loss']
    del val_features, val_labels

    # Test data
    test_features, test_labels = load_feature_data(
        f'test_data/nn_test_features-{subset_size}.npy',
        f'test_data/test-{subset_size}.csv'
    )
    nn_test_eval = nn_model.evaluate(test_features, test_labels, verbose=0, return_dict=True)
    nn_test_acc = nn_test_eval['accuracy'] if 'accuracy' in nn_test_eval else nn_test_eval.get('acc')
    nn_test_loss = nn_test_eval['loss']

    val_accuracy.append(nn_val_acc)
    test_accuracy.append(nn_test_acc)
    val_loss.append(nn_val_loss)
    test_loss.append(nn_test_loss)

    del test_features, test_labels, nn_model

    # Evaluate Hybrid model
    hybrid_model = load_model(f'models/hybrid_model-{subset_size}.keras')


    # Validation data
    val_images, val_labels = load_image_data(
        f'val_data/cnn_val_images-{subset_size}.npy',
        f'val_data/val-{subset_size}.csv'
    )
    val_features, _ = load_feature_data(
        f'val_data/nn_val_features-{subset_size}.npy',
        f'val_data/val-{subset_size}.csv'
    )
    hybrid_val_eval = hybrid_model.evaluate([val_images, val_features], val_labels, verbose=0, return_dict=True)
    hybrid_val_acc = hybrid_val_eval['accuracy'] if 'accuracy' in hybrid_val_eval else hybrid_val_eval.get('acc')
    hybrid_val_loss = hybrid_val_eval['loss']
    del val_images, val_features, val_labels

    # Test data
    test_images, test_labels = load_image_data(
        f'test_data/cnn_test_images-{subset_size}.npy',
        f'test_data/test-{subset_size}.csv'
    )
    test_features, _ = load_feature_data(
        f'test_data/nn_test_features-{subset_size}.npy',
        f'test_data/test-{subset_size}.csv'
    )
    hybrid_test_eval = hybrid_model.evaluate([test_images, test_features], test_labels, verbose=0, return_dict=True)
    hybrid_test_acc = hybrid_test_eval['accuracy'] if 'accuracy' in hybrid_test_eval else hybrid_test_eval.get('acc')
    hybrid_test_loss = hybrid_test_eval['loss']

    val_accuracy.append(hybrid_val_acc)
    test_accuracy.append(hybrid_test_acc)
    val_loss.append(hybrid_val_loss)
    test_loss.append(hybrid_test_loss)

    del test_images, test_features, test_labels, hybrid_model

    # Return results in specified format
    return {
        'models': models,
        'val_accuracy': val_accuracy,
        'test_accuracy': test_accuracy,
        'val_loss': val_loss,
        'test_loss': test_loss
    }


# Example usage:
if __name__ == "__main__":
    results = evaluate_models(
        subset_size=1000,
        cnn_model_path='cnn_model.keras',
        nn_model_path='nn_model.h5',
        hybrid_model_path='hybrid_model.keras'
    )

    # Print results
    print("Models:", results['models'])
    print("Val Accuracy:", [f"{x:.2f}" for x in results['val_accuracy']])
    print("Test Accuracy:", [f"{x:.2f}" for x in results['test_accuracy']])
    print("Val Loss:", [f"{x:.2f}" for x in results['val_loss']])
    print("Test Loss:", [f"{x:.2f}" for x in results['test_loss']])