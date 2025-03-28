from ANN_model.model_train import train_nn
from ANN_model.process_data import create_train_test_splits_and_extract_features
from CNN_model.cnn_train import train_cnn
from Hybrid_model.model_train import train_hybrid
from evaluate_models import evaluate_models
from plot_results import run_plots
import os

def run_data_extraction_pipeline():
    input_csv_path = os.path.join(os.getcwd(), "datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4/train.csv")
    dataset_dir = os.path.join(os.getcwd(),"datasets/alessandrasala79/ai-vs-human-generated-dataset/versions/4")
    test_size_value = 0.2  # % of the  data to use a test split
    random_state_value = 42  # For reproducibility
    image_size=(128,128)
    subset_sizes = [1000, 5000, 10000]
    for subset_size in subset_sizes:
        print(f"\nExtracting features for subset size: {subset_size}")
        create_train_test_splits_and_extract_features(input_csv_path, dataset_dir, test_size_value, image_size, subset_size, random_state_value)

def run_train_pipeline():
    subset_sizes = [1000, 5000, 10000]
    for subset_size in subset_sizes:
        print(f"\nTraining ANN model with subset size: {subset_size}")
        train_nn(subset_size)
        print(f"\nTraining CNN model with subset size: {subset_size}")
        train_cnn(subset_size)
        print(f"\nTraining Hybrid model with subset size: {subset_size}")
        train_hybrid(subset_size)

def run_eval_pipeline():
    cnn_model_path = 'cnn_model.keras',
    nn_model_path = 'nn_model.h5',
    hybrid_model_path = 'hybrid_model.keras'
    subset_sizes = [1000, 5000, 10000]
    for subset_size in subset_sizes:
        results = evaluate_models(subset_size, cnn_model_path, nn_model_path, hybrid_model_path)
        # Print results
        print('Subset size :', subset_size)
        print("Models:", results['models'])
        print("Val Accuracy:", [f"{x:.2f}" for x in results['val_accuracy']])
        print("Test Accuracy:", [f"{x:.2f}" for x in results['test_accuracy']])
        print("Val Loss:", [f"{x:.2f}" for x in results['val_loss']])
        print("Test Loss:", [f"{x:.2f}" for x in results['test_loss']])
        run_plots(results['test_accuracy'], results['val_accuracy'], results['test_loss'], results['val_loss'], subset_size)


if __name__ == '__main__':
    run_eval_pipeline()
