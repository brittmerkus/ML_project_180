import kagglehub
import os

current_dir = os.getcwd()
os.environ['KAGGLEHUB_CACHE'] = current_dir
# Download latest version
path = kagglehub.dataset_download("alessandrasala79/ai-vs-human-generated-dataset")

print("Path to dataset files:", path)