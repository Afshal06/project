import joblib
import os
from src.models.classifier import LeafClassifier
from src.models.regeneration import LeafRegenerator

def export_model(model, model_name, export_path):
    if not os.path.exists(export_path):
        os.makedirs(export_path)
    model_file_path = os.path.join(export_path, f"{model_name}.pkl")
    joblib.dump(model, model_file_path)
    print(f"Model exported to {model_file_path}")

if __name__ == "__main__":
    # Load the trained models (replace with actual loading logic)
    classifier = LeafClassifier()  # Load your trained classifier model
    regenerator = LeafRegenerator()  # Load your trained regenerator model

    # Specify the export path
    export_path = os.path.join(os.getcwd(), 'models', 'checkpoints')

    # Export the models
    export_model(classifier, "leaf_classifier", export_path)
    export_model(regenerator, "leaf_regenerator", export_path)