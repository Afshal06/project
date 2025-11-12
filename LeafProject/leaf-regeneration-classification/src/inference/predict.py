# predict.py

import torch
from torchvision import transforms
from PIL import Image
import json
from src.models.classifier import LeafClassifier
from src.models.regeneration import LeafRegenerator

class LeafPredictor:
    def __init__(self, classifier_model_path, regenerator_model_path):
        self.classifier = LeafClassifier()
        self.classifier.load_state_dict(torch.load(classifier_model_path))
        self.classifier.eval()

        self.regenerator = LeafRegenerator()
        self.regenerator.load_state_dict(torch.load(regenerator_model_path))
        self.regenerator.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            classification_output = self.classifier(image)
            regeneration_output = self.regenerator(image)

        predicted_class = classification_output.argmax(dim=1).item()
        return predicted_class, regeneration_output

    def get_leaf_info(self, predicted_class):
        # Placeholder for actual implementation
        leaf_info = {
            "name": "Leaf Name",
            "biological_name": "Biological Name",
            "medicinal_purpose": "Medicinal Purpose",
            "disease_recognition": "Disease Recognized",
            "recovery_methods": "Recovery Methods"
        }
        return leaf_info

if __name__ == "__main__":
    classifier_model_path = "path/to/classifier/model.pth"
    regenerator_model_path = "path/to/regenerator/model.pth"
    predictor = LeafPredictor(classifier_model_path, regenerator_model_path)

    image_path = "path/to/leaf/image.jpg"
    predicted_class, regeneration_output = predictor.predict(image_path)
    leaf_info = predictor.get_leaf_info(predicted_class)

    print(json.dumps(leaf_info, indent=4))