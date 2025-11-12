from flask import Flask, request, jsonify
from src.models.classifier import LeafClassifier
from src.models.regeneration import LeafRegenerator

app = Flask(__name__)

# Load the trained models
classifier = LeafClassifier()
regenerator = LeafRegenerator()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image = data.get('image')

    # Perform classification
    leaf_name, biological_name = classifier.predict(image)

    return jsonify({
        'leaf_name': leaf_name,
        'biological_name': biological_name
    })

@app.route('/regenerate', methods=['POST'])
def regenerate():
    data = request.json
    image = data.get('image')

    # Perform regeneration
    regenerated_image = regenerator.regenerate(image)

    return jsonify({
        'regenerated_image': regenerated_image
    })

@app.route('/disease_recognition', methods=['POST'])
def disease_recognition():
    data = request.json
    image = data.get('image')

    # Perform disease recognition
    disease_info = classifier.recognize_disease(image)

    return jsonify(disease_info)

@app.route('/disease_recovery', methods=['POST'])
def disease_recovery():
    data = request.json
    disease_name = data.get('disease_name')

    # Provide recovery methods
    recovery_methods = classifier.get_recovery_methods(disease_name)

    return jsonify({
        'recovery_methods': recovery_methods
    })

if __name__ == '__main__':
    app.run(debug=True)