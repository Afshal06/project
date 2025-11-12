# Leaf Regeneration and Classification Project

## Overview
This project focuses on the regeneration and classification of leaves using machine learning techniques. The goal is to identify various types of leaves, their biological names, medicinal purposes, recognize diseases affecting them, and suggest recovery methods.

## Project Structure
- **data/**: Contains raw and processed leaf images and annotations.
  - **raw/**: Raw leaf images and data collected for the project.
  - **processed/**: Cleaned and transformed datasets ready for training.
  - **annotations/**: Annotation files providing labels and metadata for the leaf images.
  
- **notebooks/**: Jupyter notebooks for data exploration, augmentation, and model training.
  - **01-data-exploration.ipynb**: Explore the dataset and visualize data distribution.
  - **02-augmentation-and-regeneration.ipynb**: Data augmentation techniques and leaf regeneration methods.
  - **03-model-training.ipynb**: Training machine learning models for classification and regeneration.

- **src/**: Source code for data handling, model definitions, training, inference, and utilities.
  - **data/**: Data loading and preprocessing.
    - **dataset.py**: Class for handling leaf images and annotations.
    - **augmentations.py**: Functions for data augmentation techniques.
  - **models/**: Model architectures for classification and regeneration.
    - **backbone.py**: Backbone architecture for feature extraction.
    - **classifier.py**: Classification model for predicting leaf types.
    - **regeneration.py**: Class for regenerating leaves based on input data.
  - **training/**: Training logic and evaluation.
    - **train.py**: Training loop and optimization.
    - **evaluate.py**: Functions for evaluating model performance.
  - **inference/**: Making predictions on new leaf images.
    - **predict.py**: Functionality for predictions using trained models.
  - **api/**: REST API for serving model predictions.
    - **serve.py**: API setup for serving predictions and regeneration results.
  - **utils/**: Utility functions for metrics.
    - **metrics.py**: Functions for calculating evaluation metrics.
  - **types/**: Custom types and interfaces for type checking.
    - **index.py**: Type definitions.

- **models/**: Stores model checkpoints during training.
  - **checkpoints/**: Directory for model checkpoints.

- **scripts/**: Scripts for data preparation and model export.
  - **prepare_data.py**: Prepares raw data for processing.
  - **export_model.py**: Exports the trained model for deployment.

- **experiments/**: Configuration for experiments.
  - **config.yaml**: Hyperparameters and settings for model training.

- **requirements.txt**: Lists Python dependencies required for the project.

- **environment.yml**: Conda environment file with specified dependencies.

- **setup.py**: Packaging script for the project.

- **.gitignore**: Specifies files and directories to ignore in Git.

## Expected Outcomes
1. **Leaf Name**: Common name of the leaf.
2. **Biological Name**: Scientific name of the leaf.
3. **Medicinal Purpose**: Information on the medicinal uses of the leaf.
4. **Disease Recognition**: Identification of diseases affecting the leaf.
5. **Recovery Methods**: Suggested methods for recovering from identified diseases.

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd leaf-regeneration-classification
pip install -r requirements.txt
```

Alternatively, you can create a conda environment using:

```bash
conda env create -f environment.yml
```

## Usage
Follow the Jupyter notebooks in the `notebooks/` directory to explore the data, apply augmentations, and train the models. Use the API in `src/api/serve.py` to serve predictions for new leaf images.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any suggestions or improvements.

## License
This project is licensed under the MIT License. See the LICENSE file for details.