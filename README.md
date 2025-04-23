# SplitFace: Cross-Attention Learning for Twin Face Separation.

This project aims to develop a model for discriminating between twin faces using a Vision Transformer (ViT) architecture augmented with a Part Selection Module (PSM). The model is trained to identify subtle differences between images of twins, leveraging advanced techniques in deep learning.

## Project Structure

```
SplitFace
├── src
│   ├── data_loading
│   │   ├── dataset.py          # Handles loading images and transformations
│   │   └── transforms.py       # Defines image transformation functions
│   ├── models
│   │   ├── transfg.py          # Implements the TransFG model architecture
│   │   ├── part_selection.py    # Contains the Part Selection Module
│   │   └── relation_encoding.py  # Implements the Relation Encoding module
│   ├── utils
│   │   ├── visualization.py     # Utility functions for visualizations
│   │   └── metrics.py          # Functions for calculating evaluation metrics
│   ├── losses
│   │   ├── contrastive_loss.py  # Defines the Contrastive Loss function
│   │   └── combined_loss.py     # Implements the Combined Loss function
│   ├── trainer.py               # Orchestrates the training process
│   └── inference.py             # Functions for running inference
├── configs
│   ├── default.yaml             # Default configuration settings
│   └── training.yaml            # Specific training configurations
├── scripts
│   ├── train.py                 # Entry point for training the model
│   ├── evaluate.py              # Evaluates the trained model
│   └── predict.py               # Makes predictions on new images
├── requirements.txt             # Required Python packages
└── README.md                    # Project documentation
```

## Installation

To set up the project, clone the repository and install the required packages:

```bash
git clone <repository-url>
cd twin-face-discrimination
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the model, run the following command:

```bash
python scripts/train.py --config configs/training.yaml
```

### Evaluating the Model

To evaluate the trained model on a validation or test dataset, use:

```bash
python scripts/evaluate.py --config configs/default.yaml
```

### Making Predictions

To make predictions on new images, execute:

```bash
python scripts/predict.py --image <path_to_image>
```

## Modules

- **Data Loading**: Handles loading and transforming images from the specified directory structure.
- **Models**: Implements the TransFG architecture, including the backbone, PSM, and relation encoding.
- **Losses**: Defines the loss functions used during training.
- **Trainer**: Manages the training process, including logging and validation.
- **Inference**: Provides functionality for running inference on new images.

## Acknowledgments

This project builds upon the Vision Transformer architecture and incorporates advanced techniques for part selection and relation encoding to enhance the model's ability to discriminate between twin faces.
