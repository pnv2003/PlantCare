# Plant Disease Detection using Deep Learning

## Overview
This project implements a deep learning pipeline for detecting plant diseases using the PlantVillage dataset. It leverages PyTorch for training and testing convolutional neural networks (CNNs) on plant leaf images, providing accurate classification of various plant diseases.

## Features
- Supports multiple pre-trained models, including MobileNetV2, SqueezeNet 1.1, EfficientNet-B0, and ShuffleNet V2.
- Implements transfer learning with optional freezing of model layers.
- Early stopping and learning rate scheduling to optimize training.
- Evaluation metrics: Accuracy, Precision, Recall, and F1 Score.
- Visualization of training statistics (loss and accuracy).

## Requirements
- Python 3.7+
- PyTorch 1.10+
- torchvision
- NumPy
- Matplotlib
- argparse
- PlantVillage dataset (downloaded automatically)

## Setup
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure GPU support by installing the appropriate version of PyTorch for your system from [pytorch.org](https://pytorch.org/).

## Usage
### Commands
The script provides two primary commands:
- `train`: Train a model on the PlantVillage dataset.
- `test`: Evaluate a trained model on the test dataset.

### Training
Run the following command to train a model:
```bash
python main.py train --model <model_name> --freeze
```
#### Parameters
- `--model`: Specify the model architecture (e.g., `mobilenet_v2`, `squeezenet1_1`, `efficientnet_b0`, `shufflenet_v2`).
- `--freeze`: Optional flag to freeze the pre-trained model's layers.

Example:
```bash
python main.py train --model mobilenet_v2 --freeze
```

### Testing
To evaluate a trained model, run:
```bash
python main.py test --model <model_name> --freeze
```
#### Parameters
- `--model`: Specify the model architecture used during training (e.g., `mobilenet_v2`, `squeezenet1_1`, `efficientnet_b0`, `shufflenet_v2`).
- `--freeze`: Optional flag to freeze the model's layers.

Example:
```bash
python main.py test --model mobilenet_v2 --freeze
```

### Outputs
- Trained model weights are saved in the `weights/` directory.
- Training statistics (loss and accuracy) are visualized after training.
- Evaluation metrics (Accuracy, Precision, Recall, F1 Score) are printed during testing.

## Project Structure
```
.
├── main.py              # Entry point for training and testing
├── data.py              # Dataset preparation and utilities
├── model.py             # Model architecture definitions
├── train.py             # Training logic
├── eval.py              # Evaluation and prediction utilities
├── vis.py               # Visualization utilities
├── weights/             # Directory for saving model weights
└── requirements.txt     # Required Python packages
```

## Key Hyperparameters
- **Batch size**: 64
- **Image size**: 224x224
- **Epochs**: 50
- **Learning rates**:
  - Fine-tuning: 0.0001
  - Standard training: 0.001
- **Weight decay**: 0.0001
- **Early stopping**:
  - Patience: 5 epochs
  - Delta: 0.001
- **Learning rate scheduler**:
  - Type: ReduceLROnPlateau
  - Patience: 3 epochs
  - Factor: 0.1

## Notes
- Ensure the `weights/<model_name>.pth` file exists before testing. If not, train the model first.
- The PlantVillage dataset is automatically downloaded and prepared by the script.

## References
- [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
- [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946)
- [ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design](https://arxiv.org/abs/1807.11164)
- [Transfer Learning with PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgments
- The creators of the PlantVillage dataset for their valuable contributions to plant disease research.
- The PyTorch community for providing extensive documentation and tools.

