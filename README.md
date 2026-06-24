# CIFAR-10 Image Classifier

A CNN-based image classifier for the CIFAR-10 dataset, built as part of the
NITSOL Bangladesh Limited Trainee AI Engineer assessment task.

## Project Overview

This project trains a deep learning model to classify images into 10 categories
(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck) using the
CIFAR-10 dataset. It covers the full pipeline: data loading and augmentation,
model training and evaluation, and a simple web app for real-world inference.

## Architecture

A compact CNN with 3 convolutional blocks (Conv-BN-ReLU x2 -> MaxPool each),
followed by Global Average Pooling and a dropout + linear classifier head.
~940K parameters — small enough to train quickly on CPU/MPS while still
achieving strong accuracy on CIFAR-10.

## Results

- **Validation accuracy: 91%** after 30 epochs
- Confusion matrix: see `outputs/confusion_matrix.png`
- The most common confusion pair was **cat/dog** (102 cats misclassified as
  dogs, 65 dogs misclassified as cats), by far the largest source of error in
  the matrix. A secondary pattern appears among vehicle classes, e.g.
  automobile/truck and airplane/ship, where shared silhouette shapes at
  32x32 resolution make fine-grained distinction harder. Both patterns are
  consistent with well-known difficulty patterns on CIFAR-10.

## Project Structure
cifar10-classifier/

├── data/                          # CIFAR-10 dataset (auto-downloaded, gitignored)

├── src/

│   ├── dataset.py                 # Data loading + augmentation pipeline

│   ├── model.py                   # CNN architecture

│   ├── train.py                   # Training loop with checkpointing

│   ├── evaluate.py                # Evaluation: classification report + confusion matrix

│   └── augmentation_preview.py    # Visualizes augmentation effects

├── app/

│   └── streamlit_app.py           # Web app for inference on uploaded images

├── checkpoints/                   # Saved model weights (gitignored)

├── outputs/                       # Generated reports, plots, training history

├── requirements.txt

└── README.md

## Setup Instructions

1. Clone the repository:
```bash
   git clone https://github.com/ratul-podder99/cifar10-classifier.git
   cd cifar10-classifier
```

2. Create and activate a virtual environment:
```bash
   python3 -m venv venv
   source venv/bin/activate
```

3. Install dependencies:
```bash
   pip install -r requirements.txt
```

## Training Instructions

Run training from the `src/` directory or project root (paths are resolved
relative to the project root automatically):

```bash
python src/train.py
```

This will:
- Automatically download the CIFAR-10 dataset on first run (~170MB)
- Train for 30 epochs using Adam optimizer with a step learning rate scheduler
- Use Apple Silicon MPS acceleration if available (falls back to CPU otherwise)
- Save the best-performing checkpoint to `checkpoints/best_model.pt`
- Save training history to `outputs/training_history.json`

To evaluate the trained model and generate a full performance report:

```bash
python src/evaluate.py
```

To visualize the augmentation pipeline:

```bash
python src/augmentation_preview.py
```

## Running the Inference App

```bash
streamlit run app/streamlit_app.py
```

Upload any image, and the app will display the predicted class along with
confidence scores for all 10 categories.

## Data Augmentation Techniques Used

- **Random crop (32x32, padding 4):** simulates translation invariance
- **Random horizontal flip:** most CIFAR-10 classes are flip-invariant
- **Color jitter (brightness/contrast):** improves robustness to lighting variation

See `outputs/augmentation_preview.png` for a visual example.

## Future Improvements

- Experiment with residual connections (ResNet-style) for potentially higher accuracy
- Try mixup/cutmix augmentation
- Hyperparameter tuning via grid/random search on learning rate and weight decay
- Model quantization for faster inference in the Streamlit app