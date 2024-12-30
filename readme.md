# ResNet-50 Training on Tiny ImageNet

This project demonstrates training a ResNet-50 model from scratch on the Tiny ImageNet dataset. It includes model creation, training, evaluation, saving, and deploying a demo using Gradio.

---

## Features
- **Data Loading**: Loads and preprocesses the Tiny ImageNet dataset using Hugging Face's `datasets` library.
- **Custom ResNet-50 Model**: Modifies the final fully connected (FC) layer for the required number of classes.
- **Training Pipeline**: Includes configurable hyperparameters for batch size, learning rate, and epochs.
- **Model Saving**: Saves the trained model for future use.
- **Hugging Face Integration**: Uploads the trained model to the Hugging Face Model Hub.
- **Gradio Demo**: Interactive Gradio interface for real-time predictions.

---

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- pip

Install required libraries using:
```bash
pip install -r requirements.txt
```

---

## Directory Structure
```plaintext
project/
├── train.py          # Script for training the model
├── model.py          # Script for defining the model
├── data.py           # Script for loading the dataset
├── app.py    # Script for running Gradio demo
├── utils.py          # Utility functions
├── config.py         # Configuration file
├── requirements.txt  # List of dependencies
└── README.md         # Project documentation
```

---

## Configuration
Modify the `config.py` file to update hyperparameters and paths:
```python
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "resnet50_tinyimagenet.pt"
HF_MODEL_REPO = "resnet50-tinyimagenet"
```

---

## Steps to Run

### 1. **Train the Model**
Train the ResNet-50 model on Tiny ImageNet:
```bash
python train.py
```

### 2. **Save the Model**
The trained model will be saved to the path specified in `config.py` (default: `resnet50_tinyimagenet.pt`).

### 3. **Upload to Hugging Face**
(Optional) Upload the model to the Hugging Face Model Hub:
```python
from huggingface_hub import HfApi

api = HfApi()
api.upload_file(
    path_or_fileobj="resnet50_tinyimagenet.pt",
    path_in_repo="resnet50_tinyimagenet.pt",
    repo_id=f"<YOUR_USERNAME>/resnet50-tinyimagenet",
    repo_type="model",
    commit_message="Upload trained ResNet-50"
)
```

### 4. **Run Gradio Demo**
Launch the Gradio interface for real-time image classification:
```bash
python app.py
```
Access the demo at the URL provided in the terminal.

---

## Tiny ImageNet Dataset
The Tiny ImageNet dataset contains:
- 200 classes
- 500 training images per class
- 50 validation images per class

This dataset is automatically downloaded and preprocessed using Hugging Face's `datasets` library.

---

## Gradio Demo
The Gradio demo allows you to:
1. Upload an image.
2. Get the predicted class.

Launch the demo using `app.py`.

---

## Requirements
See `requirements.txt` for all dependencies:
```plaintext
torch>=2.0.0
torchvision>=0.15.0
datasets>=2.13.0
huggingface-hub>=0.16.4
gradio>=3.30.0
matplotlib>=3.8.0
```

Install them using:
```bash
pip install -r requirements.txt
```

---

## Acknowledgments
- [PyTorch](https://pytorch.org/)
- [Hugging Face](https://huggingface.co/)
- [Gradio](https://gradio.app/)
- [Tiny ImageNet Dataset](https://www.kaggle.com/c/tiny-imagenet)

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.


Happy coding!

