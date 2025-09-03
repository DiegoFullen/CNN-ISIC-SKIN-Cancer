# Convolutional Neural Network
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Keras](https://img.shields.io/badge/Keras-3.0-red)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.4-green)
![NumPy](https://img.shields.io/badge/NumPy-1.26-lightblue)
![Pandas](https://img.shields.io/badge/Pandas-2.2-purple)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8-yellow)
![Seaborn](https://img.shields.io/badge/Seaborn-0.13-teal)

A deep learning system for **skin cancer detection using a Convolutional Neuronal Network (CNN)** trained on the ISIC dataset of dermoscopic images. The goal is to classify skin lesions and support early diagnosis.

## Key Features
- **CNN-based image classification** for skin lesion detection
- **Preprocessing pipeline** (resizing, normalization, augmentation)
- **Custom-trained CNN model** with performance metrics
- **Evaluation with accuracy, precision, recall, and F1-score**
- **Visualization tools** (loss/accuracy curves, confusion matrix)

## Project Structure
```
/skin-cancer-cnn-isic/
├── CNN/
│   └── CNN.py                # Configuration and architecture of CNN
├── Datasets/                 # CSVs with metadata of training and testing
├── Graphics/                 # Training curves (accuracy/loss), confusion matrix, model predictions (menu option 3)
├── Test/                     # Test images (separate dataset)
├── Training/                 # Training images
├── Utils/
│   ├── Test_No_GPU.py        # Check if GPU is available
│   ├── Use.py                # Use a trained model on new images
│   └── evaluation.py         # Generates the graphics in Graphics/
├── Config/
│   ├── data.json             # Version control (current model metadata)
│   └── Settings.py           # Handling JSON and global configuration
├── main.py                   # Main menu (iteration, downgrade, prediction, WIP metrics)
├── requirements.txt          # Dependencies
└── README.md
```

## How to Run
### 1. Clone repository
```bash
git clone https://github.com/DiegoFullen/CNN-ISIC-SKIN-Cancer.git
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the system
```bash
python main.py
```

## Dataset
This project uses the **ISIC Archive** (International Skin Imaging Collaboration): https://www.isic-archive.com/

Classes may include (depending on selected subset):
- Melanoma
- Nevus
- Basal Cell Carcinoma
- Benign Keratosis
- Dermatofibroma
- Vascular Lesion

## Technical Stack
Code libraries used:
- **Deep Learning**: Tensorflow, Keras
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn

## Notes
- Dataset is not included due to size; must be downloaded from ISIC.
- Model results depend on chosen CNN architecture and preprocessing.
- For reproducibility, all versions are pinned in requirements.txt.

## Project Context
Academic project developed for **Deep Learning course - CETI Colomos**.
Not intended for clinical or commercial use.

## Future Roadmap
- Improve accuracy
- Fine-Tune the model

## Author
Diego Salvador Candia Fullen
