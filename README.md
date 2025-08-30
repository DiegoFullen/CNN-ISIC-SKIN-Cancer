# Convolutional Neuronal Network
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

## Project Strucure
/skin-cancer-cnn-isic/
  │
  ├── datasets/             # ISIC dataset (not included in repo, link provided)
  ├── preprocessing/        # Image preprocessing and augmentation scripts
  ├── training/             # CNN model definition and training scripts
  ├── evaluation/           # Performance evaluation and visualization
  ├── models/               # Saved trained models (.h5)
  ├── main.py               # Entry point for training/testing pipeline
  ├── requirements.txt      # Dependencies
  └── README.md

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

Technical Stack
Code libraries used:
- Deep Learning: Tensorflow, Keras
- Data Processing: pandas, numpy, scikit-learn
- Visualization: matplotlib, seaborn

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
