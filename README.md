
# 🌍 Name Predictor by Country

This project is a **Name Predictor** that determines the likely country or origin of a given name using a Recurrent Neural Network (RNN). It provides predictions with adjustable top-N probabilities, allowing users to see the most probable countries for a given name.

## 🚀 Features

- **Predicts the country of origin** for a given name.
- Supports **adjustable top-N predictions** with probabilities.
- **Early stopping** to prevent overfitting during training.
- Utilizes **data augmentation** to handle class imbalance.
- Simple command-line interface for predictions.

## 📂 Project Structure

```plaintext
.
├── models/
│   └── best_model.pt      # Trained model saved here
├── data/
│   ├── names/
│   │   ├── English.txt    # Sample data file
│   │   ├── French.txt
│   │   └── ...            # Additional country data files
├── train.py               # Training script with early stopping
├── predict.py             # Prediction script with adjustable top-N predictions
├── model.py               # RNN model definition
├── data.py                # Data processing and augmentation
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## 📦 Prerequisites

Ensure you have the following installed:

- Python 3.7+
- `pip` (Python package installer)

## 🛠️ Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/name_predictor_from_country.git
   cd name_predictor_from_country
   ```

2. **Install the dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your dataset:**

   - Place your dataset files in the `data/names/` directory. Each file should be named according to the country it represents (e.g., `English.txt`, `French.txt`).
   - Each file should contain one name per line.

## 📊 Training the Model

To train the model, simply run:

```bash
python train.py
```

### Training Options

- The model includes **early stopping** to prevent overfitting.
- **Data augmentation** is applied to balance imbalanced datasets.
- The best model will be saved in the `models/` directory as `best_model.pt`.

#### Example Output:

```bash
5000 5% (0m 5s) 2.7754 Vourlis / Greek ✓
10000 10% (0m 10s) 3.1260 CrespA / Italian ✗ (Portuguese)
...
Early stopping at epoch 37. Best loss: 2.5479
Training complete. Best model saved with loss: 2.5479
```

## 🔮 Making Predictions

To predict the country of origin for a name, use the `predict.py` script.

### Usage:

```bash
python predict.py <name> [num_predictions]
```

- `<name>`: The name you want to predict the country for.
- `[num_predictions]` (optional): Number of top predictions to show (default is 3).

### Example:

```bash
python predict.py anitha 5
```

#### Example Output:

```bash
Top 5 predictions for the name 'anitha':
1: Indian with probability 0.4597
2: Japanese with probability 0.1845
3: Portuguese with probability 0.1139
4: Arabic with probability 0.0767
5: Spanish with probability 0.0734
```
