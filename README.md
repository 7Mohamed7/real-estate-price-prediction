# Real Estate Price Prediction

A Machine Learning model built with Python and Scikit-Learn to predict real estate prices based on property attributes such as age, distance to the nearest MRT station, and number of convenience stores.

## Model Performance

- **Algorithm:** Linear Regression
- **Training Accuracy ($R^2$ Score):** 57.63%
- **Testing Accuracy ($R^2$ Score):** 57.97%
- **Mean Absolute Error (MAE):** 5.90
- **Root Mean Squared Error (RMSE):** 8.17

## Repository Structure

- `data/`: Contains the housing dataset (`Real_estate.csv`).
- `src/train.py`: Data preprocessing, model training, evaluation, and saving model artifacts.
- `src/predict.py`: Inference script to predict prices for new property inputs.
- `models/`: Saved `.pkl` files for the model and scaler.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/Real-Estate-Price-Prediction.git](https://github.com/YOUR_USERNAME/Real-Estate-Price-Prediction.git)
   cd Real-Estate-Price-Prediction