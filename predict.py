import os
import joblib
import pandas as pd


def predict_house_price(sample_features):
    model_path = os.path.join('models', 'linear_regression_model.pkl')
    scaler_path = os.path.join('models', 'scaler.pkl')

    # Check if model files exist
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(
            '[ERROR] Saved model or scaler not found. Run train.py first!'
        )

    # Load saved artifacts
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Scale input and predict
    sample_scaled = scaler.transform(sample_features)
    predicted_price = model.predict(sample_scaled)

    return predicted_price[0]


if __name__ == '__main__':
    # Define features of a new house to predict
    # Columns: ['X1 transaction date', 'X2 house age', 'X3 distance to nearest MRT',
    #           'X4 number of convenience stores', 'X5 latitude', 'X6 longitude']

    new_house_data = pd.DataFrame([{
        'X1 transaction date': 2013.250,
        'X2 house age': 13.3,
        'X3 distance to the nearest MRT station': 561.98450,
        'X4 number of convenience stores': 5,
        'X5 latitude': 24.98746,
        'X6 longitude': 121.54391,
    }])

    predicted = predict_house_price(new_house_data)

    print('=' * 45)
    print('            REAL ESTATE PRICE PREDICTION      ')
    print('=' * 45)
    print(f'Estimated Price per Unit Area: {predicted:.2f}')
    print('=' * 45)