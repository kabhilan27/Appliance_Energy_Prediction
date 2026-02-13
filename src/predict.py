import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# ----------------------------
# Paths to models and scalers
# ----------------------------
current_dir = os.path.dirname(os.path.abspath(__file__))  # src folder
models_dir = os.path.join(current_dir, "../models")

model_path = os.path.join(models_dir, "lstm_optimized_model.keras")
scaler_X_path = os.path.join(models_dir, "scaler_X.pkl")
scaler_y_path = os.path.join(models_dir, "scaler_y.pkl")

# Check files exist
for path in [model_path, scaler_X_path, scaler_y_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

# Load model and scalers
model = load_model(model_path)
scaler_X = joblib.load(scaler_X_path)
scaler_y = joblib.load(scaler_y_path)

# ----------------------------
# Prediction function
# ----------------------------
def make_prediction(input_data, timesteps=1):
    """
    input_data: 2D array (1 sample, 10 features)
    timesteps: number of timesteps for LSTM
    Returns: predicted energy consumption (non-negative)
    """
    # Scale input
    input_scaled = scaler_X.transform(input_data)  # shape: (1, 10)

    # Repeat input across timesteps if needed
    if timesteps == 1:
        input_scaled_3d = np.reshape(input_scaled, (1, 1, 10))
    else:
        input_scaled_3d = np.repeat(input_scaled[:, np.newaxis, :], timesteps, axis=1)

    # Predict
    pred_scaled = model.predict(input_scaled_3d)

    # Inverse scale
    pred = scaler_y.inverse_transform(pred_scaled)

    # Clip negative predictions to zero
    pred = np.maximum(pred, 0)
    return pred
