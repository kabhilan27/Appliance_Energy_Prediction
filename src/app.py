import streamlit as st
import numpy as np
from predict import make_prediction

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(
    page_title="Appliance Energy Prediction",
    page_icon=None,
    layout="wide"
)

st.title("Appliance Energy Prediction")
st.markdown(
    "Predict energy consumption for your appliance based on recent readings. "
    "Use the sliders below to input realistic values. All values are automatically scaled for prediction."
)

st.divider()

# ----------------------------
# Feature labels, min/max, default, and tooltips
# ----------------------------
feature_info = {
    "Energy 10 min ago (kWh)": {"min": 10, "max": 1080, "default": 100, "tooltip": "Energy consumption 10 minutes ago"},
    "Energy 30 min ago (kWh)": {"min": 10, "max": 1080, "default": 90, "tooltip": "Energy consumption 30 minutes ago"},
    "Std Dev last 6 readings": {"min": 0, "max": 435, "default": 50, "tooltip": "Standard deviation of the last 6 readings"},
    "Avg last 6 readings": {"min": 26.66, "max": 720, "default": 100, "tooltip": "Average of the last 6 readings"},
    "Current Relative Humidity (%)": {"min": -2.92, "max": 2.89, "default": 0, "tooltip": "Current relative humidity"},
    "Avg last 3 readings": {"min": 16.66, "max": 850, "default": 50, "tooltip": "Average of the last 3 readings"},
    "Std Dev last 3 readings": {"min": 0, "max": 450, "default": 50, "tooltip": "Standard deviation of the last 3 readings"},
    "Temperature Difference (°C)": {"min": -2.32, "max": 1.23, "default": 0, "tooltip": "Temperature difference from previous reading"},
    "Avg last 12 readings": {"min": 32.5, "max": 517.5, "default": 50, "tooltip": "Average of the last 12 readings"},
    "Std Dev last 12 readings": {"min": 2.88, "max": 366.46, "default": 50, "tooltip": "Standard deviation of the last 12 readings"}
}

# ----------------------------
# Layout sliders in two columns
# ----------------------------
cols = st.columns(2)
user_input = []

for i, (label, info) in enumerate(feature_info.items()):
    with cols[i % 2]:
        value = st.slider(
            label,
            min_value=float(info["min"]),
            max_value=float(info["max"]),
            value=float(info["default"]),
            step=0.01,
            help=info["tooltip"]
        )
    user_input.append(value)

st.divider()

# ----------------------------
# Predict button
# ----------------------------
st.markdown("### Prediction")
if st.button("Predict"):
    input_data = np.array([user_input])  # shape: (1, 10)
    try:
        # timesteps=1 for demo
        prediction = make_prediction(input_data, timesteps=1)
        st.success(f"Predicted Energy Consumption: {prediction[0][0]:.2f} kWh")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
