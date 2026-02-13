Appliance Energy Prediction System
==================================

An end-to-end machine learning system to predict appliance energy consumption using historical sensor data. Built with **Python**, **TensorFlow**, **Scikit-learn**, and deployed on **Streamlit Cloud** for interactive use.

**Folder Structure**
--------------------

APPLIANCE_ENERGY_PREDICTION/
│
├── data/
│   ├── raw/                 # Raw input datasets
│   └── processed/           # Cleaned and processed datasets
│
├── models/                  # Trained machine learning models
│   ├── gradient_booster_model.pkl
│   ├── gru_model.keras
│   ├── linear_regression_model.pkl
│   ├── Istm_optimized_model.h5
│   ├── Istm_optimized_model.keras
│   ├── optimized_model.pkl
│   ├── random_forest_model.pkl
│   ├── scaler_X.pkl         # Feature scaler
│   └── scaler_y.pkl         # Target scaler
│
├── notebooks/               # Jupyter notebooks for analysis and modeling
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   └── Model.ipynb
│
├── reports/                 # Analysis reports, figures, and outputs
│
├── src/                     # Source code for Streamlit app and helper scripts
│   ├── __pycache__/
│   ├── app.py               # Main Streamlit app
│   └── predict.py           # Prediction utility functions
│
├── venv/                    # Virtual environment
├── .gitignore               # Git ignore rules
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies


**Features**
------------

*   Predict appliance energy consumption based on environmental and usage features.
    
*   Supports preprocessing, feature scaling, and model inference.
    
*   Interactive web interface using **Streamlit**.
    
*   Multiple models included: Gradient Boosting, GRU, LSTM, Linear Regression, Random Forest.
    
*   Easily extensible for adding new models.
    

**Installation**
----------------

1.  git clone https://github.com//APPLIANCE\_ENERGY\_PREDICTION.gitcd APPLIANCE\_ENERGY\_PREDICTION
    
2.  python -m venv venvsource venv/bin/activate # Linux/Macvenv\\Scripts\\activate # Windows
    
3.  pip install -r requirements.txt
    

**Running the App Locally**
---------------------------

`   streamlit run src/app.py   `

*   Opens a local web interface.
    
*   Users can input appliance/environmental features to get energy consumption predictions.
    
*   Predictions use pre-trained models stored in models/.
    

**Models Included**
-------------------

| Model               | File                         | Description                        |
|--------------------|------------------------------|------------------------------------|
| Gradient Booster    | gradient_booster_model.pkl    | Tree-based ensemble model           |
| GRU                 | gru_model.keras               | Recurrent Neural Network model      |
| LSTM Optimized      | Istm_optimized_model.h5       | Optimized LSTM model                |
| Linear Regression   | linear_regression_model.pkl   | Baseline linear model               |
| Random Forest       | random_forest_model.pkl       | Ensemble tree model                 |
| Feature Scalers     | scaler_X.pkl, scaler_y.pkl    | Input and output normalization      |

**Notebooks**
-------------

*   EDA.ipynb – Exploratory Data Analysis
    
*   Feature\_Engineering.ipynb – Feature creation and preprocessing
    
*   Model.ipynb – Model training and evaluation
    

**Deployment**
--------------

The app is deployed on **Streamlit Cloud**:

🌐 View Live App

*   Users can interact with the model online without any local setup.
    
*   Automatically updates on pushing changes to the main branch.
    

**Dependencies**
----------------

*   pandas, numpy – Data manipulation
    
*   matplotlib, seaborn – Visualization
    
*   scikit-learn – Feature scaling, preprocessing
    
*   tensorflow, torch – Deep learning models
    
*   streamlit==1.22.0 – Web app interface
    
*   protobuf==6.33.5 – TensorFlow compatibility
    
*   joblib, h5py – Model serialization
    

> **Note:** Dependency conflicts (e.g., Streamlit vs Protobuf) may occur. Streamlit Cloud usually handles this automatically.

**Known Issues**
----------------

*   GPU support may fail on Streamlit Cloud; the app defaults to CPU.
    
*   Scikit-learn version mismatch may show warnings when loading pickled models. Safe for testing.
    

**How to Contribute**
---------------------

1.  Fork the repository.
    
2.  Create a feature branch: git checkout -b feature/
    
3.  Commit your changes: git commit -m "Add feature"
    
4.  Push: git push origin feature/
    
5.  Open a pull request.
    

**License**
-----------

This project is licensed under the MIT License – see the LICENSE file for details.