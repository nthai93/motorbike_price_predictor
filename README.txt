🏍️ MOTORBIKE PRICE PREDICTOR – PANDORA STYLE
====================================================

A Streamlit-based web application for predicting used motorbike prices in Vietnam,
powered by XGBoost and real market data (2020–2024).


🚀 FEATURES
----------------------------------------------------
- Predict reasonable selling prices based on 15+ key features:
  • Age, mileage, brand, model, origin, condition, engine segment, region, etc.
- Automatic insight & recommendation engine:
  • Detects overpriced or underpriced listings.
  • Suggests fair price range for quick sales.
- Brand-level market visualization (Histogram + Boxplot)
- Built-in Model Evaluation Dashboard (MAE, RMSE, R²)
- Clean UI styled in Pandora brand color 💙


🧩 PROJECT STRUCTURE
----------------------------------------------------
motorbike_price_predictor/
│
├── app.py                     → Main Streamlit dashboard
├── requirements.txt            → Dependencies
├── README.md                   → Project documentation
│
├── output_datasets/            → Model artifacts & cleaned dataset
│   ├── best_model_XGBoost.pkl
│   ├── scaler_XGBoost.pkl
│   ├── motorbike_final_dataset_clean.csv
│
├── mappings/                   → Encoded label mappings (JSON)
│   ├── thuong_hieu.json
│   ├── dong_xe.json
│   ├── tinh_trang.json
│   ├── xuat_xu.json
│   ├── phan_khuc_dung_tich.json
│   ├── Vung_mien.json
│
├── model_training/             → Training scripts & notebooks
│   ├── train_model_xgboost.py
│   ├── anomaly_detector.py
│
└── processor/                  → Data preprocessing module
    └── feature_preprocessor.py


⚙️ INSTALLATION & LOCAL RUN
----------------------------------------------------
1️⃣ Clone the repository
   git clone https://github.com/nthai93/motorbike_price_predictor.git
   cd motorbike_price_predictor

2️⃣ Install dependencies
   pip install -r requirements.txt

3️⃣ Run the Streamlit dashboard
   streamlit run app.py

Then open http://localhost:8501/


☁️ DEPLOYMENT (OPTIONAL)
----------------------------------------------------
You can deploy seamlessly to:
- Hugging Face Spaces
- Streamlit Cloud
- Render.com

All dependencies are already defined in requirements.txt.
No environment variables required.


📊 MODEL INFORMATION
----------------------------------------------------
Algorithm: XGBoost Regressor
---------------------------------
MAE:   ≈ 6.5M VND
RMSE:  ≈ 16.4M VND
R²(Log): ≈ 0.81

Dataset: 7,000+ verified listings (2020–2024)
Preprocessing: log-transform, scaling, mean-price features, anomaly filtering


🖥️ TECHNOLOGY STACK
----------------------------------------------------
- Python 3.11+
- Streamlit 1.51+
- XGBoost, Scikit-learn, Pandas, NumPy
- Plotly for visualization
- Joblib for model serialization


🧠 MODULES OVERVIEW
----------------------------------------------------
Module                          Purpose
----------------------------------------------------
processor/feature_preprocessor  Clean & encode input features
model_training/train_model_xgboost   Train XGBoost regression model
model_training/anomaly_detector      Detect price anomalies
app.py                         Streamlit UI + prediction logic
mappings/                      Encoded label maps
output_datasets/               Saved model + scaler + dataset


👨‍💻 AUTHOR
----------------------------------------------------
Nguyễn Thanh Hải 
Contact: nthai93  
Location: Vietnam  
Note: Project for educational & research purposes.


🪪 LICENSE
----------------------------------------------------
MIT License – Free for use and modification with attribution.

⭐ If you find this useful, please star the repository!
