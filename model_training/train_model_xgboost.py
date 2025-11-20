import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# 1️⃣ Setup path
# ============================================================
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "output_datasets"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Ưu tiên file v2, fallback sang clean
if os.path.exists(os.path.join(OUTPUT_DIR, "motorbike_final_dataset_v2.csv")):
    DATA_PATH = os.path.join(OUTPUT_DIR, "motorbike_final_dataset_v2.csv")
else:
    DATA_PATH = os.path.join(OUTPUT_DIR, "motorbike_final_dataset_clean.csv")

print("📄 DATA_PATH:", DATA_PATH)

# ============================================================
# 2️⃣ Load data
# ============================================================
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ============================================================
# 3️⃣ Tự động khôi phục feature nếu thiếu
# ============================================================
# Xác định cột số km, có thể là "So_Km_da_di" hoặc "Số Km đã đi"
km_col = next((c for c in df.columns if "Km" in c and "đi" in c), None)

# Log_Gia & Log_So_Km_da_di phải tồn tại
if "Log_Gia" not in df.columns:
    if "Gia" in df.columns:
        df["Log_Gia"] = np.log1p(df["Gia"])
    else:
        raise KeyError("⚠️ Dataset thiếu cả 'Log_Gia' và 'Gia' — không thể train.")

if "Log_So_Km_da_di" not in df.columns:
    if km_col:
        df["Log_So_Km_da_di"] = np.log1p(df[km_col])
    else:
        raise KeyError("⚠️ Không tìm thấy cột số km đã đi.")

# Tự tính các feature còn thiếu
if "Gia_tren_km" not in df.columns:
    df["Gia_tren_km"] = np.expm1(df["Log_Gia"]) / (np.expm1(df["Log_So_Km_da_di"]) + 1)
if "Tuoi_xe_x_Km" not in df.columns:
    df["Tuoi_xe_x_Km"] = df["Tuoi_xe"] * df["Log_So_Km_da_di"]
if "Km_moi_nam" not in df.columns:
    df["Km_moi_nam"] = np.expm1(df["Log_So_Km_da_di"]) / (df["Tuoi_xe"] + 0.1)

print("✅ Features ensured:", [c for c in ["Gia_tren_km","Tuoi_xe_x_Km","Km_moi_nam"] if c in df.columns])

# ============================================================
# 4️⃣ Train-test split
# ============================================================
FEATURES = [
    "Tuoi_xe","Log_So_Km_da_di","Km_moi_nam","Gia_tren_km","Tuoi_xe_x_Km",
    "TinhTrang_x_XuatXu","LoaiXe_x_PhanKhuc","Thuong_hieu_code",
    "Loai_xe_code","Tinh_trang_code","Xuat_xu_code","Phan_khuc_dung_tich_code"
]
TARGET = "Log_Gia"

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise KeyError(f"⚠️ Dataset thiếu các cột cần thiết: {missing}")

X, y = df[FEATURES], df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📊 Train: {X_train.shape}, Test: {X_test.shape}")

# ============================================================
# 5️⃣ Train model
# ============================================================
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_lambda=1.2,
    n_jobs=-1,
    random_state=42
)
print("🔧 Training model...")
model.fit(X_train, y_train)
print("✅ Training complete!")

# ============================================================
# 6️⃣ Evaluate
# ============================================================
y_pred = model.predict(X_test)
y_true_vnd, y_pred_vnd = np.expm1(y_test), np.expm1(y_pred)

mae = mean_absolute_error(y_true_vnd, y_pred_vnd)
rmse = np.sqrt(mean_squared_error(y_true_vnd, y_pred_vnd))
r2 = r2_score(y_true_vnd, y_pred_vnd)

print("\n📈 Model Performance:")
print(f"   MAE  = {mae:,.0f} VND")
print(f"   RMSE = {rmse:,.0f} VND")
print(f"   R²   = {r2:.3f}")

# ============================================================
# 7️⃣ Save model
# ============================================================
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_XGBoost.pkl")
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved → {MODEL_PATH}")
