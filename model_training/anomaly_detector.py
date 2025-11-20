import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import r2_score, mean_squared_error
import joblib, os
import seaborn as sns
import matplotlib.pyplot as plt

# ============================================================
# 1️⃣ LOAD DATA & MODEL
# ============================================================
print("🚀 Loading dataset and model...")

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "output_datasets"))
DATA_PATH = os.path.join(OUTPUT_DIR, "motorbike_final_dataset_clean.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model_XGBoost.pkl")
SCALER_PATH = os.path.join(OUTPUT_DIR, "scaler_XGBoost.pkl")

print("📄 DATA_PATH:", DATA_PATH)
print("📄 MODEL_PATH:", MODEL_PATH)

df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

print(f"✅ Dataset shape: {df.shape}")
print(f"✅ Model loaded: {type(model).__name__}")

# --- Clean dữ liệu lỗi Log_Gia ---
before = df.shape[0]
df = df[df["Log_Gia"].notna() & (df["Log_Gia"] > 0)]
after = df.shape[0]
print(f"🧹 Removed {before - after} invalid rows → {after} rows remaining")

if "Dong_xe_code" not in df.columns:
    raise KeyError("❌ Missing 'Dong_xe_code' in CSV. Must use motorbike_final_dataset_clean.csv")
else:
    print("✅ Found Dong_xe_code in dataset")

# ============================================================
# 2️⃣ TẠO FEATURE GIỐNG FILE TRAIN
# ============================================================
FEATURES = [
    "Tuoi_xe", "Log_So_Km_da_di", "Km_moi_nam", "Gia_tren_km", "Tuoi_xe_x_Km",
    "TinhTrang_x_XuatXu", "LoaiXe_x_PhanKhuc",
    "Thuong_hieu_code", "Dong_xe_code", "Loai_xe_code",
    "Tinh_trang_code", "Xuat_xu_code", "Phan_khuc_dung_tich_code"
]

for col in ["Gia_tren_km", "Tuoi_xe_x_Km", "Km_moi_nam"]:
    if col not in df.columns:
        print(f"⚙️ Recomputing {col} ...")
        if col == "Gia_tren_km":
            df["Gia_tren_km"] = np.expm1(df["Log_Gia"]) / (np.expm1(df["Log_So_Km_da_di"]) + 1)
        elif col == "Tuoi_xe_x_Km":
            df["Tuoi_xe_x_Km"] = df["Tuoi_xe"] * df["Log_So_Km_da_di"]
        elif col == "Km_moi_nam":
            df["Km_moi_nam"] = np.expm1(df["Log_So_Km_da_di"]) / (df["Tuoi_xe"] + 0.1)

df_model = df[FEATURES].copy()

# === Scale đúng như lúc train ===
df_model[["Km_moi_nam", "Tuoi_xe_x_Km"]] = scaler.transform(df_model[["Km_moi_nam", "Tuoi_xe_x_Km"]])
print("✅ Scaler applied for Km_moi_nam & Tuoi_xe_x_Km")

# ============================================================
# ============================================================
# 3️⃣ DỰ ĐOÁN VÀ ĐÁNH GIÁ LẠI MODEL
# ============================================================
print("\n🔍 Model expects:", model.get_booster().feature_names)

# Dự đoán log-price
y_pred_log = model.predict(df_model.astype(np.float32))
df["Predicted_Log_Gia"] = y_pred_log

# Chuyển log → giá thực
df["Gia_du_doan"] = np.expm1(y_pred_log)
df["Gia_thuc_te"] = np.expm1(df["Log_Gia"])

# Tính phần dư và độ lệch %
df["Residual"] = df["Log_Gia"] - df["Predicted_Log_Gia"]
df["Do_lech_%"] = (df["Gia_thuc_te"] - df["Gia_du_doan"]) / df["Gia_du_doan"] * 100

# --- Đánh giá ĐÃ SỬA ---

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 1️⃣ R² trên log-scale
r2_log = r2_score(df["Log_Gia"], df["Predicted_Log_Gia"])

# 2️⃣ Lọc dữ liệu hợp lệ (loại giá 0, NaN, quá lớn)
mask = (df["Gia_thuc_te"] > 0) & (df["Gia_thuc_te"] < 5e8) & df["Gia_du_doan"].notna()
df_eval = df[mask]

# 3️⃣ RMSE & MAE trên giá thực tế
rmse_vnd = np.sqrt(mean_squared_error(df_eval["Gia_thuc_te"], df_eval["Gia_du_doan"]))
mae_vnd = mean_absolute_error(df_eval["Gia_thuc_te"], df_eval["Gia_du_doan"])

# 4️⃣ R² trên giá thực tế (VNĐ)
r2_vnd = r2_score(df_eval["Gia_thuc_te"], df_eval["Gia_du_doan"])

print(f"📈 R² (Log-Scale) = {r2_log:.3f}")
print(f"📈 R² (VNĐ-Scale) = {r2_vnd:.3f}")
print(f"📊 MAE  = {mae_vnd/1e6:.1f} triệu VND")
print(f"📊 RMSE = {rmse_vnd/1e6:.1f} triệu VND")





# ============================================================
# 4️⃣ CHUẨN HÓA PHẦN DƯ THEO PHÂN KHÚC
# ============================================================
df["Residual_z"] = df.groupby("Phan_khuc_dung_tich_code")["Residual"].transform(
    lambda x: (x - x.mean()) / (x.std() + 1e-6)
)

# ============================================================
# 5️⃣ QUY TẮC GIÁ P10–P90
# ============================================================
bounds = (
    df.groupby("Phan_khuc_dung_tich_code")["Gia_thuc_te"]
    .quantile([0.1, 0.9])
    .unstack()
    .rename(columns={0.1: "P10", 0.9: "P90"})
)
df = df.merge(bounds, on="Phan_khuc_dung_tich_code", how="left")
df["Rule_P10P90"] = (df["Gia_thuc_te"] < df["P10"]) | (df["Gia_thuc_te"] > df["P90"])

# ============================================================
# 6️⃣ ISOLATION FOREST + ONE-CLASS SVM
# ============================================================
features_unsup = [
    "Tuoi_xe", "Log_So_Km_da_di", "Residual_z",
    "Thuong_hieu_code", "Loai_xe_code", "Tinh_trang_code",
    "Xuat_xu_code", "Phan_khuc_dung_tich_code"
]
X_unsup = df[features_unsup].fillna(0)
X_scaled = StandardScaler().fit_transform(X_unsup)

iso = IsolationForest(contamination=0.05, random_state=42)
df["IsoScore"] = -iso.fit(X_scaled).decision_function(X_scaled)

svm = OneClassSVM(kernel="rbf", nu=0.05)
df["SvmScore"] = -svm.fit(X_scaled).score_samples(X_scaled)

print("✅ Isolation Forest & OneClassSVM finished")

# ============================================================
# 7️⃣ ANOMALY SCORE
# ============================================================
weights = {"Residual_z": 0.4, "IsoScore": 0.3, "SvmScore": 0.2, "Rule_P10P90": 0.1}
combined = (
    weights["Residual_z"] * np.abs(df["Residual_z"]) +
    weights["IsoScore"] * df["IsoScore"] +
    weights["SvmScore"] * df["SvmScore"] +
    weights["Rule_P10P90"] * df["Rule_P10P90"].astype(int)
)

df["AnomalyScore"] = MinMaxScaler((0, 100)).fit_transform(combined.values.reshape(-1, 1))
threshold = df["AnomalyScore"].quantile(0.95)
df["Anomaly_Flag"] = (df["AnomalyScore"] >= threshold).astype(int)
print(f"🚨 {df['Anomaly_Flag'].sum()} xe có giá bất thường (top 5%)")

# ============================================================
# 8️⃣ VISUALIZATION
# ============================================================
sns.set(style="whitegrid", font="Arial")
df["Trạng_thái"] = df["Anomaly_Flag"].map({1: "Bất thường", 0: "Bình thường"})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
(ax1, ax2), (ax3, ax4) = axes

sns.histplot(df["AnomalyScore"], bins=40, kde=True, color="skyblue", ax=ax1)
ax1.axvline(threshold, color="red", linestyle="--", label="Top 5%")
ax1.set_title("Phân phối điểm bất thường", fontsize=12, weight="bold")
ax1.legend()

sns.scatterplot(
    data=df,
    x=np.log1p(df["Gia_du_doan"]),
    y=np.log1p(df["Gia_thuc_te"]),
    hue="Trạng_thái",
    palette={"Bất thường": "red", "Bình thường": "blue"},
    alpha=0.6, s=40, ax=ax2
)
ax2.plot(
    [np.log1p(df["Gia_du_doan"]).min(), np.log1p(df["Gia_du_doan"]).max()],
    [np.log1p(df["Gia_du_doan"]).min(), np.log1p(df["Gia_du_doan"]).max()],
    color="green", linestyle="--", linewidth=1.3
)
ax2.set_xlabel("log(Giá dự đoán)")
ax2.set_ylabel("log(Giá thực tế)")
ax2.set_title("Giá thực tế vs Giá dự đoán (log-scale)", fontsize=12, weight="bold")

sns.boxplot(data=df, x="Thuong_hieu_code", y="AnomalyScore", palette="Set2", ax=ax3)
ax3.axhline(threshold, color='r', linestyle='--', label="Top 5%")
ax3.set_title("Phân bố điểm bất thường theo thương hiệu", fontsize=12, weight="bold")
ax3.tick_params(axis='x', rotation=45)
ax3.legend()

ax4.axis('off')
plt.tight_layout()
plt.show()

# ============================================================
# 9️⃣ SAVE RESULT
# ============================================================
out_path = os.path.join(OUTPUT_DIR, "motorbike_anomaly_detected.csv")
df.to_csv(out_path, index=False)
print(f"💾 Saved anomaly detection result → {out_path}")
