import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, json, os

class FeaturePreprocessor:
    def __init__(self):
        os.makedirs("mappings", exist_ok=True)
        os.makedirs("output_datasets", exist_ok=True)

    def clean_and_transform(self, df):
        df = df.copy()

        # ============================================================
        # 1️⃣ Làm sạch cơ bản
        # ============================================================
        df["Gia"] = (
            df["Giá"]
            .astype(str)
            .str.replace(r"[^0-9]", "", regex=True)
            .replace("", np.nan)
            .astype(float)
        )
        df["Nam_dang_ky"] = pd.to_numeric(df["Năm đăng ký"], errors="coerce")
        df["So_Km_da_di"] = pd.to_numeric(df["Số Km đã đi"], errors="coerce")

        df = df.dropna(subset=["Gia"])
        df.fillna({
            "Thương hiệu": "Không rõ",
            "Dòng xe": "Không rõ",
            "Tình trạng": "Không rõ",
            "Loại xe": "Không rõ",
            "Xuất xứ": "Không rõ",
            "Phân khúc dung tích": "Không rõ"
        }, inplace=True)

        # ============================================================
        # 2️⃣ Tạo các feature cơ bản
        # ============================================================
        df["Tuoi_xe"] = 2025 - df["Nam_dang_ky"]
        df["Log_So_Km_da_di"] = np.log1p(df["So_Km_da_di"])
        df["Log_Gia"] = np.log1p(df["Gia"])
        df["Km_moi_nam"] = df["So_Km_da_di"] / (df["Tuoi_xe"] + 0.1)
        df["Gia_tren_km"] = df["Gia"] / (df["So_Km_da_di"] + 1)
        df["Tuoi_xe_x_Km"] = df["Tuoi_xe"] * df["Log_So_Km_da_di"]

        # ============================================================
        # 3️⃣ Mã hóa biến phân loại
        # ============================================================
        cat_cols = ["Thương hiệu","Loại xe","Tình trạng","Xuất xứ","Phân khúc dung tích"]
        for col in cat_cols:
            le = LabelEncoder()
            df[col + "_code"] = le.fit_transform(df[col].astype(str))
            json.dump(
                dict(zip(le.classes_, le.transform(le.classes_))),
                open(f"mappings/{col.lower()}.json", "w", encoding="utf-8"),
                ensure_ascii=False, indent=2
            )

        # ============================================================
        # 4️⃣ Feature tương tác
        # ============================================================
        df["TinhTrang_x_XuatXu"] = df["Tình trạng_code"] * df["Xuất xứ_code"]
        df["LoaiXe_x_PhanKhuc"] = df["Loại xe_code"] * df["Phân khúc dung tích_code"]

        # ============================================================
        # 5️⃣ Mean price features (mới cho v3/v4)
        # ============================================================
        df["Brand_mean_price"] = df["Thương hiệu_code"].map(
            df.groupby("Thương hiệu_code")["Log_Gia"].mean().to_dict()
        )
        df["Dong_mean_price"] = df["Dòng xe"].map(
            df.groupby("Dòng xe")["Log_Gia"].mean().to_dict()
        )
        df["Segment_mean_price"] = df["Phân khúc dung tích_code"].map(
            df.groupby("Phân khúc dung tích_code")["Log_Gia"].mean().to_dict()
        )

        # Fill NaN bằng mean toàn cục
        global_mean = df["Log_Gia"].mean()
        df[["Brand_mean_price", "Dong_mean_price", "Segment_mean_price"]] = (
            df[["Brand_mean_price", "Dong_mean_price", "Segment_mean_price"]]
            .fillna(global_mean)
        )

        # ============================================================
        # 6️⃣ Scale numeric features
        # ============================================================
        scale_cols = ["Tuoi_xe", "Log_So_Km_da_di", "Km_moi_nam", "Tuoi_xe_x_Km"]
        scaler = StandardScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        joblib.dump(scaler, "output_datasets/scaler_XGBoost.pkl")

        # ============================================================
        # 7️⃣ Xuất file kết quả
        # ============================================================
        output_path = "output_datasets/motorbike_user_input_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"💾 Saved cleaned dataset → {output_path}")

        return df
