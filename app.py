# ============================================================
# 🏍️ MOTORBIKE PRICE PREDICTOR DASHBOARD (Stable v4)
# ============================================================
# Author: Hai Nguyen
# Version: v4 (Model R²≈0.84 – 16 features)
# Description:
#   - Streamlit dashboard for motorbike price prediction.
#   - Model: best_model_XGBoost.pkl
#   - Scaler: scaler_XGBoost.pkl
#   - Compatible with motorbike_final_dataset_clean.csv
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json, os, warnings
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# ============================================================
# 🧭 PAGE CONFIGURATION & THEME
# ============================================================
st.set_page_config(
    page_title="🏍️ Motorbike Price Predictor 🏍️",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS (Pandora style sidebar and theme) ---
st.markdown("""
<style>
body {
    background-color: #F7F9FB;
    color: #1B1E23;
    font-family: "Segoe UI", sans-serif;
}
h1, h2, h3, h4 {
    font-family: 'Segoe UI Semibold', sans-serif;
    color: #004AAD;
}
aside[data-testid="stSidebar"] {
    background-color: #E9F1FB !important;
    color: #003366 !important;
    font-weight: 500;
    width: 270px !important;
    min-width: 270px !important;
    border-right: 1px solid #d0d9e6;
}
.main .block-container {
    padding: 2rem 4rem !important;
}
div.stButton > button:first-child {
    background-color: #004AAD;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
div.stButton > button:first-child:hover {
    background-color: #0077FF;
}
[data-testid="stMetricValue"] {
    color: #004AAD;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 🧭 SIDEBAR NAVIGATION
# ============================================================
st.title("🏍️ Motorbike Price Prediction Dashboard")

menu = st.sidebar.radio(
    "🧭 Menu",
    [
        "🏁 Business Overview",
        "📊 Model Evaluation",
        "🧮 New Prediction",
        "📈 Market Analysis"
    ],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.caption("💡 Bấm mũi tên **< / >** góc trái để ẩn hoặc hiện menu.")

# ============================================================
# 1️⃣ LOAD MODEL, SCALER, AND DATA
# ============================================================

@st.cache_resource
def load_model_and_scaler():
    """Load trained XGBoost model and StandardScaler."""
    model = joblib.load("output_datasets/best_model_XGBoost.pkl")
    scaler = joblib.load("output_datasets/scaler_XGBoost.pkl")
    return model, scaler


@st.cache_resource
def load_mappings():
    """Load label encoding mappings from JSON files."""
    mappings = {}
    for name in ["thuong_hieu", "loai_xe", "tinh_trang", "xuat_xu", "dong_xe", "phan_khuc_dung_tich"]:
        with open(f"mappings/{name}.json", "r", encoding="utf-8") as f:
            mappings[name] = json.load(f)
    return mappings


@st.cache_data
def load_reference_data():
    """Load and preprocess reference dataset used for model evaluation."""
    df = pd.read_csv("output_datasets/motorbike_final_dataset_clean.csv")

    # --- Feature reconstruction (to ensure consistency) ---
    if "Log_Gia" not in df.columns and "Gia" in df.columns:
        df["Log_Gia"] = np.log1p(df["Gia"])
    if "Log_So_Km_da_di" not in df.columns and "So_Km_da_di" in df.columns:
        df["Log_So_Km_da_di"] = np.log1p(df["So_Km_da_di"])

    df["Gia_tren_km"] = np.expm1(df["Log_Gia"]) / (np.expm1(df["Log_So_Km_da_di"]) + 1)
    df["Tuoi_xe_x_Km"] = df["Tuoi_xe"] * df["Log_So_Km_da_di"]
    df["Km_moi_nam"] = np.expm1(df["Log_So_Km_da_di"]) / (df["Tuoi_xe"] + 0.1)
    df["Gia_thuc"] = np.expm1(df["Log_Gia"])

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=["Log_Gia", "Log_So_Km_da_di", "Tuoi_xe"], inplace=True)

    # --- Mean log price features ---
    df["Brand_mean_price"] = df["Thuong_hieu_code"].map(df.groupby("Thuong_hieu_code")["Log_Gia"].mean().to_dict())
    df["Dong_mean_price"] = df["Dong_xe_code"].map(df.groupby("Dong_xe_code")["Log_Gia"].mean().to_dict())
    df["Segment_mean_price"] = df["Phan_khuc_dung_tich_code"].map(df.groupby("Phan_khuc_dung_tich_code")["Log_Gia"].mean().to_dict())

    global_mean = df["Log_Gia"].mean()
    df[["Brand_mean_price", "Dong_mean_price", "Segment_mean_price"]] = (
        df[["Brand_mean_price", "Dong_mean_price", "Segment_mean_price"]].fillna(global_mean)
    )
    return df

# ============================================================
# 2️⃣ PREDICTION FUNCTION
# ============================================================

def predict_price(user_input_df, model, scaler):
    """Predict market price (VNĐ) for given user input."""
    FEATURES = [
        "Tuoi_xe", "Log_So_Km_da_di", "Km_moi_nam", "Tuoi_xe_x_Km",
        "TinhTrang_x_XuatXu", "LoaiXe_x_PhanKhuc",
        "Thuong_hieu_code", "Dong_xe_code", "Loai_xe_code",
        "Tinh_trang_code", "Xuat_xu_code", "Phan_khuc_dung_tich_code",
        "Vung_mien_code", "Brand_mean_price", "Dong_mean_price", "Segment_mean_price"
    ]
    scale_cols = ["Tuoi_xe", "Log_So_Km_da_di", "Km_moi_nam", "Tuoi_xe_x_Km"]

    X = user_input_df.copy()
    X[scale_cols] = scaler.transform(X[scale_cols])
    y_pred = model.predict(X[FEATURES].astype(np.float32))
    return float(np.expm1(y_pred[0]))

# --- Load model, scaler, and mappings ---
model, scaler = load_model_and_scaler()
mappings = load_mappings()
df_ref = load_reference_data()

# ============================================================
# 3️⃣ TAB 1: BUSINESS OVERVIEW
# ============================================================
if menu == "🏁 Business Overview":
    st.header("🎯 Business Problem & App Purpose")
    st.markdown("""
    Ứng dụng này giúp **dự đoán giá hợp lý của xe máy cũ tại Việt Nam**,
    hỗ trợ người **bán** đặt giá tối ưu và người **mua** tránh mua hớ.

    **Mục tiêu chính:**
    - Xây dựng mô hình học máy (XGBoost) dự đoán giá từ 16 đặc trưng.
    - Gợi ý giá đăng bán hợp lý so với thị trường thực tế.
    - Phát hiện giá **bất thường** (cao hoặc thấp bất hợp lý).

    **Công nghệ:**
    - Python (Pandas, NumPy, XGBoost, Plotly)
    - Streamlit UI đơn giản, thân thiện.
    - Dữ liệu: >7.000 xe giai đoạn 2020–2024
    """)

# ============================================================
# 4️⃣ TAB 2: MODEL EVALUATION
# ============================================================
elif menu == "📊 Model Evaluation":
    st.header("📈 Model Evaluation & Diagnostics")

    # --- Feature & dataset setup ---
    df_eval = df_ref.copy()
    FEATURES = [
        "Tuoi_xe", "Log_So_Km_da_di", "Km_moi_nam", "Tuoi_xe_x_Km",
        "TinhTrang_x_XuatXu", "LoaiXe_x_PhanKhuc",
        "Thuong_hieu_code", "Dong_xe_code", "Loai_xe_code",
        "Tinh_trang_code", "Xuat_xu_code", "Phan_khuc_dung_tich_code",
        "Vung_mien_code", "Brand_mean_price", "Dong_mean_price", "Segment_mean_price"
    ]
    TARGET = "Log_Gia"
    scale_cols = ["Tuoi_xe", "Log_So_Km_da_di", "Km_moi_nam", "Tuoi_xe_x_Km"]

    X_full = df_eval[FEATURES].copy()
    X_full[scale_cols] = scaler.transform(X_full[scale_cols])
    y_full = df_eval[TARGET]

    # --- Predict & evaluate ---
    y_pred_log = model.predict(X_full.astype(np.float32))
    y_true_vnd = np.expm1(y_full)
    y_pred_vnd = np.expm1(y_pred_log)

    # --- Clean invalid predictions ---
    mask_valid = (y_true_vnd > 0) & (y_true_vnd < 5e8) & (y_pred_vnd > 0) & (y_pred_vnd < 5e8)
    y_true_vnd, y_pred_vnd = y_true_vnd[mask_valid], y_pred_vnd[mask_valid]

    mae = mean_absolute_error(y_true_vnd, y_pred_vnd)
    rmse = np.sqrt(mean_squared_error(y_true_vnd, y_pred_vnd))
    r2_log = r2_score(y_full, y_pred_log)
    r2_vnd = r2_score(y_true_vnd, y_pred_vnd)

    # --- Display metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"≈ {mae/1e6:.1f} triệu VNĐ")
    col2.metric("RMSE", f"≈ {rmse/1e6:.1f} triệu VNĐ")
    col3.metric("R² (Log)", f"{r2_log:.3f}")
    st.caption(f"💰 R² (VNĐ thực tế): {r2_vnd:.3f}")

    # --- Feature importance & residuals ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    sns.barplot(x=imp.values[:10], y=imp.index[:10], ax=axes[0], color="#1565C0", edgecolor="black")
    axes[0].set_title("Top 10 Feature Importance")

    residuals = y_true_vnd - y_pred_vnd
    sns.scatterplot(x=y_pred_vnd, y=residuals, ax=axes[1], color="#1E88E5", alpha=0.4)
    axes[1].axhline(0, color="red", linestyle="--")
    axes[1].set_title("Residual Plot (VNĐ)")
    st.pyplot(fig)

# ============================================================
# 5️⃣ TAB 3: NEW PREDICTION
# ============================================================
elif menu == "🧮 New Prediction":
    st.header("🧮 Dự đoán giá hợp lý cho xe của bạn")

    col1, col2 = st.columns(2)
    with col1:
        thuong_hieu = st.selectbox("Thương hiệu", sorted(mappings["thuong_hieu"].keys()))
        loai_xe = st.selectbox("Loại xe", sorted(mappings["loai_xe"].keys()))
        dong_xe = st.selectbox("Dòng xe", sorted(mappings["dong_xe"].keys()))
        tinh_trang = st.selectbox("Tình trạng", sorted(mappings["tinh_trang"].keys()))
        xuat_xu = st.selectbox("Xuất xứ", sorted(mappings["xuat_xu"].keys()))
        phan_khuc = st.selectbox("Phân khúc dung tích", sorted(mappings["phan_khuc_dung_tich"].keys()))
        vung_mien = st.selectbox("Vùng miền", ["Miền Bắc", "Miền Trung", "Miền Nam"], index=0)
        nam_dang_ky = st.number_input("Năm đăng ký", 1980, 2025, 2020)
        so_km = st.number_input("Số km đã đi", 0, 500000, 30000, step=1000)
        gia_dang_ban = st.number_input("💰 Giá bạn muốn đăng bán (VNĐ)", 0, 500_000_000, 35_000_000, step=500_000)

    with col2:
        tuoi_xe = 2025 - nam_dang_ky
        log_km = np.log1p(so_km)
        km_moi_nam = np.expm1(log_km) / (tuoi_xe + 0.1)
        tuoi_xe_x_Km = tuoi_xe * log_km
        gia_tren_km = gia_dang_ban / (so_km + 1)
        st.metric("Tuổi xe (năm)", tuoi_xe)
        st.metric("Km trung bình/năm", f"{km_moi_nam:,.0f}")
        st.metric("Giá/km", f"{gia_tren_km:,.0f} đ/km")

    vung_mien_map = {"Miền Bắc": 0, "Miền Trung": 1, "Miền Nam": 2}

    user_input = {
        "Tuoi_xe": tuoi_xe,
        "Log_So_Km_da_di": log_km,
        "Km_moi_nam": km_moi_nam,
        "Tuoi_xe_x_Km": tuoi_xe_x_Km,
        "TinhTrang_x_XuatXu": mappings["tinh_trang"][tinh_trang] * mappings["xuat_xu"][xuat_xu],
        "LoaiXe_x_PhanKhuc": mappings["loai_xe"][loai_xe] * mappings["phan_khuc_dung_tich"][phan_khuc],
        "Thuong_hieu_code": mappings["thuong_hieu"][thuong_hieu],
        "Dong_xe_code": mappings["dong_xe"][dong_xe],
        "Loai_xe_code": mappings["loai_xe"][loai_xe],
        "Tinh_trang_code": mappings["tinh_trang"][tinh_trang],
        "Xuat_xu_code": mappings["xuat_xu"][xuat_xu],
        "Phan_khuc_dung_tich_code": mappings["phan_khuc_dung_tich"][phan_khuc],
        "Vung_mien_code": vung_mien_map[vung_mien],
        "Brand_mean_price": df_ref["Gia_thuc"].mean(),
        "Dong_mean_price": df_ref["Gia_thuc"].mean(),
        "Segment_mean_price": df_ref["Gia_thuc"].mean()
    }

    if st.button("🚀 Dự đoán giá "):
        brand_mean = df_ref.loc[df_ref["Thuong_hieu_code"] == mappings["thuong_hieu"][thuong_hieu], "Log_Gia"].mean()
        dong_mean = df_ref.loc[df_ref["Dong_xe_code"] == mappings["dong_xe"][dong_xe], "Log_Gia"].mean()
        segment_mean = df_ref.loc[df_ref["Phan_khuc_dung_tich_code"] == mappings["phan_khuc_dung_tich"][phan_khuc], "Log_Gia"].mean()
        global_mean = df_ref["Log_Gia"].mean()

        user_input["Brand_mean_price"] = brand_mean if not np.isnan(brand_mean) else global_mean
        user_input["Dong_mean_price"] = dong_mean if not np.isnan(dong_mean) else global_mean
        user_input["Segment_mean_price"] = segment_mean if not np.isnan(segment_mean) else global_mean

        user_df = pd.DataFrame([user_input])
        predicted_price = predict_price(user_df, model, scaler)
        diff = gia_dang_ban - predicted_price
        pct = (diff / predicted_price) * 100

        st.success(f"✅ Giá dự đoán: **{predicted_price:,.0f} VNĐ**")
        if abs(pct) < 10:
            st.info(f"💡 Giá đăng bán hợp lý (chênh {pct:+.1f}%)")
        elif pct > 10:
            st.warning(f"⚠️ Giá đăng bán cao hơn thị trường {pct:+.1f}%")
        else:
            st.success(f"🟢 Giá thấp hơn thị trường {pct:+.1f}% (thu hút khách)")

        st.markdown(f"### 💰 Gợi ý giá : {predicted_price*0.95:,.0f} – {predicted_price*1.05:,.0f} VNĐ")

        st.session_state.predicted_price = predicted_price
        st.session_state.thuong_hieu = thuong_hieu

# ============================================================
# ============================================================
# ============================================================
# ============================================================
# 6️⃣ TAB 4: MARKET ANALYSIS (v4.4 – Cleaned & Robust)
# ============================================================
elif menu == "📈 Market Analysis":
    st.header("📊 Phân tích giá thị trường & vị trí xe của bạn")

    if "predicted_price" not in st.session_state:
        st.warning("⚠️ Vui lòng dự đoán giá trước ở tab **🧮 New Prediction**.")
    else:
        predicted_price = st.session_state.predicted_price
        thuong_hieu = st.session_state.thuong_hieu

        # --- Filter theo thương hiệu ---
        df_brand = df_ref[df_ref["Thuong_hieu_code"] == mappings["thuong_hieu"][thuong_hieu]].copy()

        # --- Nếu không đủ dữ liệu thì cảnh báo ---
        if df_brand.shape[0] < 20:
            st.warning(f"⚠️ Thương hiệu {thuong_hieu} có ít dữ liệu ({df_brand.shape[0]} mẫu). Kết quả chỉ mang tính tham khảo.")
        else:
            # ============================================================
            # 🔹 1. Làm sạch dữ liệu (loại ngoại lai)
            # ============================================================
            low, high = df_brand["Gia_thuc"].quantile([0.01, 0.99])
            df_clean = df_brand[(df_brand["Gia_thuc"] >= low) & (df_brand["Gia_thuc"] <= high)].copy()

            # --- Tính thống kê sau khi clean ---
            mean_price = df_clean["Gia_thuc"].mean()
            median_price = df_clean["Gia_thuc"].median()
            min_price = df_clean["Gia_thuc"].min()
            max_price = df_clean["Gia_thuc"].max()

            # ============================================================
            # 🔹 2. Hiển thị thông tin tổng quan
            # ============================================================
            col1, col2, col3 = st.columns(3)
            col1.metric("💰 Giá dự đoán ", f"{predicted_price:,.0f} ₫")
            col2.metric("📈 Trung vị thương hiệu (ổn định)", f"{median_price:,.0f} ₫")
            col3.metric("📊 Khoảng giá phổ biến", f"{min_price:,.0f} – {max_price:,.0f} ₫")

            st.markdown("---")

            # ============================================================
            # ============================================================
            # 🔹 3. Biểu đồ song song (Histogram + Boxplot, có legend & tránh đè label)
            # ============================================================
            colA, colB = st.columns(2)

            # --- Histogram ---
            with colA:
                fig_hist = px.histogram(
                    df_clean, x="Gia_thuc", nbins=25, color_discrete_sequence=["#004AAD"]
                )

                # Thêm 2 đường tham chiếu
                fig_hist.add_vline(
                    x=predicted_price,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Xe bạn: {predicted_price:,.0f} ₫",
                    annotation_position="top left",
                    annotation_font=dict(color="orange", size=12, family="Segoe UI Semibold"),
                )
                fig_hist.add_vline(
                    x=median_price,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Trung vị: {median_price:,.0f} ₫",
                    annotation_position="bottom right",
                    annotation_font=dict(color="green", size=12, family="Segoe UI Semibold"),
                )

                fig_hist.update_xaxes(range=[low, high])
                fig_hist.update_layout(
                    title=f"Phân bố giá thị trường – {thuong_hieu}",
                    xaxis_title="Giá thực tế (VNĐ)",
                    yaxis_title="Số lượng xe",
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font=dict(family="Segoe UI", size=12),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=-0.3,
                        xanchor="center", x=0.5,
                        bgcolor="rgba(255,255,255,0)",
                        bordercolor="rgba(0,0,0,0)",
                    ),
                    shapes=[
                        dict(type="line", x0=predicted_price, x1=predicted_price, y0=0, y1=1,
                            yref="paper", line=dict(color="orange", dash="dash", width=2)),
                        dict(type="line", x0=median_price, x1=median_price, y0=0, y1=1,
                            yref="paper", line=dict(color="green", dash="dot", width=2))
                    ],
                    annotations=[
                        dict(x=predicted_price, y=1.05, xref="x", yref="paper",
                            text="Xe bạn", showarrow=False, font=dict(color="orange", size=11)),
                        dict(x=median_price, y=1.05, xref="x", yref="paper",
                            text="Trung vị thị trường", showarrow=False, font=dict(color="green", size=11))
                    ]
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            # --- Boxplot ---
            with colB:
                fig_box = px.box(
                    df_clean, y="Gia_thuc", color_discrete_sequence=["#1565C0"], points=False
                )

                fig_box.add_hline(
                    y=predicted_price,
                    line_dash="dot",
                    line_color="orange",
                    annotation_text=f"Xe bạn: {predicted_price:,.0f} ₫",
                    annotation_position="top right",
                    annotation_font=dict(color="orange", size=12)
                )
                fig_box.add_hline(
                    y=median_price,
                    line_dash="dot",
                    line_color="green",
                    annotation_text=f"Trung vị: {median_price:,.0f} ₫",
                    annotation_position="bottom left",
                    annotation_font=dict(color="green", size=12)
                )

                fig_box.update_yaxes(range=[low, high])
                fig_box.update_layout(
                    title=f"Phân bố giá – {thuong_hieu} (Boxplot, đã clean)",
                    yaxis_title="Giá (VNĐ)",
                    plot_bgcolor="#FFFFFF",
                    paper_bgcolor="#FFFFFF",
                    font=dict(family="Segoe UI", size=12),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom", y=-0.3,
                        xanchor="center", x=0.5,
                        bgcolor="rgba(255,255,255,0)",
                        bordercolor="rgba(0,0,0,0)"
                    ),
                    annotations=[
                        dict(xref="paper", yref="paper", x=0.5, y=-0.25,
                            text="<b>Màu cam:</b> Xe bạn  <b>Màu xanh lá:</b> Trung vị thị trường",
                            showarrow=False, font=dict(size=11, color="#333"))
                    ]
                )
                st.plotly_chart(fig_box, use_container_width=True)


            st.markdown("---")

            # ============================================================
            # 🔹 4. Bảng thống kê & nhận xét tự động
            # ============================================================
            st.subheader("📋 Thống kê nhanh thương hiệu ")

            summary = df_clean["Gia_thuc"].describe(percentiles=[0.25, 0.5, 0.75])
            stats = pd.DataFrame({
                "Thấp nhất": [f"{summary['min']:,.0f} ₫"],
                "Trung vị": [f"{summary['50%']:,.0f} ₫"],
                "Trung bình": [f"{summary['mean']:,.0f} ₫"],
                "Cao nhất": [f"{summary['max']:,.0f} ₫"]
            })
            st.table(stats)

            # --- Phân tích vị trí giá xe ---
            quantile_pos = (df_clean["Gia_thuc"] < predicted_price).mean() * 100
            if quantile_pos < 25:
                insight = "🟢 Xe bạn nằm **vùng giá thấp** → dễ bán, thu hút khách."
            elif 25 <= quantile_pos <= 75:
                insight = "🟡 Xe bạn nằm **vùng giá trung bình** → giá hợp lý so với thị trường."
            else:
                insight = "🔴 Xe bạn nằm **vùng giá cao** → có thể khó cạnh tranh, nên xem xét điều chỉnh."
            
            st.markdown("### 💡 Nhận xét tự động:")
            st.info(f"{insight}\n\n📊 Vị trí giá xe bạn nằm ở **top {quantile_pos:.1f}%** trong phân bố giá của {thuong_hieu}.")

            # ============================================================
            # 🔹 5. Gợi ý hành động
            # ============================================================
            st.markdown("---")
            st.markdown("""
            **📈 Gợi ý hành động:**
            - Nếu giá nằm cao hơn 75% thị trường → xem xét giảm ~5–10% để dễ bán hơn.  
            - Nếu giá nằm vùng trung bình → giữ nguyên, có thể thêm mô tả chi tiết để nổi bật.  
            - Nếu giá thấp → đảm bảo không bỏ sót thông tin (đời xe, bảo dưỡng, giấy tờ...) để tránh mất giá trị.
            """)


