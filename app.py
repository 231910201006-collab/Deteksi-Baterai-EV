"""
EV Battery Health Detection — Streamlit App (Simpel)
Hanya Dashboard + Prediksi
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

st.set_page_config(
    page_title="EV Battery Health Detection",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    .main { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    h1, h2, h3, h4, p, label, div { color: #e6edf3; }
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 8px;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #8b949e; margin-top: 4px; }
    .result-healthy {
        background: #1a4731; border: 2px solid #3fb950;
        border-radius: 14px; padding: 28px; text-align: center;
    }
    .result-replace {
        background: #3d1a1a; border: 2px solid #f85149;
        border-radius: 14px; padding: 28px; text-align: center;
    }
    .stButton > button {
        width: 100%; background: #238636; color: white;
        border: none; border-radius: 8px; padding: 12px;
        font-weight: 600; font-size: 1rem;
    }
    .stButton > button:hover { background: #2ea043; }
    .stSelectbox label, .stSlider label { color: #e6edf3 !important; }
    div[data-testid="stTab"] button { color: #8b949e !important; }
    div[data-testid="stTab"] button[aria-selected="true"] { color: #e6edf3 !important; border-bottom-color: #58a6ff !important; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

# ─── LOAD & TRAIN ────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_excel("batteryev.xlsx")
    le_bt = LabelEncoder(); le_ds = LabelEncoder(); le_bs = LabelEncoder()
    df["Battery_Type_enc"]   = le_bt.fit_transform(df["Battery_Type"])
    df["Driving_Style_enc"]  = le_ds.fit_transform(df["Driving_Style"])
    df["Battery_Status_enc"] = le_bs.fit_transform(df["Battery_Status"])

    feat_cols = [
        "Battery_Type_enc", "Battery_Capacity_kWh", "Vehicle_Age_Months",
        "Total_Charging_Cycles", "Avg_Temperature_C", "Fast_Charge_Ratio",
        "Avg_Discharge_Rate_C", "Driving_Style_enc", "Internal_Resistance_Ohm",
    ]

    df_enc      = df[feat_cols + ["Battery_Status_enc"]].copy()
    df_majority = df_enc[df_enc["Battery_Status_enc"] == 0]
    df_minority = df_enc[df_enc["Battery_Status_enc"] == 1]
    df_min_up   = resample(df_minority, replace=True, n_samples=500, random_state=42)
    df_balanced = pd.concat([df_majority, df_min_up])

    X = df_balanced[feat_cols]
    y = df_balanced["Battery_Status_enc"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    return clf, le_bt, le_ds, le_bs, feat_cols, df

clf, le_bt, le_ds, le_bs, feat_cols, df_raw = load_and_train()

# ─── SIDEBAR ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔋 EV Battery Health")
    st.markdown("---")
    st.markdown("### ⚙️ Parameter Baterai")

    battery_type   = st.selectbox("Tipe Baterai", ["NMC", "LFP"])
    capacity       = st.slider("Kapasitas (kWh)", 26.0, 88.0, 75.0, 0.5)
    age_months     = st.slider("Usia Kendaraan (Bulan)", 1, 120, 36)
    charge_cycles  = st.slider("Total Siklus Pengisian", 0, 1500, 400)
    avg_temp       = st.slider("Suhu Rata-rata (°C)", -10.0, 60.0, 25.0, 0.5)
    fast_ratio     = st.slider("Rasio Pengisian Cepat", 0.0, 1.0, 0.5, 0.01)
    discharge_rate = st.slider("Laju Discharge (C)", 0.1, 3.0, 1.5, 0.05)
    driving_style  = st.selectbox("Gaya Mengemudi", ["Conservative", "Moderate", "Aggressive"])
    internal_res   = st.slider("Resistansi Internal (Ohm)", 0.010, 0.100, 0.030, 0.001)

    st.markdown("---")
    predict_btn = st.button("🔎 Prediksi Sekarang")

# ─── HEADER ──────────────────────────────────────────────────
st.markdown("# 🔋 EV Battery Health Detection")
st.markdown("---")

# ─── TABS ────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📊 Dashboard", "🔮 Prediksi"])

# ════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Ringkasan Dataset")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#58a6ff">{len(df_raw):,}</div>
            <div class="metric-label">Total Data</div></div>""", unsafe_allow_html=True)
    with c2:
        healthy_count = df_raw["Battery_Status"].value_counts()["Healthy"]
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#3fb950">{healthy_count:,}</div>
            <div class="metric-label">Healthy</div></div>""", unsafe_allow_html=True)
    with c3:
        replace_count = df_raw["Battery_Status"].value_counts()["Replace Required"]
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#f85149">{replace_count}</div>
            <div class="metric-label">Replace Required</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value" style="color:#d29922">{len(feat_cols)}</div>
            <div class="metric-label">Fitur Input</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Distribusi Tipe Baterai")
        fig1, ax1 = plt.subplots(figsize=(5, 3.5))
        fig1.patch.set_facecolor('#161b22'); ax1.set_facecolor('#161b22')
        counts = df_raw["Battery_Type"].value_counts()
        bars = ax1.bar(counts.index, counts.values, color=["#58a6ff","#3fb950"], edgecolor='#30363d', width=0.5)
        ax1.set_ylabel("Jumlah", color="#e6edf3")
        ax1.tick_params(colors="#e6edf3")
        for b, v in zip(bars, counts.values):
            ax1.text(b.get_x()+b.get_width()/2, v+50, str(v), ha='center', color='#e6edf3', fontweight='bold')
        for sp in ax1.spines.values(): sp.set_edgecolor('#30363d')
        st.pyplot(fig1, use_container_width=True)
        plt.close()

    with col_b:
        st.markdown("#### Distribusi Kelas Target")
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        fig2.patch.set_facecolor('#161b22'); ax2.set_facecolor('#161b22')
        vc = df_raw["Battery_Status"].value_counts()
        ax2.bar(vc.index, vc.values, color=["#3fb950","#f85149"], edgecolor='#30363d', width=0.5)
        ax2.set_ylabel("Jumlah", color="#e6edf3")
        ax2.tick_params(colors="#e6edf3")
        for i, (idx, val) in enumerate(vc.items()):
            ax2.text(i, val+50, str(val), ha='center', color='#e6edf3', fontweight='bold')
        for sp in ax2.spines.values(): sp.set_edgecolor('#30363d')
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    st.markdown("---")
    st.markdown("#### 🗃️ Sample Data")
    st.dataframe(
        df_raw.head(8).style.set_properties(**{"background-color":"#161b22","color":"#e6edf3"}),
        use_container_width=True
    )

# ════════════════════════════════════════════════════════════
# TAB 2 — PREDIKSI
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔮 Prediksi Kondisi Baterai")
    st.markdown("Atur parameter baterai di **sidebar kiri** lalu klik tombol prediksi.")

    if predict_btn:
        input_data = {
            "Battery_Type_enc"      : le_bt.transform([battery_type])[0],
            "Battery_Capacity_kWh"  : capacity,
            "Vehicle_Age_Months"    : age_months,
            "Total_Charging_Cycles" : charge_cycles,
            "Avg_Temperature_C"     : avg_temp,
            "Fast_Charge_Ratio"     : fast_ratio,
            "Avg_Discharge_Rate_C"  : discharge_rate,
            "Driving_Style_enc"     : le_ds.transform([driving_style])[0],
            "Internal_Resistance_Ohm": internal_res,
        }
        X_input    = pd.DataFrame([input_data])[feat_cols]
        pred_enc   = clf.predict(X_input)[0]
        pred_prob  = clf.predict_proba(X_input)[0]
        pred_label = le_bs.inverse_transform([pred_enc])[0]
        conf       = pred_prob[pred_enc] * 100

        st.markdown("---")
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if pred_label == "Healthy":
                st.markdown(f"""
<div class="result-healthy">
  <div style="font-size:3.5rem">✅</div>
  <div style="font-size:2rem;font-weight:700;color:#3fb950;margin:8px 0">HEALTHY</div>
  <div style="color:#e6edf3">Baterai dalam kondisi BAIK</div>
  <div style="color:#8b949e;margin-top:8px">Kepercayaan: <b style="color:#3fb950">{conf:.1f}%</b></div>
</div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
<div class="result-replace">
  <div style="font-size:3.5rem">⚠️</div>
  <div style="font-size:2rem;font-weight:700;color:#f85149;margin:8px 0">REPLACE REQUIRED</div>
  <div style="color:#e6edf3">Baterai PERLU DIGANTI segera!</div>
  <div style="color:#8b949e;margin-top:8px">Kepercayaan: <b style="color:#f85149">{conf:.1f}%</b></div>
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            st.markdown("#### Parameter yang Dimasukkan")
            display_input = {
                "Tipe Baterai"        : battery_type,
                "Kapasitas (kWh)"     : f"{capacity} kWh",
                "Usia Kendaraan"      : f"{age_months} bulan",
                "Siklus Pengisian"    : f"{charge_cycles} siklus",
                "Suhu Rata-rata"      : f"{avg_temp} °C",
                "Rasio Fast Charge"   : f"{fast_ratio:.2f}",
                "Laju Discharge"      : f"{discharge_rate:.2f} C",
                "Gaya Mengemudi"      : driving_style,
                "Resistansi Internal" : f"{internal_res:.4f} Ω",
            }
            st.table(pd.DataFrame(list(display_input.items()), columns=["Parameter", "Nilai"]))

        with col_p2:
            st.markdown("#### Probabilitas Prediksi")
            fig_p, ax_p = plt.subplots(figsize=(5, 2.5))
            fig_p.patch.set_facecolor('#161b22'); ax_p.set_facecolor('#161b22')
            class_names = le_bs.classes_
            bars_p = ax_p.barh(class_names, pred_prob,
                               color=["#3fb950","#f85149"], edgecolor='#30363d', height=0.4)
            ax_p.set_xlim(0, 1.1)
            ax_p.tick_params(colors='#e6edf3')
            ax_p.set_xlabel("Probabilitas", color='#e6edf3')
            for b, v in zip(bars_p, pred_prob):
                ax_p.text(v+0.01, b.get_y()+b.get_height()/2,
                          f"{v*100:.1f}%", va='center', color='#e6edf3', fontweight='bold')
            for sp in ax_p.spines.values(): sp.set_edgecolor('#30363d')
            st.pyplot(fig_p, use_container_width=True)
            plt.close()

    else:
        st.info("👈 Atur parameter baterai di sidebar kiri, lalu klik **🔎 Prediksi Sekarang**")
        st.markdown("""
#### Panduan Parameter
| Parameter | Baterai Sehat | Baterai Perlu Diganti |
|-----------|--------------|----------------------|
| Resistansi Internal | < 0.05 Ohm | > 0.075 Ohm |
| Siklus Pengisian | < 500 | > 900 |
| Usia Kendaraan | < 60 bulan | > 65 bulan |
| Suhu Rata-rata | 15–35 °C | > 38 °C |
| Rasio Fast Charge | < 0.4 | > 0.7 |
""")

st.markdown("---")
st.markdown("""<div style="text-align:center;color:#8b949e;font-size:0.85rem">
🔋 EV Battery Health Detection &nbsp;|&nbsp; Random Forest Classifier &nbsp;|&nbsp; Tugas Machine Learning
</div>""", unsafe_allow_html=True)
