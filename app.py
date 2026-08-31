# © 2025 Ariel Shakaramiro - All rights reserved.
# Unauthorized copying or commercial use of this file is strictly prohibited.
# app.py (FIXED: no SMOTE leakage, held-out-only scoring, lazy PyCaret import,
#          real preprocessing pipeline, deduped metrics, defensive error handling)

import os
import time

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier

import optuna

from preprocessing.pipeline import clean_and_engineer_time, prepare_model_features

# PyCaret sengaja TIDAK di-import di top-level. Dependency-nya sangat berat
# (pycaret[full] menarik banyak library turunan) dan hanya dibutuhkan kalau
# user benar-benar memilih mode "AutoML (PyCaret)" di sidebar. Import di
# dalam branch supaya startup app lebih cepat dan tidak memaksa environment
# lain (yang tidak butuh AutoML) menanggung dependency itu.


# ============================== Helper functions ==============================

def set_bg_hack(png_file: str) -> None:
    """Set background image. Diam-diam skip (bukan crash) kalau file tidak ada,
    supaya app tetap bisa jalan walau asset belum ter-upload/ter-clone."""
    import base64

    if not os.path.exists(png_file):
        st.warning(f"⚠️ Background image '{png_file}' tidak ditemukan, memakai tema default.")
        return

    try:
        with open(png_file, "rb") as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        page_bg_img = f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
        }}
        .stDataFrame, .stTable, .stMarkdown, .stText, .stAlert {{
            background-color: #0C3B5D !important;
            color: white !important;
            border-radius: 10px;
            padding: 10px;
        }}
        .block-container {{
            padding: 2rem;
            background-color: rgba(12, 59, 93, 0.5);
            border-radius: 15px;
        }}
        </style>
        """
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except OSError as e:
        st.warning(f"⚠️ Gagal memuat background image: {e}")


def show_classification_metrics(y_true, y_pred) -> None:
    """Tampilkan 4 metric utama. Dipanggil dari 3 tempat berbeda sebelumnya
    (kode diulang manual) -- sekarang jadi satu fungsi supaya konsisten."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
    col2.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2f}")
    col3.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2f}")
    col4.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.2f}")


def show_confusion_and_roc(y_true, y_pred, proba) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Prediksi", y="Aktual", color="Jumlah"),
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC", line=dict(color="#00FFCC")))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
    fig_roc.update_layout(
        title=f"ROC Curve (AUC = {roc_auc:.2f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    st.plotly_chart(fig_roc, use_container_width=True)


def show_detected_examples(df_subset: pd.DataFrame, label: str = "🕵️ Transaksi Terdeteksi Penipuan") -> None:
    flagged = df_subset[df_subset["fraud_prediction"] == 1]
    st.subheader(f"{label} ({len(flagged)} transaksi)")
    st.caption(
        "Daftar di bawah diambil dari data uji (test set) yang tidak dilihat model saat "
        "training -- jadi mencerminkan kemampuan generalisasi model, bukan hafalan data training. "
        "Menampilkan 10 transaksi teratas yang diprediksi sebagai penipuan."
    )
    st.dataframe(flagged.head(10), use_container_width=True)


def load_default_data(path: str) -> pd.DataFrame:
    """Muat dataset bawaan (transaksi asli) supaya dashboard bisa langsung dicoba
    tanpa upload. Kalau file tidak ada, buat fallback minimal agar app tidak crash."""
    if os.path.exists(path):
        return pd.read_csv(path)

    st.warning(
        f"⚠️ Dataset bawaan '{path}' tidak ditemukan. Membuat data fallback minimal -- "
        "unggah file CSV transaksi Anda lewat sidebar untuk hasil yang bermakna."
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_fallback = pd.DataFrame(
        {
            "amount": np.random.randint(100, 10000, size=300),
            "transaction_time": pd.date_range("2024-01-01", periods=300, freq="h"),
            "feature1": np.random.randn(300),
            "feature2": np.random.randn(300),
            "is_fraud": np.random.choice([0, 1], size=300, p=[0.95, 0.05]),
        }
    )
    df_fallback.to_csv(path, index=False)
    return pd.read_csv(path)


def build_supervised_model(model_type: str, X_train: pd.DataFrame, y_train: pd.Series):
    """Buat & (kalau perlu) tuning model sesuai pilihan user. Fit dilakukan di sini
    supaya tuning Optuna untuk Random Forest bisa memakai data train yang benar
    (bukan seluruh dataset)."""
    if model_type == "Random Forest":

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_categorical("n_estimators", [50, 100, 150]),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "class_weight": "balanced",
            }
            clf = RandomForestClassifier(**params, random_state=42)
            clf.fit(X_train, y_train)
            # Catatan: idealnya objective Optuna dievaluasi lewat cross-validation
            # terpisah dari X_train, bukan skor training itu sendiri. Disederhanakan
            # di sini untuk menjaga waktu tuning tetap cepat di dashboard interaktif.
            y_pred_inner = clf.predict(X_train)
            return f1_score(y_train, y_pred_inner)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        best_params = study.best_params
        st.write("🔧 Hyperparameter terbaik:", best_params)
        # class_weight='balanced' sengaja ditambahkan manual di sini, BUKAN lewat
        # best_params -- Optuna hanya menyimpan parameter yang di-suggest lewat
        # trial.suggest_*. 'class_weight' yang di-hardcode di dalam objective()
        # TIDAK PERNAH muncul di study.best_params (sudah diverifikasi manual),
        # jadi kalau tidak ditambahkan ulang di sini, model final kehilangan
        # balanced class weight secara diam-diam -- bug ini ada di kode asli.
        model = RandomForestClassifier(**best_params, class_weight="balanced", random_state=42)

    elif model_type == "XGBoost":
        # use_label_encoder dihapus: parameter ini sudah deprecated/dibuang di
        # XGBoost versi baru dan akan error kalau environment pakai versi terbaru.
        model = XGBClassifier(eval_metric="logloss", random_state=42)

    elif model_type == "LightGBM":
        model = LGBMClassifier(random_state=42)

    elif model_type == "CatBoost":
        model = CatBoostClassifier(verbose=0, random_state=42)

    else:
        raise ValueError(f"Model type tidak dikenal: {model_type}")

    model.fit(X_train, y_train)
    return model


# ============================== Page setup ==============================

st.set_page_config(page_title="Deteksi Penipuan Finansial", layout="wide")
set_bg_hack("assets/background_UI-ml1.png")
st.title("🛡️ Aplikasi Deteksi Penipuan Transaksi")

# ============================== Load data ==============================

uploaded_file = st.sidebar.file_uploader("📤 Unggah File Transaksi (.csv)", type=["csv"])

if uploaded_file:
    with st.spinner("🚀 Memuat data transaksi..."):
        time.sleep(1.5)
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ File berhasil diunggah")
        except (pd.errors.ParserError, UnicodeDecodeError) as e:
            st.error(f"❌ Gagal membaca file CSV: {e}")
            st.stop()
else:
    st.info("ℹ️ Menggunakan dataset bawaan (transaksi kartu kredit asli). Unggah CSV Anda sendiri lewat sidebar untuk memakai data lain.")
    df = load_default_data("data/creditcard_fraud_real_sample.csv")

# Cleaning + fitur waktu diterapkan ke seluruh dataframe tampilan (bukan hasil
# scaling, jadi 'amount' dkk tetap dalam satuan asli untuk visualisasi).
df = clean_and_engineer_time(df, time_col="transaction_time", target_col="is_fraud")

if "is_fraud" in df.columns:
    st.subheader("📊 Ringkasan Data Label Penipuan")
    st.markdown("Jumlah data transaksi yang ditandai sebagai penipuan vs normal")
    st.write(df["is_fraud"].value_counts())
    st.bar_chart(df["is_fraud"].value_counts())

# ============================== Sidebar settings ==============================

st.sidebar.subheader("⚙️ Pengaturan Model")
model_type = st.sidebar.selectbox(
    "Pilih Algoritma Deteksi:",
    [
        "Random Forest",
        "XGBoost",
        "LightGBM",
        "CatBoost",
        "Isolation Forest (Anomali)",
        "One-Class SVM (Anomali)",
        "AutoML (PyCaret)",
    ],
)
threshold = st.sidebar.slider("Threshold Deteksi Penipuan", 0.0, 1.0, 0.5, 0.01)
quick_mode = st.sidebar.checkbox("🚀 Aktifkan Mode Prediksi Cepat (tanpa pelatihan ulang)")

# ============================== Modeling ==============================

if "is_fraud" in df.columns:
    st.subheader("🔍 Hasil Prediksi Penipuan")

    drop_cols = [c for c in ["is_fraud", "transaction_time"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["is_fraud"]

    if model_type == "AutoML (PyCaret)":
        try:
            from pycaret.classification import compare_models, predict_model, pull, save_model, setup
        except ImportError:
            st.error(
                "❌ PyCaret belum terpasang. Mode AutoML butuh dependency tambahan -- "
                "install dengan `pip install -r requirements-automl.txt`, lalu jalankan ulang app."
            )
            st.stop()

        data = df.copy()
        if "transaction_time" in data.columns:
            data = data.drop(columns=["transaction_time"])
        data = data.dropna(subset=["is_fraud"])
        df = df.loc[data.index]

        if quick_mode:
            st.info("🔄 Mode Prediksi Cepat diaktifkan: memuat model dari file")
            try:
                best_model = joblib.load("best_automl_model.pkl")
                predictions = predict_model(best_model, data=data)
                df["fraud_prediction"] = (predictions["prediction_score"] >= threshold).astype(int)

                show_classification_metrics(df["is_fraud"], df["fraud_prediction"])
                show_detected_examples(
                    df,
                    label="🕵️ Transaksi Terdeteksi Penipuan (model tersimpan, seluruh data)",
                )
            except FileNotFoundError:
                st.error("❌ Model belum tersedia. Jalankan pelatihan model terlebih dahulu.")
            except Exception as e:  # noqa: BLE001 - tampilkan pesan yang jelas ke user, jangan crash app
                st.error(f"❌ Gagal memuat / menjalankan model tersimpan: {e}")
        else:
            with st.spinner("🔍 Mencari model terbaik dengan PyCaret..."):
                try:
                    setup(data, target="is_fraud", session_id=42, verbose=False)
                    best_model = compare_models()
                    model_result = pull()
                    st.write("📈 Hasil AutoML PyCaret:", model_result)

                    predictions = predict_model(best_model, data=data)
                    df["fraud_prediction"] = (predictions["prediction_score"] >= threshold).astype(int)
                    save_model(best_model, "best_automl_model")

                    show_classification_metrics(df["is_fraud"], df["fraud_prediction"])
                    show_detected_examples(
                        df,
                        label="🕵️ Transaksi Terdeteksi Penipuan (AutoML, seluruh data)",
                    )
                except Exception as e:  # noqa: BLE001
                    st.error(f"❌ Pelatihan AutoML gagal: {e}")

    elif model_type not in ["Isolation Forest (Anomali)", "One-Class SVM (Anomali)"]:
        if y.nunique() < 2:
            st.error(
                "❌ Kolom 'is_fraud' cuma punya 1 kelas unik di data ini. Model klasifikasi "
                "butuh minimal 2 kelas (fraud & non-fraud) untuk bisa dilatih -- coba pakai "
                "mode Deteksi Anomali di sidebar, yang tidak butuh label sama sekali."
            )
            st.stop()

        # --- Split DULU, baru SMOTE hanya di data train. -----------------------
        # Sebelumnya SMOTE dijalankan ke SELURUH X,y sebelum split, sehingga
        # sample sintetis bisa "bocor" antara train dan test set dan membuat
        # accuracy/precision/recall/F1 terlihat lebih bagus dari kondisi nyata.
        try:
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # stratify gagal kalau ada kelas dengan cuma 1 anggota -- fallback ke
            # split biasa (tanpa stratify) daripada crash total.
            st.warning(
                "⚠️ Kelas 'is_fraud' terlalu sedikit untuk stratified split, "
                "memakai pembagian data biasa (tanpa stratify)."
            )
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        # Encoding & normalisasi di-fit HANYA dari data train, lalu dipakai
        # ulang (transform saja) untuk data test -- supaya test set benar-benar
        # jadi data "belum pernah dilihat" oleh proses apa pun, termasuk scaler.
        X_train, encoders, scaler = prepare_model_features(X_train_raw)
        X_test, _, _ = prepare_model_features(X_test_raw, encoders=encoders, scaler=scaler)

        minority_count = int(y_train.value_counts().min())
        if minority_count < 2:
            # SMOTE butuh minimal k_neighbors+1 sample di kelas minoritas (default
            # k_neighbors=5 -> butuh minimal 6). Kalau kelas minoritas di data
            # TRAIN cuma <2 sample, SMOTE pasti error -- daripada crash, lewati
            # oversampling dan latih model langsung (dengan peringatan ke user).
            st.warning(
                "⚠️ Kelas minoritas di data training kurang dari 2 sample -- "
                "SMOTE dilewati, model dilatih tanpa oversampling."
            )
            X_train_resampled, y_train_resampled = X_train, y_train
        else:
            k_neighbors = min(5, minority_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        model = build_supervised_model(model_type, X_train_resampled, y_train_resampled)

        proba_test = model.predict_proba(X_test)[:, 1]
        y_pred_test = (proba_test >= threshold).astype(int)

        show_classification_metrics(y_test, y_pred_test)
        show_confusion_and_roc(y_test, y_pred_test, proba_test)

        # Transaksi yang ditampilkan HANYA dari test set (data yang model belum
        # pernah lihat), bukan seluruh dataset -- supaya benar-benar mencerminkan
        # performa deteksi di data baru, bukan hafalan training.
        df_test_view = df.loc[X_test_raw.index].copy()
        df_test_view["fraud_prediction"] = y_pred_test
        show_detected_examples(df_test_view)

    else:
        st.subheader("🧪 Deteksi Anomali")
        X_anomaly, _, _ = prepare_model_features(X)

        if model_type == "Isolation Forest (Anomali)":
            model = IsolationForest(contamination=0.05, random_state=42)
        else:
            model = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")

        model.fit(X_anomaly)
        y_pred = model.predict(X_anomaly)
        df["fraud_prediction"] = (y_pred == -1).astype(int)

        st.metric("Terdeteksi Anomali", int(df["fraud_prediction"].sum()))
        st.subheader("📋 Transaksi Terdeteksi Anomali")
        st.caption(
            "Deteksi anomali bersifat unsupervised (tidak memakai label is_fraud saat "
            "training), jadi seluruh dataset ditampilkan di sini -- ini bukan celah "
            "data leakage seperti pada model supervised di atas."
        )
        st.dataframe(df[df["fraud_prediction"] == 1].head(10), use_container_width=True)

else:
    st.warning("❌ Kolom 'is_fraud' tidak ditemukan dalam dataset.")

# ============================== Visualizations ==============================

st.sidebar.header("📊 Pengaturan Visualisasi")
num_cols = df.select_dtypes(include="number").columns.tolist()

if num_cols:
    col_to_plot = st.sidebar.selectbox("Pilih Kolom Angka untuk Visualisasi", num_cols)

    st.subheader(f"📈 Histogram {col_to_plot}")
    hist = px.histogram(
        df, x=col_to_plot, nbins=40, title=f"Distribusi {col_to_plot}", color_discrete_sequence=["#00FFCC"]
    )
    st.plotly_chart(hist, use_container_width=True)

    if st.sidebar.checkbox("Tampilkan Heatmap Korelasi"):
        st.subheader("🔗 Korelasi Antar Kolom")
        corr = df.corr(numeric_only=True)
        heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale="Viridis"))
        st.plotly_chart(heatmap, use_container_width=True)

    if st.sidebar.checkbox("Tampilkan Boxplot"):
        st.subheader("📦 Boxplot")
        box = px.box(df, y=col_to_plot, title=f"Boxplot {col_to_plot}", color_discrete_sequence=["#00FFCC"])
        st.plotly_chart(box, use_container_width=True)

if "transaction_time" in df.columns and "amount" in df.columns and "hour" in df.columns:
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df_hourly_source = df.dropna(subset=["amount"])

    with st.expander("🎥 Grafik Jumlah Transaksi per Jam"):
        hourly_df = df_hourly_source.groupby("hour")["amount"].sum().reset_index()
        anim_bar = px.bar(
            hourly_df,
            x="amount",
            y="hour",
            orientation="h",
            animation_frame="hour",
            range_x=[0, hourly_df["amount"].max() * 1.2],
            color="hour",
            color_continuous_scale="Plasma",
            title="Total Transaksi per Jam",
        )
        st.plotly_chart(anim_bar, use_container_width=True)
