# app.py (FINAL with PyCaret AutoML)

import streamlit as st
import pandas as pd
import numpy as np
import base64
import os
import time
import plotly.express as px
import plotly.graph_objects as go
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.svm import OneClassSVM
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import optuna
from pycaret.classification import setup, compare_models, predict_model, pull, save_model

# === UI Styling ===
def set_bg_hack(png_file):
    with open(png_file, "rb") as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = f'''
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
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

st.set_page_config(page_title="Deteksi Penipuan Finansial", layout="wide")
set_bg_hack("assets/background_UI-ml1.png")
st.title("🛡️ Aplikasi Deteksi Penipuan Transaksi")

uploaded_file = st.sidebar.file_uploader("📤 Unggah File Transaksi (.csv)", type=["csv"])

if uploaded_file:
    with st.spinner("🚀 Memuat data transaksi..."):
        time.sleep(1.5)
        df = pd.read_csv(uploaded_file)
    st.success("✅ File berhasil diunggah")
else:
    st.info("ℹ️ Menggunakan data contoh bawaan")
    dummy_path = "data/processed/transactions_processed.csv"
    os.makedirs("data/processed", exist_ok=True)

    if not os.path.exists(dummy_path):
        df_dummy = pd.DataFrame({
            'amount': np.random.randint(100, 10000, size=300),
            'transaction_time': pd.date_range("2024-01-01", periods=300, freq='H'),
            'feature1': np.random.randn(300),
            'feature2': np.random.randn(300),
            'is_fraud': np.random.choice([0, 1], size=300, p=[0.95, 0.05])
        })
        df_dummy.to_csv(dummy_path, index=False)
        st.warning("⚠️ File contoh dibuat otomatis karena tidak ditemukan.")

    df = pd.read_csv(dummy_path)

if 'transaction_time' in df.columns:
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    df['hour'] = df['transaction_time'].dt.hour

if 'is_fraud' in df.columns:
    st.subheader("📊 Ringkasan Data Label Penipuan")
    st.markdown("Jumlah data transaksi yang ditandai sebagai penipuan vs normal")
    st.write(df['is_fraud'].value_counts())
    st.bar_chart(df['is_fraud'].value_counts())

st.sidebar.subheader("⚙️ Pengaturan Model")
model_type = st.sidebar.selectbox("Pilih Algoritma Deteksi:", [
    "Random Forest", "XGBoost", "LightGBM", "CatBoost", "Isolation Forest (Anomali)", "One-Class SVM (Anomali)", "AutoML (PyCaret)"])
threshold = st.sidebar.slider("Threshold Deteksi Penipuan", 0.0, 1.0, 0.5, 0.01)

if 'is_fraud' in df.columns:
    st.subheader("🔍 Hasil Prediksi Penipuan")
    X = df.drop(columns=['is_fraud', 'transaction_time']) if 'transaction_time' in df.columns else df.drop(columns=['is_fraud'])
    y = df['is_fraud']

    if model_type == "AutoML (PyCaret)":
        data = df.copy()
        if 'transaction_time' in data.columns:
            data = data.drop(columns=['transaction_time'])

        with st.spinner("🔍 Mencari model terbaik dengan PyCaret..."):
            s = setup(data, target='is_fraud', session_id=42, verbose=False)
            best_model = compare_models()
            model_result = pull()
            st.write("📈 Hasil AutoML PyCaret:", model_result)
            predictions = predict_model(best_model, data=data)
            df['fraud_prediction'] = (predictions['prediction_score'] >= threshold).astype(int)
            save_model(best_model, 'best_automl_model')

            y_true = df['is_fraud']
            y_pred = df['fraud_prediction']

            st.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.2f}")
            st.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.2f}")
            st.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.2f}")
            st.metric("F1 Score", f"{f1_score(y_true, y_pred, zero_division=0):.2f}")
            st.subheader("🕵️ Contoh Transaksi Terdeteksi Penipuan")
            st.dataframe(df[df['fraud_prediction'] == 1].head(10), use_container_width=True)

    elif model_type not in ["Isolation Forest (Anomali)", "One-Class SVM (Anomali)"]:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        if model_type == "Random Forest":
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_categorical('n_estimators', [50, 100, 150]),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                    'class_weight': 'balanced'
                }
                clf = RandomForestClassifier(**params, random_state=42)
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                return f1_score(y_test, y_pred)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=20)
            best_params = study.best_params
            st.write("🔧 Hyperparameter terbaik:", best_params)
            model = RandomForestClassifier(**best_params, random_state=42)
        elif model_type == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif model_type == "LightGBM":
            model = LGBMClassifier()
        elif model_type == "CatBoost":
            model = CatBoostClassifier(verbose=0)

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        df['fraud_prediction'] = (model.predict_proba(X)[:, 1] >= threshold).astype(int)

        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
        st.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.2f}")
        st.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.2f}")
        st.metric("F1 Score", f"{f1_score(y_test, y_pred, zero_division=0):.2f}")

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                           labels=dict(x="Prediksi", y="Aktual", color="Jumlah"))
        st.plotly_chart(fig_cm, use_container_width=True)

        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC', line=dict(color='#00FFCC')))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(title=f"ROC Curve (AUC = {roc_auc:.2f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig_roc, use_container_width=True)

        st.subheader("🕵️ Contoh Transaksi Terdeteksi Penipuan")
        st.dataframe(df[df['fraud_prediction'] == 1].head(10), use_container_width=True)

    else:
        st.subheader("🧪 Deteksi Anomali")
        if model_type == "Isolation Forest (Anomali)":
            model = IsolationForest(contamination=0.05, random_state=42)
        elif model_type == "One-Class SVM (Anomali)":
            model = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')

        model.fit(X)
        y_pred = model.predict(X)
        df['fraud_prediction'] = (y_pred == -1).astype(int)

        st.metric("Terdeteksi Anomali", int(df['fraud_prediction'].sum()))
        st.subheader("📋 Contoh Anomali")
        st.dataframe(df[df['fraud_prediction'] == 1].head(10), use_container_width=True)
else:
    st.warning("❌ Kolom 'is_fraud' tidak ditemukan dalam dataset.")

st.sidebar.header("📊 Pengaturan Visualisasi")
num_cols = df.select_dtypes(include='number').columns.tolist()
if num_cols:
    col_to_plot = st.sidebar.selectbox("Pilih Kolom Angka untuk Visualisasi", num_cols)

    st.subheader(f"📈 Histogram {col_to_plot}")
    hist = px.histogram(df, x=col_to_plot, nbins=40, title=f"Distribusi {col_to_plot}", color_discrete_sequence=["#00FFCC"])
    st.plotly_chart(hist, use_container_width=True)

    if st.sidebar.checkbox("Tampilkan Heatmap Korelasi"):
        st.subheader("🔗 Korelasi Antar Kolom")
        corr = df.corr(numeric_only=True)
        heatmap = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Viridis'))
        st.plotly_chart(heatmap, use_container_width=True)

    if st.sidebar.checkbox("Tampilkan Boxplot"):
        st.subheader("📦 Boxplot")
        box = px.box(df, y=col_to_plot, title=f"Boxplot {col_to_plot}", color_discrete_sequence=["#00FFCC"])
        st.plotly_chart(box, use_container_width=True)

if 'transaction_time' in df.columns and 'amount' in df.columns:
    with st.expander("🎥 Grafik Jumlah Transaksi per Jam"):
        hourly_df = df.groupby('hour')['amount'].sum().reset_index()
        anim_bar = px.bar(hourly_df, x='amount', y='hour', orientation='h',
                          animation_frame='hour', range_x=[0, hourly_df['amount'].max()*1.2],
                          color='hour', color_continuous_scale='Plasma',
                          title="Total Transaksi per Jam")
        st.plotly_chart(anim_bar, use_container_width=True)
