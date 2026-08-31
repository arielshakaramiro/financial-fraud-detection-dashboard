# financial-fraud-detection-dashboard
# 🛡️ Financial Fraud Detection Dashboard

🚀 Proyek ini adalah pipeline machine learning lengkap untuk mendeteksi transaksi keuangan yang berpotensi sebagai penipuan, dengan tampilan dashboard Streamlit yang futuristik dan interaktif.

---

## 🔍 Fitur Utama

- ✅ Pembersihan data: menangani nilai hilang dan outlier dengan metode IQR  
- ⚙️ Rekayasa fitur: encoding, normalisasi, dan ekstraksi fitur waktu  
- 🧠 Prediksi ML: pilihan model Random Forest, XGBoost, LightGBM, CatBoost, deteksi anomali (Isolation Forest / One-Class SVM), serta AutoML (PyCaret, opsional)  
- ⚖️ Penanganan data tidak seimbang: SMOTE hanya pada data train (bebas kebocoran/leakage)  
- 📊 Visualisasi: histogram, heatmap, boxplot, animasi volume transaksi  
- 🌌 Tampilan UI futuristik: tema warna `#0C3B5D` dan latar kustom  
- 🧪 Evaluasi model: akurasi, presisi, recall, F1 score, ROC curve

---

## 📁 Struktur Folder

financial-fraud-detection-dashboard/
├── data/
│ ├── creditcard_fraud_real_sample.csv   # dataset bawaan (dipakai app secara default)
│ └── creditcard_fraud_real_full.csv     # dataset lengkap (arsip)
├── preprocessing/
│ ├── pipeline.py
│ ├── cleaning.py
│ └── feature_engineering.py
├── assets/
│ └── background_UI-ml1.png
├── app.py
├── requirements.txt
├── requirements-automl.txt
└── .gitignore


---

## 📂 Dataset

Aplikasi ini memakai dataset **Credit Card Fraud Detection (ULB)** -- 284.807
transaksi kartu kredit nyata dari nasabah bank di Eropa (September 2013), dengan
492 transaksi penipuan. Kolom `V1`–`V28` adalah hasil transformasi PCA (disamarkan
demi kerahasiaan), ditambah `amount`, `transaction_time`, dan label `is_fraud`.

- `data/creditcard_fraud_real_sample.csv` — versi ringkas (12.492 baris, semua 492
  fraud + 12.000 transaksi normal). **Dipakai app secara default** supaya training
  interaktif tetap cepat. Sudah ikut di repo.
- `data/creditcard_fraud_real_full.csv` — versi lengkap (284.807 baris). Tidak ikut
  di repo karena ukurannya ~107 MB (di atas limit file GitHub); unduh sendiri dari
  sumber di bawah.

### ⬇️ Download dataset

- **Sumber resmi (Kaggle):** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Unduh langsung CSV lengkap (tanpa login):**
  https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv

  ```bash
  # unduh dataset lengkap langsung ke folder data/
  curl -L -o data/creditcard.csv \
    https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv
  ```

  Catatan: file dari sumber di atas memakai kolom asli `Time`, `V1`–`V28`, `Amount`,
  `Class`. App mengharapkan `transaction_time`, `amount`, dan `is_fraud`, jadi ganti
  nama kolomnya (`Class` → `is_fraud`, `Amount` → `amount`, `Time` → `transaction_time`)
  sebelum dipakai, atau langsung pakai `data/creditcard_fraud_real_sample.csv` yang
  sudah disesuaikan.

Untuk memakai data sendiri, cukup unggah file CSV lewat panel di sidebar kiri
(harus punya kolom `is_fraud`; kolom `transaction_time` dan `amount` opsional
tapi dianjurkan).

---

## 🧪 Teknologi yang Digunakan

- Python, pandas, numpy, scikit-learn  
- XGBoost, LightGBM, CatBoost, imbalanced-learn (SMOTE), Optuna  
- Streamlit, Plotly, joblib

---

## 🏁 Cara Menjalankan

```bash
git clone https://github.com/arielshakaramiro/financial-fraud-detection-dashboard
cd financial-fraud-detection-dashboard
pip install -r requirements.txt
streamlit run app.py

