# financial-fraud-detection-dashboard
# 🛡️ Financial Fraud Detection Dashboard

🚀 Proyek ini adalah pipeline machine learning lengkap untuk mendeteksi transaksi keuangan yang berpotensi sebagai penipuan, dengan tampilan dashboard Streamlit yang futuristik dan interaktif.

---

## 🔍 Fitur Utama

- ✅ Pembersihan data: menangani nilai hilang dan outlier dengan metode IQR  
- ⚙️ Rekayasa fitur: encoding, normalisasi, dan ekstraksi fitur waktu  
- 🧠 Prediksi ML: model Random Forest Classifier  
- 📊 Visualisasi: histogram, heatmap, boxplot, animasi volume transaksi  
- 🌌 Tampilan UI futuristik: tema warna `#0C3B5D` dan latar kustom  
- 🧪 Evaluasi model: akurasi, presisi, recall, F1 score, ROC curve

---

## 📁 Struktur Folder

financial-fraud-preprocessing/
├── data/
│ ├── raw/
│ └── processed/
├── preprocessing/
│ ├── pipeline.py
│ ├── cleaning.py
│ ├── feature_engineering.py
│ └── utils.py
├── notebooks/
│ └── eda.ipynb
├── assets/
│ └── background_UI-ml1.png
├── app.py
├── requirements.txt
└── .gitignore


---

## 🧪 Teknologi yang Digunakan

- Python, pandas, scikit-learn  
- Streamlit, Plotly, joblib  
- seaborn, matplotlib

---

## 🏁 Cara Menjalankan

```bash
git clone https://github.com/namakamu/financial-fraud-detection-dashboard
cd financial-fraud-detection-dashboard
pip install -r requirements.txt
streamlit run app.py

