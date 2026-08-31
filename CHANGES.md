# Ringkasan Perbaikan

Perbaikan ini menyasar semua temuan dari review sebelumnya. File yang diubah/ditambah:
`app.py`, `preprocessing/cleaning.py` (baru), `preprocessing/feature_engineering.py` (baru),
`preprocessing/pipeline.py` (diisi, sebelumnya stub kosong), `requirements.txt`,
`requirements-automl.txt` (baru), `.gitignore`, `README.md`, `SETUP.md`, dan dataset
bawaan di `data/` (lihat bagian 0).

## 0. Dataset asli menggantikan data dummy acak (perubahan terbaru)
**Sebelum:** tanpa upload, app memanggil `load_dummy_data()` yang meng-*generate*
data 100% acak (`np.random` untuk `feature1`, `feature2`, dan label `is_fraud`
tanpa korelasi apa pun) ke `data/processed/transactions_processed.csv`. Akibatnya
model tidak punya pola nyata untuk dipelajari -- precision/recall/F1 = 0.00 dan
ROC AUC 0.19 (lebih buruk dari tebak acak), yang membingungkan karena terlihat
seperti app-nya rusak padahal datanya yang tidak bermakna.

**Sesudah:** app memakai dataset **Credit Card Fraud Detection (ULB)** -- transaksi
kartu kredit nyata dari bank Eropa (Sept 2013). Ditambahkan dua file di `data/`:
- `creditcard_fraud_real_sample.csv` (12.492 baris, semua 492 fraud + 12.000 normal)
  -- dipakai app secara default supaya training interaktif tetap cepat.
- `creditcard_fraud_real_full.csv` (284.807 baris) -- arsip lengkap, di-*ignore*
  git karena ~107 MB (di atas limit file GitHub 100 MB).

Fungsi `load_dummy_data()` diganti `load_default_data()`: memuat dataset asli, dan
hanya membuat data fallback acak (dengan warning jelas) kalau file dataset hilang --
sekadar jaring pengaman supaya app tidak crash. Dengan data asli, model Random Forest
kini mencapai F1 0.96 dan ROC AUC 1.00 di test set (terverifikasi end-to-end).

Sekalian, semua label/teks UI yang memakai kata "Contoh"/"dummy" yang menyesatkan
dirapikan: judul tabel jadi "Transaksi Terdeteksi Penipuan (N transaksi)" dengan
jumlah nyata dan caption "menampilkan 10 teratas"; banner data default berbunyi
"Menggunakan dataset bawaan (transaksi kartu kredit asli)". `README.md` dan
`SETUP.md` diselaraskan (struktur folder, daftar model, bagian Dataset baru).

## 1. Data leakage SMOTE (bug paling kritis)
**Sebelum:** `SMOTE.fit_resample(X, y)` dijalankan ke seluruh dataset, baru setelah itu
`train_test_split`. Sample sintetis bisa bocor antara train dan test set, bikin
accuracy/precision/recall/F1 terlihat lebih tinggi dari performa sebenarnya.

**Sesudah:** `train_test_split` dulu, SMOTE cuma diterapkan ke `X_train`/`y_train`.
Test set tidak pernah disentuh proses oversampling. Sudah diverifikasi dengan test
manual: index train dan test benar-benar disjoint (`set(...).isdisjoint(...)` → `True`).

## 2. Contoh transaksi terdeteksi pakai seluruh data, bukan test set
**Sebelum:** `df['fraud_prediction'] = model.predict_proba(X)[:, 1] >= threshold`
diprediksi ke seluruh X, termasuk baris yang dipakai model saat training.

**Sesudah:** tabel "Contoh Transaksi Terdeteksi Penipuan" untuk model supervised
sekarang hanya menampilkan hasil prediksi pada `X_test` (data yang belum pernah
dilihat model), dengan catatan penjelasan di UI supaya jelas bedanya dengan mode
AutoML/anomaly detection yang memang scoring seluruh data secara sengaja.

## 3. Fitur rekayasa fitur yang sebelumnya cuma janji di README
`preprocessing/pipeline.py` sebelumnya isinya cuma komentar
`# Preprocessing pipeline logic goes here` -- tidak pernah dipanggil dari `app.py`,
padahal README mengklaim ada "encoding, normalisasi, ekstraksi fitur waktu".

Sekarang benar-benar diimplementasikan dan dipanggil dari `app.py`:
- `preprocessing/cleaning.py` -- isi nilai kosong, hapus outlier (IQR) dengan
  pengecualian: baris fraud tidak dibuang meski secara statistik outlier.
- `preprocessing/feature_engineering.py` -- ekstraksi `hour`/`day_of_week`/`is_weekend`,
  label encoding kolom kategorikal, standardisasi kolom numerik.
- `preprocessing/pipeline.py` -- orkestrasi dua tahap: `clean_and_engineer_time()` untuk
  seluruh dataframe tampilan (tidak di-scale, supaya nilai `amount` dkk tetap dalam
  satuan asli di grafik), dan `prepare_model_features()` khusus untuk fitur yang masuk
  ke model (encoder/scaler di-fit dari train, dipakai ulang -- bukan fit ulang -- ke test).

## 4. `XGBClassifier(use_label_encoder=False, ...)`
Parameter `use_label_encoder` sudah dihapus di XGBoost versi baru dan akan error kalau
environment install versi terbaru (requirements sebelumnya tidak mengunci versi).
Parameter itu dihapus dari pemanggilan.

## 5. Dependency berat & tidak dikunci versi
- Semua versi library di `requirements.txt` sekarang dikunci (pin) ke versi yang teruji.
- `pycaret[full]` dipindah ke `requirements-automl.txt` terpisah, dan import-nya di
  `app.py` dipindah dari top-level ke dalam branch AutoML saja (lazy import) --
  supaya mode lain tidak ikut menanggung dependency berat itu, dan startup app lebih cepat.
- Kalau PyCaret belum ter-install tapi user pilih mode AutoML, sekarang muncul pesan
  error yang jelas (bukan `ImportError` traceback mentah).

## 6. File binary/log yang ke-commit ke git
`best_automl_model.pkl`, isi folder `model/*.pkl`, dan `logs.log` sebelumnya ikut
ter-commit ke repo. `.gitignore` sudah ditambah `*.pkl` dan `*.log`, tapi **catatan
penting**: menambah `.gitignore` TIDAK menghapus file yang sudah pernah di-commit dari
histori git. Untuk benar-benar membersihkannya, jalankan dari root repo:

```bash
git rm --cached best_automl_model.pkl logs.log model/*.pkl
git commit -m "chore: stop tracking model/log artifacts"
```

(File tetap ada di disk lokal, cuma berhenti di-track git.)

## 7. Duplikasi kode metric & error handling
- Blok tampilan `Accuracy/Precision/Recall/F1` yang sebelumnya diulang manual 3x sekarang
  jadi satu fungsi `show_classification_metrics()`.
- Confusion matrix + ROC curve jadi `show_confusion_and_roc()`.
- `set_bg_hack()` sekarang tidak crash kalau file background tidak ada -- cuma warning
  dan lanjut pakai tema default.
- Loading model tersimpan (`joblib.load`) dan `pd.read_csv` dari upload dibungkus
  try/except dengan pesan error yang jelas ke user, bukan traceback mentah/crash app.

## 8. Bonus: bug forward-compat pandas
Data dummy bawaan pakai `pd.date_range(..., freq="H")`. Alias `"H"` sudah dihapus di
pandas 3.x (harus huruf kecil `"h"|`). Karena `requirements.txt` sebelumnya tidak
mengunci versi pandas, app bisa crash total di environment dengan pandas terbaru.
Sudah diganti ke `"h"` (kompatibel di pandas lama maupun baru).

## Soal lisensi -- tidak ada yang perlu diubah
Di review sebelumnya sempat saya duga komentar copyright di `app.py` ("All rights
reserved... unauthorized copying prohibited") kontradiksi dengan adanya file `LICENSE`.
Setelah dicek isi `LICENSE`-nya, ternyata memang proprietary/all-rights-reserved juga --
jadi konsisten, tidak ada yang perlu diperbaiki di sini. Koreksi dari saya.

## Yang sengaja TIDAK diubah
- Struktur UI/UX dan daftar model tetap sama seperti sebelumnya.
- Objective Optuna untuk Random Forest masih dievaluasi dari skor training itu sendiri,
  bukan cross-validation terpisah -- disederhanakan supaya tuning tetap cepat untuk
  dashboard interaktif. Kalau butuh tuning yang lebih rigorous, ganti jadi
  `cross_val_score` di dalam `objective()`.
