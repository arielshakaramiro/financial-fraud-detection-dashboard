# Cara Menjalankan di VS Code

1. **Extract ZIP**, lalu buka foldernya di VS Code (`File > Open Folder...`).

2. **Buat virtual environment** (buka terminal di VS Code, `Ctrl+` ` `):
   ```bash
   python -m venv venv
   ```
   Aktifkan:
   - Windows (PowerShell): `venv\Scripts\Activate.ps1`
   - macOS / Linux: `source venv/bin/activate`

   Di VS Code, pastikan interpreter Python yang dipilih (pojok kanan bawah) adalah
   yang di dalam `venv/` -- kalau belum, `Ctrl+Shift+P` → "Python: Select Interpreter".

3. **Install dependency dasar:**
   ```bash
   pip install -r requirements.txt
   ```
   Kalau mau pakai mode "AutoML (PyCaret)" di sidebar (opsional, dependency-nya berat):
   ```bash
   pip install -r requirements-automl.txt
   ```

4. **Jalankan app:**
   ```bash
   streamlit run app.py
   ```
   Browser akan otomatis kebuka ke `http://localhost:8501`. Kalau tidak, buka manual.

5. **Tanpa upload file apa pun**, app langsung memakai dataset bawaan berisi
   transaksi kartu kredit asli di `data/creditcard_fraud_real_sample.csv`
   (12.492 transaksi, 492 di antaranya penipuan) supaya dashboard bisa langsung
   dicoba dengan data yang bermakna. Untuk pakai data sendiri, upload CSV lewat
   panel di sidebar kiri.

## Kalau ada error

- `ModuleNotFoundError: No module named 'xxx'` → dependency belum ke-install, ulangi
  langkah 3 (pastikan venv aktif dulu).
- Warning "Background image tidak ditemukan" → pastikan folder `assets/` ikut ter-extract
  utuh dari ZIP (jangan cuma `app.py`-nya saja yang dipindah).
- Detail lengkap semua perbaikan dari versi sebelumnya ada di `CHANGES.md`.
