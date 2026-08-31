# © 2025 Ariel Shakaramiro - All rights reserved.
# Unauthorized copying or commercial use of this file is strictly prohibited.
"""
preprocessing/pipeline.py

Titik masuk (entrypoint) preprocessing yang dipanggil dari app.py.
Sengaja dipecah jadi dua fungsi, bukan satu fungsi besar:

1. `clean_and_engineer_time(df)`
   Dipakai pada SELURUH dataframe yang tampil di dashboard (histogram,
   boxplot, heatmap, grafik per jam). Hanya membersihkan data dan menambah
   fitur waktu -- TIDAK melakukan scaling, supaya nilai seperti 'amount'
   tetap dalam satuan aslinya (mata uang) dan enak dibaca di visualisasi.

2. `prepare_model_features(X, target_col)`
   Dipakai HANYA pada fitur (X) tepat sebelum masuk ke model (encoding +
   normalisasi). Tidak menyentuh dataframe yang dipakai untuk visualisasi,
   supaya nilai yang ditampilkan ke user tidak berubah jadi angka z-score
   yang tidak bermakna.

Pemisahan ini juga yang membuat langkah SMOTE di app.py aman dari data
leakage: prepare_model_features() dipanggil terpisah untuk X_train dan
X_test (scaler/encoder di-fit dari data train, lalu dipakai untuk transform
data test) -- bukan di-fit ke seluruh dataset sekaligus sebelum split.
"""

from __future__ import annotations

import pandas as pd

from preprocessing.cleaning import handle_missing_values, remove_outliers_iqr
from preprocessing.feature_engineering import (
    encode_categorical,
    extract_time_features,
    normalize_numeric,
)


def clean_and_engineer_time(
    df: pd.DataFrame,
    time_col: str = "transaction_time",
    target_col: str = "is_fraud",
) -> pd.DataFrame:
    """Langkah 1: cleaning + fitur waktu. Dipakai untuk seluruh dataframe tampilan."""
    df = extract_time_features(df, time_col=time_col)
    df = handle_missing_values(df)
    df = remove_outliers_iqr(df, target_col=target_col)
    return df


def prepare_model_features(
    X: pd.DataFrame,
    encoders: dict | None = None,
    scaler=None,
):
    """
    Langkah 2: encoding + normalisasi, khusus untuk fitur yang masuk ke model.

    Jika `encoders`/`scaler` diberikan (hasil fit dari data train), fungsi ini
    akan MEMAKAI ULANG objek tersebut (transform saja, tidak fit ulang) --
    supaya X_test diproses dengan parameter yang sama persis dengan X_train,
    bukan di-fit terpisah (yang juga termasuk bentuk data leakage).

    Jika tidak diberikan, fungsi akan fit encoder/scaler baru dari X yang
    diberikan (dipakai untuk fit dari X_train).

    Return: (X_hasil, encoders_dict, scaler)
    """
    X = X.copy()

    if encoders is None:
        X, encoders = encode_categorical(X)
    else:
        for col, le in encoders.items():
            if col in X.columns:
                # Kategori baru yang tidak pernah dilihat saat fit -> map ke -1
                known = set(le.classes_)
                X[col] = X[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
                X[col] = le.transform(X[col])

    if scaler is None:
        X, scaler = normalize_numeric(X)
    else:
        num_cols = list(getattr(scaler, "feature_names_in_", []))
        num_cols = [c for c in num_cols if c in X.columns]
        if num_cols:
            X[num_cols] = scaler.transform(X[num_cols])

    return X, encoders, scaler
