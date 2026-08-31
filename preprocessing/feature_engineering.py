# © 2025 Ariel Shakaramiro - All rights reserved.
# Unauthorized copying or commercial use of this file is strictly prohibited.
"""
preprocessing/feature_engineering.py

Fungsi-fungsi rekayasa fitur:
- Ekstraksi fitur waktu dari kolom timestamp
- Encoding kolom kategorikal
- Normalisasi (standardisasi) kolom numerik

Fungsi encode/normalize mengembalikan objek encoder/scaler yang sudah di-fit,
supaya bisa dipakai ulang secara konsisten saat scoring data baru
(bukan sekadar fit_transform sekali pakai lalu dibuang).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def extract_time_features(df: pd.DataFrame, time_col: str = "transaction_time") -> pd.DataFrame:
    """
    Tambahkan fitur turunan dari kolom waktu: hour, day_of_week, is_weekend.
    Kolom waktu asli tetap dipertahankan (dipakai untuk visualisasi),
    hanya fitur turunannya yang ditambahkan.
    """
    df = df.copy()
    if time_col not in df.columns:
        return df

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df["hour"] = df[time_col].dt.hour
    df["day_of_week"] = df[time_col].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def encode_categorical(
    df: pd.DataFrame, exclude: set[str] | None = None
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Label-encode semua kolom bertipe object/category, kecuali yang ada di `exclude`
    (misalnya kolom target atau kolom waktu mentah yang tidak dipakai model).

    Return: (df_hasil, dict berisi LabelEncoder per kolom -- untuk dipakai ulang
             saat scoring data baru dengan model yang sama).
    """
    df = df.copy()
    exclude = exclude or set()
    cat_cols = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]

    encoders: dict[str, LabelEncoder] = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders


def normalize_numeric(
    df: pd.DataFrame, exclude: set[str] | None = None
) -> tuple[pd.DataFrame, StandardScaler | None]:
    """
    Standardisasi (z-score) kolom numerik, kecuali yang ada di `exclude`
    (biasanya kolom target/biner seperti is_fraud).

    Return: (df_hasil, scaler -- None jika tidak ada kolom numerik untuk di-scale).
    """
    df = df.copy()
    exclude = exclude or set()
    num_cols = [c for c in df.select_dtypes(include="number").columns if c not in exclude]

    if not num_cols:
        return df, None

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    return df, scaler
