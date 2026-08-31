# © 2025 Ariel Shakaramiro - All rights reserved.
# Unauthorized copying or commercial use of this file is strictly prohibited.
"""
preprocessing/cleaning.py

Fungsi-fungsi pembersihan data:
- Penanganan nilai kosong (missing values)
- Penghapusan outlier dengan metode IQR

Catatan penting:
Saat menghapus outlier, baris dengan is_fraud == 1 (jika kolom itu ada)
SENGAJA tidak dibuang meskipun terdeteksi sebagai outlier secara statistik.
Transaksi fraud secara alami sering muncul sebagai outlier (nominal tidak wajar,
jam tidak wajar, dsb) -- justru itu yang ingin dideteksi model. Membuangnya
akan menghilangkan sinyal fraud yang paling berguna.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """
    Isi nilai kosong.
    - Kolom numerik  -> median atau mean (sesuai `strategy`)
    - Kolom kategorikal -> modus (nilai paling sering muncul)

    Tidak mengubah df asli (return copy baru).
    """
    if strategy not in ("median", "mean"):
        raise ValueError("strategy harus 'median' atau 'mean'")

    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(exclude=np.number).columns

    for col in numeric_cols:
        if df[col].isna().any():
            fill_value = df[col].median() if strategy == "median" else df[col].mean()
            df[col] = df[col].fillna(fill_value)

    for col in categorical_cols:
        if df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "unknown"
            df[col] = df[col].fillna(fill_value)

    return df


def remove_outliers_iqr(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    factor: float = 1.5,
    target_col: str = "is_fraud",
) -> pd.DataFrame:
    """
    Hapus baris outlier pada kolom numerik menggunakan metode IQR.

    Baris dengan `target_col == 1` (transaksi fraud) selalu dipertahankan,
    meski secara statistik outlier -- lihat catatan di docstring modul.

    Args:
        df: DataFrame input.
        columns: kolom numerik yang dicek. Default: semua kolom numerik
                 kecuali `target_col`.
        factor: pengali IQR (default 1.5, standar Tukey's fences).
        target_col: nama kolom label fraud, jika ada di df.
    """
    df = df.copy()

    if columns is None:
        columns = df.select_dtypes(include=np.number).columns.tolist()
        columns = [c for c in columns if c != target_col]

    keep_mask = pd.Series(True, index=df.index)
    for col in columns:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        keep_mask &= df[col].between(lower, upper)

    if target_col in df.columns:
        keep_mask = keep_mask | (df[target_col] == 1)

    return df[keep_mask].reset_index(drop=True)
