# Preprocessing bersama untuk semua model: konstanta, cleaning, fitur turunan, dan BasePreprocessor.

import numpy as np
import pandas as pd

TARGET = 'is_canceled'
ID_COL = 'id'
CAT_COLS = ['meal_package', 'room_category', 'booking_channel']


# Buang baris duplikat persis (fitur dan target). Hanya untuk data train.
def clean_train(df):
    feature_cols = [c for c in df.columns if c != ID_COL]
    return df.drop_duplicates(subset=feature_cols).reset_index(drop=True)


# Ubah nilai yang tidak masuk akal (num_adults = 0) menjadi missing.
def mark_invalid(df):
    df = df.copy()
    df.loc[df['num_adults'] == 0, 'num_adults'] = np.nan
    return df


# Tambah fitur turunan per baris, aman dipakai pada train dan test secara terpisah.
def add_features(df):
    df = df.copy()
    df['total_nights'] = df['weekend_nights'] + df['weekday_nights']
    df['total_guests'] = df['num_adults'] + df['num_children']
    df['is_free'] = (df['nightly_rate_idr'] == 0).astype(float)
    df.loc[df['nightly_rate_idr'].isna(), 'is_free'] = np.nan
    df['rate_per_guest'] = df['nightly_rate_idr'] / df['total_guests'].replace(0, np.nan)

    arrival = pd.to_datetime(
        dict(year=df['arrival_year'], month=df['arrival_month'], day=df['arrival_day']),
        errors='coerce',
    )
    df['arrival_weekday'] = arrival.dt.weekday.astype(float)
    df['arrival_dayofyear'] = arrival.dt.dayofyear.astype(float)
    df['arrival_index'] = ((arrival - pd.Timestamp('2023-01-01')).dt.days).astype(float)
    return df


# Dasar preprocessor tiap model, statistik dipelajari dari data train saja.
# Kelas turunan cukup override _fit_extra dan _transform_extra.
class BasePreprocessor:

    # feature_engineering: pakai add_features atau tidak; rare_min_count: batas kategori langka.
    def __init__(self, feature_engineering=True, rare_min_count=20, drop_cols=None):
        self.feature_engineering = feature_engineering
        self.rare_min_count = rare_min_count
        self.drop_cols = list(drop_cols or [])

    # Pelajari kolom dan kategori dari data train, lalu statistik khusus model.
    def fit(self, df):
        df = self._base(df)
        self.cat_cols_ = [c for c in CAT_COLS if c in df.columns]
        self.num_cols_ = [c for c in df.columns if c not in self.cat_cols_]
        self.keep_cats_ = {}
        for c in self.cat_cols_:
            counts = df[c].value_counts()
            self.keep_cats_[c] = sorted(counts[counts >= self.rare_min_count].index)
        self._fit_extra(self._group_rare(df))
        return self

    # Terapkan transformasi yang sudah dipelajari ke data validasi atau test.
    def transform(self, df):
        return self._transform_extra(self._group_rare(self._base(df)))

    # Gabungan fit lalu transform.
    def fit_transform(self, df):
        return self.fit(df).transform(df)

    # Statistik khusus model, default tidak ada.
    def _fit_extra(self, df):
        pass

    # Transformasi khusus model, default data dikembalikan apa adanya.
    def _transform_extra(self, df):
        return df

    # Tandai nilai janggal, tambah fitur turunan (opsional), buang kolom id dan target.
    def _base(self, df):
        df = mark_invalid(df)
        if self.feature_engineering:
            df = add_features(df)
        drop = [ID_COL, TARGET] + self.drop_cols
        return df.drop(columns=[c for c in drop if c in df.columns])

    # Ganti kategori yang tidak ada di daftar train menjadi 'Other'.
    def _group_rare(self, df):
        df = df.copy()
        for c in self.cat_cols_:
            known = df[c].isin(self.keep_cats_[c]) | df[c].isna()
            df.loc[~known, c] = 'Other'
        return df
