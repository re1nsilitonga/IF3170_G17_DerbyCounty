"""Tahap cleaning, feature engineering, dan transformasi data pemesanan penginapan."""

import numpy as np
import pandas as pd

TARGET = 'is_canceled'
ID_COL = 'id'
CAT_COLS = ['meal_package', 'room_category', 'booking_channel']


def clean_train(df):
    """Bersihkan data train: buang baris yang duplikat persis (termasuk targetnya).

    Hanya dipakai pada data train. Data test tidak boleh dibuang barisnya karena
    setiap id wajib punya prediksi.
    """
    feature_cols = [c for c in df.columns if c != ID_COL]
    return df.drop_duplicates(subset=feature_cols).reset_index(drop=True)


def mark_invalid(df):
    """Ubah nilai yang tidak masuk akal menjadi missing. Berlaku untuk train dan test."""
    df = df.copy()
    # Pemesanan tanpa tamu dewasa dianggap data salah input
    df.loc[df['num_adults'] == 0, 'num_adults'] = np.nan
    return df


def add_features(df):
    """Tambah fitur turunan yang bisa dihitung per baris tanpa statistik dataset.

    Karena hanya memakai nilai pada baris itu sendiri, fungsi ini aman dipakai
    pada train dan test secara terpisah.
    """
    df = df.copy()
    df['total_nights'] = df['weekend_nights'] + df['weekday_nights']
    df['total_guests'] = df['num_adults'] + df['num_children']
    df['is_free'] = (df['nightly_rate_idr'] == 0).astype(float)
    df.loc[df['nightly_rate_idr'].isna(), 'is_free'] = np.nan
    df['rate_per_guest'] = df['nightly_rate_idr'] / df['total_guests'].replace(0, np.nan)

    # Fitur kalender dari tanggal kedatangan
    arrival = pd.to_datetime(
        dict(year=df['arrival_year'], month=df['arrival_month'], day=df['arrival_day']),
        errors='coerce',
    )
    df['arrival_weekday'] = arrival.dt.weekday.astype(float)
    df['arrival_dayofyear'] = arrival.dt.dayofyear.astype(float)
    # Urutan hari absolut, sehingga musim di tahun berbeda bisa dibedakan
    df['arrival_index'] = ((arrival - pd.Timestamp('2023-01-01')).dt.days).astype(float)
    return df


class Preprocessor:
    """Transformasi fitur yang statistiknya dipelajari dari data train saja.

    mode='tree'   : missing value dibiarkan NaN dan kolom kategorikal tetap berupa
                    teks, karena Decision Tree from scratch bisa menanganinya langsung.
    mode='scaled' : imputasi median/modus, indikator missing, one-hot encoding,
                    lalu standardisasi. Dipakai untuk model berbasis jarak atau bobot
                    seperti KNN, Logistic Regression, dan SVM.
    """

    def __init__(self, mode='tree', feature_engineering=True, rare_min_count=20, drop_cols=None):
        """Simpan konfigurasi.

        feature_engineering=False berarti fitur turunan dari add_features tidak dibuat.
        rare_min_count adalah batas jumlah kemunculan agar kategori tidak dianggap langka.
        """
        self.mode = mode
        self.feature_engineering = feature_engineering
        self.rare_min_count = rare_min_count
        self.drop_cols = list(drop_cols or [])

    def fit(self, df):
        """Pelajari statistik (kategori, median, modus, mean, std) dari data train."""
        df = self._base(df)
        self.cat_cols_ = [c for c in CAT_COLS if c in df.columns]
        self.num_cols_ = [c for c in df.columns if c not in self.cat_cols_]

        # Kategori yang terlalu jarang digabung menjadi 'Other'
        self.keep_cats_ = {}
        for c in self.cat_cols_:
            counts = df[c].value_counts()
            self.keep_cats_[c] = sorted(counts[counts >= self.rare_min_count].index)
        df = self._group_rare(df)

        if self.mode == 'scaled':
            self.medians_ = df[self.num_cols_].median()
            self.modes_ = {c: df[c].mode().iloc[0] for c in self.cat_cols_}
            # Indikator missing hanya dibuat untuk kolom yang memang punya NaN di train
            self.missing_cols_ = [c for c in df.columns if df[c].isna().any()]
            encoded = self._encode(df)
            self.columns_ = list(encoded.columns)
            self.means_ = encoded.mean()
            self.stds_ = encoded.std().replace(0, 1.0)
        return self

    def transform(self, df):
        """Terapkan transformasi yang sudah dipelajari ke data baru (validasi atau test)."""
        df = self._group_rare(self._base(df))
        if self.mode == 'tree':
            return df
        encoded = self._encode(df).reindex(columns=self.columns_, fill_value=0.0)
        return (encoded - self.means_) / self.stds_

    def fit_transform(self, df):
        """Gabungan fit lalu transform pada data yang sama."""
        return self.fit(df).transform(df)

    # ------------------------------------------------------------ helper

    def _base(self, df):
        """Tandai nilai janggal, tambah fitur turunan (opsional), lalu buang kolom yang tidak dipakai."""
        df = mark_invalid(df)
        if self.feature_engineering:
            df = add_features(df)
        drop = [ID_COL, TARGET] + self.drop_cols
        return df.drop(columns=[c for c in drop if c in df.columns])

    def _group_rare(self, df):
        """Ganti kategori di luar daftar kategori yang disimpan menjadi 'Other'."""
        df = df.copy()
        for c in self.cat_cols_:
            known = df[c].isin(self.keep_cats_[c]) | df[c].isna()
            df.loc[~known, c] = 'Other'
        return df

    def _encode(self, df):
        """Imputasi, indikator missing, dan one-hot encoding (khusus mode scaled)."""
        out = pd.DataFrame(index=df.index)
        for c in self.missing_cols_:
            out[f'{c}_missing'] = df[c].isna().astype(float)
        out[self.num_cols_] = df[self.num_cols_].fillna(self.medians_)
        for c in self.cat_cols_:
            filled = df[c].fillna(self.modes_[c])
            for cat in self.keep_cats_[c] + ['Other']:
                out[f'{c}_{cat}'] = (filled == cat).astype(float)
        return out.astype(float)
