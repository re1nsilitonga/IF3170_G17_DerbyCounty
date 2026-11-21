# Preprocessor untuk KNN.

import pandas as pd

from .common import BasePreprocessor


# Indikator missing, imputasi median/modus, one-hot, lalu standardisasi.
class KNNPreprocessor(BasePreprocessor):

    # Pelajari median, modus, kolom hasil encoding, serta mean dan std dari data train.
    def _fit_extra(self, df):
        # Kolom tanpa observasi train memakai konstanta, bukan statistik validasi/test.
        self.medians_ = df[self.num_cols_].median().fillna(0.0)
        self.modes_ = {}
        for c in self.cat_cols_:
            modes = df[c].mode()
            self.modes_[c] = modes.iloc[0] if not modes.empty else 'Other'
        self.missing_cols_ = [c for c in df.columns if df[c].isna().any()]
        encoded = self._encode(df)
        self.columns_ = list(encoded.columns)
        self.means_ = encoded.mean()
        # Std sampel juga tidak terdefinisi jika train hanya berisi satu baris.
        self.stds_ = encoded.std().replace(0, 1.0).fillna(1.0)

    # Encode lalu standardisasi dengan statistik dari train.
    def _transform_extra(self, df):
        encoded = self._encode(df).reindex(columns=self.columns_, fill_value=0.0)
        return (encoded - self.means_) / self.stds_

    # Indikator missing, imputasi, dan one-hot encoding.
    def _encode(self, df):
        out = pd.DataFrame(index=df.index)
        for c in self.missing_cols_:
            out[f'{c}_missing'] = df[c].isna().astype(float)
        out[self.num_cols_] = df[self.num_cols_].fillna(self.medians_)
        for c in self.cat_cols_:
            filled = df[c].fillna(self.modes_[c])
            for cat in dict.fromkeys(self.keep_cats_[c] + ['Other']):
                out[f'{c}_{cat}'] = (filled == cat).astype(float)
        return out.astype(float)
