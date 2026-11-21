# Preprocessor untuk SVM dengan imputasi, indikator missing, one-hot, dan standardisasi.

from .knn import KNNPreprocessor


# Gunakan encoding dan transformasi yang sudah tersedia di KNNPreprocessor.
class SVMPreprocessor(KNNPreprocessor):
    # Pelajari statistik dari train untuk transform data validasi dan test.
    def _fit_extra(self, df):
        # Pelajari nilai imputasi, dengan fallback untuk kolom yang seluruhnya kosong.
        self.medians_ = df[self.num_cols_].median().fillna(0.0)
        self.modes_ = {c: df[c].mode().get(0, 'Other') for c in self.cat_cols_}

        # Simpan indikator missing dan susunan kolom hasil encoding.
        self.missing_cols_ = [c for c in df.columns if df[c].isna().any()]
        encoded = self._encode(df)
        self.columns_ = list(encoded.columns)

        # Simpan statistik scaling, std nol atau tidak tersedia diganti satu.
        self.means_ = encoded.mean()
        self.stds_ = encoded.std().fillna(1.0).replace(0, 1.0)
