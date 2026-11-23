import pandas as pd

from .common import BasePreprocessor


BINS = {
    'days_before_arrival': [-float('inf'), 30, 90, float('inf')],
    'nightly_rate_idr': [-float('inf'), 1_000_000, 2_000_000, float('inf')],
}
MISSING = '__MISSING__'


# based on hari dan tarif terus encode per kolom
class NBPreprocessor(BasePreprocessor):

    # features=None pake semua fitur asli
    def __init__(self, features=None):
        super().__init__(feature_engineering=False)
        self.features = features

    # kelompokin angka ke rentang if needed terus pisahin nilai kosong
    def _as_categories(self, df, column):
        values = df[column]
        if column in BINS:
            values = pd.cut(values, bins=BINS[column], labels=False)
        if column in self.num_cols_:
            # 1 sama 1.0 disamain terus pertahanin nilai kosong
            values = values.astype('Int64')
        return values.astype('string').fillna(MISSING)

    # buat kamus kode / kolom dari train data
    def _fit_extra(self, df):
        self.columns_ = list(df.columns if self.features is None else self.features)
        self.category_codes_ = {}

        for column in self.columns_:
            values = self._as_categories(df, column)
            codes = {'Other': 0, MISSING: 1}
            for value in sorted(values.unique()):
                if value not in codes:
                    codes[value] = len(codes)
            self.category_codes_[column] = codes

    # pake kamus train data trs map kategori baru ke Other
    def _transform_extra(self, df):
        encoded = {}
        for column in self.columns_:
            values = self._as_categories(df, column)
            codes = self.category_codes_[column]
            encoded[column] = values.map(codes).fillna(0).astype(int)
        return pd.DataFrame(encoded, index=df.index)
