import numpy as np
import pandas as pd

from .common import TARGET, BasePreprocessor

# Fitur waktu yang berulang (siklik) beserta periodenya
CYCLIC_COLS = {'arrival_month': 12, 'arrival_weekday': 7, 'arrival_dayofyear': 366}

# Hari pertama pada data sebagai nol untuk indeks hari
ORIGIN_DAY = pd.Timestamp('2023-01-01')


# Cari batas bin satu fitur dengan pemisahan biner berulang yang paling menurunkan impuritas Gini terhadap target
# Pemisahan terbaik dipilih lebih dulu (best-first) sampai terbentuk n_bins kelompok dengan tiap kelompok minimal min_leaf baris
# Idenya diadaptasi dari kriteria pemisahan CART oleh Breiman dkk 1984 dalam Classification and Regression Trees
# Inovasi ini adalah binning terawasi yang menaruh batas bin di tempat rasio batal benar-benar berubah
def target_edges(x, t, n_bins, min_leaf):
    order = np.argsort(x, kind='mergesort')
    xs, ts = x[order], t[order].astype(float)
    cum = np.concatenate([[0.0], np.cumsum(ts)])

    # Pemisahan terbaik pada segmen [lo, hi) dengan hasil berupa (penurunan impuritas, posisi potong)
    def best_split(lo, hi):
        if hi - lo < 2 * min_leaf:
            return None
        cut = np.arange(lo + min_leaf, hi - min_leaf + 1)
        cut = cut[xs[cut - 1] != xs[np.minimum(cut, len(xs) - 1)]]
        if len(cut) == 0:
            return None
        n_l, n_r = cut - lo, hi - cut
        pos_l, pos_r = cum[cut] - cum[lo], cum[hi] - cum[cut]
        pos, n = cum[hi] - cum[lo], hi - lo
        # n * Gini / 2 = pos * (n - pos) / n
        after = pos_l * (n_l - pos_l) / n_l + pos_r * (n_r - pos_r) / n_r
        gain = pos * (n - pos) / n - after
        i = int(np.argmax(gain))
        return float(gain[i]), int(cut[i])

    segments = {(0, len(xs)): best_split(0, len(xs))}
    edges = []
    while len(edges) < n_bins - 1:
        open_ = {k: v for k, v in segments.items() if v is not None}
        if not open_:
            break
        (lo, hi), (_, cut) = max(open_.items(), key=lambda kv: kv[1][0])
        edges.append((xs[cut - 1] + xs[cut]) / 2.0)
        del segments[(lo, hi)]
        segments[(lo, cut)] = best_split(lo, cut)
        segments[(cut, hi)] = best_split(cut, hi)
    return np.array(sorted(edges))


# Model linear hanya bisa memberi satu bobot per fitur sehingga data disiapkan agar hubungan fitur dengan log-odds mendekati garis lurus
# Urutannya adalah indikator missing, imputasi median, log1p untuk fitur miring, encoding siklik
# Setelah itu binning opsional, one-hot kategori, standardisasi, dan fitur interaksi opsional
# Inovasi dibanding preprocessing biasa adalah binning berbasis target, sel dua dimensi, interaksi fitur, dan booking_index
# Inovasi lain adalah indikator missing, log1p untuk fitur miring, dan encoding siklik untuk bulan serta hari
class LRPreprocessor(BasePreprocessor):

    # log_skewed menerapkan log1p pada fitur non-negatif dengan skewness di atas skew_threshold
    # cyclic mengganti fitur waktu siklik dengan pasangan sin dan cos
    # bin_cols adalah fitur numerik yang ditambah versi one-hot per kelompok dengan jumlah kelompok n_bins
    # onehot_cols adalah fitur numerik diskrit yang di-one-hot per nilai dan nilai besar digabung ke batas atas
    # interaction_cols adalah kolom hasil encoding yang masing-masing dikalikan dengan seluruh kolom lain
    # booking_index menambah fitur hari pemesanan dibuat yaitu hari kedatangan dikurangi days_before_arrival
    # bin_method 'quantile' membagi merata dan 'target' mencari batas yang paling memisahkan kelas pada data train
    # bin_min_leaf adalah jumlah baris minimal pada setiap bin untuk bin_method 'target'
    # cross_bins adalah pasangan fitur numerik yang dijadikan sel dua dimensi dengan satu kolom 0 atau 1 untuk setiap kombinasi kelompok
    # cross_n_bins adalah jumlah kelompok per fitur dan cross_min_count adalah batas minimal baris train agar sebuah sel dibuatkan kolom
    def __init__(self, feature_engineering=True, rare_min_count=20, drop_cols=None, log_skewed=True,
                 skew_threshold=1.0, cyclic=True, bin_cols=None, n_bins=10, onehot_cols=None,
                 interaction_cols=None, booking_index=False, bin_method='quantile', bin_min_leaf=200,
                 cross_bins=None, cross_n_bins=6, cross_min_count=100):
        super().__init__(feature_engineering, rare_min_count, drop_cols)
        self.log_skewed = log_skewed
        self.skew_threshold = skew_threshold
        self.cyclic = cyclic
        self.bin_cols = list(bin_cols or [])
        self.n_bins = n_bins
        self.onehot_cols = dict(onehot_cols or {})
        self.interaction_cols = list(interaction_cols or [])
        self.booking_index = booking_index
        self.bin_method = bin_method
        self.bin_min_leaf = bin_min_leaf
        self.cross_bins = [tuple(p) for p in (cross_bins or [])]
        self.cross_n_bins = cross_n_bins
        self.cross_min_count = cross_min_count

    # Periksa pengaturan binning dan simpan target train lalu lanjut ke fit biasa
    def fit(self, df):
        if self.bin_method not in ('quantile', 'target'):
            raise ValueError(f'bin_method tidak dikenal: {self.bin_method}')
        if self.bin_method == 'target' and TARGET not in df.columns:
            raise ValueError("bin_method 'target' membutuhkan kolom target pada data fit")
        self._target = df[TARGET] if TARGET in df.columns else None
        return super().fit(df)

    # Batas bin satu fitur berupa kuantil merata atau pemisahan terbaik terhadap target pada data train
    def _bin_edges(self, x, n_bins):
        if self.bin_method == 'target':
            return target_edges(x.to_numpy(), self._target.loc[x.index].to_numpy(), n_bins, self.bin_min_leaf)
        return np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)[1:-1]))

    # Nomor sel dua dimensi untuk setiap baris dari kelompok fitur pertama dan kedua
    def _cross_ids(self, filled, a, b):
        edges = self.cross_edges_[(a, b)]
        ia = np.searchsorted(edges[a], filled[a].to_numpy(), side='right')
        ib = np.searchsorted(edges[b], filled[b].to_numpy(), side='right')
        return ia * (len(edges[b]) + 1) + ib

    # Tambahkan fitur booking_index setelah langkah dasar jika diminta
    def _base(self, df):
        out = super()._base(df)
        if not self.booking_index:
            return out
        return out.assign(booking_index=self._booking_index(df))

    # Hari pemesanan dibuat dihitung dari hari awal data, dan nilai yang tidak bisa dihitung dibiarkan NaN
    @staticmethod
    def _booking_index(df):
        arrival = pd.to_datetime(
            dict(year=df['arrival_year'], month=df['arrival_month'], day=df['arrival_day']), errors='coerce')
        booking = arrival - pd.to_timedelta(df['days_before_arrival'], unit='D')
        return (booking - ORIGIN_DAY).dt.days.astype(float)

    # Pelajari seluruh statistik dari data train yaitu median, kolom miring, batas bin, kategori acuan, mean, dan std
    def _fit_extra(self, df):
        # Indikator missing hanya untuk fitur numerik karena kategori kosong sudah menjadi level 'Missing'
        self.missing_cols_ = [c for c in self.num_cols_ if df[c].isna().any()]
        self.medians_ = df[self.num_cols_].median()
        filled = df[self.num_cols_].fillna(self.medians_)

        # Fitur biner tidak ikut di-log karena hanya bernilai 0 dan 1
        self.log_cols_ = []
        if self.log_skewed:
            for c in self.num_cols_:
                x = filled[c]
                if x.min() >= 0 and x.nunique() > 2 and x.skew() > self.skew_threshold:
                    self.log_cols_.append(c)

        self.bin_edges_ = {c: self._bin_edges(filled[c], self.n_bins) for c in self.bin_cols}

        # Pada sel dua dimensi batas kelompok tiap fitur dicari terpisah dengan jumlah kelompok yang lebih kecil
        self.cross_edges_, self.cross_cells_ = {}, {}
        for a, b in self.cross_bins:
            self.cross_edges_[(a, b)] = {c: self._bin_edges(filled[c], self.cross_n_bins) for c in (a, b)}
            cells, counts = np.unique(self._cross_ids(filled, a, b), return_counts=True)
            self.cross_cells_[(a, b)] = cells[counts >= self.cross_min_count]

        # Kategori paling sering dijadikan acuan dan tidak dibuat kolomnya
        # Dengan begitu kolom one-hot tidak kolinear dengan bias
        self.levels_ = {}
        for c in self.cat_cols_:
            values = df[c].fillna('Missing')
            levels = list(self.keep_cats_[c]) + ['Other', 'Missing']
            reference = values.value_counts().idxmax()
            self.levels_[c] = [v for v in levels if v != reference]

        encoded = self._encode(df)
        # Kolom yang konstan di train dibuang karena tidak membawa informasi
        # Kolom yang identik dengan kolom sebelumnya juga dibuang, misalnya is_free_missing dan nightly_rate_idr_missing
        encoded = encoded.loc[:, encoded.std() > 0]
        encoded = encoded.loc[:, ~encoded.T.duplicated()]
        self.columns_ = list(encoded.columns)
        self.means_ = encoded.mean()
        self.stds_ = encoded.std()

        self._fit_interactions((encoded - self.means_) / self.stds_)

    # Encode, standardisasi, lalu tambah fitur interaksi dengan statistik dari train
    def _transform_extra(self, df):
        encoded = self._encode(df).reindex(columns=self.columns_, fill_value=0.0)
        scaled = (encoded - self.means_) / self.stds_
        if not self.pairs_:
            return scaled
        return (self._interactions(scaled) - self.inter_means_) / self.inter_stds_

    # Pilih pasangan kolom interaksi dan pelajari mean serta std hasil perkaliannya
    # Kolom dari fitur yang sama tidak dipasangkan, misalnya days_before_arrival dengan days_before_arrival_bin3
    # Setiap pasangan hanya dibuat sekali
    # Inovasi ini membuat model linear bisa menangkap pengaruh fitur yang saling bergantung
    def _fit_interactions(self, scaled):
        left = [c for c in self.interaction_cols if c in self.columns_]
        missing = set(self.interaction_cols) - set(left)
        if missing:
            raise ValueError(f'kolom interaksi tidak ditemukan: {sorted(missing)}')

        origin = {c: self._origin(c) for c in self.columns_}
        self.pairs_, seen = [], set()
        for a in left:
            for b in self.columns_:
                key = frozenset((a, b))
                if origin[a] == origin[b] or key in seen:
                    continue
                seen.add(key)
                self.pairs_.append((a, b))
        if not self.pairs_:
            return

        # Hasil perkalian yang konstan di train dibuang lalu seluruh kolom akhir distandardisasi ulang
        expanded = self._interactions(scaled)
        keep = expanded.std() > 0
        self.pairs_ = [p for p, k in zip(self.pairs_, keep.iloc[len(self.columns_):]) if k]
        expanded = expanded.loc[:, keep]
        self.inter_means_ = expanded.mean()
        self.inter_stds_ = expanded.std()

    # Gabungkan kolom asli dengan hasil perkalian setiap pasangan
    def _interactions(self, scaled):
        if not self.pairs_:
            return scaled
        values = scaled.to_numpy()
        index = {c: i for i, c in enumerate(self.columns_)}
        products = np.column_stack([values[:, index[a]] * values[:, index[b]] for a, b in self.pairs_])
        names = [f'{a}*{b}' for a, b in self.pairs_]
        return pd.concat([scaled, pd.DataFrame(products, index=scaled.index, columns=names)], axis=1)

    # Nama fitur asal dari sebuah kolom hasil encoding dan indikator missing dianggap fitur tersendiri
    def _origin(self, col):
        if col.endswith('_missing'):
            return col
        for c in sorted(self.num_cols_ + self.cat_cols_, key=len, reverse=True):
            if col == c or col.startswith(c + '_'):
                return c
        return col

    # Ubah DataFrame hasil _base menjadi matriks numerik sebelum standardisasi
    def _encode(self, df):
        out = {}
        for c in self.missing_cols_:
            out[f'{c}_missing'] = df[c].isna().astype(float)

        filled = df[self.num_cols_].fillna(self.medians_)
        for c in self.num_cols_:
            if self.cyclic and c in CYCLIC_COLS:
                angle = 2 * np.pi * filled[c] / CYCLIC_COLS[c]
                out[f'{c}_sin'] = np.sin(angle)
                out[f'{c}_cos'] = np.cos(angle)
                continue
            out[c] = np.log1p(filled[c]) if c in self.log_cols_ else filled[c]

        for c, edges in self.bin_edges_.items():
            bins = np.searchsorted(edges, filled[c].to_numpy(), side='right')
            # Bin pertama dijadikan acuan
            for k in range(1, len(edges) + 1):
                out[f'{c}_bin{k}'] = (bins == k).astype(float)

        for (a, b), cells in self.cross_cells_.items():
            ids = self._cross_ids(filled, a, b)
            for cell in cells:
                out[f'{a}_x_{b}_{cell}'] = (ids == cell).astype(float)

        for c, max_value in self.onehot_cols.items():
            clipped = filled[c].clip(upper=max_value)
            # Nilai 0 dijadikan acuan
            for v in range(1, max_value + 1):
                out[f'{c}_{v}'] = (clipped == v).astype(float)

        for c in self.cat_cols_:
            values = df[c].fillna('Missing')
            for v in self.levels_[c]:
                out[f'{c}_{v}'] = (values == v).astype(float)

        return pd.DataFrame(out, index=df.index).astype(float)
