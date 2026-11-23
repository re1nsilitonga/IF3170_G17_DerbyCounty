# Implementasi KNN from scratch untuk klasifikasi biner.
# Referensi:
# [1] E. Ferdian, "KNN", salindia IF3170 Inteligensi Artifisial, ITB, 2026
#     (algoritma dasar: simpan data train, cari k tetangga terdekat, voting mayoritas).
# [2] P. Cunningham dan S. J. Delany, "k-Nearest Neighbour Classifiers: 2nd Edition
#     (with Python examples)", arXiv:2004.04523, 2020 (jarak Minkowski dan voting
#     terbobot jarak 1/d).

import numpy as np


# KNN dengan jarak Minkowski (p=1 Manhattan, p=2 Euclidean), input harus numerik dan sudah di-scaling.
class KNN:

    # weights: 'uniform' atau 'distance'; batch_size: jumlah query per hitungan jarak.
    def __init__(self, k=5, p=2, weights='uniform', batch_size=256):
        self.k = k
        self.p = p
        self.weights = weights
        self.batch_size = batch_size

    # Lazy learner, training cukup menyimpan data.
    def fit(self, X, y):
        X = self._validate_X(X)
        if len(X) == 0:
            raise ValueError('Data train tidak boleh kosong.')
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError('y harus satu dimensi dengan jumlah sampel sama seperti X.')
        if y.dtype.kind not in 'biuf' or not np.isfinite(y).all() or not np.isin(y, [0, 1]).all():
            raise ValueError('y harus berisi label biner 0/1 yang finite.')
        self._validate_k(self.k, len(X))
        if (isinstance(self.p, (bool, np.bool_))
                or not isinstance(self.p, (int, float, np.integer, np.floating))
                or not np.isfinite(self.p) or self.p < 1):
            raise ValueError('p harus bilangan finite >= 1.')
        if self.weights not in ('uniform', 'distance'):
            raise ValueError("weights harus 'uniform' atau 'distance'.")
        if (isinstance(self.batch_size, (bool, np.bool_))
                or not isinstance(self.batch_size, (int, np.integer)) or self.batch_size < 1):
            raise ValueError('batch_size harus bilangan bulat positif.')
        # Salin agar perubahan input oleh pemanggil tidak mengubah model hasil fit.
        self.X_ = X.copy()
        self.y_ = y.astype(int, copy=True)
        return self

    @staticmethod
    def _validate_X(X):
        if np.iscomplexobj(X):
            raise ValueError('X harus berisi bilangan real.')
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0:
            raise ValueError('X harus dua dimensi dengan minimal satu fitur.')
        if not np.isfinite(X).all():
            raise ValueError('X harus berisi nilai finite, tanpa NaN/inf.')
        return X

    @staticmethod
    def _validate_k(k, n_train):
        if (isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer))
                or not 1 <= k <= n_train):
            raise ValueError('k harus bilangan bulat dengan 1 <= k <= n_train.')
        return int(k)

    def _check_fitted(self):
        if not hasattr(self, 'X_') or not hasattr(self, 'y_'):
            raise ValueError('Panggil fit sebelum mencari tetangga atau membuat prediksi.')

    _EUCLID_BLOCK = 32

    # Jarak setiap baris Q ke seluruh data train.
    def _distances(self, Q):
        if self.p == 2:
            # d(q, x) = sqrt(sum_j (q_j - x_j)^2)
            XT = np.ascontiguousarray(self.X_.T)
            sq = np.zeros((len(Q), len(self.X_)), dtype=float)
            for s in range(0, len(Q), self._EUCLID_BLOCK):
                block = sq[s:s + self._EUCLID_BLOCK]
                q = Q[s:s + self._EUCLID_BLOCK]
                buf = np.empty_like(block)
                for col in range(XT.shape[0]):
                    np.subtract(q[:, col, None], XT[col][None, :], out=buf)
                    np.multiply(buf, buf, out=buf)
                    block += buf
            return np.sqrt(sq, out=sq)
        diff = np.abs(Q[:, None, :] - self.X_[None, :, :])
        return (diff ** self.p).sum(2) ** (1.0 / self.p)

    # Indeks k jarak terkecil per baris
    @staticmethod
    def _select_k(d, k):
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        sel = np.take_along_axis(d, idx, axis=1)
        # Urutkan kandidat berdasarkan jarak, lalu indeks train terkecil untuk jarak sama.
        order = np.lexsort((idx, sel), axis=1)
        idx = np.take_along_axis(idx, order, axis=1)
        # argpartition tidak menjamin indeks terkecil saat tie memotong batas ke-k;
        # baris tersebut dipilih ulang dari seluruh kandidat dengan jarak <= jarak ke-k.
        kth = np.take_along_axis(d, idx[:, -1:], axis=1)
        tied = np.flatnonzero((d <= kth).sum(1) > k)
        for r in tied:
            cand = np.flatnonzero(d[r] <= kth[r, 0])
            idx[r] = cand[np.argsort(d[r, cand], kind='stable')[:k]]
        return idx

    # Cari k tetangga terdekat, hasilnya (jarak, indeks) terurut dari yang terdekat.
    def kneighbors(self, X, k=None):
        self._check_fitted()
        k = self._validate_k(self.k if k is None else k, len(self.X_))
        X = self._validate_X(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('Jumlah fitur query harus sama seperti data train.')
        if len(X) == 0:
            return np.empty((0, k), dtype=float), np.empty((0, k), dtype=int)
        bs = self.batch_size if self.p == 2 else max(1, self.batch_size // 16)
        all_d, all_i = [], []
        for s in range(0, len(X), bs):
            d = self._distances(X[s:s + bs])
            # Jarak sama: indeks train terkecil didahulukan, termasuk saat tie
            # memotong batas tetangga ke-k. Copy melepaskan view hasil argpartition.
            idx = self._select_k(d, k).copy()
            all_i.append(idx)
            all_d.append(np.take_along_axis(d, idx, axis=1))
        # Simpan hanya k tetangga per query; matriks jarak penuh tiap batch dilepas.
        return np.vstack(all_d), np.vstack(all_i)

    # Probabilitas kelas 1 dari hasil kneighbors, bisa dipakai ulang untuk k yang lebih kecil.
    def proba_from_neighbors(self, dist, idx, k=None):
        self._check_fitted()
        k = self._validate_k(self.k if k is None else k, len(self.X_))
        if k > dist.shape[1]:
            raise ValueError('Jumlah tetangga tersimpan harus minimal k.')
        dist, idx = dist[:, :k], idx[:, :k]
        if len(dist) == 0:
            return np.empty(0, dtype=float)
        labels = self.y_[idx]
        if self.weights == 'distance':
            w = 1.0 / (dist + 1e-9)
            return (w * labels).sum(1) / w.sum(1)
        return labels.mean(1)

    # Probabilitas kelas 1, yaitu proporsi (terbobot) tetangga berlabel 1.
    def predict_proba(self, X):
        dist, idx = self.kneighbors(X)
        return self.proba_from_neighbors(dist, idx)

    # Label 0/1 berdasarkan threshold.
    def predict(self, X, threshold=0.5):
        """Hasil integer 0/1; probabilitas tepat pada threshold memilih kelas 1."""
        if (isinstance(threshold, (bool, np.bool_))
                or not isinstance(threshold, (int, float, np.integer, np.floating))
                or not np.isfinite(threshold) or not 0 <= threshold <= 1):
            raise ValueError('threshold harus bilangan finite dalam rentang [0, 1].')
        return (self.predict_proba(X) >= threshold).astype(int)
