# Implementasi KNN from scratch untuk klasifikasi biner.

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

    # Jarak setiap baris Q ke seluruh data train.
    def _distances(self, Q):
        if self.p == 2:
            # Akumulasi sqrt(sum((q-x)^2)) tanpa pengurangan dua norma besar.
            # Per fitur agar tidak membentuk tensor query x train x fitur.
            distances = np.zeros((len(Q), len(self.X_)), dtype=float)
            for col in range(self.X_.shape[1]):
                diff = Q[:, col, None] - self.X_[None, :, col]
                np.hypot(distances, diff, out=distances)
            return distances
        diff = np.abs(Q[:, None, :] - self.X_[None, :, :])
        return (diff ** self.p).sum(2) ** (1.0 / self.p)

    # Cari k tetangga terdekat, hasilnya (jarak, indeks) terurut dari yang terdekat.
    def kneighbors(self, X, k=None):
        if not hasattr(self, 'X_'):
            raise ValueError('Panggil fit sebelum mencari tetangga.')
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
            # Stable sort mempertahankan indeks train terkecil untuk jarak sama,
            # termasuk saat tie memotong batas tetangga ke-k.
            idx = np.argsort(d, axis=1, kind='stable')[:, :k]
            all_i.append(idx)
            all_d.append(np.take_along_axis(d, idx, axis=1))
        return np.vstack(all_d), np.vstack(all_i)

    # Probabilitas kelas 1 dari hasil kneighbors, bisa dipakai ulang untuk k yang lebih kecil.
    def proba_from_neighbors(self, dist, idx, k=None):
        k = self._validate_k(self.k if k is None else k, len(self.X_))
        if k > dist.shape[1]:
            raise ValueError('Jumlah tetangga tersimpan harus minimal k.')
        dist, idx = dist[:, :k], idx[:, :k]
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
        return (self.predict_proba(X) >= threshold).astype(int)
