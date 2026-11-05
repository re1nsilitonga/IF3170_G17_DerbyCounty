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
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y).astype(int)
        return self

    # Jarak setiap baris Q ke seluruh data train.
    def _distances(self, Q):
        if self.p == 2:
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab, lebih cepat dari selisih langsung
            sq = (Q ** 2).sum(1)[:, None] + (self.X_ ** 2).sum(1)[None, :] - 2.0 * Q @ self.X_.T
            return np.sqrt(np.maximum(sq, 0.0))
        diff = np.abs(Q[:, None, :] - self.X_[None, :, :])
        return (diff ** self.p).sum(2) ** (1.0 / self.p)

    # Cari k tetangga terdekat, hasilnya (jarak, indeks) terurut dari yang terdekat.
    def kneighbors(self, X, k=None):
        k = k or self.k
        X = np.asarray(X, dtype=float)
        bs = self.batch_size if self.p == 2 else max(1, self.batch_size // 16)
        all_d, all_i = [], []
        for s in range(0, len(X), bs):
            d = self._distances(X[s:s + bs])
            part = np.argpartition(d, k - 1, axis=1)[:, :k]
            dk = np.take_along_axis(d, part, axis=1)
            order = np.argsort(dk, axis=1, kind='mergesort')
            all_i.append(np.take_along_axis(part, order, axis=1))
            all_d.append(np.take_along_axis(dk, order, axis=1))
        return np.vstack(all_d), np.vstack(all_i)

    # Probabilitas kelas 1 dari hasil kneighbors, bisa dipakai ulang untuk k yang lebih kecil.
    def proba_from_neighbors(self, dist, idx, k=None):
        k = k or self.k
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
