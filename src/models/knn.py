"""Implementasi k-Nearest Neighbor (KNN) from scratch untuk klasifikasi biner.

Mengacu pada slide kuliah IF3170 "k-Nearest Neighbor (KNN)": jarak Minkowski
(p=1 Manhattan, p=2 Euclidean) dan voting mayoritas k tetangga terdekat.
Opsi weights='distance' mengikuti P. Cunningham dan S. J. Delany,
"k-Nearest Neighbour Classifiers", arXiv:2004.04523, 2020.
"""

import numpy as np


class KNN:
    """KNN dengan jarak Minkowski. Fitur harus sudah numerik dan sudah di-scaling.

    Parameter
    ---------
    k : jumlah tetangga
    p : orde jarak Minkowski (1 Manhattan, 2 Euclidean)
    weights : 'uniform' (suara sama) atau 'distance' (suara dibobot 1/jarak)
    batch_size : jumlah baris query yang dihitung jaraknya sekaligus, untuk hemat memori
    """

    def __init__(self, k=5, p=2, weights='uniform', batch_size=256):
        """Simpan hyperparameter."""
        self.k = k
        self.p = p
        self.weights = weights
        self.batch_size = batch_size

    def fit(self, X, y):
        """KNN adalah lazy learner, jadi training cukup menyimpan data."""
        self.X_ = np.asarray(X, dtype=float)
        self.y_ = np.asarray(y).astype(int)
        return self

    def _distances(self, Q):
        """Hitung jarak dari setiap baris Q ke seluruh data train."""
        if self.p == 2:
            # ||a - b||^2 = ||a||^2 + ||b||^2 - 2ab, jauh lebih cepat dari selisih langsung
            sq = (Q ** 2).sum(1)[:, None] + (self.X_ ** 2).sum(1)[None, :] - 2.0 * Q @ self.X_.T
            return np.sqrt(np.maximum(sq, 0.0))
        diff = np.abs(Q[:, None, :] - self.X_[None, :, :])
        return (diff ** self.p).sum(2) ** (1.0 / self.p)

    def kneighbors(self, X, k=None):
        """Cari k tetangga terdekat. Mengembalikan (jarak, indeks) terurut dari yang terdekat."""
        k = k or self.k
        X = np.asarray(X, dtype=float)
        bs = self.batch_size if self.p == 2 else max(1, self.batch_size // 16)
        all_d, all_i = [], []
        for s in range(0, len(X), bs):
            d = self._distances(X[s:s + bs])
            # argpartition mengambil k terkecil tanpa mengurutkan semua data
            part = np.argpartition(d, k - 1, axis=1)[:, :k]
            dk = np.take_along_axis(d, part, axis=1)
            order = np.argsort(dk, axis=1, kind='mergesort')
            all_i.append(np.take_along_axis(part, order, axis=1))
            all_d.append(np.take_along_axis(dk, order, axis=1))
        return np.vstack(all_d), np.vstack(all_i)

    def proba_from_neighbors(self, dist, idx, k=None):
        """Hitung probabilitas kelas positif dari hasil kneighbors (bisa dipakai ulang untuk k kecil)."""
        k = k or self.k
        dist, idx = dist[:, :k], idx[:, :k]
        labels = self.y_[idx]
        if self.weights == 'distance':
            w = 1.0 / (dist + 1e-9)
            return (w * labels).sum(1) / w.sum(1)
        return labels.mean(1)

    def predict_proba(self, X):
        """Probabilitas kelas positif, yaitu proporsi (terbobot) tetangga berlabel 1."""
        dist, idx = self.kneighbors(X)
        return self.proba_from_neighbors(dist, idx)

    def predict(self, X, threshold=0.5):
        """Label 0/1 berdasarkan threshold probabilitas."""
        return (self.predict_proba(X) >= threshold).astype(int)
