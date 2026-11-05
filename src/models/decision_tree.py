# Implementasi Decision Tree Learning (CART) from scratch untuk klasifikasi biner.

import numpy as np
import pandas as pd


# Satu node pohon. Daun hanya memakai value (proporsi kelas 1) dan n_samples.
class Node:
    __slots__ = ('feature', 'threshold', 'is_cat', 'left_cats', 'seen_cats',
                 'missing_left', 'left', 'right', 'value', 'n_samples', 'gain')

    # Buat node dengan proporsi kelas 1 dan jumlah sampelnya.
    def __init__(self, value, n_samples):
        self.value = value
        self.n_samples = n_samples
        self.feature = None
        self.threshold = None
        self.is_cat = False
        self.left_cats = None
        self.seen_cats = None
        self.missing_left = True
        self.left = None
        self.right = None
        self.gain = 0.0

    # True jika node tidak punya anak.
    @property
    def is_leaf(self):
        return self.left is None


# Impurity gini atau entropy secara vektor dari jumlah positif dan total sampel.
def _impurity(pos, n, criterion):
    n = np.asarray(n, dtype=float)
    p = np.divide(pos, n, out=np.zeros_like(n), where=n > 0)
    if criterion == 'gini':
        return 2.0 * p * (1.0 - p)
    # Entropy biner, dengan konvensi 0 * log(0) = 0
    q = 1.0 - p
    with np.errstate(divide='ignore', invalid='ignore'):
        h = -(np.where(p > 0, p * np.log2(p), 0.0) + np.where(q > 0, q * np.log2(q), 0.0))
    return h


# CART biner: split numerik dengan threshold, kategori diurutkan berdasarkan proporsi kelas 1,
# missing value diarahkan ke cabang dengan impurity terkecil.
class DecisionTree:
    # min_impurity_decrease: penurunan impurity terbobot minimum agar split diterima.
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, min_impurity_decrease=0.0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease

    # ------------------------------------------------------------ training

    # Latih pohon. Kolom bertipe teks pada X dianggap kategorikal.
    def fit(self, X, y):
        X = pd.DataFrame(X)
        self.feature_names_ = list(X.columns)
        self.is_cat_ = np.array([not pd.api.types.is_numeric_dtype(X[c]) for c in X.columns])
        # Kamus kategori ke kode integer, dipelajari dari data train
        self.cat_maps_ = {}
        for j, c in enumerate(self.feature_names_):
            if self.is_cat_[j]:
                cats = sorted(X[c].dropna().unique())
                self.cat_maps_[c] = {v: i for i, v in enumerate(cats)}

        self._X = self._to_array(X)
        self._y = np.asarray(y).astype(int)
        self.n_total_ = len(self._y)
        self.feature_importances_ = np.zeros(len(self.feature_names_))

        self.root_ = self._build(np.arange(self.n_total_), depth=0)

        total = self.feature_importances_.sum()
        if total > 0:
            self.feature_importances_ /= total
        del self._X, self._y
        return self

    # Ubah DataFrame jadi array float: kategori jadi kode, missing jadi NaN, kategori baru jadi -2.
    def _to_array(self, X):
        out = np.empty((len(X), len(self.feature_names_)), dtype=float)
        for j, c in enumerate(self.feature_names_):
            if self.is_cat_[j]:
                # Kategori yang tidak dikenal saat train diberi kode -2
                codes = X[c].map(self.cat_maps_[c])
                codes = codes.where(X[c].isna() | codes.notna(), -2)
                out[:, j] = codes.astype(float).to_numpy()
            else:
                out[:, j] = pd.to_numeric(X[c], errors='coerce').to_numpy(dtype=float)
        return out

    # Bangun pohon secara rekursif dari indeks baris idx.
    def _build(self, idx, depth):
        y = self._y[idx]
        n, pos = len(y), int(y.sum())
        node = Node(value=pos / n, n_samples=n)

        # Kondisi berhenti (pre-pruning)
        if (pos == 0 or pos == n
                or n < self.min_samples_split
                or n < 2 * self.min_samples_leaf
                or (self.max_depth is not None and depth >= self.max_depth)):
            return node

        best = self._best_split(idx, y)
        if best is None:
            return node

        gain, f, thr, left_cats, seen_cats, miss_left = best
        # Penurunan impurity dibobot proporsi sampel node terhadap seluruh data
        weighted_gain = gain * n / self.n_total_
        if weighted_gain < self.min_impurity_decrease:
            return node

        go_left = self._go_left(self._X[idx, f], thr, left_cats, seen_cats, miss_left,
                                self.is_cat_[f])
        node.feature, node.threshold = f, thr
        node.is_cat = bool(self.is_cat_[f])
        node.left_cats, node.seen_cats = left_cats, seen_cats
        node.missing_left = miss_left
        node.gain = gain
        self.feature_importances_[f] += weighted_gain

        node.left = self._build(idx[go_left], depth + 1)
        node.right = self._build(idx[~go_left], depth + 1)
        return node

    # Cari split terbaik dari semua fitur, None jika tidak ada split valid.
    def _best_split(self, idx, y):
        n = len(y)
        parent = _impurity(y.sum(), n, self.criterion)
        best_gain, best = 1e-12, None

        for f in range(self._X.shape[1]):
            x = self._X[idx, f]
            miss = np.isnan(x)
            n_miss = int(miss.sum())
            pos_miss = int(y[miss].sum())
            xv, yv = x[~miss], y[~miss]
            if len(xv) < 2:
                continue

            if self.is_cat_[f]:
                # Urutkan kategori berdasarkan proporsi positif, lalu ganti nilai x dengan peringkatnya
                cats, inv = np.unique(xv, return_inverse=True)
                if len(cats) < 2:
                    continue
                cnt = np.bincount(inv)
                pos_rate = np.bincount(inv, weights=yv) / cnt
                order = np.argsort(pos_rate, kind='mergesort')
                rank = np.empty(len(cats), dtype=int)
                rank[order] = np.arange(len(cats))
                xv = rank[inv].astype(float)

            sort = np.argsort(xv, kind='mergesort')
            xs, ys = xv[sort], yv[sort]
            # Kandidat titik potong: hanya di antara dua nilai yang berbeda
            cut = np.where(xs[:-1] != xs[1:])[0]
            if len(cut) == 0:
                continue
            cum_pos = np.cumsum(ys)
            n_left = cut + 1.0
            p_left = cum_pos[cut].astype(float)
            n_right = len(xs) - n_left
            p_right = cum_pos[-1] - p_left

            options = [True, False] if n_miss > 0 else [True]
            for miss_left in options:
                nl = n_left + (n_miss if miss_left else 0)
                pl = p_left + (pos_miss if miss_left else 0)
                nr = n_right + (0 if miss_left else n_miss)
                pr = p_right + (0 if miss_left else pos_miss)
                valid = (nl >= self.min_samples_leaf) & (nr >= self.min_samples_leaf)
                if not valid.any():
                    continue
                child = (nl * _impurity(pl, nl, self.criterion)
                         + nr * _impurity(pr, nr, self.criterion)) / n
                gain = np.where(valid, parent - child, -np.inf)
                i = int(np.argmax(gain))
                if gain[i] > best_gain:
                    best_gain = float(gain[i])
                    if self.is_cat_[f]:
                        # Kategori dengan peringkat <= titik potong masuk ke kiri
                        left_cats = cats[order[: int(xs[cut[i]]) + 1]]
                        best = (best_gain, f, None, left_cats, cats, miss_left)
                    else:
                        thr = (xs[cut[i]] + xs[cut[i] + 1]) / 2.0
                        best = (best_gain, f, thr, None, None, miss_left)
        return best

    # Tentukan baris yang masuk cabang kiri. Missing dan kategori tak dikenal ikut arah default.
    @staticmethod
    def _go_left(x, thr, left_cats, seen_cats, miss_left, is_cat):
        if is_cat:
            # Missing dan kategori yang tidak pernah terlihat di node ini ikut arah default
            unknown = np.isnan(x) | ~np.isin(x, seen_cats)
            return np.where(unknown, miss_left, np.isin(x, left_cats))
        return np.where(np.isnan(x), miss_left, x <= thr)

    # ------------------------------------------------------------ prediksi

    # Probabilitas kelas 1 (proporsi kelas 1 di daun) untuk tiap baris.
    def predict_proba(self, X):
        Xa = self._to_array(pd.DataFrame(X))
        out = np.empty(len(Xa))
        stack = [(self.root_, np.arange(len(Xa)))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf or len(idx) == 0:
                out[idx] = node.value
                continue
            go_left = self._go_left(Xa[idx, node.feature], node.threshold, node.left_cats,
                                    node.seen_cats, node.missing_left, node.is_cat)
            stack.append((node.left, idx[go_left]))
            stack.append((node.right, idx[~go_left]))
        return out

    # Label 0/1 berdasarkan threshold.
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------ inspeksi

    # Kedalaman pohon hasil training.
    def get_depth(self):
        def depth(node):
            return 0 if node.is_leaf else 1 + max(depth(node.left), depth(node.right))
        return depth(self.root_)

    # Jumlah daun pohon.
    def get_n_leaves(self):
        def leaves(node):
            return 1 if node.is_leaf else leaves(node.left) + leaves(node.right)
        return leaves(self.root_)

    # Teks kondisi split sebuah node, misalnya 'days_before_arrival <= 150.5'.
    def describe_split(self, node):
        name = self.feature_names_[node.feature]
        if node.is_cat:
            inv = {v: k for k, v in self.cat_maps_[name].items()}
            cats = ', '.join(str(inv[int(c)]) for c in node.left_cats)
            return f'{name} in {{{cats}}}'
        return f'{name} <= {node.threshold:.4g}'
