import numpy as np


# input = kode kategori dari NBPreprocessor
class NaiveBayes:

    # alpha = 1 pake laplace smoothing
    def __init__(self, alpha=1.0):
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError('alpha harus positif dan berhingga')
        self.alpha = alpha

    # itung peluang awal kelas dan peluang setiap kategori di dalam kelas
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or X.shape[1] == 0 or y.shape != (len(X),):
            raise ValueError('X harus berbentuk tabel fitur dan y harus sejajar dengan baris X')
        if not np.issubdtype(X.dtype, np.integer) or (X < 0).any():
            raise ValueError('X harus berisi kode kategori bilangan bulat nonnegatif')
        if not np.array_equal(np.unique(y), [0, 1]):
            raise ValueError('Data latih harus punya label 0 dan 1')

        class_counts = np.bincount(y.astype(int), minlength=2)
        self.log_prior_ = np.log(class_counts / len(y))
        self.log_likelihoods_ = []

        for column in range(X.shape[1]):
            codes = X[:, column]
            # code 0 untuk Other, code 1 buat nilai kosong selalu tersedia
            n_categories = max(2, int(codes.max()) + 1)
            probabilities = np.empty((2, n_categories))

            for label in (0, 1):
                counts = np.bincount(codes[y == label].astype(int), minlength=n_categories)
                # rumus smoothing based on docs
                probabilities[label] = (counts + self.alpha) / (
                    class_counts[label] + self.alpha * n_categories
                )

            self.log_likelihoods_.append(np.log(probabilities))

        return self

    # probabilitas kelas 1 based on prior sama kode kategori tiap baris
    def predict_proba(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(self.log_likelihoods_):
            raise ValueError('Jumlah dan urutan fitur harus sama dengan saat fit')
        if not np.issubdtype(X.dtype, np.integer) or (X < 0).any():
            raise ValueError('X harus berisi kode kategori bilangan bulat nonnegatif')

        log_scores = np.empty((len(X), 2))
        for label in (0, 1):
            log_scores[:, label] = self.log_prior_[label]
            for column, table in enumerate(self.log_likelihoods_):
                # totalin log setara with kaliin peluang asli
                log_scores[:, label] += table[label, X[:, column]]

        # pergeseran yang sama di kedua kelas biar jaga perbandingan skornya
        weights = np.exp(log_scores - log_scores.max(axis=1, keepdims=True))
        return weights[:, 1] / weights.sum(axis=1)

    # label 0/1 based on threshold
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
