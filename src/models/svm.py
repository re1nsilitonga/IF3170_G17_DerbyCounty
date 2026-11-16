# Implementasi SVM Linear from scratch untuk klasifikasi biner.

import warnings

import numpy as np


class SVM:

    # C mengatur penalti margin, tol toleransi optimasi, dan max_iter batas putaran training.
    def __init__(self, C=1.0, tol=1e-3, max_iter=1000, seed=42):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed

    # Latih bobot dan bias dengan optimasi pasangan alpha menggunakan SMO.
    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError('y harus berupa vektor label numerik finite sepanjang X.')
        classes, encoded = np.unique(y, return_inverse=True)
        if len(classes) != 2:
            raise ValueError('SVM membutuhkan tepat dua kelas pada data train.')
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError('C harus positif dan finite.')
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError('tol harus positif dan finite.')
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError('max_iter harus berupa bilangan bulat positif.')

        # Inisialisasi bobot dan alpha, lalu ubah label menjadi -1 dan +1.
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.coef_ = np.zeros(self.n_features_in_)
        self.intercept_ = 0.0
        self.alpha_ = np.zeros(len(X))
        signed_y = 2.0 * encoded - 1.0
        errors = -signed_y.copy()
        rng = np.random.default_rng(self.seed)
        eps = 1e-8 * min(1.0, self.C)

        # Perbarui pasangan alpha sampai tidak ada perubahan atau batas iterasi tercapai.
        for iteration in range(self.max_iter):
            changed = 0
            for i in rng.permutation(len(X)):
                # Cari sampel yang masih melanggar kondisi optimalitas KKT.
                residual = signed_y[i] * errors[i]
                violates = ((residual < -self.tol and self.alpha_[i] < self.C - eps)
                            or (residual > self.tol and self.alpha_[i] > eps))
                if not violates:
                    continue

                # Pilih pasangan dengan selisih error terbesar, lalu coba kandidat lainnya.
                non_bound = np.flatnonzero((self.alpha_ > eps) & (self.alpha_ < self.C - eps))
                if len(non_bound) > 1:
                    best = non_bound[np.argmax(np.abs(errors[i] - errors[non_bound]))]
                else:
                    best = int(np.argmax(np.abs(errors[i] - errors)))
                candidates = np.concatenate(([best], rng.permutation(non_bound),
                                             rng.permutation(len(X))))
                for j in candidates:
                    if self._take_step(X, signed_y, errors, i, j, eps):
                        changed += 1
                        break

            self.n_iter_ = iteration + 1
            # Hitung ulang error untuk mengurangi selisih pembulatan selama pembaruan.
            errors = X @ self.coef_ + self.intercept_ - signed_y
            if changed == 0:
                break

        # Periksa konvergensi dari kondisi KKT pada hasil akhir training.
        residuals = signed_y * errors
        violations = ((residuals < -self.tol) & (self.alpha_ < self.C - eps)
                      | (residuals > self.tol) & (self.alpha_ > eps))
        self.converged_ = not violations.any()
        # Simpan support vector, yaitu sampel dengan alpha positif.
        self.support_ = np.flatnonzero(self.alpha_ > eps)
        self.support_vectors_ = X[self.support_].copy()
        self.dual_coef_ = self.alpha_[self.support_] * signed_y[self.support_]
        if not self.converged_:
            warnings.warn('SVM berhenti sebelum seluruh kondisi KKT memenuhi tol. '
                          'Periksa scaling atau naikkan max_iter.', RuntimeWarning, stacklevel=2)
        return self

    # Optimasi dua alpha sekaligus mempertahankan sum(alpha_i * y_i) = 0.
    def _take_step(self, X, y, errors, i, j, eps):
        if i == j:
            return False
        old_i, old_j = self.alpha_[i], self.alpha_[j]
        # Tentukan batas alpha_j agar kedua alpha tetap berada dalam rentang 0 sampai C.
        if y[i] != y[j]:
            lower = max(0.0, old_j - old_i)
            upper = min(self.C, self.C + old_j - old_i)
        else:
            lower = max(0.0, old_i + old_j - self.C)
            upper = min(self.C, old_i + old_j)
        if upper - lower <= eps:
            return False

        # Hitung alpha_j baru dari jarak kedua sampel dan selisih errornya.
        difference = X[i] - X[j]
        eta = difference @ difference
        slope = y[j] * (errors[i] - errors[j])
        if eta > 0:
            new_j = np.clip(old_j + slope / eta, lower, upper)
        else:
            # Untuk titik identik, pilih ujung rentang tanpa membagi dengan nol.
            new_j = upper if slope > 0 else lower
        if abs(new_j - old_j) <= eps:
            return False
        # Sesuaikan alpha_i agar sum(alpha_i * y_i) tetap nol, lalu hitung perubahan bobot.
        new_i = old_i + y[i] * y[j] * (old_j - new_j)
        delta_i, delta_j = new_i - old_i, new_j - old_j
        weight_change = y[i] * delta_i * X[i] + y[j] * delta_j * X[j]

        # Tentukan bias baru dari sampel yang alphanya berada di antara 0 dan C.
        bias_i = self.intercept_ - errors[i] - X[i] @ weight_change
        bias_j = self.intercept_ - errors[j] - X[j] @ weight_change
        if eps < new_i < self.C - eps:
            bias = bias_i
        elif eps < new_j < self.C - eps:
            bias = bias_j
        else:
            bias = (bias_i + bias_j) / 2.0

        # Terapkan perubahan bobot dan bias ke model serta error seluruh sampel.
        errors += X @ weight_change + bias - self.intercept_
        self.alpha_[i], self.alpha_[j] = new_i, new_j
        self.coef_ += weight_change
        self.intercept_ = float(bias)
        return True

    # Pastikan fitur berupa matriks numerik tanpa NaN atau infinity.
    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError('X harus berupa matriks numerik finite dengan minimal satu fitur.')
        return X

    # Hitung score w.x + b, score positif mengarah ke kelas kedua.
    def decision_function(self, X):
        if not hasattr(self, 'coef_'):
            raise RuntimeError('SVM harus di-fit sebelum melakukan prediksi.')
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Jumlah fitur harus sama dengan data train.')
        return X @ self.coef_ + self.intercept_

    # Ubah tanda score menjadi label asli.
    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]
