# Implementasi SVM from scratch dengan kernel Linear, Polynomial, dan Gaussian.

import warnings
from functools import lru_cache

import numpy as np


class SVM:

    # C mengatur penalti margin, tol toleransi optimasi, dan max_iter batas putaran training.
    def __init__(self, C=1.0, tol=1e-3, max_iter=1000, seed=42,
                 kernel='linear', degree=3, gamma='scale', coef0=0.0):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

    # Latih SVM, lalu pelajari kalibrasi probabilitas dari holdout di dalam data train.
    def fit(self, X, y):
        X = self._validate_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != len(X) or not np.isfinite(y).all():
            raise ValueError('y harus berupa vektor label numerik finite sepanjang X.')
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError('SVM membutuhkan tepat dua kelas pada data train.')
        if not np.isfinite(self.C) or self.C <= 0:
            raise ValueError('C harus positif dan finite.')
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError('tol harus positif dan finite.')
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError('max_iter harus berupa bilangan bulat positif.')
        if self.kernel not in ('linear', 'poly', 'rbf'):
            raise ValueError("kernel harus 'linear', 'poly', atau 'rbf'.")
        if not isinstance(self.degree, (int, np.integer)) or self.degree < 1:
            raise ValueError('degree harus berupa bilangan bulat positif.')
        if not np.isfinite(self.coef0) or self.coef0 < 0:
            raise ValueError('coef0 harus nonnegatif dan finite.')
        if isinstance(self.gamma, str):
            if self.gamma not in ('scale', 'auto'):
                raise ValueError("gamma harus 'scale', 'auto', atau angka positif.")
        elif not np.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError('gamma harus positif dan finite.')

        self._fit_svm(X, y)
        self._calibrate(X, y)
        return self

    # Optimasi pasangan alpha menggunakan SMO untuk kernel yang dipilih.
    def _fit_svm(self, X, y):
        classes, encoded = np.unique(y, return_inverse=True)
        if self.gamma == 'scale':
            variance = X.var()
            self.gamma_ = 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
        elif self.gamma == 'auto':
            self.gamma_ = 1.0 / X.shape[1]
        else:
            self.gamma_ = float(self.gamma)

        # Inisialisasi bobot dan alpha, lalu ubah label menjadi -1 dan +1.
        self.classes_ = classes
        self.n_features_in_ = X.shape[1]
        self.coef_ = np.zeros(self.n_features_in_) if self.kernel == 'linear' else None
        self.intercept_ = 0.0
        self.alpha_ = np.zeros(len(X))
        signed_y = 2.0 * encoded - 1.0
        errors = -signed_y.copy()
        rng = np.random.default_rng(self.seed)
        eps = 1e-8 * min(1.0, self.C)

        # Simpan sebagian kolom kernel agar memori tidak tumbuh menjadi matriks n x n.
        @lru_cache(maxsize=128)
        def kernel_column(index):
            return self._kernel(X, X[index:index + 1]).ravel()

        # Acak kandidat berikutnya hanya jika kandidat sebelumnya tidak bisa dipakai.
        def candidates(best, non_bound):
            yield best
            yield from rng.permutation(non_bound)
            yield from rng.permutation(len(X))

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
                for j in candidates(best, non_bound):
                    if self._take_step(X, signed_y, errors, i, j, eps, kernel_column):
                        changed += 1
                        break

            self.n_iter_ = iteration + 1
            if self.kernel == 'linear':
                errors = X @ self.coef_ + self.intercept_ - signed_y
            if changed == 0:
                break

        # Simpan support vector, yaitu sampel dengan alpha positif.
        self.support_ = np.flatnonzero(self.alpha_ > eps)
        self.support_vectors_ = X[self.support_].copy()
        self.dual_coef_ = self.alpha_[self.support_] * signed_y[self.support_]
        kernel_column.cache_clear()

        # Periksa kondisi KKT menggunakan score akhir untuk memastikan konvergensi.
        errors = self.decision_function(X) - signed_y
        residuals = signed_y * errors
        violations = ((residuals < -self.tol) & (self.alpha_ < self.C - eps)
                      | (residuals > self.tol) & (self.alpha_ > eps))
        self.converged_ = not violations.any()
        if not self.converged_:
            warnings.warn('SVM berhenti sebelum seluruh kondisi KKT memenuhi tol. '
                          'Periksa scaling atau naikkan max_iter.', RuntimeWarning, stacklevel=2)
        return self

    # Optimasi dua alpha sekaligus mempertahankan sum(alpha_i * y_i) = 0.
    def _take_step(self, X, y, errors, i, j, eps, kernel_column):
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

        # Hitung perubahan alpha dari nilai kernel kedua sampel dan selisih errornya.
        if self.kernel == 'linear':
            difference = X[i] - X[j]
            eta = difference @ difference
        else:
            column_i, column_j = kernel_column(i), kernel_column(j)
            eta = max(column_i[i] + column_j[j] - 2.0 * column_i[j], 0.0)
        slope = y[j] * (errors[i] - errors[j])
        if eta > 0:
            new_j = np.clip(old_j + slope / eta, lower, upper)
        else:
            # Untuk titik identik, pilih ujung rentang tanpa membagi dengan nol.
            new_j = upper if slope > 0 else lower
        if abs(new_j - old_j) <= eps:
            return False
        # Sesuaikan alpha_i agar sum(alpha_i * y_i) tetap nol.
        new_i = old_i + y[i] * y[j] * (old_j - new_j)
        delta_i, delta_j = new_i - old_i, new_j - old_j
        if self.kernel == 'linear':
            # Gabungkan perubahan bobot agar pasangan yang diterima cukup satu perkalian matriks.
            weight_change = y[i] * delta_i * X[i] + y[j] * delta_j * X[j]
            score_change = self._kernel(X, weight_change[None, :]).ravel()
        else:
            score_change = y[i] * delta_i * column_i + y[j] * delta_j * column_j

        # Tentukan bias baru dari sampel yang alphanya berada di antara 0 dan C.
        bias_i = self.intercept_ - errors[i] - score_change[i]
        bias_j = self.intercept_ - errors[j] - score_change[j]
        if eps < new_i < self.C - eps:
            bias = bias_i
        elif eps < new_j < self.C - eps:
            bias = bias_j
        else:
            bias = (bias_i + bias_j) / 2.0

        # Terapkan perubahan score dan bias, bobot eksplisit hanya dipakai kernel linear.
        errors += score_change + bias - self.intercept_
        self.alpha_[i], self.alpha_[j] = new_i, new_j
        if self.kernel == 'linear':
            self.coef_ += weight_change
        self.intercept_ = float(bias)
        return True

    # Hitung kemiripan seluruh pasangan baris pada X dan Y.
    def _kernel(self, X, Y):
        with np.errstate(over='ignore', invalid='ignore'):
            product = X @ Y.T
            if self.kernel == 'poly':
                result = (self.gamma_ * product + self.coef0) ** self.degree
            elif self.kernel == 'rbf':
                distance = (X ** 2).sum(axis=1)[:, None] + (Y ** 2).sum(axis=1) - 2.0 * product
                result = np.exp(-self.gamma_ * np.maximum(distance, 0.0))
            else:
                result = product
        if not np.isfinite(result).all():
            raise ValueError('Perhitungan kernel tidak finite, periksa scaling dan parameter kernel.')
        return result

    # Gunakan 20 persen dari tiap kelas untuk kalibrasi model sementara.
    def _calibrate(self, X, y):
        positive = y == self.classes_[1]
        counts = np.bincount(positive.astype(int), minlength=2)
        if counts.min() < 2:
            self.probA_ = 0.0
            self.probB_ = float(np.log((counts[0] + 1.0) / (counts[1] + 1.0)))
            warnings.warn('Sampel tiap kelas terlalu sedikit untuk holdout kalibrasi. '
                          'Probabilitas memakai prior kelas.', RuntimeWarning, stacklevel=2)
            return

        rng = np.random.default_rng(self.seed)
        holdout = []
        for label in self.classes_:
            indices = rng.permutation(np.flatnonzero(y == label))
            size = max(1, len(indices) // 5)
            holdout.extend(indices[:size])
        holdout = np.asarray(holdout)
        training = np.ones(len(X), dtype=bool)
        training[holdout] = False

        calibration_model = SVM(C=self.C, tol=self.tol, max_iter=self.max_iter,
                                seed=self.seed, kernel=self.kernel, degree=self.degree,
                                gamma=self.gamma_, coef0=self.coef0)
        calibration_model._fit_svm(X[training], y[training])
        scores = calibration_model.decision_function(X[holdout])
        self._fit_sigmoid(scores, positive[holdout])

    # Pelajari sigmoid P(kelas kedua) = 1 / (1 + exp(A * score + B)).
    def _fit_sigmoid(self, scores, positive):
        n_positive = int(positive.sum())
        n_negative = len(positive) - n_positive
        target = np.where(positive, (n_positive + 1.0) / (n_positive + 2.0),
                          1.0 / (n_negative + 2.0))
        center = scores.mean()
        scale = scores.std()
        if scale == 0:
            scale = 1.0
        design = np.column_stack(((scores - center) / scale, np.ones(len(scores))))
        params = np.array([0.0, np.log((n_negative + 1.0) / (n_positive + 1.0))])

        # Newton memperbarui A dan B, line search memastikan loss turun.
        for _ in range(100):
            logits = design @ params
            probability = np.exp(-np.logaddexp(0.0, logits))
            gradient = design.T @ (target - probability)
            if np.max(np.abs(gradient)) < 1e-5:
                break
            weights = probability * (1.0 - probability)
            hessian = design.T @ (weights[:, None] * design) + np.eye(2) * 1e-12
            direction = np.linalg.solve(hessian, gradient)
            loss = np.sum(target * np.logaddexp(0.0, logits)
                          + (1.0 - target) * np.logaddexp(0.0, -logits))
            step = 1.0
            while step >= 1e-10:
                candidate = params - step * direction
                candidate_logits = design @ candidate
                candidate_loss = np.sum(target * np.logaddexp(0.0, candidate_logits)
                                        + (1.0 - target) * np.logaddexp(0.0, -candidate_logits))
                if candidate_loss <= loss - 1e-4 * step * (gradient @ direction):
                    params = candidate
                    break
                step *= 0.5
            if step < 1e-10:
                break

        self.probA_ = float(params[0] / scale)
        self.probB_ = float(params[1] - self.probA_ * center)

    # Pastikan fitur berupa matriks numerik tanpa NaN atau infinity.
    @staticmethod
    def _validate_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] == 0 or not np.isfinite(X).all():
            raise ValueError('X harus berupa matriks numerik finite dengan minimal satu fitur.')
        return X

    # Kernel linear memakai w.x + b, kernel lainnya memakai kontribusi support vector.
    def decision_function(self, X):
        if not hasattr(self, 'support_'):
            raise RuntimeError('SVM harus di-fit sebelum melakukan prediksi.')
        X = self._validate_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError('Jumlah fitur harus sama dengan data train.')
        if getattr(self, 'kernel', 'linear') == 'linear':
            return X @ self.coef_ + self.intercept_
        scores = np.empty(len(X))
        for start in range(0, len(X), 256):
            kernel = self._kernel(X[start:start + 256], self.support_vectors_)
            scores[start:start + 256] = kernel @ self.dual_coef_ + self.intercept_
        return scores

    # Probabilitas kelas kedua dalam array satu dimensi untuk pipeline validasi tim.
    def predict_proba(self, X):
        if not hasattr(self, 'probA_'):
            raise RuntimeError('SVM harus di-fit dengan kalibrasi sebelum menghitung probabilitas.')
        logits = self.probA_ * self.decision_function(X) + self.probB_
        return np.exp(-np.logaddexp(0.0, logits))

    # Gunakan tanda score secara default, atau threshold probabilitas jika diberikan.
    def predict(self, X, threshold=None):
        if threshold is not None:
            if not np.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError('threshold harus berada dalam rentang 0 sampai 1.')
            probability = self.predict_proba(X)
            return self.classes_[(probability >= threshold).astype(int)]
        scores = self.decision_function(X)
        return self.classes_[(scores >= 0).astype(int)]
