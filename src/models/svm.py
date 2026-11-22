# SVM dual dengan kernel Linear, Polynomial, dan Gaussian serta solver SciPy.

import warnings

import numpy as np
from scipy.optimize import minimize


class SVM:

    # C mengatur penalti margin, tol toleransi optimasi, dan max_iter batas putaran training.
    def __init__(self, C=1.0, tol=1e-3, max_iter=1000, seed=42,
                 kernel='linear', degree=3, gamma='scale', coef0=0.0, device='cpu'):
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.seed = seed
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.device = device

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

    # Pecah QP dual menjadi blok kecil agar tidak menyimpan matriks kernel n x n.
    def _fit_svm(self, X, y):
        self.device_ = self._resolve_device()
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        if self.gamma == 'scale':
            variance = X.var()
            self.gamma_ = 1.0 / (X.shape[1] * variance) if variance > 0 else 1.0
        elif self.gamma == 'auto':
            self.gamma_ = 1.0 / X.shape[1]
        else:
            self.gamma_ = float(self.gamma)

        signed_y = 2.0 * encoded - 1.0
        self.alpha_ = np.zeros(len(X))
        gradient = -np.ones(len(X))
        eps = 1e-8 * min(1.0, self.C)
        features = None
        if self.device_ == 'cuda':
            try:
                features = self._gpu_features(X)
            except MemoryError:
                if self.device != 'auto':
                    raise
                warnings.warn('Memori GPU tidak cukup; training beralih ke CPU.', RuntimeWarning)
                self.device_ = 'cpu'

        steps_per_pass = (len(X) + 63) // 64
        self.n_iter_ = 0
        for iteration in range(self.max_iter * steps_per_pass):
            # Dua himpunan kandidat memberi batas bawah dan atas bias pada kondisi KKT.
            values = -signed_y * gradient
            up = np.flatnonzero(((signed_y > 0) & (self.alpha_ < self.C - eps))
                                | ((signed_y < 0) & (self.alpha_ > eps)))
            low = np.flatnonzero(((signed_y > 0) & (self.alpha_ > eps))
                                 | ((signed_y < 0) & (self.alpha_ < self.C - eps)))
            if values[up].max() - values[low].min() <= 2.0 * self.tol:
                break
            working = np.unique(np.concatenate((up[np.argsort(values[up])[-32:]],
                                                low[np.argsort(values[low])[:32]])))
            if features is None:
                kernel = self._kernel(X, X[working])
            else:
                import torch
                try:
                    with torch.inference_mode():
                        kernel = self._kernel(features, features[working], xp=torch).cpu().numpy()
                except torch.cuda.OutOfMemoryError as error:
                    if self.device != 'auto':
                        raise MemoryError('VRAM tidak cukup untuk blok kernel SVM.') from error
                    warnings.warn('Memori GPU tidak cukup; training beralih ke CPU.', RuntimeWarning)
                    features = None
                    self.device_ = 'cpu'
                    kernel = self._kernel(X, X[working])
            columns = signed_y[:, None] * kernel * signed_y[working]
            block = columns[working]
            old = self.alpha_[working].copy()
            labels = signed_y[working]
            linear = gradient[working] - block @ old

            # SciPy menangani optimasi QP beserta batas alpha dan kendala jumlah alpha*y = 0.
            result = minimize(lambda a: (0.5 * a @ block @ a + linear @ a, block @ a + linear),
                              old, jac=True, method='SLSQP', bounds=[(0.0, self.C)] * len(old),
                              constraints={'type': 'eq', 'fun': lambda a: labels @ (a - old),
                                           'jac': lambda a: labels},
                              options={'ftol': min(1e-9, self.tol ** 2 * 1e-3), 'maxiter': 1000})
            if not result.success:
                raise RuntimeError(f'Solver QP SVM gagal: {result.message}')
            self.alpha_[working] = result.x
            gradient += columns @ (result.x - old)
            self.n_iter_ = iteration // steps_per_pass + 1

        values = -signed_y * gradient
        up = ((signed_y > 0) & (self.alpha_ < self.C - eps)) | ((signed_y < 0) & (self.alpha_ > eps))
        low = ((signed_y > 0) & (self.alpha_ > eps)) | ((signed_y < 0) & (self.alpha_ < self.C - eps))
        self.intercept_ = float((values[up].max() + values[low].min()) / 2.0)
        self.support_ = np.flatnonzero(self.alpha_ > eps)
        self.support_vectors_ = X[self.support_].copy()
        self.dual_coef_ = self.alpha_[self.support_] * signed_y[self.support_]
        self.coef_ = self.dual_coef_ @ self.support_vectors_ if self.kernel == 'linear' else None
        residual = signed_y * self.decision_function(X) - 1.0
        self.converged_ = not (((residual < -self.tol) & (self.alpha_ < self.C - eps))
                              | ((residual > self.tol) & (self.alpha_ > eps))).any()
        if not self.converged_:
            warnings.warn('SVM berhenti sebelum seluruh kondisi KKT memenuhi tol. '
                          'Periksa scaling atau naikkan max_iter.', RuntimeWarning, stacklevel=2)
        return self

    @staticmethod
    def _gpu_features(X, torch_device='cuda'):
        import torch
        try:
            return torch.as_tensor(X, dtype=torch.float64, device=torch_device)
        except torch.cuda.OutOfMemoryError as error:
            raise MemoryError('VRAM tidak cukup untuk data SVM.') from error

    # PyTorch opsional: mode CPU dan prediksi model tersimpan tidak mengimpornya.
    def _resolve_device(self):
        device = getattr(self, 'device', 'cpu')
        if device not in ('cpu', 'cuda', 'auto'):
            raise ValueError("device harus 'cpu', 'cuda', atau 'auto'.")
        if device == 'cpu':
            return 'cpu'
        try:
            import torch
        except ImportError as error:
            if device == 'auto':
                return 'cpu'
            raise RuntimeError('device=cuda memerlukan PyTorch dengan dukungan CUDA.') from error
        if torch.cuda.is_available():
            return 'cuda'
        if device == 'auto':
            return 'cpu'
        raise RuntimeError('CUDA tidak tersedia. Aktifkan GPU dan gunakan PyTorch CUDA, atau device=cpu.')

    # Hitung kemiripan seluruh pasangan baris pada X dan Y.
    def _kernel(self, X, Y, xp=np):
        with np.errstate(over='ignore', invalid='ignore'):
            product = X @ Y.T
            if self.kernel == 'poly':
                result = (self.gamma_ * product + self.coef0) ** self.degree
            elif self.kernel == 'rbf':
                distance = (X ** 2).sum(axis=1)[:, None] + (Y ** 2).sum(axis=1) - 2.0 * product
                result = xp.exp(-self.gamma_ * distance.clip(0.0, None))
            else:
                result = product
        if not xp.isfinite(result).all():
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
                                gamma=self.gamma_, coef0=self.coef0, device=self.device_)
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

        def objective(params):
            logits = design @ params
            probability = np.exp(-np.logaddexp(0.0, logits))
            loss = np.sum(np.logaddexp(0.0, -logits) + target * logits)
            return loss, design.T @ (target - probability)

        result = minimize(objective, params, jac=True, method='BFGS',
                          options={'gtol': 1e-5, 'maxiter': 100})
        if not result.success and np.max(np.abs(result.jac)) > 1e-5:
            warnings.warn(f'Kalibrasi sigmoid belum konvergen: {result.message}', RuntimeWarning)
        params = result.x
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
