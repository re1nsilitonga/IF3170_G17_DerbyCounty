import numpy as np


# Sigmoid yang stabil secara numerik sehingga tidak overflow untuk z yang sangat besar atau sangat kecil
# Fungsi ini memakai modul array yang sama dengan z sehingga bisa berjalan di NumPy maupun CuPy
def sigmoid(z):
    xp = _array_module(z)
    e = xp.exp(-xp.abs(z))
    return xp.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


# Ambil modul array dari sebuah array, yaitu cupy untuk array GPU dan numpy untuk lainnya
def _array_module(a):
    if type(a).__module__.split('.')[0] == 'cupy':
        import cupy
        return cupy
    return np


# Pilih modul array untuk training dengan 'cpu' memakai NumPy dan 'gpu' memakai CuPy
# CuPy adalah pustaka array dengan sintaks seperti NumPy yang berjalan di GPU NVIDIA
def _select_backend(device):
    if device not in ('cpu', 'gpu'):
        raise ValueError(f"device tidak dikenal: {device}")
    if device == 'cpu':
        return np
    try:
        import cupy
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
        error = 'tidak ada GPU NVIDIA yang terdeteksi'
    except Exception as e:
        error = f'CuPy tidak bisa dipakai ({type(e).__name__})'
    raise RuntimeError(f"device='gpu' diminta tetapi {error}. Pasang CuPy (misalnya 'pip install cupy-cuda13x[ctk]') "
                       f"atau pakai device='cpu'.")


# Salin array NumPy atau CuPy ke memori CPU sebagai array NumPy float64
def _to_host(a):
    return np.asarray(a.get() if hasattr(a, 'get') else a, dtype=float)


# Logistic Regression berbasis gradien dengan regularisasi L1, L2, dan Elastic Net
# Fungsi objektif disamakan dengan scikit-learn agar nilai C bisa dibandingkan langsung
#   J = C * sum(s_i * logloss_i) + (1 - rho) / 2 * ||w||^2 + rho * ||w||_1
# Objektif tersebut dibagi C * sum(s) agar skala gradien tidak bergantung pada jumlah data
# s_i adalah bobot sampel dan rho bernilai 0 untuk L2, 1 untuk L1, dan l1_ratio untuk Elastic Net
# Bias tidak ikut diregularisasi
# Sumber FISTA adalah Beck dan Teboulle 2009 dalam SIAM Journal on Imaging Sciences
# Sumber adaptive restart adalah O'Donoghue dan Candes 2015 dalam Foundations of Computational Mathematics
# Inovasi di luar gradient descent biasa adalah FISTA, adaptive restart, langkah 1 / L, dan proximal step untuk L1
# Inovasi lain adalah bobot kelas untuk cost-sensitive dan backend GPU lewat CuPy
class LogisticRegression:

    # penalty bernilai None, 'l1', 'l2', atau 'elasticnet'
    # C adalah kebalikan kekuatan regularisasi dan l1_ratio adalah porsi L1 pada Elastic Net
    # solver bernilai 'gd' untuk gradient descent biasa atau 'fista'
    # learning_rate berupa angka atau 'auto' untuk 1 dibagi konstanta Lipschitz
    # tol menghentikan training jika perubahan parameter terbesar dalam satu iterasi lebih kecil dari nilai ini
    # class_weight bernilai None atau dict {kelas: bobot} yang dipakai untuk cost-sensitive
    # record_params menyimpan bias dan bobot di setiap iterasi untuk visualisasi lintasan
    # device bernilai 'cpu' untuk NumPy atau 'gpu' untuk CuPy
    # Hanya training yang berjalan di GPU dan hasilnya (coef_, intercept_) disalin ke CPU
    # dtype bernilai 'float64' atau 'float32' dan float32 lebih cepat di GPU tetapi hasilnya tidak persis sama
    def __init__(self, penalty='l2', C=1.0, l1_ratio=0.5, solver='fista', learning_rate='auto',
                 max_iter=5000, tol=1e-5, class_weight=None, record_params=False, device='cpu',
                 dtype='float64'):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.class_weight = class_weight
        self.record_params = record_params
        self.device = device
        self.dtype = dtype

    # Latih bobot dan bias, nama kolom disimpan di feature_names_ jika X berupa DataFrame
    # Kedua solver memakai langkah yang sama yaitu gradien log-loss dan L2 lalu proximal step untuk L1
    # Bedanya FISTA menghitung gradien di titik momentum Nesterov sehingga konvergen lebih cepat pada fitur berkorelasi
    def fit(self, X, y):
        if self.solver not in ('gd', 'fista'):
            raise ValueError(f'solver tidak dikenal: {self.solver}')
        if self.dtype not in ('float64', 'float32'):
            raise ValueError(f'dtype tidak dikenal: {self.dtype}')
        xp = _select_backend(self.device)
        self.feature_names_ = list(X.columns) if hasattr(X, 'columns') else None
        # Data disalin ke perangkat sekali di awal dan seluruh iterasi berjalan di perangkat yang sama
        X = xp.asarray(np.asarray(X, dtype=self.dtype))
        y = xp.asarray(np.asarray(y).astype(self.dtype))

        s = self._sample_weights(xp, y)
        S = float(s.sum())
        lam1, lam2 = self._penalty_strength(S)
        lr = self._step_size(xp, X, s, S, lam2)

        # (w, b) adalah parameter sekarang dan (wv, bv) adalah titik tempat gradien dihitung
        # z dan zv disimpan agar X @ w cukup dihitung sekali per iterasi
        w, b = xp.zeros(X.shape[1], dtype=X.dtype), 0.0
        wv, bv = w.copy(), b
        z = xp.zeros(len(y), dtype=X.dtype)
        zv = z.copy()
        t = 1.0

        self.loss_history_ = [self._objective(xp, z, y, s, S, w, lam1, lam2)]
        self.param_history_ = [np.r_[b, _to_host(w)]] if self.record_params else None
        self.converged_ = False

        for it in range(1, self.max_iter + 1):
            err = s * (sigmoid(zv) - y) / S
            w_new = wv - lr * (X.T @ err + lam2 * wv)
            b_new = bv - lr * float(err.sum())

            # Bagian L1 tidak terdiferensiasi di nol sehingga dipakai soft-thresholding (proximal step)
            # Dengan begitu bobot yang tidak penting benar-benar bernilai nol
            if lam1 > 0:
                w_new = xp.sign(w_new) * xp.maximum(xp.abs(w_new) - lr * lam1, 0.0)

            z_new = X @ w_new + b_new
            delta = max(float(xp.abs(w_new - w).max()), abs(b_new - b))

            if self.solver == 'fista':
                # Momentum di-reset jika arah langkah berbalik terhadap momentum (adaptive restart)
                # Reset ini mencegah parameter berosilasi di sekitar titik optimum
                if float((wv - w_new) @ (w_new - w)) + (bv - b_new) * (b_new - b) > 0:
                    t = 1.0
                t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
                beta = (t - 1.0) / t_next
                t = t_next
            else:
                beta = 0.0

            wv = w_new + beta * (w_new - w)
            bv = b_new + beta * (b_new - b)
            zv = z_new + beta * (z_new - z)
            w, b, z = w_new, b_new, z_new

            self.loss_history_.append(self._objective(xp, z, y, s, S, w, lam1, lam2))
            if self.record_params:
                self.param_history_.append(np.r_[b, _to_host(w)])
            if delta < self.tol:
                self.converged_ = True
                break

        self.coef_ = _to_host(w)
        self.intercept_ = float(b)
        self.n_iter_ = it
        self.learning_rate_ = lr
        if self.record_params:
            self.param_history_ = np.array(self.param_history_)
        return self

    # Nilai z = Xw + b sebelum masuk sigmoid
    def decision_function(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_

    # Probabilitas kelas 1
    def predict_proba(self, X):
        return sigmoid(self.decision_function(X))

    # Label 0 atau 1 berdasarkan threshold
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    # Bobot tiap sampel dari class_weight untuk cost-sensitive dengan nilai 1 untuk kelas yang tidak disebut
    def _sample_weights(self, xp, y):
        if self.class_weight is not None and not isinstance(self.class_weight, dict):
            raise ValueError('class_weight harus None atau dict {kelas: bobot}')
        s = xp.ones(len(y), dtype=y.dtype)
        for c, wc in (self.class_weight or {}).items():
            s[y == c] *= wc
        return s

    # Kekuatan regularisasi L1 dan L2 setelah objektif dibagi C * sum(s)
    def _penalty_strength(self, S):
        if self.penalty is None:
            return 0.0, 0.0
        rho = {'l1': 1.0, 'l2': 0.0, 'elasticnet': self.l1_ratio}.get(self.penalty)
        if rho is None:
            raise ValueError(f"penalty tidak dikenal: {self.penalty}")
        scale = 1.0 / (self.C * S)
        return rho * scale, (1.0 - rho) * scale

    # Learning rate 'auto' adalah 1 / L dengan L konstanta Lipschitz gradien bagian yang halus
    # Turunan kedua sigmoid paling besar 0,25 sehingga L <= 0,25 * eig_max(X'SX / sum(s)) + lam2
    # Dengan langkah ini gradient descent dijamin turun tanpa mencoba learning rate satu per satu
    def _step_size(self, xp, X, s, S, lam2):
        if self.learning_rate != 'auto':
            return float(self.learning_rate)
        Xb = xp.hstack([xp.ones((len(X), 1), dtype=X.dtype), X])
        hessian_bound = 0.25 * (Xb.T * s) @ Xb / S
        return 1.0 / (float(xp.linalg.eigvalsh(hessian_bound).max()) + lam2)

    # Nilai fungsi objektif yaitu rata-rata log-loss berbobot ditambah penalti
    # log(1 + e^z) - y * z sama dengan log-loss dan tidak menghasilkan log(0)
    def _objective(self, xp, z, y, s, S, w, lam1, lam2):
        logloss = float(xp.sum(s * (xp.logaddexp(0.0, z) - y * z))) / S
        return logloss + lam1 * float(xp.abs(w).sum()) + 0.5 * lam2 * float(w @ w)
