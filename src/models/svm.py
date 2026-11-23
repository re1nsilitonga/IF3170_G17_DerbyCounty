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
        # Alias lama dipertahankan agar pemanggil dan model tersimpan tetap dapat digunakan.
        if self.kernel not in ('linear', 'poly', 'gaussian', 'rbf'):
            raise ValueError("kernel harus 'linear', 'poly', atau 'gaussian'.")
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
            elif self.kernel in ('gaussian', 'rbf'):
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


# Visualisasi dan ekspor hasil SVM; library plotting dimuat saat diperlukan.
def show_plot(fig, path):
    """Simpan PNG dan tampilkan grafik interaktif dengan fallback gambar di notebook."""
    import base64
    import json
    from pathlib import Path
    from IPython.display import display

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.update_layout(template='plotly_white', font=dict(family='Arial', size=13, color='#243447'),
                      margin=dict(l=70, r=35, t=100, b=110),
                      legend=dict(orientation='h', y=-0.22, x=0.5, xanchor='center'))
    png = fig.to_image(format='png', scale=1.5)
    path.write_bytes(png)
    display({'application/vnd.plotly.v1+json': json.loads(fig.to_json()),
             'image/png': base64.b64encode(png).decode('ascii')}, raw=True)


def boundary_plot(X, y, models, names):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    low, high = X.min(axis=0) - 0.5, X.max(axis=0) + 0.5
    x, v = np.linspace(low[0], high[0], 160), np.linspace(low[1], high[1], 160)
    xx, yy = np.meshgrid(x, v)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    fig = make_subplots(rows=1, cols=3, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[f'{names[k]} · {len(m.support_)} support vector'
                                        for k, m in models.items()], horizontal_spacing=0.05)
    for col, model in enumerate(models.values(), 1):
        score = model.decision_function(grid).reshape(xx.shape)
        assert score.min() < -1 and score.max() > 1
        fig.add_trace(go.Heatmap(x=x, y=v, z=(score >= 0).astype(int),
                                colorscale=[[0, '#eaf3fa'], [0.49, '#eaf3fa'],
                                            [0.5, '#fcf0e5'], [1, '#fcf0e5']],
                                showscale=False, hoverinfo='skip'), row=1, col=col)
        for start, end, step, name, dash, width in [(-1, 1, 2, 'Margin f=±1', 'dash', 1.4),
                                                    (0, 0, 1, 'Batas f=0', 'solid', 2.6)]:
            fig.add_trace(go.Contour(x=x, y=v, z=score, autocontour=False,
                                     contours=dict(start=start, end=end, size=step,
                                                   coloring='none', showlabels=True),
                                     line=dict(color='#34465a', dash=dash, width=width),
                                     showscale=False, name=name, legendgroup=name,
                                     showlegend=col == 1, hovertemplate='f=%{z:.3f}<extra></extra>'),
                          row=1, col=col)
        for label, color, symbol, name in [(0, '#246c9c', 'circle', 'Tidak batal'),
                                            (1, '#bf671f', 'triangle-up', 'Batal')]:
            mask = y == label
            fig.add_trace(go.Scatter(x=X[mask, 0], y=X[mask, 1], mode='markers',
                                     marker=dict(color=color, symbol=symbol, size=6, opacity=0.65),
                                     name=name, legendgroup=name, showlegend=col == 1,
                                     hovertemplate='Jeda: %{x:.2f}<br>Tarif: %{y:.2f}<extra>%{fullData.name}</extra>'),
                          row=1, col=col)
        fig.add_trace(go.Scatter(x=model.support_vectors_[:, 0], y=model.support_vectors_[:, 1],
                                 mode='markers', marker=dict(symbol='circle-open', size=10,
                                                            color='#243447', line=dict(width=1)),
                                 customdata=model.alpha_[model.support_], name='Support vector',
                                 legendgroup='sv', showlegend=col == 1,
                                 hovertemplate='Jeda: %{x:.2f}<br>Tarif: %{y:.2f}<br>Alpha: %{customdata:.4f}<extra>Support vector</extra>'),
                      row=1, col=col)
    fig.update_xaxes(title_text='Jeda kedatangan (z-score)', range=[low[0], high[0]])
    fig.update_yaxes(range=[low[1], high[1]])
    fig.update_yaxes(title_text='Tarif kamar (z-score)', row=1, col=1)
    fig.update_layout(title=f'Batas dan margin SVM · {len(X)} baris train', width=1450, height=620)
    return fig


def pso_plot(history, comparison, dev_size, holdout_size):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f'Fitness pencarian · {dev_size} baris', 'Baseline dan pemenang PSO'], horizontal_spacing=0.12)
    fig.add_trace(go.Scatter(x=list(range(1, len(history) + 1)), y=history,
                             mode='lines+markers', name='Fitness terbaik', line=dict(color='#246c9c')),
                  row=1, col=1)
    fig.add_hline(y=comparison.iloc[0]['oof_macro_f1'], line_dash='dash',
                  line_color='#7c8795', annotation_text='Baseline', row=1, col=1)
    for field, name, color in [('oof_macro_f1', 'OOF seleksi', '#246c9c'),
                                ('holdout_macro_f1', f'Holdout {holdout_size} baris', '#bf671f')]:
        fig.add_trace(go.Bar(x=comparison.index, y=comparison[field], name=name,
                             marker_color=color, text=comparison[field], texttemplate='%{text:.4f}',
                             textposition='outside', hovertemplate='%{x}<br>Macro F1: %{y:.4f}<extra>%{fullData.name}</extra>'),
                      row=1, col=2)
    fig.update_xaxes(title_text='Generasi', dtick=1, row=1, col=1)
    fig.update_yaxes(title_text='Macro F1 OOF terbaik', row=1, col=1)
    fig.update_yaxes(title_text='Macro F1', range=[0, 1], row=1, col=2)
    fig.update_layout(title='Optimasi PSO · skor seleksi dan holdout terpisah',
                      barmode='group', width=1150, height=520)
    return fig


def kernel_plot(results, details, names, rows):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=[f'Macro F1 OOF · {rows:,} baris',
                                                      'Kestabilan pada lima fold'], horizontal_spacing=0.12)
    for impl, label, color in [('from_scratch', 'From scratch', '#246c9c'),
                                ('sklearn', 'scikit-learn', '#bf671f')]:
        part = results[results.implementation == impl].set_index('kernel').loc[list(names)]
        fig.add_trace(go.Bar(x=list(names.values()), y=part.oof_macro_f1, name=label,
                             marker_color=color, text=part.oof_macro_f1,
                             texttemplate='%{text:.4f}', textposition='outside'), row=1, col=1)
    for kernel, color, symbol in [('linear', '#246c9c', 'circle'), ('poly', '#7955a3', 'square'),
                                   ('gaussian', '#27856a', 'triangle-up')]:
        part = details[(details.kernel == kernel) & (details.implementation == 'from_scratch')]
        fig.add_trace(go.Scatter(x=part.fold, y=part.macro_f1, name=names[kernel],
                                 mode='lines+markers', line=dict(color=color), marker=dict(symbol=symbol)),
                      row=1, col=2)
    fig.update_yaxes(title_text='Macro F1')
    fig.update_yaxes(range=[0, 1], row=1, col=1)
    fig.update_xaxes(title_text='Fold', dtick=1, row=1, col=2)
    fig.update_layout(title='Perbandingan ketiga kernel SVM', barmode='group', width=1200, height=520)
    return fig
