import sys
import unittest
import os
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from models import SVM


class SVMTests(unittest.TestCase):
    def test_device_selection_and_cpu_without_torch(self):
        X = np.array([[-1.], [-2.], [1.], [2.]])
        y = np.array([0, 0, 1, 1])
        with patch.dict(sys.modules, {'torch': None}), threadpool_limits(limits=1):
            for device in ('cpu', 'auto'):
                model = SVM(device=device).fit(X, y)
                self.assertEqual(model.device_, 'cpu')
                np.testing.assert_array_equal(model.predict(X), y)
            with self.assertRaisesRegex(RuntimeError, 'PyTorch'):
                SVM(device='cuda').fit(X, y)
        no_cuda = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict(sys.modules, {'torch': no_cuda}):
            self.assertEqual(SVM(device='auto')._resolve_device(), 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'CUDA tidak tersedia'):
                SVM(device='cuda').fit(X, y)
        with self.assertRaisesRegex(ValueError, 'device'):
            SVM(device='invalid').fit(X, y)

    def test_gpu_memory_failure_falls_back_only_in_auto_mode(self):
        X = np.array([[-1.], [-2.], [1.], [2.]])
        y = np.array([0, 0, 1, 1])
        with patch.object(SVM, '_resolve_device', lambda m: 'cpu' if m.device == 'cpu' else 'cuda'), \
                patch.object(SVM, '_gpu_features', side_effect=MemoryError('VRAM')), \
                threadpool_limits(limits=1):
            with self.assertWarnsRegex(RuntimeWarning, 'beralih ke CPU'):
                model = SVM(device='auto').fit(X, y)
            self.assertEqual(model.device_, 'cpu')
            np.testing.assert_array_equal(model.predict(X), y)
            with self.assertRaisesRegex(MemoryError, 'VRAM'):
                SVM(device='cuda').fit(X, y)

    def _check_gpu_solver(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(80, 4))
        y = (X[:, 0] * X[:, 1] + 0.3 * X[:, 2] > 0).astype(int)
        for kernel in ('linear', 'poly', 'rbf'):
            with self.subTest(kernel=kernel), threadpool_limits(limits=1):
                params = dict(kernel=kernel, degree=2, coef0=1., C=1., tol=1e-5)
                cpu = SVM(**params, device='cpu').fit(X, y)
                gpu = SVM(**params, device='cuda').fit(X, y)
                self.assertEqual(gpu.device_, 'cuda')
                self.assertTrue(gpu.converged_)
                np.testing.assert_allclose(gpu.decision_function(X), cpu.decision_function(X), atol=2e-3, rtol=0)
                np.testing.assert_allclose(gpu.predict_proba(X), cpu.predict_proba(X), atol=2e-3, rtol=0)
                np.testing.assert_array_equal(gpu.predict(X), cpu.predict(X))
                self.assertTrue(np.all((gpu.alpha_ >= -1e-10) & (gpu.alpha_ <= gpu.C + 1e-10)))
                self.assertAlmostEqual(float(gpu.alpha_ @ (2 * y - 1)), 0., places=9)
                probability = gpu.predict_proba(X)
                payload = pickle.dumps(gpu)
                with patch.dict(sys.modules, {'torch': None}):
                    loaded = pickle.loads(payload)
                    np.testing.assert_array_equal(loaded.predict_proba(X), probability)
                repeated = SVM(**params, device='cuda').fit(X, y)
                np.testing.assert_array_equal(repeated.predict_proba(X), probability)
        duplicates = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]] * 3)
        labels = np.array([-2, 7, 7, -2] * 3)
        for kernel in ('poly', 'rbf'):
            model = SVM(kernel=kernel, degree=2, C=10, device='cuda').fit(duplicates, labels)
            self.assertTrue(model.converged_)
            np.testing.assert_array_equal(model.predict(duplicates), labels)
        identical = SVM(kernel='rbf', device='cuda').fit(np.zeros((8, 2)), np.array([0, 1] * 4))
        self.assertTrue(identical.converged_)
        self.assertTrue(np.isfinite(identical.predict_proba(np.zeros((2, 2)))).all())

    def test_gpu_solver_math_with_torch_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Optional PyTorch is not installed')
        original = SVM._gpu_features
        def on_cpu(X):
            return original(X, torch_device='cpu')
        original_resolve = SVM._resolve_device
        def resolve(model):
            return 'cuda' if model.device == 'cuda' else original_resolve(model)
        with patch.object(SVM, '_gpu_features', staticmethod(on_cpu)), patch.object(SVM, '_resolve_device', resolve):
            self._check_gpu_solver()

    def test_cuda_hardware(self):
        try:
            import torch
            available = torch.cuda.is_available()
        except ImportError:
            available = False
        if not available:
            if os.environ.get('SVM_REQUIRE_CUDA') == '1':
                self.fail('CUDA hardware verification requires a working CUDA PyTorch build')
            self.skipTest('No CUDA PyTorch available')
        self._check_gpu_solver()

    def test_linear_matches_sklearn_and_dual_scores(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(60, 4))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        with threadpool_limits(limits=1):
            model = SVM(C=0.1, tol=1e-5, seed=42).fit(X, y)
            reference = SVC(kernel='linear', C=0.1, tol=1e-5).fit(X, y)
            repeated = SVM(C=0.1, tol=1e-5, seed=42).fit(X, y)
        self.assertTrue(model.converged_)
        np.testing.assert_allclose(model.decision_function(X), reference.decision_function(X),
                                   atol=1e-3, rtol=0)
        dual_scores = X @ model.support_vectors_.T @ model.dual_coef_ + model.intercept_
        np.testing.assert_allclose(model.decision_function(X), dual_scores, atol=1e-10)
        np.testing.assert_array_equal(model.predict(X), reference.predict(X))
        np.testing.assert_array_equal(model.predict_proba(X), repeated.predict_proba(X))
        self.assertTrue(np.all((model.predict_proba(X) >= 0) & (model.predict_proba(X) <= 1)))
        self.assertTrue(np.all((model.alpha_ >= -1e-10) & (model.alpha_ <= model.C + 1e-10)))
        self.assertAlmostEqual(float(model.alpha_ @ (2 * y - 1)), 0.0, places=10)

    def test_nonlinear_kernels_still_separate_xor(self):
        X = np.array([[-1., -1.], [-1., 1.], [1., -1.], [1., 1.]] * 3)
        y = np.array([0, 1, 1, 0] * 3)
        for kernel in ['poly', 'rbf']:
            with self.subTest(kernel=kernel), threadpool_limits(limits=1):
                model = SVM(kernel=kernel, degree=2, C=10, seed=42).fit(X, y)
                self.assertTrue(model.converged_)
                np.testing.assert_array_equal(model.predict(X), y)
                self.assertTrue(np.isfinite(model.predict_proba(X)).all())

    def test_iteration_limit_and_solver_failure_are_visible(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(150, 4))
        y = (X[:, 0] * X[:, 1] > 0).astype(int)
        with threadpool_limits(limits=1), self.assertWarnsRegex(RuntimeWarning, 'KKT'):
            model = SVM(kernel='rbf', max_iter=1).fit(X, y)
        self.assertFalse(model.converged_)
        self.assertEqual(model.n_iter_, 1)
        self.assertTrue(np.isfinite(model.predict_proba(X)).all())
        failure = SimpleNamespace(success=False, message='iteration limit')
        with patch('models.svm.minimize', return_value=failure):
            with self.assertRaisesRegex(RuntimeError, 'Solver QP SVM gagal: iteration limit'):
                SVM().fit(X, y)


if __name__ == '__main__':
    unittest.main()
