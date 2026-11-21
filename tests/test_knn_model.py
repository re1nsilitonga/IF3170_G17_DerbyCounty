import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.knn import KNN


class KNNModelTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0., 0.], [3., 4.], [0., 4.], [3., 0.]])
        self.y = np.array([0, 1, 1, 0])

    def test_fit_returns_self_and_stores_independent_numeric_binary_data(self):
        X, y = self.X.copy(), self.y.astype(float)
        model = KNN(k=2)
        self.assertIs(model.fit(pd.DataFrame(X), pd.Series(y)), model)
        self.assertEqual(model.X_.dtype.kind, 'f')
        self.assertEqual(model.y_.dtype.kind, 'i')
        np.testing.assert_array_equal(model.X_, X)
        np.testing.assert_array_equal(model.y_, self.y)
        # Periksa ndarray juga agar penyimpanan referensi input terdeteksi.
        model.fit(X, y)
        X[:] = 99
        y[:] = 0
        np.testing.assert_array_equal(model.X_, self.X)
        np.testing.assert_array_equal(model.y_, self.y)

    def test_invalid_training_features_are_rejected(self):
        cases = [[], [1, 2], np.zeros((2, 1, 1)), np.empty((0, 2)),
                 np.empty((2, 0)), [[np.nan]], [[np.inf]], [[-np.inf]],
                 [['text']], [[1 + 2j]]]
        for X in cases:
            with self.subTest(X=X), self.assertRaises(ValueError):
                KNN(k=1).fit(X, [0])

    def test_invalid_labels_are_rejected_before_integer_conversion(self):
        cases = [0, [0], [[0], [1]], [0, .5], [0, -1], [0, 2],
                 [0, np.nan], [0, np.inf], ['0', '1'], [0, None], [0, 1j]]
        for y in cases:
            with self.subTest(y=y), self.assertRaises(ValueError):
                KNN(k=1).fit([[0], [1]], y)

    def test_invalid_k_is_rejected_on_fit_and_neighbor_override(self):
        model = KNN(k=1).fit(self.X, self.y)
        distances, indices = model.kneighbors([[0, 0]], k=4)
        for k in [0, -1, 5, 1.5, 2.0, True, np.bool_(False), '2']:
            with self.subTest(k=k):
                with self.assertRaises(ValueError):
                    KNN(k=k).fit(self.X, self.y)
                with self.assertRaises(ValueError):
                    model.kneighbors([[0, 0]], k=k)
                with self.assertRaises(ValueError):
                    model.proba_from_neighbors(distances, indices, k=k)
        KNN(k=np.int64(4)).fit(self.X, self.y)

    def test_invalid_distance_weight_and_batch_parameters_are_rejected(self):
        cases = [{'p': 0}, {'p': .5}, {'p': np.nan}, {'p': np.inf}, {'p': True},
                 {'p': '2'}, {'weights': 'unknown'}, {'batch_size': 0},
                 {'batch_size': -1}, {'batch_size': 1.5}, {'batch_size': True}]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                KNN(k=1, **kwargs).fit(self.X, self.y)

    def test_euclidean_distances_match_manual_calculation(self):
        model = KNN(k=3).fit(self.X, self.y)
        actual = model._distances(np.array([[0., 0.], [3., 4.]]))
        np.testing.assert_allclose(actual, [[0, 5, 4, 3], [5, 0, 3, 4]])

    def test_euclidean_large_offsets_and_identical_points(self):
        X = np.array([[1e12, 1e12], [1e12 + 3, 1e12 + 4], [.1, .2]])
        model = KNN(k=1).fit(X, [0, 1, 0])
        d = model._distances(X)
        np.testing.assert_array_equal(d.diagonal(), [0., 0., 0.])
        self.assertEqual(d[0, 1], 5.)
        self.assertEqual(d[1, 0], 5.)

    def test_nearest_neighbors_match_manual_indices_and_order(self):
        model = KNN(k=3).fit(self.X, self.y)
        distances, indices = model.kneighbors([[0, 0], [3, 4]])
        np.testing.assert_array_equal(indices, [[0, 3, 2], [1, 2, 3]])
        np.testing.assert_allclose(distances, [[0, 3, 4], [0, 3, 4]])
        for k in (1, len(self.X)):
            d, idx = model.kneighbors([[0, 0]], k=k)
            np.testing.assert_array_equal(idx, [[0, 3, 2, 1][:k]])
            np.testing.assert_allclose(d, [[0, 3, 4, 5][:k]])
        d, idx = model.kneighbors([[0, 0]], k=1)
        with self.assertRaises(ValueError):
            model.proba_from_neighbors(d, idx, k=2)

    def test_tie_at_k_boundary_prefers_smaller_training_index(self):
        X = [[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0], [2, 0]]
        queries = [[0, 0]] * 3
        for p in (1, 2):
            for batch_size in (1, 2, 256):
                model = KNN(k=3, p=p, batch_size=batch_size).fit(X, [0, 1, 0, 1, 1, 0])
                for _ in range(2):
                    d, idx = model.kneighbors(queries)
                    np.testing.assert_array_equal(idx, [[4, 0, 1]] * 3)
                    np.testing.assert_allclose(d, [[0, 1, 1]] * 3)

    def test_duplicate_training_points_preserve_index_order(self):
        model = KNN(k=2).fit([[.1, .2]] * 4, [0, 1, 0, 1])
        d, idx = model.kneighbors([[.1, .2]])
        np.testing.assert_array_equal(idx, [[0, 1]])
        np.testing.assert_array_equal(d, [[0., 0.]])

    def test_query_validation_and_empty_query(self):
        with self.assertRaises(ValueError):
            KNN(k=1).kneighbors([[0, 0]])
        model = KNN(k=2).fit(self.X, self.y)
        for X in ([0, 0], [[0]], [[np.nan, 0]], [[0, np.inf]]):
            with self.subTest(X=X), self.assertRaises(ValueError):
                model.kneighbors(X)
        d, idx = model.kneighbors(np.empty((0, 2)))
        self.assertEqual(d.shape, (0, 2))
        self.assertEqual(idx.shape, (0, 2))

    def test_failed_refit_does_not_replace_training_data(self):
        model = KNN(k=2).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            model.fit(self.X + 5, [0, 0, .5, 1])
        np.testing.assert_array_equal(model.X_, self.X)
        np.testing.assert_array_equal(model.y_, self.y)

    def test_predict_binary_labels_and_threshold_equality(self):
        for weights in ('uniform', 'distance'):
            with self.subTest(weights=weights):
                model = KNN(k=2, weights=weights).fit([[0], [2], [4], [6]], [0, 0, 1, 1])
                query = [[0], [3], [6]]
                np.testing.assert_allclose(model.predict_proba(query), [0., .5, 1.])
                predicted = model.predict(query)
                np.testing.assert_array_equal(predicted, [0, 1, 1])
                self.assertEqual(predicted.shape, (3,))
                self.assertEqual(predicted.dtype.kind, 'i')
                np.testing.assert_array_equal(model.predict(query, threshold=np.nextafter(.5, 1)), [0, 0, 1])
                np.testing.assert_array_equal(model.predict(query, threshold=0), [1, 1, 1])
                np.testing.assert_array_equal(model.predict(query, threshold=1), [0, 0, 1])

    def test_invalid_thresholds_are_rejected(self):
        model = KNN(k=1).fit(self.X, self.y)
        for threshold in (-.1, 1.1, np.nan, np.inf, -np.inf, True, '.5', [.5]):
            with self.subTest(threshold=threshold), self.assertRaisesRegex(ValueError, 'threshold'):
                model.predict([[0, 0]], threshold=threshold)

    def test_all_inference_methods_require_fit(self):
        model = KNN(k=1)
        calls = [lambda: model.kneighbors([[0, 0]]),
                 lambda: model.predict_proba([[0, 0]]),
                 lambda: model.predict([[0, 0]]),
                 lambda: model.proba_from_neighbors(np.zeros((1, 1)), np.zeros((1, 1), dtype=int))]
        for call in calls:
            with self.assertRaisesRegex(ValueError, 'Panggil fit'):
                call()

    def test_prediction_rejects_wrong_query_feature_counts(self):
        model = KNN(k=2).fit(self.X, self.y)
        for method in (model.kneighbors, model.predict_proba, model.predict):
            for query in ([[0]], [[0, 0, 0]], np.empty((0, 3))):
                with self.subTest(method=method.__name__, shape=np.shape(query)):
                    with self.assertRaisesRegex(ValueError, 'Jumlah fitur'):
                        method(query)

    def test_empty_query_returns_empty_outputs_without_distance_calculation(self):
        for weights in ('uniform', 'distance'):
            for p in (1, 2):
                model = KNN(k=2, p=p, weights=weights).fit(self.X, self.y)
                query = np.empty((0, 2))
                with patch.object(model, '_distances', wraps=model._distances) as distances:
                    d, idx = model.kneighbors(query)
                    proba = model.predict_proba(query)
                    predicted = model.predict(query)
                    cached = model.proba_from_neighbors(d, idx)
                    distances.assert_not_called()
                self.assertEqual(d.shape, (0, 2))
                self.assertEqual(idx.shape, (0, 2))
                for result in (proba, predicted, cached):
                    self.assertEqual(result.shape, (0,))
                self.assertEqual(proba.dtype.kind, 'f')
                self.assertEqual(predicted.dtype.kind, 'i')

    def test_batch_sizes_preserve_distances_neighbors_probabilities_and_labels(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(17, 3))
        X[1] = X[0]  # Duplikat dengan label berbeda turut menguji tie.
        y = np.arange(17) % 2
        query = np.vstack([X[:2], rng.normal(size=(11, 3))])
        for p in (1, 2, 3):
            for weights in ('uniform', 'distance'):
                reference = KNN(k=5, p=p, weights=weights, batch_size=1024).fit(X, y)
                expected_d, expected_idx = reference.kneighbors(query)
                expected_proba = reference.predict_proba(query)
                expected_labels = reference.predict(query)
                for batch_size in (1, 2, 7, 32):
                    with self.subTest(p=p, weights=weights, batch_size=batch_size):
                        model = KNN(k=5, p=p, weights=weights, batch_size=batch_size).fit(X, y)
                        d, idx = model.kneighbors(query)
                        np.testing.assert_array_equal(idx, expected_idx)
                        np.testing.assert_array_equal(d, expected_d)
                        np.testing.assert_array_equal(model.predict_proba(query), expected_proba)
                        np.testing.assert_array_equal(model.predict(query), expected_labels)

    def test_distance_computation_is_limited_to_query_batches(self):
        query = np.tile([[0., 0.], [3., 4.]], (9, 1))[:17]
        for p in (1, 2):
            model = KNN(k=2, p=p, batch_size=32).fit(self.X, self.y)
            # p != 2 memakai batch_size // 16; Euclidean memakai batch_size utuh.
            if p == 2:
                model.batch_size = 4
            effective_batch = 4 if p == 2 else 2
            with patch.object(model, '_distances', wraps=model._distances) as distances:
                predicted = model.predict(query)
                lengths = [len(call.args[0]) for call in distances.call_args_list]
            self.assertEqual(lengths, [effective_batch] * (17 // effective_batch) + [1])
            self.assertLess(max(lengths), len(query))
            self.assertEqual(sum(lengths), len(query))
            self.assertEqual(predicted.shape, (17,))

    def test_neighbor_batches_do_not_retain_full_sorted_index_matrices(self):
        model = KNN(k=3, batch_size=2).fit(np.arange(40).reshape(-1, 1), np.arange(40) % 2)
        # Periksa alokasi dasar batch yang ditahan sebelum digabung, bukan hanya shape view.
        with patch('src.models.knn.np.vstack', wraps=np.vstack) as stacking:
            d, idx = model.kneighbors([[0], [5], [10], [15], [20]])
        self.assertEqual(d.shape, (5, 3))
        self.assertEqual(idx.shape, (5, 3))
        for call in stacking.call_args_list:
            chunks = call.args[0]
            self.assertEqual([chunk.shape for chunk in chunks], [(2, 3), (2, 3), (1, 3)])
            for chunk in chunks:
                backing = chunk
                while isinstance(backing.base, np.ndarray):
                    backing = backing.base
                self.assertLessEqual(backing.nbytes, chunk.nbytes)


    def test_neighbor_selection_matches_stable_full_sort_with_many_ties(self):
        rng = np.random.default_rng(7)
        # Koordinat integer kecil menghasilkan banyak jarak sama, termasuk di batas ke-k.
        X = rng.integers(0, 3, size=(200, 3)).astype(float)
        y = rng.integers(0, 2, size=200)
        query = rng.integers(0, 3, size=(60, 3)).astype(float)
        for p in (1, 2):
            for k in (1, 4, 17, 200):
                with self.subTest(p=p, k=k):
                    model = KNN(k=k, p=p, batch_size=7).fit(X, y)
                    d, idx = model.kneighbors(query)
                    full = model._distances(query)
                    expected = np.argsort(full, axis=1, kind='stable')[:, :k]
                    np.testing.assert_array_equal(idx, expected)
                    np.testing.assert_array_equal(d, np.take_along_axis(full, expected, axis=1))

    def test_euclidean_distances_match_direct_formula_on_random_data(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(50, 6))
        query = rng.normal(size=(70, 6))  # > _EUCLID_BLOCK agar sub-blok ikut diuji.
        model = KNN(k=1).fit(X, np.arange(50) % 2)
        expected = np.sqrt(((query[:, None, :] - X[None, :, :]) ** 2).sum(2))
        np.testing.assert_allclose(model._distances(query), expected, rtol=1e-12, atol=1e-12)

    def test_manual_example_probabilities_and_labels(self):
        # Train 1D: posisi 0..5 dengan label [0, 0, 1, 1, 1, 0].
        X = np.arange(6, dtype=float).reshape(-1, 1)
        y = np.array([0, 0, 1, 1, 1, 0])
        model = KNN(k=3).fit(X, y)
        # q=0.1 -> tetangga 0,1,2 -> 1/3; q=3 -> 3,2,4 -> 1; q=5.2 -> 5,4,3 -> 2/3.
        query = [[.1], [3.], [5.2]]
        _, idx = model.kneighbors(query)
        np.testing.assert_array_equal(idx, [[0, 1, 2], [3, 2, 4], [5, 4, 3]])
        proba = model.predict_proba(query)
        np.testing.assert_allclose(proba, [1 / 3, 1., 2 / 3])
        np.testing.assert_array_equal(model.predict(query), [0, 1, 1])

    def test_k_equals_n_train_gives_global_class_proportion(self):
        model = KNN(k=4).fit(self.X, self.y)
        rng = np.random.default_rng(0)
        proba = model.predict_proba(rng.normal(size=(5, 2)) * 10)
        np.testing.assert_allclose(proba, np.full(5, self.y.mean()))

    def test_voting_tie_predicts_class_one_at_default_threshold(self):
        # k=2, tetangga berlabel 0 dan 1 -> proba 0.5 -> kelas 1 karena aturan >= threshold.
        model = KNN(k=2).fit([[0.], [1.], [10.]], [0, 1, 0])
        self.assertEqual(model.predict_proba([[.5]])[0], .5)
        self.assertEqual(model.predict([[.5]])[0], 1)

    def test_constant_feature_does_not_change_neighbors(self):
        X = np.array([[0.], [2.], [5.], [9.]])
        y = np.array([0, 1, 1, 0])
        query = np.array([[1.4], [6.]])
        base = KNN(k=2).fit(X, y)
        const = KNN(k=2).fit(np.hstack([X, np.zeros((4, 1))]), y)
        d0, i0 = base.kneighbors(query)
        d1, i1 = const.kneighbors(np.hstack([query, np.zeros((2, 1))]))
        np.testing.assert_array_equal(i0, i1)
        np.testing.assert_allclose(d0, d1)

    def test_predict_proba_is_one_dimensional_within_unit_interval(self):
        rng = np.random.default_rng(1)
        X, y = rng.normal(size=(40, 3)), rng.integers(0, 2, size=40)
        for weights in ('uniform', 'distance'):
            proba = KNN(k=7, weights=weights).fit(X, y).predict_proba(rng.normal(size=(25, 3)))
            self.assertEqual(proba.shape, (25,))
            self.assertTrue(((proba >= 0) & (proba <= 1)).all())
            if weights == 'uniform':
                # Voting uniform: proporsi tetangga berlabel 1 selalu kelipatan 1/k.
                np.testing.assert_allclose(proba * 7, np.round(proba * 7))


if __name__ == '__main__':
    unittest.main()
