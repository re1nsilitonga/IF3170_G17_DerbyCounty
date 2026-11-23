import copy
import unittest

import numpy as np
import pandas as pd

from src.preprocessing.knn import KNNPreprocessor


class KNNPreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'is_canceled': [0, 0, 1, 1],
            'num_adults': [1.0, 2.0, 3.0, 4.0],
            'value': [2.0, np.nan, 6.0, 8.0],
            'meal_package': ['A', 'A', 'B', None],
        })

    def make_prep(self):
        return KNNPreprocessor(feature_engineering=False, rare_min_count=2)

    def test_base_pipeline_and_train_imputation(self):
        self.train.loc[0, 'num_adults'] = 0.0
        prep = self.make_prep()
        result = prep.fit_transform(self.train)
        encoded = result * prep.stds_ + prep.means_
        self.assertNotIn('id', result.columns)
        self.assertNotIn('is_canceled', result.columns)
        self.assertEqual(encoded.loc[0, 'num_adults'], 3.0)
        self.assertEqual(encoded.loc[0, 'num_adults_missing'], 1.0)
        self.assertEqual(encoded.loc[1, 'value'], 6.0)
        self.assertEqual(encoded.loc[3, 'meal_package_A'], 1.0)

    def test_all_missing_numeric_uses_zero(self):
        self.train['value'] = np.nan
        prep = self.make_prep()
        result = prep.fit_transform(self.train)
        self.assertEqual(prep.medians_['value'], 0.0)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        np.testing.assert_array_equal(result['value'], np.zeros(4))
        query = self.train.iloc[:1].copy()
        query['value'] = 7.0
        self.assertEqual(prep.transform(query).iloc[0]['value'], 7.0)
        self.assertEqual(prep.medians_['value'], 0.0)

    def test_all_missing_categorical_uses_other(self):
        self.train['meal_package'] = None
        prep = self.make_prep()
        result = prep.fit_transform(self.train)
        self.assertEqual(prep.modes_['meal_package'], 'Other')
        self.assertEqual(prep.means_['meal_package_Other'], 1.0)
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        query = self.train.iloc[:1].copy()
        query['meal_package'] = 'Unseen'
        transformed = prep.transform(query)
        self.assertEqual(list(transformed.columns), list(result.columns))
        self.assertEqual(transformed.iloc[0]['meal_package_Other'], 0.0)
        # Kategori baru tetap Other, tetapi bukan missing seperti data train.
        self.assertEqual(transformed.iloc[0]['meal_package_missing'], -1.0)

    def test_rare_and_unseen_categories_share_fixed_other_column(self):
        prep = self.make_prep()
        result = prep.fit_transform(self.train)
        query = self.train.iloc[[2, 3]].drop(columns='is_canceled').copy()
        query['meal_package'] = ['Unseen', 'Other']
        query = query[query.columns[::-1]]
        transformed = prep.transform(query)
        encoded = transformed * prep.stds_ + prep.means_
        self.assertEqual(list(transformed.columns), list(result.columns))
        self.assertNotIn('meal_package_B', transformed.columns)
        np.testing.assert_array_equal(encoded['meal_package_Other'], [1.0, 1.0])
        np.testing.assert_array_equal(encoded['meal_package_A'], [0.0, 0.0])
        self.assertTrue(transformed.columns.is_unique)

    def test_transform_keeps_train_statistics_inputs_and_rows(self):
        original = self.train.copy(deep=True)
        prep = self.make_prep().fit(self.train)
        medians, means, stds = prep.medians_.copy(), prep.means_.copy(), prep.stds_.copy()
        modes = prep.modes_.copy()
        categories = {c: list(values) for c, values in prep.keep_cats_.items()}
        query = self.train.iloc[[1, 3]].copy()
        query.loc[3, 'value'] = 1000000.0
        query_original = query.copy(deep=True)
        result = prep.transform(query)
        encoded = result * prep.stds_ + prep.means_
        self.assertEqual(encoded.loc[1, 'value'], 6.0)
        self.assertEqual(list(result.index), [1, 3])
        pd.testing.assert_series_equal(prep.medians_, medians)
        pd.testing.assert_series_equal(prep.means_, means)
        pd.testing.assert_series_equal(prep.stds_, stds)
        self.assertEqual(prep.modes_, modes)
        self.assertEqual(prep.keep_cats_, categories)
        pd.testing.assert_frame_equal(self.train, original)
        pd.testing.assert_frame_equal(query, query_original)

    def test_missing_indicator_schema_is_learned_only_from_train(self):
        prep = self.make_prep()
        train = prep.fit_transform(self.train)
        self.assertEqual(prep.missing_cols_, ['value', 'meal_package'])
        query = self.train.iloc[[0, 2]].copy()
        # Missing baru pada validasi tidak boleh membuat kolom indikator baru.
        query['num_adults'] = np.nan
        query['meal_package'] = 'Unseen'
        transformed = prep.transform(query)
        self.assertEqual(prep.missing_cols_, ['value', 'meal_package'])
        self.assertNotIn('num_adults_missing', transformed)
        self.assertEqual(list(transformed.columns), list(train.columns))
        encoded = transformed * prep.stds_ + prep.means_
        np.testing.assert_array_equal(encoded['num_adults'], [2.5, 2.5])
        np.testing.assert_array_equal(encoded['value_missing'], [0.0, 0.0])
        np.testing.assert_array_equal(encoded['meal_package_missing'], [0.0, 0.0])

    def test_standardization_matches_hand_calculated_train_statistics(self):
        self.train['constant'] = 5.0
        prep = self.make_prep()
        train = prep.fit_transform(self.train)
        # Setelah imputasi, value = [2, 6, 6, 8]. Std sampel memakai ddof=1.
        expected_std = np.sqrt(19.0 / 3.0)
        self.assertEqual(prep.means_['value'], 5.5)
        self.assertAlmostEqual(prep.stds_['value'], expected_std)
        np.testing.assert_allclose(train['value'], np.array([-3.5, .5, .5, 2.5]) / expected_std)
        # Indikator [0, 1, 0, 0] juga distandardisasi memakai statistik train.
        self.assertEqual(prep.means_['value_missing'], .25)
        self.assertEqual(prep.stds_['value_missing'], .5)
        np.testing.assert_allclose(train['value_missing'], [-.5, 1.5, -.5, -.5])
        self.assertEqual(prep.stds_['constant'], 1.0)
        np.testing.assert_array_equal(train['constant'], np.zeros(4))
        query = self.train.iloc[:1].copy()
        query['value'] = 1000.0
        query['constant'] = 9.0
        transformed = prep.transform(query)
        self.assertAlmostEqual(transformed.iloc[0]['value'], (1000.0 - 5.5) / expected_std)
        self.assertEqual(transformed.iloc[0]['constant'], 4.0)
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())

    def test_transform_preserves_entire_fitted_state_with_duplicate_indices(self):
        prep = self.make_prep().fit(self.train)
        state = copy.deepcopy(vars(prep))
        query = self.train.iloc[[3, 1, 1]].copy()
        query['meal_package'] = ['Unseen', None, 'A']
        original = query.copy(deep=True)
        result = prep.transform(query)
        pd.testing.assert_frame_equal(query, original)
        self.assertEqual(list(result.index), [3, 1, 1])
        self.assertEqual(len(result), 3)
        self.assertEqual(set(vars(prep)), set(state))
        for name, expected in state.items():
            if isinstance(expected, pd.Series):
                pd.testing.assert_series_equal(getattr(prep, name), expected)
            else:
                self.assertEqual(getattr(prep, name), expected, name)
        pd.testing.assert_frame_equal(result, prep.transform(query))

    def test_single_train_row_remains_finite(self):
        prep = self.make_prep()
        result = prep.fit_transform(self.train.iloc[:1])
        self.assertTrue(np.isfinite(result.to_numpy()).all())
        np.testing.assert_array_equal(result.to_numpy(), np.zeros(result.shape))


if __name__ == '__main__':
    unittest.main()
