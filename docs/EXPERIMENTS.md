# Log Eksperimen

| Eksperimen | Pemilik | Ringkasan | OOF Macro F1 | Std fold | Threshold | Public |
| --- | --- | --- | --- | --- | --- | --- |
| dt_baseline_fe | Reinsen | CART gini, depth 12, leaf 10, dengan 7 fitur turunan | 0.8291 | 0.0076 | 0.53 | 0.83740 |
| dt_baseline_no_fe | Reinsen | sama dengan dt_baseline_fe tanpa feature engineering | 0.8271 | 0.0031 | 0.43 | - |
| svm_baseline_linear_fe | Akram | SVM linear SMO, C 0.01, tol 0.01, max_iter 1000, dengan feature engineering | 0.7656 | 0.0050 | 0.45 | - |
| svm_linear_tuning_fe | Akram | SVM linear SMO, C 0.1, tol 0.01, max_iter 1000, dengan feature engineering; train penuh per fold | 0.7659 | 0.0053 | 0.44 | - |
| svm_poly_tuning_fe | Akram | SVM poly, C 1.0, degree 3, gamma scale, coef0 1.0, FE; train 3000/fold, max_iter 200 | 0.7607 | 0.0051 | 0.37 | - |
| svm_rbf_tuning_fe | Akram | SVM rbf, C 1.0, gamma scale, FE; train 3000/fold, max_iter 200 | 0.7612 | 0.0056 | 0.33 | - |
