# Log Eksperimen

| Eksperimen | Pemilik | Ringkasan | OOF Macro F1 | Std fold | Threshold | Public |
| --- | --- | --- | --- | --- | --- | --- |
| dt_baseline_fe | Reinsen | CART gini, depth 12, leaf 10, dengan 7 fitur turunan | 0.8291 | 0.0076 | 0.53 | 0.83740 |
| dt_baseline_no_fe | Reinsen | sama dengan dt_baseline_fe tanpa feature engineering | 0.8271 | 0.0031 | 0.43 | - |
| svm_baseline_linear | Akram | SVM linear SMO, C 0.01, tol 0.01, max_iter 1000, dengan feature engineering | 0.7656 | 0.0050 | 0.45 | - |
| svm_linear_tuning | Akram | SVM linear SMO, C 0.1, tol 0.01, max_iter 1000, dengan feature engineering; train penuh per fold | 0.7659 | 0.0053 | 0.44 | - |
| svm_rbf | Akram | SVM rbf, C 1, gamma scale, FE; train penuh per fold, Kaggle CPU | 0.7907 | 0.0036 | 0.43 | - |
| svm_poly | Akram | SVM poly, C 1, degree 3, gamma scale, coef0 1, FE; train penuh per fold, Kaggle CPU | 0.7943 | 0.0034 | 0.40 | - |
