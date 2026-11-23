# Log Eksperimen

| Eksperimen | Pemilik | Ringkasan | OOF Macro F1 | Std fold | Threshold | Public |
| --- | --- | --- | --- | --- | --- | --- |
| dt_baseline_fe | Reinsen | CART gini, depth 12, leaf 10, dengan 7 fitur turunan | 0.8291 | 0.0076 | 0.53 | 0.83740 |
| dt_baseline_no_fe | Reinsen | sama dengan dt_baseline_fe tanpa feature engineering | 0.8271 | 0.0031 | 0.43 | - |
| svm_baseline_linear | Akram | SVM linear SMO, C 0.01, tol 0.01, max_iter 1000, dengan feature engineering | 0.7656 | 0.0050 | 0.45 | - |
| svm_linear_tuning | Akram | SVM linear SMO, C 0.1, tol 0.01, max_iter 1000, dengan feature engineering; train penuh per fold | 0.7659 | 0.0053 | 0.44 | - |
| svm_rbf | Akram | SVM rbf, C 1, gamma scale, FE; train penuh per fold, Kaggle CPU | 0.7907 | 0.0036 | 0.43 | - |
| svm_poly | Akram | SVM poly, C 1, degree 3, gamma scale, coef0 1, FE; train penuh per fold, Kaggle CPU | 0.7943 | 0.0034 | 0.40 | - |
| nb_categorical_two_features | Argha | Categorical NB alpha 1, kanal pemesanan dan jumlah permintaan khusus | 0.6652 | 0.0041 | 0.44 | - |
| nb_categorical_all | Argha | Categorical NB alpha 1, 17 fitur asli dengan pengelompokan hari dan tarif | 0.7342 | 0.0084 | 0.42 | 0.73697 |
| nb_baseline_finetune | Argha | Preprocessing 17 fitur sama, alpha 2, bobot permintaan khusus 2 | 0.73984 | 0.00482 | 0.49 | - |
| nb_baseline_finetune_2 | Argha | Alpha permintaan khusus 10, alpha fitur lain 2, preprocessing dan bobot fine-tuning | 0.73977 | 0.00460 | 0.49 | - |
| nb_baseline_finetune_3 | Argha | Bin kontekstual untuk hari dan tarif serta kategori gabungan | 0.80196 | 0.00729 | 0.43 | - |

Pada `nb_baseline_finetune_2`, Macro F1 OOF turun 0,00007 dan tiga dari lima fold membaik tipis. Pada `nb_baseline_finetune_3`, threshold 0,43 dipakai untuk melaporkan skor pada protokol tim. Threshold ekspor 0,45 dipilih melalui evaluasi OOF seleksi. Validasi bertingkat menghasilkan Macro F1 0,79943.
