# Log Eksperimen

| Eksperimen | Pemilik | Ringkasan | OOF Macro F1 | Std fold | Threshold | Public |
| --- | --- | --- | --- | --- | --- | --- |
| dt_baseline_fe | Reinsen | CART gini, depth 12, leaf 10, dengan 7 fitur turunan | 0.8291 | 0.0076 | 0.53 | 0.83740 |
| dt_baseline_no_fe | Reinsen | sama dengan dt_baseline_fe tanpa feature engineering | 0.8271 | 0.0031 | 0.43 | - |
| nb_categorical_two_features | Argha | Categorical NB alpha 1, kanal pemesanan dan jumlah permintaan khusus | 0.6652 | 0.0041 | 0.44 | - |
| nb_categorical_all | Argha | Categorical NB alpha 1, 17 fitur asli dengan pengelompokan hari dan tarif | 0.7342 | 0.0084 | 0.42 | 0.73697 |
