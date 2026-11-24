# IF3170_G17_DerbyCounty

Tugas Besar 2 IF3170 Inteligensi Artifisial: prediksi pembatalan pemesanan penginapan (`is_canceled`) dengan metrik Macro F1.

## Pembagian per Model

Setiap anggota memegang satu model, lengkap dengan implementasi from scratch, perbandingan scikit-learn, bonus visualisasi dan optimasi, serta bab laporannya.

| Model | Pemilik | Notebook eksperimen |
| --- | --- | --- |
| Decision Tree | Reinsen | `src/experiments/dtl/` |
| KNN | Manu | `src/experiments/knn/` |
| Logistic Regression | Juan | `src/experiments/lr/` |
| SVM | Akram | `src/experiments/svm/` |
| Naive Bayes | Argha | `src/experiments/nb/` |

## Struktur Folder

```
src/
├── main.ipynb              # notebook final, mereproduksi submission Kaggle
└── experiments/<model>/    # notebook eksperimen tiap model
data/
├── train.csv
├── test.csv
└── sample_submission.csv
output/
├── submissions/            # submission_<eksperimen>.csv
└── models/                 # <eksperimen>.pkl, dibuat saat notebook dijalankan
```

Semua kode from scratch (preprocessing, model, validasi) ditulis langsung di dalam notebook. Format notebook mengikuti template tugas.

## Eksperimen Terbaik per Model

| Model | Notebook | Submission | Public |
| --- | --- | --- | --- |
| KNN + vote booking kembar | `src/experiments/knn/knn_twin_seed5.ipynb` (sama dengan bagian KNN di `src/main.ipynb`) | `submission_knn_twin_final.csv` | 0.87535 |
| Decision Tree (C4.5) | `src/experiments/dtl/dtl_gain_ratio.ipynb` | `submission_dtl_gain_ratio.csv` | 0.84679 |
| Logistic Regression | `src/experiments/lr/lr_cross_bins.ipynb` | `submission_lr_cross_bins.csv` | 0.83542 |
| SVM | `src/experiments/svm/svm_final.ipynb` | `submission_svm_final.csv` | 0.85189 |
| Naive Bayes | `src/experiments/nb/nb_baseline_finetune_3.ipynb` | - | 0.80173 |

Semua file submission ada di `output/submissions/`.

## Notebook Final (`src/main.ipynb`)

- **Model final: KNN from scratch + vote booking kembar** (k = 17, Minkowski p = 1.5, bobot 1/d², threshold 0.4845). Menghasilkan `output/submissions/submission_knn_twin_final.csv`.
- **DTL: C4.5 (gain ratio)** dari eksperimen `src/experiments/dtl/dtl_gain_ratio.ipynb`. Menangani data numerik, kategorikal, dan missing secara langsung. Menghasilkan `output/submissions/submission_dtl_gain_ratio.csv`.
- **Logistic Regression, SVM, dan Naive Bayes** from scratch dilatih pada `X_train` hasil preprocessing bersama dan dibandingkan lewat Macro F1 holdout di bagian Model Evaluation. Skor terbaik tiap model ada di notebook eksperimennya (tabel di atas).
- Preprocessing bersama dibuat dengan fungsi `prepare(fit_df, *dfs)`. Model lain cukup memakai `X_train`, `X_val`, dan `X_test` dari bagian Preparation Pipeline.
- Bonus yang sudah ada: gambar percabangan DTL dengan parameter `top_n`, cost-complexity pruning pada DTL (algoritma optimasi di luar materi, dengan referensi), dan cost-sensitive classification (biaya FP 2 kali FN).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter

jupyter nbconvert --to notebook --execute --inplace src/main.ipynb
```

Notebook bisa dijalankan dari folder mana pun karena root repo dicari otomatis.
