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

Notebook DTL, KNN, Logistic Regression, dan Naive Bayes memuat kode from scratch secara langsung. Notebook SVM memakai implementasi bersama di `src/models/svm.py` dan helper validasi di `src/utils.py`. Preprocessing eksperimen SVM lama berada di `src/preprocessing/`, sedangkan preprocessing SVM final ditulis langsung di notebook. Format notebook mengikuti template tugas.

Hasil SVM final dan cara menjalankannya dijelaskan di [`src/experiments/svm/README.md`](src/experiments/svm/README.md). CSV final berada di `output/submissions/submission_svm_final.csv`, dengan public score Kaggle **0.85189**.

## Notebook Final (`src/main.ipynb`)

- **Model final: KNN from scratch + vote booking kembar** (k = 17, Minkowski p = 1.5, bobot 1/d², threshold 0.4845). Menghasilkan `output/submissions/submission_knn_twin_final.csv`.
- **DTL: C4.5 (gain ratio)** dari eksperimen `src/experiments/dtl/dtl_gain_ratio.ipynb`. Menangani data numerik, kategorikal, dan missing secara langsung. Menghasilkan `output/submissions/submission_dtl_gain_ratio.csv`.
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
