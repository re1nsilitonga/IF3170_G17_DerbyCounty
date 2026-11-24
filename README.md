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

## Notebook Naive Bayes

Jalankan **Run All** dari root repo setelah menjalankan setup di atas. Keempat notebook dapat dijalankan sendiri dengan `data/train.csv`, `data/test.csv`, dan `data/sample_submission.csv`. Jika data berada di folder lain, atur `DERBY_DATA_DIR` sebelum membuka kernel.

| Urutan | Notebook | Macro F1 OOF |
| --- | --- | ---: |
| 1 | [Baseline](src/experiments/nb/nb_baseline.ipynb) | 0,73422 |
| 2 | [Fine-tuning](src/experiments/nb/nb_baseline_finetune.ipynb) | 0,73984 |
| 3 | [Smoothing per fitur](src/experiments/nb/nb_baseline_finetune_2.ipynb) | 0,73977 |
| 4 | [Naive Bayes kontekstual](src/experiments/nb/nb_baseline_finetune_3.ipynb) | 0,80196 |

CSV dibuat di `output/template_nb/submissions/`. Setel `DERBY_OUTPUT_DIR` untuk memilih folder output lain. Baseline mereproduksi `submission_nb_categorical_all.csv` dengan skor Kaggle public **0,73697**. Tiga notebook berikutnya merupakan eksperimen lokal.

Smoothing per fitur mengubah alpha `special_requests` dari 2 ke 10 dengan preprocessing, bobot, dan threshold yang sama. Skor OOF turun 0,00007 dan tiga dari lima fold membaik tipis. Versi kontekstual menghasilkan Macro F1 **0,79943** pada validasi bertingkat. Threshold 0,43 dipakai untuk melaporkan skor pada protokol tim, sedangkan threshold 0,45 dipakai untuk ekspor CSV.
