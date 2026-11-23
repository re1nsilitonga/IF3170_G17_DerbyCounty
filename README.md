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

## Eksperimen DTL (`src/experiments/dtl/`)

Lima notebook yang menunjukkan perkembangan Decision Tree dari baseline sampai public tertinggi:

| Notebook | Perubahan dari sebelumnya | Public |
| --- | --- | --- |
| `dtl_baseline_no_fe` | CART gini, depth 12, leaf 10, tanpa fitur turunan | - |
| `dtl_baseline_fe` | ditambah 7 fitur turunan | 0.83740 |
| `dtl_gini_mingain` | pre-pruning minimum gain | 0.84473 |
| `dtl_entropy_mingain` | criterion entropy (information gain) | - |
| `dtl_gain_ratio` | criterion gain ratio (C4.5) | 0.84679 |

## Aturan Kerja

1. **Eksperimen hanya di model sendiri.** Notebook disimpan di `src/experiments/<model>/` dengan nama `<model>_<deskripsi>.ipynb`. Satu notebook untuk satu eksperimen.
2. **Preprocessing selalu di-fit pada data train saja**, lalu diterapkan ke validasi atau test. Train dan test tidak boleh digabung.
3. **Validasi seragam** agar skor bisa dibandingkan: 5-fold cross validation (OOF) dengan Macro F1, dan perubahan dicek di beberapa seed fold sebelum dipercaya.
4. **Submit Kaggle** hanya untuk model single from scratch, dengan keputusan dari skor lokal (OOF), bukan skor public.
5. **Reproducible**: pasang seed dan jangan memakai sumber acak tanpa seed. Notebook yang dijalankan ulang harus menghasilkan submission yang sama persis (md5 dicetak di akhir notebook).

## Git

- Satu branch per pekerjaan dari `main`, misalnya `feat/svm`, `feat/naive-bayes`, `exp/dt-pruning`.
- Pakai semantic commit: `feat(svm): ...`, `fix(knn): ...`, `docs: ...`, `refactor: ...`.
- Merge ke `main` lewat pull request.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter

jupyter nbconvert --to notebook --execute --inplace src/main.ipynb
```

Notebook bisa dijalankan dari folder mana pun karena root repo dicari otomatis.
