# IF3170_G17_DerbyCounty

## Pembagian per Model

Setiap anggota memegang satu model, lengkap dengan implementasi from scratch, perbandingan scikit-learn, bonus visualisasi dan optimasi, serta bab laporannya.

| Model | Pemilik | File yang dipegang |
| --- | --- | --- |
| Decision Tree| Reinsen | `models/decision_tree.py`, `preprocessing/dt.py`, `preprocessing/common.py` |
| KNN | Manu | `models/knn.py`, `preprocessing/knn.py` |
| Logistic Regression + cost-sensitive | Juan | `models/logistic_regression.py`, `preprocessing/lr.py` |
| SVM | Akram | `models/svm.py`, `preprocessing/svm.py` |
| Naive Bayes | Argha | `models/naive_bayes.py`, `preprocessing/nb.py` |

## Struktur Folder

```
src/
├── main.ipynb              # notebook final, mereproduksi submission Kaggle (diisi di akhir)
├── utils.py                # shared: save/load, metrik, validasi silang
├── models/                 # satu file per model from scratch
├── preprocessing/
│   ├── common.py           # shared: clean_train, mark_invalid, add_features, BasePreprocessor
│   └── <model>.py          # preprocessor milik tiap model
└── experiments/<model>/    # notebook eksperimen tiap model
output/
├── submissions/            # submission_<eksperimen>.csv
├── models/                 # <eksperimen>.pkl
└── figures/                # gambar dan video untuk bonus
docs/
└── EXPERIMENTS.md          # log seluruh eksperimen
```

## Aturan Kerja

1. **Eksperimen hanya di model sendiri.** Notebook disimpan di `src/experiments/<model>/` dengan nama `<model>_<deskripsi>.ipynb`, misalnya `svm_rbf_subsample.ipynb`. Satu notebook untuk satu eksperimen.
2. **Preprocessing khusus model** ditulis di `src/preprocessing/<model>.py` sebagai turunan `BasePreprocessor`, cukup override `_fit_extra` dan `_transform_extra` (contoh: `knn.py`). Bebas diubah oleh pemiliknya.
3. **File shared** (`preprocessing/common.py`, `utils.py`) diubah lewat pemegang pipeline. Cleaning yang membuang baris hanya boleh di `common.py`. Kalau ingin membuang outlier, lakukan di dalam `fit` saja (data train fold), data validasi dan test tetap utuh.
4. **Validasi wajib seragam** agar skor bisa dibandingkan:
   ```python
   train = clean_train(train_raw)
   folds = stratified_kfold(y, n_splits=5, seed=42)
   oof = cross_val_proba(make_model, make_prep, train, y, folds)
   threshold, score = best_threshold(y, oof)
   ```
   Model wajib punya `fit(X, y)`, `predict_proba(X)` (probabilitas kelas 1), dan `predict(X)`.
5. **Preprocessing selalu di-fit pada data train saja**, lalu `transform` ke validasi atau test. Train dan test tidak boleh digabung.
6. **Catat setiap eksperimen** di `docs/EXPERIMENTS.md`: nama, pemilik, ringkasan, OOF Macro F1, std per fold, threshold.
7. **Submit Kaggle lewat pemegang pipeline**, hanya jika model single from scratch, OOF naik minimal sekitar 0.003 dari skor terbaik, dan unggul di mayoritas fold. Keputusan diambil dari skor lokal, bukan skor public.
8. **Reproducible**: pasang seed dan jangan memakai sumber acak tanpa seed. Notebook yang dijalankan ulang harus menghasilkan submission yang sama persis.

## Git

- Satu branch per pekerjaan dari `main`, misalnya `feat/svm`, `feat/naive-bayes`, `exp/dt-pruning`.
- Pakai semantic commit: `feat(svm): ...`, `fix(knn): ...`, `docs: ...`, `refactor: ...`.
- Merge ke `main` lewat pull request.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt jupyter

jupyter nbconvert --to notebook --execute --inplace src/experiments/dt/dt_baseline_fe.ipynb
```

Notebook bisa dijalankan dari folder mana pun karena root repo dicari otomatis.
