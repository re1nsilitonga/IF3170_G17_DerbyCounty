"""Utilitas umum: save/load model, metrik evaluasi, dan validasi silang."""

import pickle
from pathlib import Path

import numpy as np


def save_model(model, path):
    """Simpan objek model (atau objek Python apa pun) ke file pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def load_model(path):
    """Muat kembali model yang sudah disimpan dengan save_model."""
    with open(path, 'rb') as f:
        return pickle.load(f)


# ---------------------------------------------------------------- Metrik

def confusion_matrix(y_true, y_pred):
    """Hitung confusion matrix biner dalam bentuk [[TN, FP], [FN, TP]]."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]])


def f1_per_class(y_true, y_pred):
    """Hitung F1 untuk kelas 0 dan kelas 1, dikembalikan sebagai array [f1_0, f1_1]."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    # F1 = 2TP / (2TP + FP + FN), untuk kelas 0 peran TP dan TN ditukar
    f1_pos = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp > 0 else 0.0
    return np.array([f1_neg, f1_pos])


def macro_f1(y_true, y_pred):
    """Macro F1, yaitu rata-rata F1 kedua kelas. Ini metrik resmi kompetisi."""
    return float(f1_per_class(y_true, y_pred).mean())


def accuracy(y_true, y_pred):
    """Proporsi prediksi yang benar."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def best_threshold(y_true, proba, grid=None):
    """Cari threshold probabilitas yang memberi Macro F1 tertinggi.

    Dipakai pada prediksi out-of-fold agar threshold tidak dipilih dari data
    yang dilihat model saat training.
    """
    if grid is None:
        grid = np.round(np.arange(0.20, 0.81, 0.01), 2)
    scores = np.array([macro_f1(y_true, (proba >= t).astype(int)) for t in grid])
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


# ---------------------------------------------------------------- Validasi

def stratified_kfold(y, n_splits=5, seed=42):
    """Bagi indeks menjadi k fold dengan proporsi kelas yang sama di tiap fold.

    Mengembalikan list berisi pasangan (train_idx, val_idx).
    """
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        # Bagikan indeks kelas c ke fold secara bergiliran
        fold_of[idx] = np.arange(len(idx)) % n_splits
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(n_splits)]


def cross_val_proba(make_model, make_prep, df, y, folds):
    """Jalankan validasi silang dan kembalikan probabilitas out-of-fold.

    make_model dan make_prep adalah fungsi tanpa argumen yang membuat model dan
    preprocessor baru. Preprocessor di-fit ulang di setiap fold hanya dengan data
    train fold tersebut, sehingga tidak ada kebocoran informasi dari fold validasi.
    """
    y = np.asarray(y)
    oof = np.zeros(len(y))
    for tr_idx, va_idx in folds:
        prep = make_prep()
        X_tr = prep.fit_transform(df.iloc[tr_idx])
        X_va = prep.transform(df.iloc[va_idx])
        model = make_model()
        model.fit(X_tr, y[tr_idx])
        oof[va_idx] = model.predict_proba(X_va)
    return oof


def fold_scores(y, proba, folds, threshold):
    """Hitung Macro F1 per fold dengan satu threshold, untuk melihat kestabilan model."""
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    return np.array([macro_f1(y[va], pred[va]) for _, va in folds])
