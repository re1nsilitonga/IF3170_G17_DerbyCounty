# Utilitas umum: save/load model, metrik evaluasi, dan validasi silang.

import pickle
from pathlib import Path

import numpy as np


# Simpan objek model ke file pickle.
def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


# Muat kembali model yang disimpan dengan save_model.
def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# Confusion matrix biner dalam bentuk [[TN, FP], [FN, TP]].
def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return np.array([[tn, fp], [fn, tp]])


# F1 untuk kelas 0 dan kelas 1, hasilnya [f1_0, f1_1].
def f1_per_class(y_true, y_pred):
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    f1_pos = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn > 0 else 0.0
    f1_neg = 2 * tn / (2 * tn + fn + fp) if tn + fn + fp > 0 else 0.0
    return np.array([f1_neg, f1_pos])


# Macro F1, rata-rata F1 kedua kelas (metrik resmi kompetisi).
def macro_f1(y_true, y_pred):
    return float(f1_per_class(y_true, y_pred).mean())


# Proporsi prediksi yang benar.
def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


# Cari threshold dengan Macro F1 tertinggi, dipakai pada prediksi out-of-fold.
def best_threshold(y_true, proba, grid=None):
    if grid is None:
        grid = np.round(np.arange(0.20, 0.81, 0.01), 2)
    scores = np.array([macro_f1(y_true, (proba >= t).astype(int)) for t in grid])
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


# Bagi indeks menjadi k fold dengan proporsi kelas sama, hasilnya list (train_idx, val_idx).
def stratified_kfold(y, n_splits=5, seed=42):
    y = np.asarray(y)
    rng = np.random.RandomState(seed)
    fold_of = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_splits
    return [(np.where(fold_of != k)[0], np.where(fold_of == k)[0]) for k in range(n_splits)]


# Validasi silang yang mengembalikan probabilitas out-of-fold.
# Preprocessor di-fit ulang tiap fold hanya dengan data train fold tersebut.
def cross_val_proba(make_model, make_prep, df, y, folds):
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


# Macro F1 per fold dengan satu threshold, untuk melihat kestabilan model.
def fold_scores(y, proba, folds, threshold):
    y = np.asarray(y)
    pred = (proba >= threshold).astype(int)
    return np.array([macro_f1(y[va], pred[va]) for _, va in folds])
