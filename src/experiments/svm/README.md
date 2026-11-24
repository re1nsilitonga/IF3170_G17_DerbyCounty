# Eksperimen SVM

Empat notebook mengikuti susunan `main.ipynb` kelompok dan memakai `SVMPreprocessor`
dari `src/preprocessing/svm.py`. Bagian model selain SVM dibiarkan kosong.

| Notebook | Konfigurasi utama | OOF historis | Public Kaggle historis |
| --- | --- | --- | --- |
| [svm_baseline_linear.ipynb](svm_baseline_linear.ipynb) | Linear, C 0,01 | 0,7656 | 0,77484 |
| [svm_linear_tuning.ipynb](svm_linear_tuning.ipynb) | Linear, C 0,1 | 0,7659 | Belum teridentifikasi |
| [svm_gaussian.ipynb](svm_gaussian.ipynb) | Gaussian, C 1, gamma scale | 0,7907 | Belum teridentifikasi |
| [svm_poly.ipynb](svm_poly.ipynb) | Polynomial, C 1, degree 3, coef0 1 | 0,7943 | 0,81212 |

OOF historis berasal dari eksperimen dengan solver SMO. Training yang dijalankan oleh
notebook sekarang memakai SLSQP dan kalibrasi BFGS. Tabel tersebut bukan hasil training ulang.
CSV Kaggle historis telah dicocokkan dengan prediksi model lama tersimpan.

## Menjalankan notebook

Jalankan dari root repo atau buka notebook di Jupyter dan jalankan semua sel secara berurutan.

```powershell
python -m pip install -r requirements.txt jupyter
python -m jupyter nbconvert --execute --to notebook --inplace src/experiments/svm/svm_baseline_linear.ipynb --ExecutePreprocessor.timeout=-1
python -m jupyter nbconvert --execute --to notebook --inplace src/experiments/svm/svm_linear_tuning.ipynb --ExecutePreprocessor.timeout=-1
python -m jupyter nbconvert --execute --to notebook --inplace src/experiments/svm/svm_gaussian.ipynb --ExecutePreprocessor.timeout=-1
python -m jupyter nbconvert --execute --to notebook --inplace src/experiments/svm/svm_poly.ipynb --ExecutePreprocessor.timeout=-1
```

Seluruh grafik memakai Matplotlib. Fungsi visualisasi ada di bagian bawah `src/models/svm.py`.
`DEVICE='cpu'` tidak memerlukan PyTorch. Pilihan `auto` atau `cuda` dapat digunakan jika
PyTorch dengan dukungan CUDA tersedia, tetapi solver SLSQP tetap bekerja pada CPU.

## Hasil eksekusi

Setiap notebook menyimpan hasil pada `output/experiments/<nama>/template_rerun/`.
Isinya mencakup model training ulang, probabilitas OOF, perbandingan scikit-learn,
error analysis, gambar bonus, hasil PSO, dan CSV training ulang.

Baseline dan polynomial juga menghasilkan CSV `_historical.csv` dari model tersimpan.
Prediksinya diperiksa terhadap CSV Kaggle asli sebelum diekspor. Public score historis
hanya merujuk keluaran ini. CSV `_rerun.csv` merupakan hasil training baru dan belum
memiliki public score. File model dan submission historis tidak ditimpa.

PSO dijalankan sebagai eksperimen tambahan Gaussian pada 800 baris pengembangan dan
400 holdout. Hasilnya tidak mengganti konfigurasi eksperimen utama. Setiap notebook
juga memvisualisasikan pemisah, margin, dan support vector ketiga kernel pada dua fitur.

## Pemeriksaan kecil

Ubah `SMOKE_TEST=True` pada sel Settings untuk menjalankan seluruh alur dengan 320
baris train, 12 baris test, dan anggaran bonus yang lebih kecil. Hasil disimpan pada
subfolder `smoke/`. Mode ini tidak memverifikasi reproduksi CSV Kaggle penuh.
`RUN_SCREENING=True` pada notebook linear tuning menjalankan kembali screening C
yang bersifat opsional.

Pemeriksaan top-to-bottom memakai data kecil dan salinan notebook yang dieksekusi
disimpan pada `output/experiments/svm_notebook_validation/`. Empat notebook historis memakai
seluruh data secara default, tetapi training ulang dan CV penuhnya belum diverifikasi.
Notebook final dengan preprocessing kelompok sudah dijalankan pada seluruh data.

## Satu eksperimen tambahan dengan preprocessing kelompok

[svm_final.ipynb](svm_final.ipynb) memakai
`FinalSVMPreprocessor` langsung di notebook sesuai fungsi
`prepare` template kelompok. Seed 5, seleksi fitur, clipping, fitur tanggal, robust
scaling, bobot fitur, one-hot 0,75, dan StratifiedKFold mengikuti template.

Sebanyak 24 kandidat Linear, Polynomial, dan Gaussian disaring dengan scikit-learn pada maksimal 1.200 sampel train. Skor screening bukan skor from scratch. Satu
kandidat tiap kernel diprioritaskan dari screening lalu diverifikasi dengan OOF penuh.
Pencarian berhenti jika OOF Polynomial lama terlampaui minimal 0,003 atau seluruh
kandidat prioritas selesai. Pemenang terverifikasi dilatih pada seluruh train.
Hanya satu CSV final `submission_svm_final.csv` yang dihasilkan,
di folder output eksperimen. Tidak ada pengiriman otomatis ke Kaggle.
Bonus PSO mengoptimasi kernel SVM yang terpilih tanpa ekspor submission tambahan.
Notebook ini menggantikan rencana dua eksperimen tambahan.

```powershell
python -m jupyter nbconvert --execute --to notebook --inplace src/experiments/svm/svm_final.ipynb --ExecutePreprocessor.timeout=-1
```

Preprocessor telah dicocokkan langsung dengan fungsi `prepare` asli pada 21.847
baris fit, 5.462 baris validasi, dan 6.844 baris test. Hasil float32 identik untuk
ketiga bagian. Notebook Gaussian dan notebook final SVM lulus eksekusi seluruh sel
pada 320 baris train dan 12 baris test. Notebook final telah dievaluasi pada seluruh 27.309 baris train bersih dengan 5 fold, seed 5.

Artefak eksperimen final memakai nama `svm_final.pkl` dan `submission_svm_final.csv`.

## Hasil final

OOF Macro F1 SVM from scratch adalah **0,832426** dengan C=0,3, gamma=1, dan
threshold 0,25. Pembanding scikit-learn memperoleh **0,832468** pada fold yang sama.
OOF historis Linear adalah 0,7656 dan Polynomial 0,794336. Pembagian fold historis
memakai seed 42, sehingga perbandingan dengan hasil final seed 5 bersifat indikatif.
Public score Kaggle notebook final adalah **0,85189**, submission **58803590**.

Model final berada di `output/models/svm_final.pkl`. CSV yang sudah disubmit berada di
`output/submissions/submission_svm_final.csv`. Statistik preprocessing disimpan sebagai
`preprocessor_state` dan dipulihkan memakai kelas yang didefinisikan dalam notebook. Visualisasi wajib memakai Matplotlib
untuk menampilkan pemisah, margin, dan support vector ketiga kernel.
