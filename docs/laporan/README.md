# Laporan (LaTeX)

Source laporan Tugas Besar 2. Hasil build disalin ke `docs/laporan.pdf`.

```
docs/laporan/
├── main.tex            # preamble, sampul, daftar isi, urutan bab
├── bab/                # isi tiap bab (edit di sini)
│   ├── 1_dasar_teori.tex
│   ├── 2_implementasi.tex
│   ├── 3_hasil.tex
│   ├── 4_kesimpulan.tex
│   ├── 5_pembagian_tugas.tex
│   ├── 6_referensi.tex
│   └── 7_lampiran.tex
├── gambar/             # semua gambar (png/pdf)
├── build.bat           # build di Windows
└── build.sh            # build di macOS / Linux
```

## Build

Hanya memakai paket LaTeX standar (tanpa `babel`, `minted`, atau `shell-escape`), jadi bisa di-build dengan distribusi LaTeX apa pun.

**Windows (MiKTeX atau TeX Live)**

1. Pasang [MiKTeX](https://miktex.org/download). Saat instalasi, pilih *Install missing packages on-the-fly: Yes*.
2. Klik dua kali `build.bat`, atau dari terminal:
   ```bat
   cd docs\laporan
   build.bat
   ```

**macOS (MacTeX)**

1. Pasang [MacTeX](https://www.tug.org/mactex/) (atau BasicTeX lalu `sudo tlmgr install geometry amsmath booktabs multirow float caption xcolor listings enumitem tabularx`).
2. Dari terminal:
   ```bash
   cd docs/laporan
   ./build.sh
   ```

**Linux (TeX Live)**: sama dengan macOS. Di Fedora, paket yang kurang bisa dipasang dengan `sudo dnf install texlive-scheme-medium`.

**Overleaf**: upload isi folder `docs/laporan/` (termasuk `gambar/` dan `bab/`), set compiler ke pdfLaTeX, main document `main.tex`.

**Manual** (semua OS): jalankan `pdflatex main.tex` tiga kali di folder ini, lalu salin `main.pdf` ke `docs/laporan.pdf`.

## Menulis

- Bagian yang belum diisi ditandai `\todo{...}` dan tampil merah di PDF. Cari dengan `grep -rn todo bab/`.
- Gambar baru: simpan di `gambar/`, lalu `\includegraphics[width=\textwidth]{nama_file.png}`.
- Kode: `\begin{lstlisting}[caption={...}] ... \end{lstlisting}`. Gunakan karakter ASCII saja di dalam kode.
- Referensi: tambah `\bibitem{ref:nama}` di `bab/6_referensi.tex`, lalu kutip dengan `\cite{ref:nama}`.
- Angka desimal ditulis dengan koma (0,8529), sesuai bahasa Indonesia.
