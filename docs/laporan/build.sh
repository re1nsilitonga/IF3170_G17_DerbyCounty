#!/bin/sh
# Build laporan di macOS / Linux: ./build.sh
set -e
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error main.tex
cp main.pdf ../laporan.pdf
echo "Selesai: docs/laporan.pdf"
