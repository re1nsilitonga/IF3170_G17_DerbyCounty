@echo off
rem Build laporan di Windows: klik dua kali build.bat atau jalankan dari terminal
cd /d "%~dp0"
for /L %%i in (1,1,3) do (
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  if errorlevel 1 (
    echo Build gagal, lihat main.log
    pause
    exit /b 1
  )
)
copy /Y main.pdf ..\laporan.pdf
echo Selesai: docs\laporan.pdf
pause
