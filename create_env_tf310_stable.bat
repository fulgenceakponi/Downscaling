@echo off
echo ============================================
echo   Creation environnement TensorFlow stable
echo ============================================

REM 1. Création environnement
conda create -y -n tf310_stable python=3.10

REM 2. Activation
call conda activate tf310_stable

REM 3. Mise à jour pip
python -m pip install --upgrade pip

REM 4. Librairies cœur (versions compatibles)
pip install ^
 numpy==1.24.4 ^
 scipy==1.10.1 ^
 pandas==1.5.3 ^
 matplotlib==3.7.3 ^
 netCDF4==1.6.5 ^
 h5py==3.9.0 ^
 xarray==2023.6.0 ^
 dask[complete]==2023.6.1

REM 5. TensorFlow (CPU stable)
pip install tensorflow==2.10.1

REM 6. Outils ML utiles
pip install scikit-learn==1.3.2 joblib==1.3.2

REM 7. Vérification
python - << EOF
import numpy as np, xarray as xr, tensorflow as tf, dask
print("NUMPY:", np.__version__)
print("XARRAY:", xr.__version__)
print("TF:", tf.__version__)
print("DASK:", dask.__version__)
print("ENV OK")
EOF

echo ============================================
echo   Environnement tf310_stable prêt
echo ============================================
pause
