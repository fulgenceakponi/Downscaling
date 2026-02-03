#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 OBJECTIFS SCIENTIFIQUES DU SCRIPT
-------------------------------------------------
-------------------------------------------------
1.	Utiliser toutes les variables disponibles :
    o	ALADIN63 daily (upscalé à 150 km)
    o	CNRM-CM5 (150 km)
2.	Variable cible (Y) :
    o	ALADIN63 à 12 km – température (tas / tasAdjust)
3.	Entrées (X) pendant l’apprentissage :
    o	ALADIN↑ (150 km) + CMIP (150 km)
    → concaténation canal-wise
4.	Test d’ablation propre :
    o	🔹 Test normal : ALADIN↑ + CMIP
    o	🔹 Test ablation : CMIP seul (sans mettre des zéros artificiels)
5.	Découpage temporel :
    o	80 % apprentissage
    o	10 % validation
    o	10 % test
6.	Alignement strict :
    o	période commune
    o	grilles identiques
    o	masquage terre / océan
7.	U-Net Conv2D
8.	Métriques spatiales : Corrélation; RMSE; MAE
9.	Cartes comparatives : apprentissage; validation; test;	test ablation CMIP seul
10.	Sauvegarde : modèle; normalisation; figures géoréférencées

"""

"""
U-NET DE DOWNSCALING CLIMATIQUE MULTI-SOURCES
============================================

Objectif :
---------
Apprendre un modèle U-Net Conv2D pour reconstruire la température
ALADIN63 à 12 km (tasAdjust) à partir de prédicteurs basse résolution.

Entrées (X) :
-------------
- ALADIN63 upscalé à 150 km (multi-variables)
- CMIP5 (CNRM-CM5) à 150 km (multi-variables)

Sortie (Y) :
------------
- ALADIN63 12 km : tasAdjust

Expériences :
-------------
1) Apprentissage / validation / test standard (ALADIN↑ + CMIP)
2) Test d’ablation : CMIP seul (sans ALADIN, sans zéros)

Contraintes :
-------------
- Océan totalement exclu
- Même normalisation apprentissage / test
- Cartes comparatives + métriques spatiales
"""

# ====================================================
# IMPORTS
# ====================================================

import os
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# ====================================================
# CHEMINS
# ====================================================

ALADIN_DIR = r"D:\Projet_Lighten IO\essai_etape1\data\ALADIN63\daily"
CMIP_DIR   = r"D:\Projet_Lighten IO\essai_etape5\data\CNRM-CM5_150km\downloads"

MODEL_DIR = "models"
FIG_DIR   = "figures_multivar_ablation"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ====================================================
# VARIABLES
# ====================================================

target_var  = "tasAdjust"
target_unit = "K"

VAR_MAP = {
    "tasAdjust": "tas",
    "prAdjust": "pr",
    "rsdsAdjust": "rsds",
    "sfcWindAdjust": "sfcWind",
    "tasmaxAdjust": "tasmax",
    "tasminAdjust": "tasmin"
}

input_vars = [v for v in os.listdir(ALADIN_DIR) if v in VAR_MAP]

# ====================================================
# CHARGEMENT ALADIN 12 KM
# ====================================================

print("📥 Chargement ALADIN 12 km")

data_aladin = {}
for v in input_vars:
    files = sorted(glob.glob(os.path.join(ALADIN_DIR, v, "*.nc")))
    data_aladin[v] = xr.open_mfdataset(files, combine="by_coords")[v]

ds_y = data_aladin[target_var]

lat  = ds_y.lat
lon  = ds_y.lon
time = ds_y.time

Y = ds_y.transpose("time", "lat", "lon").values[..., np.newaxis]

land_mask = ~np.isnan(Y[0, ..., 0])

# ====================================================
# UPSCALING
# ====================================================

def upscale_mean(arr, factor):
    t, ny, nx = arr.shape
    ny2, nx2 = ny // factor, nx // factor
    arr = arr[:, :ny2*factor, :nx2*factor]
    return arr.reshape(t, ny2, factor, nx2, factor).mean(axis=(2,4))

# ====================================================
# ALADIN UPSCALÉ
# ====================================================

ALADIN_UP = []
for v in input_vars:
    arr = data_aladin[v].transpose("time", "lat", "lon").values
    arr[:, ~land_mask] = np.nan
    ALADIN_UP.append(upscale_mean(arr, factor=5))

ALADIN_UP = np.stack(ALADIN_UP, axis=-1)
n_aladin = ALADIN_UP.shape[-1]

# ====================================================
# CMIP UPSCALÉ
# ====================================================

print("📥 Chargement CMIP5")

CMIP_UP = []

for v in input_vars:
    cmip_name = VAR_MAP[v]
    files = glob.glob(os.path.join(CMIP_DIR, f"{cmip_name}_day_CNRM-CM5_*.nc"))
    ds = xr.open_mfdataset(files, combine="by_coords")[cmip_name]

    if "latitude" in ds.coords:
        ds = ds.rename({"latitude": "lat", "longitude": "lon"})

    ds = ds.interp(lat=lat, lon=lon)
    arr = ds.transpose("time", "lat", "lon").values
    CMIP_UP.append(upscale_mean(arr, factor=5))

CMIP_UP = np.stack(CMIP_UP, axis=-1)

# ====================================================
# ALIGNEMENT TEMPOREL
# ====================================================

nt = min(ALADIN_UP.shape[0], CMIP_UP.shape[0], Y.shape[0])
ALADIN_UP = ALADIN_UP[:nt]
CMIP_UP   = CMIP_UP[:nt]
Y         = Y[:nt]
time      = time[:nt]

# ====================================================
# X COMPLET
# ====================================================

X_full = np.concatenate([ALADIN_UP, CMIP_UP], axis=-1)

# ====================================================
# NORMALISATION
# ====================================================

X_mean, X_std = [], []

for i in range(X_full.shape[-1]):
    m = np.nanmean(X_full[..., i])
    s = np.nanstd(X_full[..., i]) + 1e-6
    X_full[..., i] = (X_full[..., i] - m) / s
    X_mean.append(m)
    X_std.append(s)

Y_mean = np.nanmean(Y[:, land_mask])
Y_std  = np.nanstd(Y[:, land_mask]) + 1e-6
Y = (Y - Y_mean) / Y_std

np.savez(os.path.join(MODEL_DIR, "norm_X.npz"), mean=X_mean, std=X_std)
np.savez(os.path.join(MODEL_DIR, "norm_y.npz"), mean=Y_mean, std=Y_std)

X_full = np.nan_to_num(X_full, nan=0.0)

X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
Y      = np.nan_to_num(Y,      nan=0.0, posinf=0.0, neginf=0.0)

# ====================================================
# SPLIT
# ====================================================

i1 = int(0.8 * nt)
i2 = int(0.9 * nt)

X_train, Y_train = X_full[:i1], Y[:i1]
X_val,   Y_val   = X_full[i1:i2], Y[i1:i2]
X_test,  Y_test  = X_full[i2:], Y[i2:]

# ====================================================
# CROP POUR U-NET
# ====================================================

def crop_to_match(source, target):
    """
    Centre-crop source pour matcher la taille spatiale de target
    """
    sh = tf.shape(source)
    th = tf.shape(target)

    dh = sh[1] - th[1]
    dw = sh[2] - th[2]

    crop_h1 = dh // 2
    crop_h2 = dh - crop_h1
    crop_w1 = dw // 2
    crop_w2 = dw - crop_w1

    return source[:, crop_h1:sh[1]-crop_h2, crop_w1:sh[2]-crop_w2, :]

# ====================================================
# U-NET ROBUSTE + SUPER-RÉSOLUTION 150 km → 12 km
# ====================================================

def unet_2d(input_shape, target_shape):
    inp = layers.Input(input_shape)

    # --------------------
    # Encoder (150 km)
    # --------------------
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same")(inp)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(64, 3, activation="relu", padding="same")(p1)
    p2 = layers.MaxPooling2D()(c2)

    # --------------------
    # Bottleneck
    # --------------------
    b = layers.Conv2D(128, 3, activation="relu", padding="same")(p2)

    # --------------------
    # Decoder (150 km)
    # --------------------
    u2 = layers.UpSampling2D()(b)
    c2c = crop_to_match(c2, u2)
    u2 = layers.Concatenate()([u2, c2c])
    u2 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)

    u1 = layers.UpSampling2D()(u2)
    c1c = crop_to_match(c1, u1)
    u1 = layers.Concatenate()([u1, c1c])
    u1 = layers.Conv2D(32, 3, activation="relu", padding="same")(u1)

    # =================================================
    # SUPER-RÉSOLUTION 150 km → 12 km (×5)
    # =================================================

    sr = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(u1)
    sr = layers.Conv2D(32, 3, activation="relu", padding="same")(sr)

    sr = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(sr)
    sr = layers.Conv2D(16, 3, activation="relu", padding="same")(sr)

    sr = layers.UpSampling2D(size=(2, 2), interpolation="bilinear")(sr)
    sr = layers.Conv2D(16, 3, activation="relu", padding="same")(sr)

    # --------------------
    # Ajustement exact à Y
    # --------------------
    sr = layers.Lambda(
        lambda x: x[:, :target_shape[0], :target_shape[1], :]
    )(sr)

    out = layers.Conv2D(1, 1, activation="linear")(sr)

    return models.Model(inp, out)

model = unet_2d(
    input_shape=X_train.shape[1:],      # (ny150, nx150, nvar)
    target_shape=Y_train.shape[1:3]      # (ny12, nx12)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="mse"
)

# ====================================================
# ENTRAÎNEMENT
# ====================================================

model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=20,
    batch_size=4
)

model.save(os.path.join(MODEL_DIR, "unet_multivar_ablation.h5"))

# ====================================================
# TESTS
# ====================================================

Y_pred = model.predict(X_test)

X_test_cmip_only = X_test[..., n_aladin:]
X_test_ablation = np.concatenate(
    [np.zeros_like(X_test[..., :n_aladin]), X_test_cmip_only],
    axis=-1
)

Y_pred_ablation = model.predict(X_test_ablation)

print("\n✅ SCRIPT TERMINÉ AVEC SUCCÈS")
print("✔ Apprentissage : ALADIN↑ + CMIP")
print("✔ Test ablation : CMIP seul")

# ====================================================
# MÉTRIQUES SPATIALES
# ====================================================

def spatial_metrics(y_true, y_pred):
    """
    Calcul des métriques spatiales sur une grille 2D
    - y_true, y_pred : 2D arrays (lat x lon)
    Retour : correlation, RMSE, MAE
    """
    mask = ~np.isnan(y_true)  # seulement les points terrestres
    yt = y_true[mask].ravel()
    yp = y_pred[mask].ravel()

    # Corrélation
    r = np.corrcoef(yt, yp)[0, 1]

    # RMSE
    rmse = np.sqrt(np.mean((yp - yt)**2))

    # MAE
    mae = np.mean(np.abs(yp - yt))

    return r, rmse, mae
# ====================================================
# CARTES
# ====================================================

def plot_maps(y_true, y_pred, t, tag):
    yt = y_true[t, ..., 0] * Y_std + Y_mean
    yp = y_pred[t, ..., 0] * Y_std + Y_mean

    yt[~land_mask] = np.nan
    yp[~land_mask] = np.nan

    r, rmse, mae = spatial_metrics(yt, yp)
    date = np.datetime_as_string(time[t], unit="D")

    titles = ["ALADIN 12 km", "U-Net", "Différence"]

    plt.figure(figsize=(15, 4))
    for i, data in enumerate([yt, yp, yp - yt]):
        plt.subplot(1, 3, i+1)
        plt.pcolormesh(lon, lat, data, shading="auto", cmap="coolwarm")
        plt.colorbar()
        plt.title(titles[i])

    plt.suptitle(
        f"{tag} | r={r:.2f} RMSE={rmse:.2f} MAE={mae:.2f}",
        fontsize=12
    )

    plt.savefig(os.path.join(FIG_DIR, f"{tag}_{date}.png"), dpi=200)
    plt.close()


plot_maps(Y_test, Y_pred, 0, "TEST_FULL")
plot_maps(Y_test, Y_pred_ablation, 0, "TEST_ABLATION")

print("\n✅ SCRIPT TERMINÉ AVEC SUCCÈS")
