# ================================
# Proyecto: Análisis de tumores óseos en el fémur
# ================================

import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

# Función segura para convertir valores a entero
def safe_int(value):
    """Convierte a entero si es posible, si no devuelve 0"""
    try:
        return int(float(value))
    except:
        return 0

# 1. Cargar hoja de cálculo con estadísticas globales (Filtrado de imágenes.xlsx)
df_stats = pd.read_excel("Filtrado de imágenes.xlsx")

# Buscar la fila que contiene "Femur"
mask_femur = df_stats.astype(str).apply(lambda row: row.str.contains("Femur", case=False, na=False).any(), axis=1)
idx_femur = df_stats.index[mask_femur].tolist()

df_femur_stats = pd.DataFrame()
if idx_femur:
    # Tomamos las dos filas siguientes (Mujeres y Hombres)
    df_femur_stats = df_stats.loc[idx_femur[0]+1 : idx_femur[0]+2]

    # Seleccionamos solo las primeras 5 columnas relevantes
    df_femur_stats = df_femur_stats.iloc[:, :5]

    # Renombrar columnas
    df_femur_stats.columns = ["Grupo", "Tumor", "Benign", "Malignant", "No tumor"]

# 2. Cargar dataset de imágenes y filtrar solo las del fémur
df_meta = pd.read_excel("dataset.xlsx")
df_femur_meta = df_meta[df_meta['femur'] == 1]   # aquí se filtran las imágenes del fémur

# Carpeta de imágenes
img_path = "images/"

# Validar imágenes disponibles en tiempo real
valid_images = [f for f in df_femur_meta['image_id'].tolist() if os.path.exists(os.path.join(img_path, f))]

if valid_images:
    random_img = random.choice(valid_images)
    print("\nMostrando imagen aleatoria del fémur:", random_img)
else:
    print("⚠ No se encontraron imágenes del fémur en la carpeta.")

# Buscar la fila correspondiente a la imagen aleatoria en el dataset
meta_row = df_femur_meta[df_femur_meta['image_id'] == random_img]

if not meta_row.empty:
    tiene_tumor = meta_row['tumor'].values[0]   # 1 = tumor, 0 = no tumor
    benigno = meta_row['benign'].values[0]
    maligno = meta_row['malignant'].values[0]

    if tiene_tumor == 1:
        if benigno == 1:
            print(f"✅ La imagen {random_img} corresponde a un fémur con tumor benigno.")
        elif maligno == 1:
            print(f"⚠ La imagen {random_img} corresponde a un fémur con tumor maligno.")
        else:
            print(f"✅ La imagen {random_img} corresponde a un fémur con tumor (tipo no especificado).")
    else:
        print(f"✅ La imagen {random_img} corresponde a un fémur sin tumor.")
else:
    print("⚠ No se encontró información de tumor para esta imagen en el Excel.")

# 3. Interfaz bonita en consola (tabla clara)

# Número total de imágenes en el dataset
total_imgs = len(df_meta)

# Número de imágenes de fémur
total_femur = len(df_femur_meta)

# Extraer valores de Filtrado de imágenes.xlsx con seguridad
mujeres_row = df_femur_stats[df_femur_stats["Grupo"].astype(str).str.contains("Mujeres", case=False, na=False)]
hombres_row = df_femur_stats[df_femur_stats["Grupo"].astype(str).str.contains("Hombres", case=False, na=False)]

mujeres_benign = safe_int(mujeres_row["Benign"].values[0]) if not mujeres_row.empty else 0
mujeres_malign = safe_int(mujeres_row["Malignant"].values[0]) if not mujeres_row.empty else 0
hombres_benign = safe_int(hombres_row["Benign"].values[0]) if not hombres_row.empty else 0
hombres_malign = safe_int(hombres_row["Malignant"].values[0]) if not hombres_row.empty else 0

# Mostrar en consola
print("\n=== Estadísticas del Fémur (Filtrado de imágenes.xlsx) ===")
print(f"Total de imágenes en el dataset: {total_imgs}")
print(f"Total de imágenes de fémur: {total_femur}")
print(f"Hombres - Benignos: {hombres_benign}, Malignos: {hombres_malign}")
print(f"Mujeres - Benignos: {mujeres_benign}, Malignos: {mujeres_malign}")

# 4. Aplicamos el filtro de Sobel

ImageOriginal = cv2.imread(os.path.join(img_path, random_img), cv2.IMREAD_GRAYSCALE)

if ImageOriginal is None:
    print("⚠ No se pudo cargar la imagen:", random_img)
else:
    # Mostrar imagen original
    plt.imshow(ImageOriginal, cmap='gray')
    plt.title(f"Imagen original del fémur: {random_img}")
    plt.axis('off')
    plt.show()

Imagen_FSobel = ImageOriginal

sobel_x = np.array([
    [-2,-1,0,1,2],
    [-2,-1,0,1,2],
    [-4,-2,0,2,4],
    [-2,-1,0,1,2],
    [-2,-1,0,1,2],
])

sobel_y = np.array([
    [-2, -2, -4, -2, -2],
    [-1, -1, -2, -1, -1],
    [ 0,  0,  0,  0,  0],
    [ 1,  1,  2,  1,  1],
    [ 2,  2,  4,  2,  2]
])

# Sobel X manual
sobelx_manual = cv2.filter2D(Imagen_FSobel, -1, sobel_x)

# Sobel Y manual
sobely_manual = cv2.filter2D(Imagen_FSobel, -1, sobel_y)

# Magnitud combinada
sobel_mag = cv2.magnitude(np.float32(sobelx_manual), np.float32(sobely_manual))
sobel_abs = cv2.convertScaleAbs(sobel_mag)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1); plt.imshow(sobelx_manual, cmap='gray'); plt.title("Sobel X manual"); plt.axis('off')
plt.subplot(1,3,2); plt.imshow(sobely_manual, cmap='gray'); plt.title("Sobel Y manual"); plt.axis('off')
plt.subplot(1,3,3); plt.imshow(sobel_abs, cmap='gray'); plt.title("Magnitud combinada"); plt.axis('off')
plt.show()

