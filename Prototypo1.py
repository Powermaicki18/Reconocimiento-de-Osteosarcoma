# ================================
# Proyecto: Análisis de tumores óseos en el fémur
# Mostrar estadísticas y una imagen aleatoria del fémur
# ================================

import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

# 1. Cargar hoja de cálculo con estadísticas globales
df_stats = pd.read_excel("Filtrado de imágenes.xlsx")

# Buscar la fila que contiene "Femur"
mask_femur = df_stats.astype(str).apply(lambda row: row.str.contains("Femur", case=False, na=False).any(), axis=1)
idx_femur = df_stats.index[mask_femur].tolist()

df_femur_stats = pd.DataFrame()
if idx_femur:
    # Tomamos las dos filas siguientes (Mujeres y Hombres)
    df_femur_stats = df_stats.loc[idx_femur[0]+1 : idx_femur[0]+2]

    # Renombrar columnas según tu estructura
    df_femur_stats = df_femur_stats.rename(
        columns={df_femur_stats.columns[0]:"Grupo",   # Mujeres/Hombres
                 df_femur_stats.columns[1]:"Tumor",
                 df_femur_stats.columns[2]:"Benign",
                 df_femur_stats.columns[3]:"Malignant",
                 df_femur_stats.columns[4]:"No tumor"}
    )

# 2. Interfaz bonita en consola (tabla clara)
print("\n=== Estadísticas del Fémur ===")
print("{:<12} {:>8} {:>8} {:>10} {:>10}".format("Grupo","Tumor","Benign","Malignant","No tumor"))
print("-"*60)

for _, row in df_femur_stats.iterrows():
    grupo = str(row.get("Grupo", ""))
    tumor = row.get("Tumor", 0)
    benign = row.get("Benign", 0)
    malignant = row.get("Malignant", 0)
    no_tumor = row.get("No tumor", 0)

    # Convertir a número entero si es posible, si no dejar como texto
    try: tumor = int(float(tumor))
    except: tumor = str(tumor)
    try: benign = int(float(benign))
    except: benign = str(benign)
    try: malignant = int(float(malignant))
    except: malignant = str(malignant)
    try: no_tumor = int(float(no_tumor))
    except: no_tumor = str(no_tumor)

    print("{:<12} {:>8} {:>8} {:>10} {:>10}".format(
        grupo, tumor, benign, malignant, no_tumor
    ))

# 3. Cargar dataset de imágenes y filtrar solo las del fémur
df_meta = pd.read_excel("dataset.xlsx")
df_femur_meta = df_meta[df_meta['femur'] == 1]   # aquí se filtran las imágenes del fémur

# Carpeta de imágenes
img_path = "images/"

# Validar imágenes disponibles en tiempo real
valid_images = [f for f in df_femur_meta['image_id'].tolist() if os.path.exists(os.path.join(img_path, f))]

def show_image(img_file):
    """Carga y muestra una imagen en escala de grises"""
    img = cv2.imread(os.path.join(img_path, img_file), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠ No se pudo cargar la imagen: {img_file}")
        return
    plt.imshow(img, cmap='gray')
    plt.title(f"Imagen aleatoria del fémur: {img_file}")
    plt.axis('off')
    plt.show()

if valid_images:
    random_img = random.choice(valid_images)
    print("\nMostrando imagen aleatoria del fémur:", random_img)
else:
    print("⚠ No se encontraron imágenes del fémur en la carpeta.")

##avances para 10/03/2026 agregacion de filtros consultados para la imagen random que logramos filtrar del dataset

##Aplicamos el filtro de sovel 

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
    [-1,0,1],
    [-2,0,2],
    [-1,0,1],
])

sobel_y = np.array([
    [-1,-2,1],
    [0,0,0],
    [1,2,1],
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
