# ================================
# Proyecto: Análisis de tumores óseos en el fémur
# ================================

import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Función segura para convertir valores a entero
def safe_int(value):
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
    df_femur_stats = df_stats.loc[idx_femur[0]+1 : idx_femur[0]+2]
    df_femur_stats = df_femur_stats.iloc[:, :5]
    df_femur_stats.columns = ["Grupo", "Tumor", "Benign", "Malignant", "No tumor"]

# 2. Cargar dataset global (para validar benigno/maligno por imagen)
df_meta = pd.read_excel("dataset.xlsx")

# Carpeta de imágenes ya filtradas del fémur
img_path = "femur_images/"

# 3. Seleccionar una imagen de cada tipo
benigno_row = df_meta[(df_meta['femur'] == 1) & (df_meta['benign'] == 1)].head(1)
maligno_row = df_meta[(df_meta['femur'] == 1) & (df_meta['malignant'] == 1)].head(1)
sin_tumor_row = df_meta[(df_meta['femur'] == 1) & (df_meta['tumor'] == 0)].head(1)

imagenes = [
    ("Tumor benigno", benigno_row),
    ("Tumor maligno", maligno_row),
    ("Sin tumor", sin_tumor_row)
]

# 4. Función para aplicar Sobel y mostrar resultados
def aplicar_sobel(img_file, diagnostico):
    ruta = os.path.join(img_path, img_file)
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"⚠ No se pudo cargar la imagen: {img_file}")
        return

    # Mostrar imagen original
    plt.imshow(img, cmap='gray')
    plt.title(f"Imagen del fémur ({diagnostico}): {img_file}")
    plt.axis('off')
    plt.show()

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

    sobelx_manual = cv2.filter2D(img, -1, sobel_x)
    sobely_manual = cv2.filter2D(img, -1, sobel_y)
    sobel_mag = cv2.magnitude(np.float32(sobelx_manual), np.float32(sobely_manual))
    sobel_abs = cv2.convertScaleAbs(sobel_mag)

    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1); plt.imshow(sobelx_manual, cmap='gray'); plt.title("Sobel X"); plt.axis('off')
    plt.subplot(1,3,2); plt.imshow(sobely_manual, cmap='gray'); plt.title("Sobel Y"); plt.axis('off')
    plt.subplot(1,3,3); plt.imshow(sobel_abs, cmap='gray'); plt.title("Magnitud combinada"); plt.axis('off')
    plt.show()

# 5. Mostrar cada caso
for diagnostico, row in imagenes:
    if not row.empty:
        img_file = row['image_id'].values[0]
        print(f"\n✅ Procesando imagen {img_file} ({diagnostico})")
        aplicar_sobel(img_file, diagnostico)
    else:
        print(f"⚠ No se encontró imagen para el caso: {diagnostico}")

# 6. Mostrar estadísticas globales del fémur
mujeres_row = df_femur_stats[df_femur_stats["Grupo"].astype(str).str.contains("Mujeres", case=False, na=False)]
hombres_row = df_femur_stats[df_femur_stats["Grupo"].astype(str).str.contains("Hombres", case=False, na=False)]

mujeres_benign = safe_int(mujeres_row["Benign"].values[0]) if not mujeres_row.empty else 0
mujeres_malign = safe_int(mujeres_row["Malignant"].values[0]) if not mujeres_row.empty else 0
hombres_benign = safe_int(hombres_row["Benign"].values[0]) if not hombres_row.empty else 0
hombres_malign = safe_int(hombres_row["Malignant"].values[0]) if not hombres_row.empty else 0

print("\n=== Estadísticas del Fémur (Filtrado de imágenes.xlsx) ===")
print(f"Hombres - Benignos: {hombres_benign}, Malignos: {hombres_malign}")
print(f"Mujeres - Benignos: {mujeres_benign}, Malignos: {mujeres_malign}")

