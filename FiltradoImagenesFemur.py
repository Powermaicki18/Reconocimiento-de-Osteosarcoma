# ================================
# Script: Agrupar imágenes del fémur en una carpeta nueva
# ================================

import pandas as pd
import os
import shutil

# 1. Cargar dataset de imágenes
df_meta = pd.read_excel("dataset.xlsx")

# 2. Filtrar solo las imágenes del fémur
df_femur_meta = df_meta[df_meta['femur'] == 1]

# Carpeta original de imágenes
img_path = "images/"

# Carpeta destino para agrupar las imágenes del fémur
dest_path = "femur_images/"

# Crear carpeta destino si no existe
os.makedirs(dest_path, exist_ok=True)

# 3. Copiar las imágenes filtradas
for img_file in df_femur_meta['image_id'].tolist():
    src = os.path.join(img_path, img_file)
    dst = os.path.join(dest_path, img_file)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"✅ Copiada: {img_file}")
    else:
        print(f"⚠ No se encontró la imagen: {img_file}")

print("\n=== Proceso completado ===")
print(f"Total de imágenes de fémur copiadas: {len(df_femur_meta)}")
print(f"Carpeta destino: {dest_path}")