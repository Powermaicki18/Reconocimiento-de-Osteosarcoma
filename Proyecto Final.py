#PROYECTO FINAL - PROCESAMIENTO DE IMÁGENES
#POR LUISA SÁNCHEZ Y MIGUEL NIETO
#DETECCIÓN DE OSTEOSARCOMAS - METODO 1

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from skimage.filters import threshold_otsu

# Configuración
IMAGE_DIR = 'Prueba'
IMG_SIZE = (256, 256)
EPOCHS = 5
BATCH_SIZE = 8
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Dataset personalizado
class RadiografiaDataset(Dataset):
    def __init__(self, folder):
        self.img_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpeg')]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, IMG_SIZE)
        img = img.astype(np.float32) / 255.0
        return self.transform(img), self.img_paths[idx]

# Modelo Autoencoder
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

# Segmentación ósea
def segmentar_hueso(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    t = threshold_otsu(blur)
    otsu_mask = (blur > t).astype(np.uint8)
    edges = cv2.Canny(blur, 50, 150)
    canny_mask = (edges > 0).astype(np.uint8)
    # Combina ambas
    combined = np.clip(otsu_mask + canny_mask, 0, 1)
    return combined

# Visualización de errores
def mostrar_resultado(img_path, original_tensor, reconstruida_tensor):
    original = original_tensor.squeeze().cpu().numpy()
    reconstruida = reconstruida_tensor.squeeze().cpu().detach().numpy()
    error = np.abs(original - reconstruida)

    hueso_mask = segmentar_hueso((original * 255).astype(np.uint8))
    error *= hueso_mask  # aplicar solo en hueso

    fig, axs = plt.subplots(1, 4, figsize=(14, 4))
    axs[0].imshow(original, cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(reconstruida, cmap='gray')
    axs[1].set_title('Reconstruida')
    axs[2].imshow(hueso_mask, cmap='gray')
    axs[2].set_title('Segmentación ósea')
    axs[3].imshow(error, cmap='hot')
    axs[3].set_title('Mapa de error (tumor)')
    plt.suptitle(f"Análisis: {os.path.basename(img_path)}")
    plt.tight_layout()
    plt.show()

# Entrenamiento
def main():
    dataset = RadiografiaDataset(IMAGE_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = Autoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    print("Entrenando autoencoder...")
    model.train()
    for epoch in range(EPOCHS):
        for x, _ in loader:
            x = x.to(DEVICE)
            y = model(x)
            loss = loss_fn(y, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

    # Evaluar en imágenes
    model.eval()
    with torch.no_grad():
        for i in range(4):  # Cambiar número si se quieren más resultados
            x, path = dataset[i]
            x = x.unsqueeze(0).to(DEVICE)
            y = model(x)
            mostrar_resultado(path, x, y)

if __name__ == "__main__":
    main()
