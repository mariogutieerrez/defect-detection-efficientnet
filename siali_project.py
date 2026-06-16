# =================================================================
# Detección de defectos en botellas con visión artificial
# Autor: Mario Gutiérrez de Uzquiano
# Dataset: MVTec AD - categoría bottle
# Modelo: EfficientNet-B0 con transfer learning
# =================================================================

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt


# =================================================================
# RUTAS Y PARÁMETROS
# =================================================================
ruta = r'C:\Users\34679\OneDrive\Escritorio\bottle'

BATCH_SIZE = 16
EPOCHS     = 20
NUM_CLASES = 2  # 0 = botella OK, 1 = botella con defecto


# =================================================================
# TRANSFORMACIONES
# En train aplicamos data augmentation para generar variedad artificial
# y reducir el overfitting. En test solo normalizamos.
# =================================================================
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(            # Valores estándar de ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =================================================================
# DATASET PERSONALIZADO
# Le decimos a PyTorch dónde están las imágenes y qué etiqueta
# tiene cada una. Las muestras defectuosas se dividen 50/50
# entre entrenamiento y test para evaluar sobre datos no vistos.
# =================================================================
class BottleDataset(Dataset):
    def __init__(self, ruta, modo='train', transform=None):
        self.transform = transform
        self.imagenes = []
        self.etiquetas = []

        carpeta_good = os.path.join(ruta, 'train', 'good')

        # Botellas correctas: todas para entrenamiento
        if modo == 'train':
            for img in os.listdir(carpeta_good):
                self.imagenes.append(os.path.join(carpeta_good, img))
                self.etiquetas.append(0)

        # Defectos: mitad para train, mitad para test
        test_path = os.path.join(ruta, 'test')
        for categoria in os.listdir(test_path):
            carpeta = os.path.join(test_path, categoria)
            imgs = os.listdir(carpeta)
            mitad = len(imgs) // 2

            if categoria == 'good':
                if modo == 'test':
                    for img in imgs:
                        self.imagenes.append(os.path.join(carpeta, img))
                        self.etiquetas.append(0)
            else:
                subset = imgs[:mitad] if modo == 'train' else imgs[mitad:]
                for img in subset:
                    self.imagenes.append(os.path.join(carpeta, img))
                    self.etiquetas.append(1)

    def __len__(self):
        return len(self.imagenes)

    def __getitem__(self, idx):
        img = Image.open(self.imagenes[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.etiquetas[idx]


# =================================================================
# DATALOADERS
# Sirven las imágenes en batches de 16 sin cargar todo en memoria.
# =================================================================
train_dataset = BottleDataset(ruta, modo='train', transform=transform_train)
test_dataset  = BottleDataset(ruta, modo='test',  transform=transform_test)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Imágenes de entrenamiento: {len(train_dataset)}")
print(f"Imágenes de test: {len(test_dataset)}")


# =================================================================
# MODELO - EFFICIENTNET CON TRANSFER LEARNING
# Usamos EfficientNet-B0 preentrenado en ImageNet.
# Congelamos todo excepto los últimos 3 bloques, que se adaptan
# a nuestro problema mediante fine-tuning.
# El clasificador final se reemplaza por uno de 2 clases.
# =================================================================
modelo = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

# Congelamos todos los parámetros
for param in modelo.parameters():
    param.requires_grad = False

# Descongelamos los últimos 3 bloques para fine-tuning
for param in modelo.features[-3:].parameters():
    param.requires_grad = True

# Nuevo clasificador adaptado a 2 clases
modelo.classifier[1] = nn.Linear(modelo.classifier[1].in_features, NUM_CLASES)

device = torch.device('cpu')
modelo = modelo.to(device)
print("Modelo cargado correctamente")


# =================================================================
# FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
# CrossEntropyLoss para clasificación binaria.
# Adam con dos learning rates distintos: conservador para las capas
# preentrenadas y más alto para el clasificador nuevo.
# =================================================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': modelo.features[-3:].parameters(), 'lr': 0.0001},  # fine-tuning suave
    {'params': modelo.classifier[1].parameters(), 'lr': 0.001}    # clasificador nuevo
])


# =================================================================
# ENTRENAMIENTO
# Cada época recorre todas las imágenes en batches de 16.
# Por cada batch: predicción → error → gradientes → actualización de pesos.
# =================================================================
print("\n--- Iniciando entrenamiento ---")
for epoch in range(EPOCHS):
    modelo.train()
    running_loss = 0.0

    for imagenes, etiquetas in train_loader:
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)

        optimizer.zero_grad()                  # Limpiamos gradientes anteriores
        outputs = modelo(imagenes)             # Forward pass
        loss = criterion(outputs, etiquetas)   # Calculamos el error
        loss.backward()                        # Backward pass
        optimizer.step()                       # Actualizamos los pesos

        running_loss += loss.item()

    print(f"Época [{epoch+1:02d}/{EPOCHS}] - Loss: {running_loss/len(train_loader):.4f}")

print("--- Entrenamiento completado ---\n")


# =================================================================
# EVALUACIÓN
# Evaluamos sobre imágenes no vistas durante el entrenamiento.
# =================================================================
modelo.eval()
correctas = 0
total = 0

with torch.no_grad():
    for imagenes, etiquetas in test_loader:
        imagenes = imagenes.to(device)
        etiquetas = etiquetas.to(device)
        outputs = modelo(imagenes)
        _, predicciones = torch.max(outputs, 1)
        total += etiquetas.size(0)
        correctas += (predicciones == etiquetas).sum().item()

accuracy = 100 * correctas / total
print(f"Precisión sobre el conjunto de test: {accuracy:.2f}% ({correctas}/{total})")


# =================================================================
# GUARDADO DEL MODELO
# =================================================================
torch.save(modelo.state_dict(), r'C:\Users\34679\OneDrive\Escritorio\modelo_siali.pth')
print("Modelo guardado correctamente")


# =================================================================
# VISUALIZACIÓN DE RESULTADOS
# Cuadrícula con las predicciones sobre el conjunto de test.
# Verde = acierto | Rojo = fallo
# =================================================================
modelo.eval()
fig, axes = plt.subplots(4, 6, figsize=(18, 12))
axes = axes.flatten()

with torch.no_grad():
    for i, ax in enumerate(axes):
        if i >= len(test_dataset.imagenes):
            ax.axis('off')
            continue

        img_original = Image.open(test_dataset.imagenes[i]).convert('RGB')
        img_original = img_original.resize((224, 224))

        img_tensor = transform_test(img_original).unsqueeze(0).to(device)
        output = modelo(img_tensor)
        _, pred = torch.max(output, 1)
        pred = pred.item()
        etiqueta_real = test_dataset.etiquetas[i]

        color = 'green' if pred == etiqueta_real else 'red'
        texto_pred = 'OK' if pred == 0 else 'DEFECTO'
        texto_real = 'OK' if etiqueta_real == 0 else 'DEFECTO'

        ax.imshow(img_original)
        ax.set_title(f'Pred: {texto_pred}\nReal: {texto_real}',
                     color=color, fontsize=8, fontweight='bold')
        ax.axis('off')

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

plt.suptitle(f'Resultados sobre conjunto de test — Accuracy: {accuracy:.2f}%',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(r'C:\Users\34679\OneDrive\Escritorio\resultados_siali.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Figura guardada en el escritorio")