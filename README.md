# TraficoCNN - Clasificacion de senales de trafico con PyTorch

Proyecto desarrollado como practica de Inteligencia Artificial (AA2).
El objetivo es entrenar una red neuronal convolucional para reconocer distintas senales de trafico a partir de imagenes del dataset GTSRB (German Traffic Sign Recognition Benchmark).

---

## Estructura del proyecto

TraficoCNN/
|-- data/
|   |-- train/
|   |   |-- stop/
|   |   |-- yield/
|   |   |-- no_entry/
|   |   `-- speed_limit_30/
|   `-- val/
|       |-- stop/
|       |-- yield/
|       |-- no_entry/
|       `-- speed_limit_30/
|-- main.py
|-- main_resnet.py
|-- copiar-gtsrb.ps1
|-- requirements.txt
|-- .gitignore
`-- README.md

---

## Descripcion del modelo

Se implementaron dos enfoques:

1. CNN propia (main.py)
   - Arquitectura simple con 4 bloques convolucionales (Conv2d -> ReLU -> MaxPool2d)
   - Capas Dropout para regularizacion
   - Entrenamiento con CrossEntropyLoss y optimizador Adam
   - Data augmentation con RandomRotation, ColorJitter, Affine, etc.

2. Transfer Learning (main_resnet.py)
   - Basado en ResNet18 preentrenada en ImageNet
   - Se reemplazo la capa final (fc) por una con 4 salidas (una por clase)
   - Se congelaron las capas convolucionales base para aprovechar los pesos ya aprendidos
   - Entrenamiento mas rapido y con mejor rendimiento en validacion

---

## Dataset: GTSRB

- Fuente: https://benchmark.ini.rub.de/gtsrb_dataset.html
- Subconjunto de 4 clases utilizadas:
  - 00001 -> Speed limit (30 km/h)
  - 00013 -> Yield
  - 00014 -> Stop
  - 00017 -> No entry
- Preparacion automatica mediante el script copiar-gtsrb.ps1

---

## Requisitos e instalacion

1. Crear y activar entorno virtual:
   python -m venv .venv
   .\.venv\Scripts\Activate

2. Instalar dependencias:
   pip install -r requirements.txt

3. Ejecutar el modelo:
   python main.py          # Entrenamiento con CNN propia
   python main_resnet.py   # Transfer learning con ResNet18

---

## Resultados

CNN propia:
- Train Accuracy: ~95%
- Val Accuracy: ~60-70%
- Observacion: tendencia clara a overfitting

ResNet18 (Transfer Learning):
- Train Accuracy: ~98%
- Val Accuracy: ~85-90%
- Mejor generalizacion y convergencia mas rapida

Grafica de entrenamiento (ejemplo):
![Accuracy y Loss](docs/training_plot.png)

---

## Funcion de perdida

Se utiliza Categorical Cross Entropy, ideal para clasificacion multiclase:

L = - sum(y_i * log(p_i))

donde y_i es la etiqueta real (one-hot) y p_i la probabilidad predicha tras la funcion Softmax.
Penaliza fuertemente cuando la clase correcta tiene baja probabilidad, permitiendo gradientes mas informativos y aprendizaje mas estable.

---

## Tecnologias utilizadas

- Python 3.10+
- PyTorch
- Torchvision
- Matplotlib
- scikit-learn
- PowerShell (para preparar el dataset)

---

## Autor

Raule  
Grado en Ciencia e Ingenieria de Datos  
Universidad de Las Palmas de Gran Canaria

"El mejor modelo no es el mas complejo, sino el que generaliza mejor."
