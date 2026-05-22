# Aprendizaje Federado con MNIST

## Equipo

- Maximiliano García Suástegui - A01657689
- Carlos Alberto Gómez San Pedro - A01658377
- Arath Mendivil Mora - A01660670

Este repositorio contiene una simulación de aprendizaje federado utilizando la base de datos MNIST y un modelo desarrollado con TensorFlow/Keras. El objetivo es entrenar modelos locales sobre particiones privadas y estadísticamente equivalentes de MNIST, para después construir un modelo global mediante diferentes estrategias de agregación.

## Objetivo del proyecto

El objetivo de esta actividad es implementar un flujo de aprendizaje federado donde la base de datos MNIST se divide en 3 partes, una por cada integrante del equipo.

Cada cliente entrena localmente el mismo modelo, usando únicamente su propia partición de datos. Posteriormente, los pesos de los modelos locales se agregan para generar un modelo global.

En este proyecto se comparan tres estrategias:

1. FedAvg One-Shot
2. FedAvg Multi-Round
3. Performance-Weighted Multi-Round

## Estructura del repositorio

```text
federated-mnist/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── TheModel.py
│   ├── split_mnist_private.py
│   ├── local_training.py
│   ├── global_aggregation.py
│   └── utils.py
└── notebooks/
    └── local_training.ipynb
```

Además, durante la ejecución local del proyecto se utilizan o generan las siguientes carpetas:

```text
local_data/
local_models/
global_models/
results/
```

Estas carpetas pueden no aparecer en GitHub si están vacías, ya que Git no versiona carpetas vacías. En caso de no existir, los scripts las crean automáticamente o pueden crearse manualmente antes de ejecutar el flujo.

No es necesario subir archivos `.gitkeep`; estos solo se usan opcionalmente para mostrar carpetas vacías dentro del repositorio.

## Nota importante sobre privacidad

Las particiones locales de MNIST se consideran confidenciales para la simulación federada, por lo que no deben subirse al repositorio.

Las siguientes carpetas están excluidas mediante `.gitignore`:

```text
local_data/
local_models/
global_models/
results/
.venv/
```

El repositorio contiene el código necesario para reproducir el experimento, pero no incluye las particiones privadas, modelos entrenados ni resultados generados localmente. Estos archivos se generan al ejecutar los scripts y permanecen únicamente en el entorno local.

## Modelo utilizado

El modelo está definido en:

```text
src/TheModel.py
```

Se utilizó una red neuronal convolucional desarrollada con TensorFlow/Keras. El modelo es distinto al visto en clase, ya que incorpora:

- Bloques convolucionales adicionales
- Batch Normalization
- Dropout
- Capa densa final para clasificación multiclase

El modelo recibe imágenes de MNIST con dimensión:

```text
28 x 28 x 1
```

y predice una de las 10 clases posibles, correspondientes a los dígitos del 0 al 9.

## Configuración del entorno

### 1. Crear entorno virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn jupyter ipykernel nbformat
```

También se puede instalar desde el archivo `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Generar `requirements.txt`

En caso de ser necesario:

```bash
pip freeze > requirements.txt
```

## Ejecución del proyecto

### Paso 1: dividir MNIST en 3 particiones privadas

```bash
python src/split_mnist_private.py
```

Este script crea 3 particiones locales:

```text
local_data/client_1.npz
local_data/client_2.npz
local_data/client_3.npz
```

Cada cliente recibe 20,000 muestras de entrenamiento. La división se realiza de forma estratificada, por lo que la distribución de clases es prácticamente equivalente entre los tres clientes.

### Paso 2: entrenamiento local

```bash
python src/local_training.py
```

Este script crea primero un modelo global inicial compartido:

```text
global_models/global_initial.keras
```

Después, cada cliente carga ese mismo modelo inicial y entrena localmente con su propia partición de datos.

Esto es importante porque FedAvg requiere que los clientes partan del mismo modelo global inicial. Si cada cliente inicia con pesos aleatorios distintos, el promedio de pesos puede destruir la representación aprendida.

Este paso genera:

```text
local_models/client_1_local.keras
local_models/client_2_local.keras
local_models/client_3_local.keras
results/local_training_results.json
results/client_sample_counts.json
```

### Paso 3: cómputo del modelo global

```bash
python src/global_aggregation.py
```

Este script evalúa tres estrategias de agregación:

1. FedAvg One-Shot
2. FedAvg Multi-Round
3. Performance-Weighted Multi-Round

Los resultados finales se guardan en:

```text
results/global_aggregation_results.json
```

y los modelos globales generados se guardan en:

```text
global_models/
```

## Métodos de agregación

### 1. FedAvg One-Shot

FedAvg One-Shot es la línea base del experimento. Consiste en entrenar una vez los modelos locales y después promediar sus pesos para construir un modelo global.

Como en este experimento los tres clientes tienen el mismo número de muestras, los pesos de agregación son iguales para cada cliente.

### 2. FedAvg Multi-Round

FedAvg Multi-Round simula un escenario federado más realista.

En lugar de agregar una sola vez, el modelo global se actualiza durante varias rondas:

```text
Modelo global → entrenamiento local → agregación → nuevo modelo global
```

Este proceso se repite durante 5 rondas federadas.

### 3. Performance-Weighted Multi-Round

Performance-Weighted Multi-Round también utiliza varias rondas federadas, pero pondera a cada cliente de acuerdo con su desempeño de validación local.

Los clientes con mejor desempeño de validación reciben ligeramente más peso durante la agregación. Esto puede ayudar a reducir la influencia de actualizaciones locales menos confiables.

## Configuración del experimento

```text
Número de clientes: 3
Rondas federadas: 5
Épocas locales por ronda: 1
Batch size: 64
Validation split: 0.1
Seed: 42
```

## Resultados

| Método | Test Loss | Test Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| FedAvg One-Shot | 0.1441 | 0.9584 | 0.9582 | 0.9581 |
| FedAvg Multi-Round | 0.0302 | 0.9890 | 0.9890 | 0.9890 |
| Performance-Weighted Multi-Round | 0.0282 | 0.9899 | 0.9899 | 0.9899 |

## Interpretación de resultados

El método FedAvg One-Shot obtuvo una exactitud de 0.9584. Este resultado confirma que la agregación funciona correctamente cuando todos los clientes parten del mismo modelo global inicial.

FedAvg Multi-Round mejoró el resultado hasta 0.9890. Esto se debe a que el modelo global se actualiza de forma iterativa durante varias rondas, permitiendo que el aprendizaje de los clientes se integre progresivamente.

El mejor resultado fue obtenido por Performance-Weighted Multi-Round, con una exactitud de 0.9899. Este método dio ligeramente más peso a los clientes con mejor desempeño de validación en cada ronda. La mejora sobre FedAvg Multi-Round fue pequeña porque las particiones eran estadísticamente equivalentes y tenían el mismo tamaño, pero aun así fue el método con mejor desempeño.

## Notas finales

Este proyecto es una simulación de aprendizaje federado. En un sistema federado real, cada cliente entrenaría en un dispositivo o servidor separado, y únicamente compartiría pesos o actualizaciones del modelo. Los datos locales no se compartirían con el servidor central.

En este proyecto, las particiones locales se mantienen fuera del repositorio para respetar el requisito de confidencialidad de la actividad.
