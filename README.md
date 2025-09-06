# Helmet YOLO Reducer

Este projeto propõe treinar e **encolher** um detector de **capacete (yolo com uma 1 classe)**.

##  Metodologia
### 0) Preparar o dataset | `prepare_helmet_dataset.py`
`python prepare_helmet_dataset.py`

A pasta dadaset será baixada do Kaggle, para isso verifique sei login local. Se a pasta já existe, simplesmente vai ocorrer a filtragem da base, deixando apenas uma classe (a classe de capacete) nos labels.

### 1) Treino baseline | `train_baseline.py`
`python train_baseline.py`

Ele vai treinar YOLOv8n (pré-treinado COCO) com hiperparâmetros fixos.

### 2) Treino reduzido | `train_reduced.py`
`python train_reduced.py`

Gera yolov8n_min.yaml com depth_multiple width_multiple para 1 classe. Vamos ver se o mAP fica bom, e tentar reduzir os ainda multiplicadores.

# Útil
* [Descrição de alguns parâmetros YOLO](./YOLO.md)
* [Script](./verify-cuda.py) que verifica se a GPU e o CUDA estão disponíveis: `verify-cuda.py`