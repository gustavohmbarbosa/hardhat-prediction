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

Usa o um dos .yaml definidos na pasta models. Vamos ver se o mAP fica bom, e tentar reduzir os ainda mais a arquitetira.

### 3) Gerar tabela com resultados | `summarize_runs.py`
`python summarize_runs.py`

Vai gerar uma tabela neste estilo:
# Model Report

| Kind   | Run                           | Stage        | Model            |   ImgSize |   Epochs |   Batch | File                                          |   Size (MB) |   mAP@0.5 |   mAP@0.5:0.95 |   Train Time (min) |   Params (M) |   FLOPs (G) |   Latency (ms) |
|:-------|:------------------------------|:-------------|:-----------------|----------:|---------:|--------:|:----------------------------------------------|------------:|----------:|---------------:|-------------------:|-------------:|------------:|---------------:|
| run    | runs_train\reduced_min_320_v2 | reduced_arch | yolov8n_min.yaml |       320 |      100 |      32 | runs_train\reduced_min_320_v2\weights\best.pt |       3.109 |    0.861  |         0.5402 |              211.4 |       1.5387 |      0.5748 |          6.21  |
| run    | runs_train\reduced_min_320    | reduced_arch | yolov8n_min.yaml |       320 |       10 |      16 | runs_train\reduced_min_320\weights\best.pt    |       3.603 |    0.7638 |         0.4493 |               18.7 |       1.7992 |      0.6469 |          6.569 |
| run    | runs_train\baseline_v8n_320   |              | yolov8n.pt       |       320 |       60 |      16 | runs_train\baseline_v8n_320\weights\best.pt   |       5.923 |    0.8879 |         0.5803 |               89.7 |       3.011  |      1.0243 |          6.965 |

TODO: explicar colunas da tabela

# Útil
* [Descrição de alguns parâmetros YOLO](./YOLO.md)
* [Script](./verify-cuda.py) que verifica se a GPU e o CUDA estão disponíveis: `verify-cuda.py`