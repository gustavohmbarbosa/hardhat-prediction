# Helmet YOLO Reducer

Este projeto tem como objetivo **treinar** e, em seguida, **reduzir** o tamanho de um modelo YOLO para detecção de **capacete** (1 classe).

## ⚙️ Metodologia

### 0) Preparar o dataset | `prepare_helmet_dataset.py`

```bash
python prepare_helmet_dataset.py
```

* Faz o download do dataset do Kaggle (se já não existir em `dataset/`).
* Aplica a filtragem dos rótulos, mantendo **apenas a classe 0 (capacete)**.
* Gera uma estrutura de pastas no formato YOLO:

  ```
  dataset/
    images/{train,val,test}
    labels/{train,val,test}
  ```

>⚠️ **Nota**: é necessário ter as credenciais do Kaggle configuradas (`kaggle.json`). Saiba mais em https://www.kaggle.com/docs/api#authentication.

### 1) Treino baseline | `train_baseline.py`

```bash
python train_baseline.py
```

* Treina o **YOLOv8n (nano)** pré-treinado no COCO.
* Usa hiperparâmetros fixos definidos no código (ex.: `imgsz=320`, `epochs=60`, `batch=16`).
* Serve como **referência** para comparação com versões reduzidas.

### 2) Treino reduzido | `train_reduced.py`

```bash
python train_reduced.py
```

* Treina uma arquitetura **modificada** a partir de um `.yaml` em `models/` (ex.: `yolov8n_min.yaml`).
* O `.yaml` usa multiplicadores menores de **depth** e **width**, reduzindo:

  * **Parâmetros (M)**
  * **FLOPs (G)**
  * **Tamanho do arquivo**
* Objetivo: encontrar um ponto de equilíbrio entre **tamanho menor** e **mAP aceitável**.

### 3) Poda do modelo (descartado) | `prune_and_finetune.py`

> Objetivo: **reduzir ainda mais** o tamanho/complexidade do modelo **depois** do treino, removendo canais pouco importantes (**structured channel pruning**) e, opcionalmente, fazer um **fine-tune** curto para recuperar mAP.

#### O que o script faz

1. **Carrega** um checkpoint treinado (`best.pt`) do Ultralytics YOLO.
2. **Calcula importância** dos canais (L1/Magnitude) e seleciona **apenas** `Conv2d` com `groups==1` (evita depthwise/DFL/Detect).
3. **Tenta podar estruturalmente** (remover canais do grafo, não só zerar pesos).
4. **Salva** o resultado
5. (Opcional) **Fine-tune** rápido no modelo podado para **recuperar a acurácia**.

> Nota: em arquiteturas com **atalhos/concats** (como YOLOv8 com blocos C2f e múltiplos heads), a poda estrutural é **mais restrita**. Se o grafo não puder ser ajustado com segurança, o script evita cortes perigosos (isso pode resultar em “sem mudança de Params/FLOPs”). A alternativa mais robusta e simples é **reduzir a arquitetura pelo YAML** (width/depth), e usar **quantização** para diminuir arquivo/latência.

#### Uso

```bash
# dependências
pip install torch-pruning ultralytics thop onnx onnxruntime

# execução
python prune_and_finetune.py \
  --weights runs_train/<seu_run>/weights/best.pt \
  --data hardhat.yaml \
  --imgsz 320 \
  --epochs 30 \        # 0 = não faz fine-tune
  --batch 16 \
  --sparsity 0.35 \    # 0.2~0.35 = moderado; >0.5 = agressivo
  --device cuda:0
```

##### Argumentos principais

* `--weights`: checkpoint treinado a ser podado (ex.: `best.pt`).
* `--data`: YAML do dataset (ex.: `hardhat.yaml`) — usado no fine-tune.
* `--imgsz`: tamanho de entrada para traçar o grafo.
* `--epochs`: épocas de **fine-tune pós-poda** (0 para pular).
* `--batch`: batch size do fine-tune.
* `--sparsity`: **proporção global de canais a remover** (0.0–0.9).
* `--device`: `cuda:0` para GPU, ou `cpu`.

##### Saídas esperadas

* `pruned_full.pt` → **nn.Module** podado (estrutura menor).
* `pruned.onnx` → export estruturado (se `onnx` instalado).
* Novo run em `runs_prune/<exp>/` se `--epochs > 0` (fine-tune), com `weights/best.pt` e métricas.

#### Como avaliar

* Rode `python summarize_runs.py` para comparar **Params, FLOPs, Size, Latency e mAP** antes × depois.
* Se **Params/FLOPs não mudarem**, a poda **não foi estruturalmente aplicada** naquela topologia


### 4) Resumir resultados | `summarize_runs.py`

```bash
python summarize_runs.py
```

* Coleta automaticamente os resultados em `runs_train/*` e `runs_prune/*`.
* Gera um relatório `.csv` e `.md` com métricas dos modelos.

Exemplo:
| Kind   | Run                           | Stage        | Model            |   ImgSize |   Epochs |   Batch | File                                          |   Size (MB) |   mAP@0.5 |   mAP@0.5:0.95 |   Train Time (min) |   Params (M) |   FLOPs (G) |   Latency (ms) |
|:-------|:------------------------------|:-------------|:-----------------|----------:|---------:|--------:|:----------------------------------------------|------------:|----------:|---------------:|-------------------:|-------------:|------------:|---------------:|
| run    | runs_train\reduced_min_320_v2 | reduced_arch | yolov8n_min.yaml |       320 |      100 |      32 | runs_train\reduced_min_320_v2\weights\best.pt |       3.109 |    0.861  |         0.5402 |              211.4 |       1.5387 |      0.5748 |          6.21  |
| run    | runs_train\reduced_min_320    | reduced_arch | yolov8n_min.yaml |       320 |       10 |      16 | runs_train\reduced_min_320\weights\best.pt    |       3.603 |    0.7638 |         0.4493 |               18.7 |       1.7992 |      0.6469 |          6.569 |
| run    | runs_train\baseline_v8n_320   |              | yolov8n.pt       |       320 |       60 |      16 | runs_train\baseline_v8n_320\weights\best.pt   |       5.923 |    0.8879 |         0.5803 |               89.7 |       3.011  |      1.0243 |          6.965 |

## Explicação das colunas

* **Kind** → tipo de entrada (`run` = treino, `pruned_ckpt` = modelo podado sem fine-tune).
* **Run** → diretório do experimento.
* **Stage** → estágio do processo (baseline, reduced\_arch, prune, etc).
* **Model** → qual YAML ou checkpoint foi usado.
* **ImgSize** → resolução de entrada (pixels).
* **Epochs** → número de épocas de treino.
* **Batch** → tamanho do batch.
* **File** → caminho para o `best.pt`.
* **Size (MB)** → tamanho do arquivo em disco.
* **mAP\@0.5** → média da precisão média com IoU=0.5 (permissivo).
* **mAP\@0.5:0.95** → métrica oficial COCO (IoU=0.5:0.95, mais rigorosa).
* **Train Time (min)** → duração estimada do treino.
* **Params (M)** → número de parâmetros (em milhões).
* **FLOPs (G)** → custo computacional (em bilhões de operações).
* **Latency (ms)** → tempo médio de inferência de 1 imagem (ms).

## 🔗 Útil

* [Descrição de parâmetros YOLO](./YOLO.md)
* [Script de verificação GPU/CUDA](./verify-cuda.py):

  ```bash
  python verify-cuda.py
  ```

## Outros Scripts

### 📸 Predição em uma imagem | `predict_one.py`

**O que faz:**
Carrega o modelo escolhido, executa a **predição em uma única imagem**, salva:

* a **imagem com as detecções desenhadas** (`outdir/vis/<nome>_pred.jpg`);
* um **.txt** com as detecções (classe, confiança, x1 y1 x2 y2).

**Uso:**

```bash
python predict_one.py \
  --weights runs_train/reduced_min_320_v2/weights/best.pt \
  --image dataset/images/test/obra_101.jpg \
  --outdir predict_out \
  --imgsz 320 \
  --conf 0.25
```

**Parâmetros principais:**

* `--weights` → caminho do modelo (.pt).
* `--image` → caminho da imagem de entrada.
* `--outdir` → pasta de saída (default: `predict_out`).
* `--imgsz` → tamanho de entrada para a inferência (default: 320).
* `--conf` → limiar de confiança (default: 0.25).

**Saídas:**

* `predict_out/vis/obra_101_pred.jpg` (overlay com as caixas)
* `predict_out/obra_101_pred.txt` (linhas: `class conf x1 y1 x2 y2`)

---

### 🧪 Teste no conjunto de teste (ou subset) | `test_model.py`

Compara **Pred vs Real** em `images/test`.

**O que faz:**

* Lê `images/test` e `labels/test` a partir do `hardhat.yaml`.
* Roda predição, faz **matching por IoU** (padrão `0.5`) e contabiliza **TP/FP/FN**.
* Gera **visualizações** com **GT (verde)** vs **Pred (vermelho)**.
* Salva **CSV por imagem** e **summary.json** com métricas agregadas.

**Casos de uso:**

1. **Avaliar todo o test:**

```bash
python test_model.py \
  --weights runs_train/reduced_min_320_v2/weights/best.pt \
  --data hardhat.yaml \
  --outdir eval_out \
  --imgsz 320 \
  --conf 0.25 \
  --nms_iou 0.5 \
  --match_iou 0.5
```

2. **Avaliar somente 3 imagens específicas (pelos nomes-base):**

```bash
python test_model.py \
  --weights runs_train/reduced_min_320_v2/weights/best.pt \
  --data hardhat.yaml \
  --outdir eval_out_subset \
  --imgsz 320 \
  --conf 0.25 \
  --nms_iou 0.5 \
  --match_iou 0.5 \
  --subset obra_101,obra_202,capacete_777
```

**Parâmetros principais:**

* `--weights` → modelo (.pt) a avaliar.
* `--data` → YAML do dataset (ex.: `hardhat.yaml`).
* `--outdir` → pasta de saída (default: `eval_out`).
* `--imgsz` → tamanho de entrada da inferência (default: 320).
* `--conf` → limiar de confiança (default: 0.25).
* `--nms_iou` → IoU para NMS das predições (default: 0.5).
* `--match_iou` → IoU para considerar **TP** no matching Pred×GT (default: 0.5).
* `--subset` → lista de nomes-base (sem extensão) separados por vírgula para avaliar só um subconjunto.

**Saídas:**

* `eval_out/vis/<nome>_cmp.jpg` → imagem com **GT (verde)** e **Pred (vermelho)**.
* `eval_out/per_image.csv` → métricas por imagem (`TP/FP/FN`, `precision`, `recall`, `f1`, `mean_IoU@match_iou`).
* `eval_out/summary.json` → agregado global (totais de TP/FP/FN e métricas globais).

**Leitura rápida das métricas:**

* **precision** alto / **recall** baixo → modelo “exigente” (menos falsos positivos, mais falsos negativos).
* **recall** alto / **precision** baixo → modelo “permissivo” (mais cobertura, porém mais falsos positivos).
* **mean\_IoU\@0.5** → quão bem as caixas **corretas** sobrepõem o GT (média de IoU dos TPs).
