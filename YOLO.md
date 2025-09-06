## Comando usado no treinamento inicial
```bash
yolo detect train \
  model=yolov8n.pt \
  data=hardhat.yaml \
  imgsz=320 \
  epochs=60 \
  batch=16 \
  device=0 \
  patience=10 \
  workers=2
```

## Explicando cada parâmetro
* **`model=yolov8n.pt`**
  Modelo base que vamos usar.
---
* **`data=hardhat.yaml`**
  Arquivo de configuração do dataset.
---
* **`imgsz=320`**
  Tamanho da imagem usada no treino (redimensionada).

  * Quanto **menor**, mais rápido/leve - menos acurácia.
  * Quanto **maior** (ex.: 640), mais pesado - mais acurácia em objetos pequenos.
---
* **`epochs=60`**
  Quantidade de vezes que o modelo vai passar por todo o dataset.
  * Com o `early stopping` (pacience), nem sempre chega até o fim.
---
* **`batch=16`**
  Quantas imagens o modelo processa por vez.

  * Valores maiores usam mais GPU (VRAM).
  * Se sua GPU reclamar de memória, reduza (8, 4 ou até 2).
  * Se der pra aumentar (32+), pode melhorar estabilidade do treino.
---
* **`device=0`**
  Diz qual GPU usar.

  * `0` = primeira GPU.
  * Se não tiver GPU, será usada a CPU.
  * Se tiver várias: `device=0,1`.
---
* **`patience=10`**
  Hiperparâmetro do **early stopping**:
  * Se por 10 épocas seguidas não melhorar o `mAP` em validação, o treino para.
---
* **`workers=2`**
  Número de processos usados pra carregar imagens em paralelo.
  * Em Windows/Colab, 2 é seguro.
  * Em Linux/GPU boa, pode usar 4 ou 8.
