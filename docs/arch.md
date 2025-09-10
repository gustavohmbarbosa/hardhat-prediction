## 1. Introdução
O YAML define a arquitetura do YOLOv8, uma rede convolucional moderna para detecção de objetos. O arquivo é dividido em Parâmetros gerais, Backbone e Head.

## 2. Parâmetros Gerais
Define quantas classes serão identificadas e as escalas do modelo.
```yaml
nc: 80 # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  n: [0.33, 0.25, 1024] # YOLOv8n summary: 129 layers, 3157200 parameters, 3157184 gradients, 8.9 GFLOPS
  s: [0.33, 0.50, 1024] # YOLOv8s summary: 129 layers, 11166560 parameters, 11166544 gradients, 28.8 GFLOPS
  m: [0.67, 0.75, 768] # YOLOv8m summary: 169 layers, 25902640 parameters, 25902624 gradients, 79.3 GFLOPS
  l: [1.00, 1.00, 512] # YOLOv8l summary: 209 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPS
  x: [1.00, 1.25, 512] 
```
Cada linha tem 3 números: Profundidade (depth), Largura (width), Máx. canais (max_channels).

Eles são fatores multiplicadores, não números absolutos de camadas ou filtros.

### Profundidade (depth)
Controla quantas vezes os blocos do backbone e head serão repetidos.

Exemplo: no backbone temos blocos como `[-1, 3, C2f, [128, True]]`.

Esse 3 significa "repete 3 vezes".

Se o fator de profundidade for 0.33, ele reduz: arredonda para baixo (fica 1 repetição em vez de 3).

Se for 1.00, mantém igual.
Se fosse 2.0, duplicaria (6 repetições).

Então 0.33 não quer dizer 33 camadas, e sim "um terço da profundidade original".

### Largura (width)
Controla quantos filtros (número de canais de saída) cada convolução terá.

Exemplo: `Conv, [128, 3, 2]` tem 128 filtros.

Se o fator for 0.25, vira ~32 filtros.

Se for 0.50, vira ~64 filtros.

Se for 1.25, vira ~160 filtros.

Isso define a riqueza das features que a rede pode aprender:

Poucos filtros = mais rápido, mas pode perder detalhes.

Muitos filtros = mais preciso, mas mais pesado

### Máx. canais (max_channels)
Define o limite máximo de filtros por camada.

Mesmo que o multiplicador peça mais filtros, a rede não ultrapassa esse teto.

Exemplo: no YOLOv8n, max_channels = 1024.

Então nenhuma camada vai ter mais que 1024 filtros.





Perfeito — vamos fechar os dois blocos principais do YAML: **Backbone** e **Head**. Vou te dar um “como ler” linha-a-linha, explicar os **módulos** (Conv, C2f, SPPF, Upsample, Concat, Detect) e o que **cada um impacta** em precisão, velocidade e memória.

---

# 3) Backbone - extrator de características

É a parte da rede que transforma a imagem em **mapas de características** cada vez mais abstratos (bordas → texturas → partes → objetos).

### Como ler cada linha do YAML

Formato:
`[from, repeats, module, args]`

* **from**: de onde vem a entrada dessa camada.

  * `-1` = a saída da **camada anterior**;
  * `[a, b]` = usa **duas** entradas (p.ex. para Concat).
* **repeats**: quantas vezes **repete** aquele módulo em sequência (é escalado pelo *depth*).
* **module**: qual bloco usar (`Conv`, `C2f`, `SPPF`…).
* **args**: parâmetros do módulo (ex.: `Conv, [64, 3, 2]` → 64 filtros, kernel 3×3, stride 2).

### O que significam P1/2, P2/4, P3/8…

* **P3/8**: mapa com **stride 8** (resolução ≈ 1/8 da imagem).
  Se a imagem é 640×640, P3 ≈ 80×80; P4/16 ≈ 40×40; P5/32 ≈ 20×20.
* Quanto **maior o stride**, **menor a resolução** e **maior o campo de visão** (melhor para objetos grandes).

### Principais módulos do backbone

* **Conv\[c\_out, k, s]**
  Convolução (normalmente Conv2d + BatchNorm + SiLU).

  * **Aumentar filtros (c\_out)** → mais capacidade, mais parâmetros e GFLOPs.
  * **Stride 2** → **downsample** (reduz a resolução do mapa).
* **C2f\[c, shortcut=True]**
  Bloco **leve-residual** do YOLOv8 (evolução do CSP). Internamente tem várias convs “finas” com *skip connections* para preservar informações.

  * **Impacto**: ótimo custo/benefício em **capacidade** sem explodir parâmetros.
  * **repeats** maior = mais profundo = mais representativo (e mais pesado).
* **SPPF\[c, k]** (*Spatial Pyramid Pooling Fast*)
  Extrai padrões em **várias escalas** com *poolings* encadeados.

  * **Impacto**: melhora detecção de objetos em tamanhos variados com custo baixo.

### Efeitos práticos (backbone)

* Mais **repeats** e mais **filtros** → ↑mAP potencial, ↑tempo/VRAM.
* Tirar blocos (por ex., remover P5 no backbone) → rede mais rápida/leve, mas pode perder objetos **grandes**.

---

# 4) Head - junta escalas e prevê as caixas

Recebe mapas de várias escalas do backbone, **mescla informações** (tipo FPN/PAN) e **prediz** caixas + classes.

### Fluxo típico do head

1. **Upsample**: aumenta a resolução de um mapa profundo (ex.: P5 → P4).
2. **Concat**: concatena com o mapa da mesma escala vindo do backbone (skip lateral).
3. **C2f**: refina a fusão (convs leves com atalhos).
4. (Opcional) **Conv stride 2**: desce de volta (P3→P4) para fazer outro ramo.
5. **Detect**: cabeça final que gera **logits de classe** e **box** (YOLOv8 é **anchor-free** e usa **DFL – Distribution Focal Loss** para coordenadas).

### Módulos do head

* **nn.Upsample\[..., scale=2]**: dobra a resolução (não tem pesos).
* **Concat**: empilha canais de dois mapas (não tem pesos).
* **C2f\[c]**: mesmo bloco do backbone, mas aqui serve para **refinar a fusão multi-escala**.
* **Conv\[c, 3, 2]**: *downsample* controlado (recria um caminho mais profundo no head).
* **Detect\[nc]**: produz as saídas:

  * **classes**: `nc` (no teu caso, 1)
  * **caixas**: coordenadas **anchor-free** via DFL
  * É aplicada em **cada escala** conectada (ex.: `[P3, P4, P5]`).

### Por que usar várias escalas (P3, P4, P5)?

* **P3 (stride 8)**: objetos **pequenos** (capacete, rosto distante).
* **P4 (stride 16)**: médios.
* **P5 (stride 32)**: grandes.
  Remover P5 deixa a rede mais leve e foca em pequenos/médios — bom se seu problema quase **não tem objetos grandes**.

### Efeitos práticos (head)

* **Mais escalas** na `Detect` → ↑recall em tamanhos variados, ↑custo.
* **Menos escalas** → mais simples/leve, mas pode falhar em certos tamanhos.
* **C2f no head** melhora a **qualidade da fusão** (costuma subir mAP com custo moderado).


# 5) Dicionário rápido (pra usar no slide)

* **Backbone**: extrai **features** (o “que” está na imagem).
* **Head**: **localiza e classifica** (o “onde” e “qual”).
* **Conv\[k, s]**: aprende filtros; `s=2` reduz resolução.
* **C2f**: bloco leve com atalhos → **capacidade** sem inflar muito parâmetros.
* **SPPF**: pirâmide de *pooling* rápida → **robusto a escalas**.
* **Upsample/Concat**: mistura **alto nível** (semântico) com **alto detalhe** (espacial).
* **Detect (anchor-free + DFL)**: camadas finais que soltam **caixas + classes** por escala.
* **P3/8, P4/16, P5/32**: níveis da pirâmide; **stride** relativo à imagem.
