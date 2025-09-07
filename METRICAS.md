# 📊 O que significam mAP\@0.5 e mAP\@0.5:0.95?

### 1. AP (Average Precision)

* Em detecção de objetos, você tem **precisão** e **revocação (recall)**:

  * **Precisão** = TP / (TP + FP) → dos que previ, quantos estão certos?
  * **Recall** = TP / (TP + FN) → dos objetos reais, quantos detectei?
* Se você varia o limiar de confiança do modelo (score mínimo para considerar uma detecção), obtém diferentes pontos de (precisão, recall).
* A **curva precisão–recall (PR curve)** é gerada.
* **AP** = área sob essa curva (quanto mais perto de 1.0, melhor).

---

### 2. IoU (Intersection over Union)

* Para dizer se uma previsão acerta um objeto, usamos **IoU**:

  $$
  IoU = \frac{Área\_Interseção}{Área\_União}
  $$
* Se IoU ≥ um limiar (threshold), contamos como **TP**; senão, **FP**.

---

### 3. mAP (mean Average Precision)

* Em datasets com várias classes, calcula-se o AP de cada classe e faz-se a média: **mAP**.
* No seu caso (só capacete), mAP ≈ AP da classe 0.

---

### 4. Diferença entre @0.5 e @0.5:0.95

* **mAP\@0.5**: só considera IoU = 0.5 (50%).

  * Uma detecção é “certa” se sobrepõe pelo menos metade do objeto.
  * Métrica **mais permissiva**.
* **mAP\@0.5:0.95**: é a métrica oficial do COCO.

  * Calcula o AP em **10 thresholds**: IoU = 0.5, 0.55, 0.6, …, 0.95.
  * Faz a média de todos.
  * É **mais rígida**: exige não só encontrar o objeto, mas também localizá-lo com alta precisão.
  * Normalmente o valor é bem mais baixo que @0.5.

---

# 📐 Como são calculados na prática

1. Para cada IoU threshold:

   * Ordena as detecções por confiança.
   * Marca TP/FP com base no IoU.
   * Constrói a curva P–R.
   * Calcula a área sob a curva (AP).
2. Para mAP:

   * Faz a média dos APs de todas as classes (no seu caso, só uma).
3. Para mAP\@0.5:0.95:

   * Repete o processo em todos os thresholds 0.5–0.95 (passo 0.05).
   * Faz a média final.

---

# 🔎 Como interpretar

* **mAP\@0.5** alto (ex.: 0.88) → o modelo **encontra bem** os capacetes, mesmo que a caixa não seja perfeita.
* **mAP\@0.5:0.95** mais baixo (ex.: 0.58) → o modelo tem dificuldade em localizar com exatidão (as caixas podem estar frouxas ou deslocadas).
* **Gap grande** entre os dois → indica que o modelo detecta mas não posiciona as caixas com precisão.
* **Gap pequeno** → indica detecções bem localizadas.

---

📌 exemplo da sua tabela:

```
baseline_v8n_320 → mAP@0.5 = 0.8879, mAP@0.5:0.95 = 0.5803
```

👉 O modelo encontra quase 89% dos capacetes de forma aceitável, mas só \~58% ficam com bounding box bem justa (alta qualidade de localização).
