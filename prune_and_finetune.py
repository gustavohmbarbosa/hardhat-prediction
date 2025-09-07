#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poda estrutural de canais + fine-tune para YOLOv8 (Ultralytics), com redução REAL de parâmetros.

- Pruna Conv2d (groups==1) globalmente via L1 (MagnitudeImportance).
- Evita depthwise e a cabeça Detect.
- Mede Params/FLOPs antes e depois (se 'thop' estiver instalado).
- Salva o MÓDULO prunado (estrutura reduzida): pruned_full.pt
- Exporta ONNX do modelo prunado: pruned.onnx
- Faz fine-tune usando YOLO(base_podado) diretamente (sem YAML).

Uso:
  pip install torch-pruning ultralytics thop
  python prune_and_finetune.py \
      --weights runs_train/reduced_min_320/weights/best.pt \
      --data hardhat.yaml --imgsz 320 --epochs 30 --batch 16 \
      --sparsity 0.35 --device cuda:0
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO

# -------------------- utils --------------------
def is_prunable_conv(m: nn.Module) -> bool:
    return isinstance(m, nn.Conv2d) and m.groups == 1 and m.out_channels > 8

def measure_params_flops(module: nn.Module, imgsz: int, device: str) -> Optional[tuple[float, float]]:
    try:
        from thop import profile
    except Exception:
        return None
    module = module.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        module = module.cuda()
        x = torch.randn(1, 3, imgsz, imgsz, device="cuda")
        torch.cuda.synchronize()
    else:
        x = torch.randn(1, 3, imgsz, imgsz)
    with torch.no_grad():
        flops, params = profile(module, inputs=(x,), verbose=False)
    return round(params/1e6, 4), round(flops/1e9, 4)

def export_onnx(module: nn.Module, out: Path, imgsz: int, device: str):
    module = module.eval()
    if device.startswith("cuda") and torch.cuda.is_available():
        module = module.cuda()
        x = torch.randn(1, 3, imgsz, imgsz, device="cuda")
    else:
        x = torch.randn(1, 3, imgsz, imgsz)
    torch.onnx.export(
        module, x, out.as_posix(),
        input_names=["images"], output_names=["preds"],
        opset_version=12, do_constant_folding=True,
        dynamic_axes={"images": {0: "batch"}, "preds": {0: "batch"}}
    )

# -------------------- args --------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="checkpoint treinado (best.pt) para podar")
    ap.add_argument("--data", type=str, required=True, help="dataset yaml para fine-tune (mesmo do treino)")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--epochs", type=int, default=30, help="épocas de fine-tune pós-poda (0 = não treina)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--sparsity", type=float, default=0.35, help="proporção global de canais a remover (0.0–0.9)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--project", type=str, default="runs_prune")
    ap.add_argument("--name", type=str, default="pruned_ft")
    return ap.parse_args()

# -------------------- main --------------------
def main():
    args = parse_args()
    use_cuda = args.device.startswith("cuda") and torch.cuda.is_available()
    device = args.device if use_cuda else "cpu"
    print(f"[env] torch {torch.__version__} | CUDA avail: {torch.cuda.is_available()} | device={device}")

    # 1) carregar o modelo treinado (como módulo nn.Module do Ultralytics)
    print(f"[load] {args.weights}")
    base = YOLO(args.weights).model.eval()

    # 2) medir ANTES (opcional)
    before = measure_params_flops(base, args.imgsz, device)
    if before:
        print(f"[before] Params(M)={before[0]} | FLOPs(G)={before[1]}")

    # 3) construir entrada dummy e configurar pruner
    example = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)
    base.to(device)
    importance = tp.importance.MagnitudeImportance(p=1)

    prunable_modules = [m for m in base.modules() if is_prunable_conv(m)]
    ignored_layers = []
    # evitar Detect head
    for m in base.modules():
        if m.__class__.__name__.lower().startswith("detect"):
            ignored_layers.append(m)

    print(f"[prune] pruning_ratio={args.sparsity} | prunable_conv={len(prunable_modules)}")
    pruner = tp.pruner.MetaPruner(
        base,
        example_inputs=example,
        importance=importance,
        pruning_ratio=args.sparsity,   # <- estrutural
        global_pruning=True,
        ignored_layers=ignored_layers
    )

    # 4) aplicar poda (modifica o GRAFO: menos canais)
    pruner.step()

    # 5) pegar o MÓDULO podado e trazer pra CPU
    pruned = pruner.model.to("cpu").eval()

    # 6) medir DEPOIS (opcional)
    after = measure_params_flops(pruned, args.imgsz, "cpu")
    if after:
        print(f"[after]  Params(M)={after[0]} | FLOPs(G)={after[1]}")

    # 7) salvar o módulo completo e exportar ONNX
    pruned_full = Path("pruned_full.pt")
    torch.save(pruned, pruned_full)   # salva o módulo inteiro (estrutura reduzida)
    print(f"[save] pruned nn.Module -> {pruned_full.resolve()}")

    onnx_path = Path("pruned.onnx")
    try:
        export_onnx(pruned, onnx_path, args.imgsz, "cpu")
        print(f"[export] ONNX -> {onnx_path.resolve()}")
    except Exception as e:
        print(f"[export] ONNX falhou: {e}")

    # 8) fine-tune diretamente com o MÓDULO podado (sem YAML)
    if args.epochs > 0:
        print("[finetune] iniciando fine-tune do modelo podado...")
        # O construtor do Ultralytics aceita nn.Module
        y = YOLO(pruned)
        results = y.train(
            data=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            device=device,
            workers=args.workers,
            project=args.project,
            name=args.name,
            seed=42,
            patience=10
        )
        print(f"[done] fine-tune dir: {results.save_dir}")

if __name__ == "__main__":
    main()
