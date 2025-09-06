"""
Gera uma tabela comparando tamanho e eficácia dos modelos treinados (Ultralytics).
- Varre pastas em runs_train/*
- Lê summary.json (se existir) e results.csv do Ultralytics
- Mede tamanho (MB) do best.pt
- (opcional) Mede Params/FLOPs com thop e latência média (forward) no tamanho desejado
- Salva model_report.csv e model_report.md e também printa a tabela no terminal

Requisitos:
  pip install ultralytics pandas thop  # thop é só se for medir params/flops
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import pandas as pd
from ultralytics import YOLO

# ---------------- CONFIG ----------------
RUNS_ROOT = Path("runs_train")      # pasta onde seus treinos foram salvos
IMGSZ     = 320                     # resolução p/ FLOPs/latência
MEASURE_PARAMS_FLOPS = True         # requer thop
MEASURE_LATENCY      = True         # faz 100 forwards e tira média (sintético)
LATENCY_WARMUP       = 10
LATENCY_ITERS        = 100
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

def read_ultralytics_results_csv(run_dir: Path) -> Dict[str, Any]:
    """
    Lê o último registro do results.csv do Ultralytics e retorna métricas úteis.
    Campos variam por versão; tentamos pegar mAP50 e mAP50-95.
    """
    out: Dict[str, Any] = {}
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return out
    try:
        # pandas ajuda a pegar a última linha facilmente
        df = pd.read_csv(results_csv)
        last = df.iloc[-1].to_dict()
        # chaves comuns (podem variar por versão):
        # 'metrics/mAP50(B)' ou 'metrics/mAP50' / 'metrics/mAP50-95(B)' ou 'metrics/mAP50-95'
        for k in ["metrics/mAP50(B)", "metrics/mAP50", "mAP50", "val/mAP50"]:
            if k in last:
                out["mAP@0.5"] = float(last[k])
                break
        for k in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "val/mAP50-95"]:
            if k in last:
                out["mAP@0.5:0.95"] = float(last[k])
                break
        # opcional: precisão/recall
        for k in ["metrics/precision(B)", "metrics/recall(B)", "precision", "recall"]:
            if k in last:
                out[k.split("/")[-1]] = float(last[k])
        return out
    except Exception:
        # fallback manual
        try:
            with results_csv.open("r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    last = rows[-1]
                    if "metrics/mAP50(B)" in last:
                        out["mAP@0.5"] = float(last["metrics/mAP50(B)"])
                    if "metrics/mAP50-95(B)" in last:
                        out["mAP@0.5:0.95"] = float(last["metrics/mAP50-95(B)"])
        except Exception:
            pass
        return out

def measure_params_flops(model_path: Path, imgsz: int, device: str) -> Dict[str, float]:
    out = {}
    try:
        from thop import profile  # import aqui pra permitir rodar sem thop
    except Exception:
        return out
    try:
        m = YOLO(str(model_path)).model.eval()
        if device.startswith("cuda") and torch.cuda.is_available():
            m = m.cuda()
            x = torch.randn(1, 3, imgsz, imgsz, device="cuda")
        else:
            x = torch.randn(1, 3, imgsz, imgsz)
        flops, params = profile(m, inputs=(x,), verbose=False)
        out["Params (M)"] = round(params / 1e6, 4)
        out["FLOPs (G)"]  = round(flops  / 1e9, 4)
    except Exception:
        pass
    return out

def measure_latency(model_path: Path, imgsz: int, device: str,
                    warmup: int = 10, iters: int = 100) -> Optional[float]:
    """
    Mede latência média (ms) de forward() puro (sem dataloader) — sintético.
    Útil p/ comparar versões; números absolutos não representam pipeline completo.
    """
    try:
        m = YOLO(str(model_path)).model.eval()
        if device.startswith("cuda") and torch.cuda.is_available():
            m = m.cuda()
            x = torch.randn(1, 3, imgsz, imgsz, device="cuda")
            torch.cuda.synchronize()
        else:
            x = torch.randn(1, 3, imgsz, imgsz)

        # warmup
        for _ in range(warmup):
            with torch.no_grad():
                _ = m(x)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(iters):
            with torch.no_grad():
                _ = m(x)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
        return round(dt * 1000.0, 3)  # ms
    except Exception:
        return None

def read_summary_json(run_dir: Path) -> Dict[str, Any]:
    out = {}
    sj = run_dir / "summary.json"
    if sj.exists():
        try:
            out = json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            pass
    return out

def collect_runs(root: Path) -> List[Path]:
    runs = []
    if not root.exists():
        return runs
    for p in sorted(root.glob("*")):
        if p.is_dir():
            # só considera se tiver pesos
            if (p / "weights" / "best.pt").exists():
                runs.append(p)
    return runs

def main():
    print(f"[env] Torch {torch.__version__} | CUDA avail: {torch.cuda.is_available()} | Device={DEVICE}")
    runs = collect_runs(RUNS_ROOT)
    if not runs:
        print(f"[warn] Nenhum run encontrado em {RUNS_ROOT.resolve()}")
        return

    rows = []
    for run in runs:
        best = run / "weights" / "best.pt"
        size_mb = round(best.stat().st_size / (1024 * 1024), 3) if best.exists() else None
        res = read_ultralytics_results_csv(run)
        summ = read_summary_json(run)

        row = {
            "Run": run.name,
            "Stage": summ.get("stage") or "",
            "Model": summ.get("model") or "",
            "ImgSize": summ.get("imgsz") or IMGSZ,
            "Epochs": summ.get("epochs") or "",
            "Batch":  summ.get("batch")  or "",
            "File (best.pt)": str(best),
            "Size (MB)": size_mb,
            "mAP@0.5": res.get("mAP@0.5"),
            "mAP@0.5:0.95": res.get("mAP@0.5:0.95"),
        }

        if MEASURE_PARAMS_FLOPS and best.exists():
            pf = measure_params_flops(best, IMGSZ, DEVICE)
            row.update(pf)

        if MEASURE_LATENCY and best.exists():
            lat = measure_latency(best, IMGSZ, DEVICE, LATENCY_WARMUP, LATENCY_ITERS)
            if lat is not None:
                row["Latency (ms)"] = lat

        rows.append(row)

    # DataFrame e salvamento
    df = pd.DataFrame(rows)
    # ordenar por Params (se existir), senão por Size
    if "Params (M)" in df.columns:
        df = df.sort_values(by="Params (M)", ascending=True, na_position="last")
    elif "Size (MB)" in df.columns:
        df = df.sort_values(by="Size (MB)", ascending=True, na_position="last")

    csv_path = RUNS_ROOT / "model_report.csv"
    md_path  = RUNS_ROOT / "model_report.md"
    df.to_csv(csv_path, index=False)

    # Markdown bonito
    def df_to_markdown(d: pd.DataFrame) -> str:
        # limita casas decimais em floats
        d2 = d.copy()
        for c in d2.columns:
            if pd.api.types.is_float_dtype(d2[c]):
                d2[c] = d2[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        return d2.to_markdown(index=False)

    md = "# Model Report\n\n" + df_to_markdown(df) + "\n"
    md_path.write_text(md, encoding="utf-8")

    print(f"\n[done] Tabela gerada:")
    print(df_to_markdown(df))
    print(f"\nArquivos salvos:\n- {csv_path}\n- {md_path}")

if __name__ == "__main__":
    main()
