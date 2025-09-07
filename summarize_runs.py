#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumo de modelos YOLO (Ultralytics) com 'Train Time (min)':
- Coleta runs em runs_train/* e runs_prune/*
- Lê metrics do results.csv e summary.json (se existirem)
- Mede tamanho do best.pt, Params/FLOPs (thop) e latência sintética
- Inclui checkpoints "pruned*.pt" (pré-fine-tune) na raiz
- Adiciona coluna 'Train Time (min)' estimada via timestamps dos arquivos do run

Requisitos:
  pip install ultralytics pandas thop
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import torch
import pandas as pd
from ultralytics import YOLO

# ---------------- CONFIG ----------------
RUNS_DIRS = [Path("runs_train"), Path("runs_prune")]
ROOT      = Path(".")
IMGSZ     = 320
MEASURE_PARAMS_FLOPS = True
MEASURE_LATENCY      = True
LAT_WARMUP = 10
LAT_ITERS  = 100
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

def read_ultralytics_results_csv(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        return out
    try:
        df = pd.read_csv(results_csv)
        last = df.iloc[-1].to_dict()
        for k in ["metrics/mAP50(B)", "metrics/mAP50", "mAP50", "val/mAP50"]:
            if k in last:
                out["mAP@0.5"] = float(last[k]); break
        for k in ["metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95", "val/mAP50-95"]:
            if k in last:
                out["mAP@0.5:0.95"] = float(last[k]); break
        for k in ["metrics/precision(B)", "precision"]:
            if k in last: out["precision"] = float(last[k])
        for k in ["metrics/recall(B)", "recall"]:
            if k in last: out["recall"] = float(last[k])
    except Exception:
        try:
            with results_csv.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                if "metrics/mAP50(B)" in last: out["mAP@0.5"] = float(last["metrics/mAP50(B)"])
                if "metrics/mAP50-95(B)" in last: out["mAP@0.5:0.95"] = float(last["metrics/mAP50-95(B)"])
        except Exception:
            pass
    return out

def read_summary_json(run_dir: Path) -> Dict[str, Any]:
    sj = run_dir / "summary.json"
    if sj.exists():
        try:
            return json.loads(sj.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def measure_params_flops(model_path: Path, imgsz: int, device: str) -> Dict[str, float]:
    out = {}
    try:
        from thop import profile
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
        out["Params (M)"] = round(params/1e6, 4)
        out["FLOPs (G)"]  = round(flops/1e9, 4)
    except Exception:
        pass
    return out

def measure_latency(model_path: Path, imgsz: int, device: str,
                    warmup: int = 10, iters: int = 100) -> Optional[float]:
    try:
        m = YOLO(str(model_path)).model.eval()
        if device.startswith("cuda") and torch.cuda.is_available():
            m = m.cuda()
            x = torch.randn(1, 3, imgsz, imgsz, device="cuda")
            torch.cuda.synchronize()
        else:
            x = torch.randn(1, 3, imgsz, imgsz)
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
        return round((time.perf_counter() - t0) / iters * 1000, 3)
    except Exception:
        return None

def compute_run_duration_minutes(run_dir: Path) -> Optional[float]:
    """
    Estima a duração do treino pelo intervalo [primeiro_timestamp, último_timestamp]
    dos arquivos dentro do diretório do run (recursivo).
    """
    try:
        times: List[float] = []
        for p in run_dir.rglob("*"):
            if p.is_file():
                st = p.stat()
                times.append(st.st_mtime)
                # incluir st_ctime em Windows pode “empurrar” início para trás; preferimos mtime
        if not times:
            return None
        dt = max(times) - min(times)
        return round(dt / 60.0, 1)  # minutos, uma casa decimal
    except Exception:
        return None

def collect_runs(dirs: List[Path]) -> List[Path]:
    items = []
    for d in dirs:
        if not d.exists(): continue
        for p in sorted(d.glob("*")):
            if (p / "weights" / "best.pt").exists():
                items.append(p)
    return items

def collect_pruned_checkpoints(root: Path) -> List[Path]:
    paths = []
    for patt in ["pruned.pt", "pruned_*.pt", "pruned_full.pt"]:
        paths += list(root.glob(patt))
    return [p for p in paths if p.is_file()]

def main():
    print(f"[env] Torch {torch.__version__} | CUDA avail: {torch.cuda.is_available()} | Device={DEVICE}")

    rows = []

    # 1) Runs (train e fine-tune)
    runs = collect_runs(RUNS_DIRS)
    for run in runs:
        best = run / "weights" / "best.pt"
        size_mb = round(best.stat().st_size / (1024*1024), 3) if best.exists() else None
        res  = read_ultralytics_results_csv(run)
        summ = read_summary_json(run)
        dur_min = compute_run_duration_minutes(run)

        row = {
            "Kind": "run",
            "Run": str(run),
            "Stage": summ.get("stage", ""),
            "Model": summ.get("model", ""),
            "ImgSize": summ.get("imgsz", IMGSZ),
            "Epochs": summ.get("epochs", ""),
            "Batch":  summ.get("batch", ""),
            "File": str(best),
            "Size (MB)": size_mb,
            "mAP@0.5": res.get("mAP@0.5"),
            "mAP@0.5:0.95": res.get("mAP@0.5:0.95"),
            "Train Time (min)": dur_min,
        }

        if MEASURE_PARAMS_FLOPS and best.exists():
            row.update(measure_params_flops(best, IMGSZ, DEVICE))
        if MEASURE_LATENCY and best.exists():
            lat = measure_latency(best, IMGSZ, DEVICE, LAT_WARMUP, LAT_ITERS)
            if lat is not None:
                row["Latency (ms)"] = lat

        rows.append(row)

    # 2) Checkpoints podados pré-fine-tune
    pruned_list = collect_pruned_checkpoints(ROOT)
    for ckpt in pruned_list:
        try:
            size_mb = round(ckpt.stat().st_size / (1024*1024), 3)
        except Exception:
            size_mb = None

        row = {
            "Kind": "pruned_ckpt",
            "Run": "",
            "Stage": "pruned_pre_ft",
            "Model": "",
            "ImgSize": IMGSZ,
            "Epochs": "",
            "Batch": "",
            "File": str(ckpt),
            "Size (MB)": size_mb,
            "mAP@0.5": None,
            "mAP@0.5:0.95": None,
            "Train Time (min)": None,
        }

        if MEASURE_PARAMS_FLOPS:
            row.update(measure_params_flops(ckpt, IMGSZ, DEVICE))
        if MEASURE_LATENCY:
            lat = measure_latency(ckpt, IMGSZ, DEVICE, LAT_WARMUP, LAT_ITERS)
            if lat is not None:
                row["Latency (ms)"] = lat

        rows.append(row)

    if not rows:
        print("[warn] Nenhum modelo encontrado.")
        return

    df = pd.DataFrame(rows)
    if "Params (M)" in df.columns:
        df = df.sort_values(by=["Kind", "Params (M)", "Size (MB)"], ascending=[True, True, True], na_position="last")
    else:
        df = df.sort_values(by=["Kind", "Size (MB)"], ascending=[True, True], na_position="last")

    out_csv = Path("model_report.csv")
    out_md  = Path("model_report.md")
    df.to_csv(out_csv, index=False)

    def df_to_md(d: pd.DataFrame) -> str:
        d2 = d.copy()
        for c in d2.columns:
            if pd.api.types.is_float_dtype(d2[c]):
                d2[c] = d2[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        return d2.to_markdown(index=False)

    md = "# Model Report\n\n" + df_to_md(df) + "\n"
    out_md.write_text(md, encoding="utf-8")

    print(df_to_md(df))
    print(f"\nSalvo:\n- {out_csv}\n- {out_md}")

if __name__ == "__main__":
    main()
