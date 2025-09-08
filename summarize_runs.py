#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resumo de modelos YOLO (Ultralytics) com 'Train Time (min)' + GRÁFICOS:
- Coleta runs em runs_train/* e runs_prune/*
- Lê metrics do results.csv e summary.json (se existirem)
- Mede tamanho do best.pt, Params/FLOPs (thop) e latência sintética
- Inclui checkpoints "pruned*.pt" (pré-fine-tune) na raiz
- Adiciona coluna 'Train Time (min)' estimada via timestamps dos arquivos do run
- Gera gráficos comparativos (PNG) e referencia no model_report.md

Requisitos:
  pip install ultralytics pandas thop matplotlib
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
    try:
        times: List[float] = []
        for p in run_dir.rglob("*"):
            if p.is_file():
                st = p.stat()
                times.append(st.st_mtime)
        if not times:
            return None
        dt = max(times) - min(times)
        return round(dt / 60.0, 1)
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

# ---------------- PLOTS (versão com Pareto + labels) ----------------
def make_plots(df: pd.DataFrame, out_dir: Path) -> List[str]:
    """
    Gera PNGs comparativos e retorna a lista de caminhos (strings) para inserir no .md.
    Inclui:
      - Scatter Params x mAP@0.5 (com Pareto e labels) -> scatter_params_vs_map50.png
      - Scatter Params x mAP@0.5:0.95 (com Pareto e labels) -> scatter_params_vs_map5095.png
      - Scatter Latência x mAP@0.5 (com Pareto e labels) -> scatter_latency_vs_map50.png
      - Barras: Params por modelo -> bar_params.png
      - Barras: Tamanho por modelo -> bar_size.png
      - Barras: Latência por modelo -> bar_latency.png
      - Barras: Eficiência (mAP@0.5 / Params) -> bar_eff_map50_per_param.png
      - Barras: Eficiência (mAP@0.5 / MB) -> bar_eff_map50_per_mb.png
    """
    saved = []
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"[plots] matplotlib ausente ou falhou ({e}); pulando gráficos.")
        return saved

    out_dir.mkdir(parents=True, exist_ok=True)

    # Apenas runs (com métricas) para comparar desempenho
    dfr = df[df["Kind"] == "run"].copy()

    def label_from_run(v: str) -> str:
        try:
            return Path(v).name
        except Exception:
            return str(v)

    dfr["Label"] = dfr["Run"].map(label_from_run)

    def fmt2(x):
        return f"{float(x):.2f}"

    # ---------------- helpers ----------------
    def annotate_points(ax, x, y, labels, extra_vals=None):
        """
        Desenha rótulos com Nome + valores (2 casas). extra_vals pode ser dict {"xname": arr, "yname": arr}
        """
        for i in range(len(x)):
            text = labels[i]
            if extra_vals is not None:
                parts = []
                if "x" in extra_vals:
                    parts.append(fmt2(extra_vals["x"][i]))
                if "y" in extra_vals:
                    parts.append(fmt2(extra_vals["y"][i]))
                if parts:
                    text += f" ({', '.join(parts)})"
            ax.annotate(text, (x[i], y[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    def pareto_indices_cost_minimize_benefit_maximize(cost, benefit):
        """
        Retorna índices da fronteira de Pareto quando queremos MINIMIZAR cost e MAXIMIZAR benefit.
        Estratégia: ordenar por cost asc, manter apenas pontos que sejam 'recorde' em benefit.
        """
        order = np.argsort(cost)
        best_benefit = -np.inf
        keep = []
        for idx in order:
            b = benefit[idx]
            if b > best_benefit + 1e-12:
                keep.append(idx)
                best_benefit = b
        return np.array(keep, dtype=int)

    # ------------- Scatter: Params × mAP@0.5 (com Pareto) -------------
    if {"Params (M)", "mAP@0.5"}.issubset(dfr.columns):
        d0 = dfr.dropna(subset=["Params (M)", "mAP@0.5"]).copy()
        if len(d0) >= 2:
            x = d0["Params (M)"].to_numpy(dtype=float)
            y = d0["mAP@0.5"].to_numpy(dtype=float)
            sz = d0["Size (MB)"].fillna(d0["Size (MB)"].median()).clip(0.1, None).to_numpy(dtype=float)
            labels = d0["Label"].tolist()
            plt.figure(figsize=(9, 5))
            ax = plt.gca()
            sc = ax.scatter(x, y, s=sz * 12.0, alpha=0.85)
            annotate_points(ax, x, y, labels, extra_vals={"x": x, "y": y})
            # Pareto
            pidx = pareto_indices_cost_minimize_benefit_maximize(x, y)
            px, py = x[pidx], y[pidx]
            order = np.argsort(px)
            ax.plot(px[order], py[order], marker="o", linewidth=1.5)
            ax.set_xlabel("Parâmetros (M)")
            ax.set_ylabel("mAP@0.5")
            ax.set_title("Params (M) vs mAP@0.5 (bolha ~ tamanho do arquivo)")
            ax.grid(True, alpha=0.3)
            p = out_dir / "scatter_params_vs_map50.png"
            plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
            saved.append(str(p))

    # ------------- Scatter: Params × mAP@0.5:0.95 (com Pareto) -------------
    if {"Params (M)", "mAP@0.5:0.95"}.issubset(dfr.columns):
        d0 = dfr.dropna(subset=["Params (M)", "mAP@0.5:0.95"]).copy()
        if len(d0) >= 2:
            x = d0["Params (M)"].to_numpy(dtype=float)
            y = d0["mAP@0.5:0.95"].to_numpy(dtype=float)
            sz = d0["Size (MB)"].fillna(d0["Size (MB)"].median()).clip(0.1, None).to_numpy(dtype=float)
            labels = d0["Label"].tolist()
            plt.figure(figsize=(9, 5))
            ax = plt.gca()
            ax.scatter(x, y, s=sz * 12.0, alpha=0.85)
            annotate_points(ax, x, y, labels, extra_vals={"x": x, "y": y})
            pidx = pareto_indices_cost_minimize_benefit_maximize(x, y)
            px, py = x[pidx], y[pidx]
            order = np.argsort(px)
            ax.plot(px[order], py[order], marker="o", linewidth=1.5)
            ax.set_xlabel("Parâmetros (M)")
            ax.set_ylabel("mAP@0.5:0.95")
            ax.set_title("Params (M) vs mAP@0.5:0.95 (bolha ~ tamanho do arquivo)")
            ax.grid(True, alpha=0.3)
            p = out_dir / "scatter_params_vs_map5095.png"
            plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
            saved.append(str(p))

    # ------------- Scatter: Latência × mAP@0.5 (com Pareto) -------------
    if {"Latency (ms)", "mAP@0.5"}.issubset(dfr.columns):
        d0 = dfr.dropna(subset=["Latency (ms)", "mAP@0.5"]).copy()
        if len(d0) >= 2:
            x = d0["Latency (ms)"].to_numpy(dtype=float)
            y = d0["mAP@0.5"].to_numpy(dtype=float)
            sz = d0["Size (MB)"].fillna(d0["Size (MB)"].median()).clip(0.1, None).to_numpy(dtype=float)
            labels = d0["Label"].tolist()
            plt.figure(figsize=(9, 5))
            ax = plt.gca()
            ax.scatter(x, y, s=sz * 12.0, alpha=0.85)
            annotate_points(ax, x, y, labels, extra_vals={"x": x, "y": y})
            pidx = pareto_indices_cost_minimize_benefit_maximize(x, y)
            px, py = x[pidx], y[pidx]
            order = np.argsort(px)  # menor latência à esquerda
            ax.plot(px[order], py[order], marker="o", linewidth=1.5)
            ax.set_xlabel("Latência (ms)")
            ax.set_ylabel("mAP@0.5")
            ax.set_title("Latência vs mAP@0.5 (bolha ~ tamanho do arquivo)")
            ax.grid(True, alpha=0.3)
            p = out_dir / "scatter_latency_vs_map50.png"
            plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
            saved.append(str(p))

    # ------------- Barras: Params -------------
    if "Params (M)" in dfr.columns and dfr["Params (M)"].notna().any():
        d0 = dfr.dropna(subset=["Params (M)"]).copy().sort_values("Params (M)")
        plt.figure(figsize=(10, 5))
        plt.bar(d0["Label"], d0["Params (M)"])
        for i, v in enumerate(d0["Params (M)"]):
            plt.text(i, v, fmt2(v), ha="center", va="bottom", fontsize=8)
        plt.ylabel("Parâmetros (M)")
        plt.title("Parâmetros por modelo (runs)")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)
        p = out_dir / "bar_params.png"
        plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
        saved.append(str(p))

    # ------------- Barras: Tamanho -------------
    if "Size (MB)" in dfr.columns and dfr["Size (MB)"].notna().any():
        d0 = dfr.dropna(subset=["Size (MB)"]).copy().sort_values("Size (MB)")
        plt.figure(figsize=(10, 5))
        plt.bar(d0["Label"], d0["Size (MB)"])
        for i, v in enumerate(d0["Size (MB)"]):
            plt.text(i, v, fmt2(v), ha="center", va="bottom", fontsize=8)
        plt.ylabel("Tamanho do arquivo (MB)")
        plt.title("Tamanho do checkpoint (runs)")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)
        p = out_dir / "bar_size.png"
        plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
        saved.append(str(p))

    # ------------- Barras: Latência -------------
    if "Latency (ms)" in dfr.columns and dfr["Latency (ms)"].notna().any():
        d0 = dfr.dropna(subset=["Latency (ms)"]).copy().sort_values("Latency (ms)")
        plt.figure(figsize=(10, 5))
        plt.bar(d0["Label"], d0["Latency (ms)"])
        for i, v in enumerate(d0["Latency (ms)"]):
            plt.text(i, v, fmt2(v), ha="center", va="bottom", fontsize=8)
        plt.ylabel("Latência (ms) @1x")
        plt.title("Latência por modelo (runs)")
        plt.xticks(rotation=30, ha="right")
        plt.grid(axis="y", alpha=0.3)
        p = out_dir / "bar_latency.png"
        plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
        saved.append(str(p))

    # ------------- Barras: Eficiência (mAP@0.5 por M de params) -------------
    if {"mAP@0.5", "Params (M)"}.issubset(dfr.columns):
        d0 = dfr.dropna(subset=["mAP@0.5", "Params (M)"]).copy()
        if not d0.empty:
            d0["eff_map50_per_param"] = d0["mAP@0.5"] / d0["Params (M)"].replace(0, np.nan)
            d0 = d0.dropna(subset=["eff_map50_per_param"]).sort_values("eff_map50_per_param", ascending=False)
            if not d0.empty:
                plt.figure(figsize=(10, 5))
                plt.bar(d0["Label"], d0["eff_map50_per_param"])
                for i, v in enumerate(d0["eff_map50_per_param"]):
                    plt.text(i, v, fmt2(v), ha="center", va="bottom", fontsize=8)
                plt.ylabel("mAP@0.5 por M de parâmetros")
                plt.title("Eficiência de acurácia por parâmetros")
                plt.xticks(rotation=30, ha="right")
                plt.grid(axis="y", alpha=0.3)
                p = out_dir / "bar_eff_map50_per_param.png"
                plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
                saved.append(str(p))

    # ------------- Barras: Eficiência (mAP@0.5 por MB) -------------
    if {"mAP@0.5", "Size (MB)"}.issubset(dfr.columns):
        d0 = dfr.dropna(subset=["mAP@0.5", "Size (MB)"]).copy()
        if not d0.empty:
            d0["eff_map50_per_mb"] = d0["mAP@0.5"] / d0["Size (MB)"].replace(0, np.nan)
            d0 = d0.dropna(subset=["eff_map50_per_mb"]).sort_values("eff_map50_per_mb", ascending=False)
            if not d0.empty:
                plt.figure(figsize=(10, 5))
                plt.bar(d0["Label"], d0["eff_map50_per_mb"])
                for i, v in enumerate(d0["eff_map50_per_mb"]):
                    plt.text(i, v, fmt2(v), ha="center", va="bottom", fontsize=8)
                plt.ylabel("mAP@0.5 por MB")
                plt.title("Eficiência de acurácia por tamanho de arquivo")
                plt.xticks(rotation=30, ha="right")
                plt.grid(axis="y", alpha=0.3)
                p = out_dir / "bar_eff_map50_per_mb.png"
                plt.tight_layout(); plt.savefig(p, dpi=160); plt.close()
                saved.append(str(p))

    return saved
# ---------------- FIM PLOTS ----------------

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

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "model_report.csv"
    out_md  = reports_dir / "model_report.md"
    df.to_csv(out_csv, index=False)

    # GERA GRÁFICOS
    pngs = make_plots(df, reports_dir)

    # Markdown tabelado + imagens
    def df_to_md(d: pd.DataFrame) -> str:
        d2 = d.copy()
        for c in d2.columns:
            if pd.api.types.is_float_dtype(d2[c]):
                d2[c] = d2[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        return d2.to_markdown(index=False)

    md = "# Model Report\n\n" + df_to_md(df) + "\n\n"
    if pngs:
        md += "## Gráficos\n\n"
        for p in pngs:
            rel = Path(p).as_posix()
            md += f"![{Path(p).stem}]({rel})\n\n"

    out_md.write_text(md, encoding="utf-8")

    print(df_to_md(df))
    print(f"\nSalvo:\n- {out_csv}\n- {out_md}")
    if pngs:
        print("Gráficos:")
        for p in pngs:
            print(f"- {p}")

if __name__ == "__main__":
    main()
