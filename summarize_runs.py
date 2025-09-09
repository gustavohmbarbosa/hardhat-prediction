"""
Resumo de modelos YOLO (Ultralytics) + gráficos e score composto.

- Coleta runs em runs_train/* e runs_prune/*
- Lê metrics do results.csv e summary.json (se existirem)
- Mede tamanho do best.pt, Params/FLOPs (thop) e latência sintética
- Inclui checkpoints "pruned*.pt" (pré-fine-tune) na raiz
- Adiciona 'Train Time (min)' por timestamps
- Gera gráficos comparativos (PNG) e calcula score composto

Requisitos:
  pip install ultralytics pandas thop matplotlib
"""

import csv
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
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

# Pesos do score composto (A > B > C)
SCORE_WA = 0.60  # mAP
SCORE_WB = 0.25  # Params (baixo é melhor)
SCORE_WC = 0.15  # Latência (baixo é melhor)

# Títulos curtos
MAX_LABEL_LEN = 60
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
    except Exception as e:
        print(f"[warn] THOP/profile falhou em '{model_path.name}': {e}")
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
    except Exception as e:
        print(f"[warn] Latency falhou em '{model_path.name}': {e}")
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


# ----------------- GRÁFICOS & SCORE -----------------

def shorten(s: str, n: int = MAX_LABEL_LEN) -> str:
    s = s.replace("\\", "/")
    return s if len(s) <= n else ("…" + s[-(n-1):])


def run_display_name(row: pd.Series) -> str:
    """
    Nome curto e consistente: apenas o nome do diretório do run.
    Ex.: runs_train/reduced_min_320_v3 -> 'reduced_min_320_v3'
    """
    run_path = str(row.get("Run", "")).replace("\\", "/")
    run_name = Path(run_path).name or Path(run_path).parent.name or run_path
    return run_name

def add_caption(fig, text: str):
    fig.text(0.01, 0.01, text, ha="left", va="bottom", fontsize=9, wrap=True)


def annotate_xy(ax, xs, ys):
    for x, y in zip(xs, ys):
        ax.annotate(f"({x:.2f}, {y:.2f})", (x, y),
                    textcoords="offset points", xytext=(6,6), fontsize=8)


def pareto_frontier(xs, ys, minimize_x=True, maximize_y=True):
    idx = np.argsort(xs if minimize_x else -xs)
    best = -np.inf if maximize_y else np.inf
    frontier = []
    for i in idx:
        y = ys[i]
        if maximize_y:
            if y > best:
                best = y
                frontier.append(i)
        else:
            if y < best:
                best = y
                frontier.append(i)
    return frontier


def safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def plot_scatter_params_map(df: pd.DataFrame, outdir: Path):
    d = df.dropna(subset=["Params (M)", "mAP@0.5", "Size (MB)"])
    if d.empty: return
    x = d["Params (M)"].to_numpy()
    y = d["mAP@0.5"].to_numpy()
    s = (d["Size (MB)"].to_numpy() * 30).clip(10, None)  # bolha proporcional ao tamanho
    labels = [run_display_name(r) for _, r in d.iterrows()]

    fig, ax = plt.subplots(figsize=(9,6.5))
    ax.scatter(x, y, s=s, alpha=0.75, edgecolors="none")
    annotate_xy(ax, x, y)

    # rótulo textual com o nome da série
    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), xytext=(6,-10), textcoords="offset points", fontsize=8)

    # fronteira de Pareto (min Params, max mAP)
    pareto_idx = pareto_frontier(x, y, minimize_x=True, maximize_y=True)
    if len(pareto_idx) >= 2:
        px, py = x[pareto_idx], y[pareto_idx]
        order = np.argsort(px)
        ax.plot(px[order], py[order], linestyle="--", linewidth=1)

    ax.set_xlabel("Params (M)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Params (M) vs mAP@0.5  -  bolha ∝ Size (MB)")
    add_caption(
        fig,
        "Descrição: este gráfico mostra o equilíbrio entre tamanho do modelo (eixo X) e acerto (eixo Y). "
        "O ideal é ficar mais à esquerda e mais alto: menos parâmetros com boa precisão. "
        "O tamanho da bolha indica o tamanho do arquivo no disco (MB). "
        "A linha tracejada liga os modelos que são bons “trocas” nesse equilíbrio (fronteira de Pareto)."
    )
    fig.tight_layout(rect=[0,0.05,1,1])
    fig.savefig(outdir / "scatter_params_map.png", dpi=180)
    plt.close(fig)


def plot_scatter_latency_map(df: pd.DataFrame, outdir: Path):
    d = df.dropna(subset=["Latency (ms)", "mAP@0.5"])
    if d.empty: return
    x = d["Latency (ms)"].to_numpy()
    y = d["mAP@0.5"].to_numpy()
    labels = [run_display_name(r) for _, r in d.iterrows()]

    fig, ax = plt.subplots(figsize=(9,6.5))
    ax.scatter(x, y, alpha=0.75, edgecolors="none")
    annotate_xy(ax, x, y)

    for xi, yi, lab in zip(x, y, labels):
        ax.annotate(lab, (xi, yi), xytext=(6,-10), textcoords="offset points", fontsize=8)

    pareto_idx = pareto_frontier(x, y, minimize_x=True, maximize_y=True)
    if len(pareto_idx) >= 2:
        px, py = x[pareto_idx], y[pareto_idx]
        order = np.argsort(px)
        ax.plot(px[order], py[order], linestyle="--", linewidth=1)

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("mAP@0.5")
    ax.set_title("Latency (ms) vs mAP@0.5")
    add_caption(
        fig,
        "Descrição: comparamos rapidez e acerto. Quanto mais à esquerda (menor latência) e mais alto (maior mAP), melhor. "
        "A linha tracejada destaca os modelos com melhores compromissos entre velocidade e precisão."
    )
    fig.tight_layout(rect=[0,0.05,1,1])
    fig.savefig(outdir / "scatter_latency_map.png", dpi=180)
    plt.close(fig)


def plot_bars(df: pd.DataFrame, col: str, title: str, fname: str, outdir: Path):
    d = df.dropna(subset=[col])
    if d.empty: return
    # ordenar crescente (ou decrescente para métricas de acurácia/eficiência/score)
    descending_cols = {"mAP@0.5", "mAP@0.5:0.95", "Efficiency (mAP/Params)",
                       "Efficiency (mAP/MB)", "Score"}
    ascending = False if col in descending_cols else True
    d = d.sort_values(col, ascending=ascending)
    names = [run_display_name(r) for _, r in d.iterrows()]
    vals = d[col].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, len(names)*0.7), 6.2))
    bars = ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(col)

    # rótulos numéricos (duas casas)
    for b, v in zip(bars, vals):
        ax.annotate(f"{v:.2f}", xy=(b.get_x() + b.get_width()/2, b.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    # descrição rápida
    hint = {
        "Params (M)": "Descrição: mede a complexidade do modelo. Menos parâmetros tendem a custar menos e rodar melhor em edge.",
        "Size (MB)": "Descrição: espaço ocupado no disco/flash. Útil para atualização OTA e dispositivos com memória limitada.",
        "Latency (ms)": "Descrição: tempo médio para processar uma imagem. Menor é melhor para cenários de tempo real.",
        "mAP@0.5": "Descrição: taxa de acerto com critério mais permissivo (IoU 0.5). Serve como noção rápida de precisão.",
        "mAP@0.5:0.95": "Descrição: métrica mais exigente (IoU de 0.5 a 0.95). Retrata melhor a qualidade geral do detector.",
        "Efficiency (mAP/Params)": "Descrição: quanta precisão obtemos por milhão de parâmetros - eficiência do modelo.",
        "Efficiency (mAP/MB)": "Descrição: precisão por megabyte de arquivo - bom para escolher modelos compactos.",
        "Score": "Descrição: Score composto (0–100) que pondera mAP (60%), número de parâmetros (25%) e latência (15%), priorizando precisão mas premiando modelos menores e mais rápidos.",
    }.get(col, "")
    if hint:
        add_caption(fig, hint)

    fig.tight_layout(rect=[0,0.06,1,1])
    fig.savefig(outdir / fname, dpi=180)
    plt.close(fig)


def _norm01(x: pd.Series) -> pd.Series:
    x = pd.to_numeric(x, errors="coerce")
    xmin, xmax = x.min(skipna=True), x.max(skipna=True)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
        return pd.Series(np.nan, index=x.index)
    return (x - xmin) / (xmax - xmin)

def compute_efficiencies_and_score(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    p = safe_series(d, "Params (M)")
    mb = safe_series(d, "Size (MB)")
    m = safe_series(d, "mAP@0.5")
    lat = safe_series(d, "Latency (ms)")

    # Eficiências auxiliares
    d["Efficiency (mAP/Params)"] = np.where((p > 0) & m.notna(), m / p, np.nan)
    d["Efficiency (mAP/MB)"]     = np.where((mb > 0) & m.notna(), m / mb, np.nan)

    # Utilidades normalizadas [0,1]
    # A: mAP alto -> melhor
    uA = _norm01(m)

    # B: menos parâmetros melhor (usamos log para suavizar escala)
    logp = pd.Series(np.where(p > 0, np.log(p), np.nan), index=d.index)
    uB = _norm01(-logp)  # inverte: menor logp => maior utilidade

    # C: menor latência melhor
    uC = _norm01(-lat)

    # Score final (0-100)
    score = 100.0 * (SCORE_WA * uA + SCORE_WB * uB + SCORE_WC * uC)
    d["Score"] = score

    # Extras úteis para depurar
    d["uA_mAP"]     = uA
    d["uB_params"]  = uB
    d["uC_latency"] = uC

    return d



# ----------------- MAIN -----------------

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

    # Ordenação para a tabela
    if "Params (M)" in df.columns:
        df = df.sort_values(by=["Kind", "Params (M)", "Size (MB)"], ascending=[True, True, True], na_position="last")
    else:
        df = df.sort_values(by=["Kind", "Size (MB)"], ascending=[True, True], na_position="last")

    reports_dir = ROOT / "reports"
    plots_dir = reports_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_csv = reports_dir / "model_report.csv"
    out_md  = reports_dir / "model_report.md"
    df.to_csv(out_csv, index=False)

    # Markdown
    def df_to_md(d: pd.DataFrame) -> str:
        d2 = d.copy()
        for c in d2.columns:
            if pd.api.types.is_float_dtype(d2[c]):
                d2[c] = d2[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        return d2.to_markdown(index=False)

    md = "# Model Report\n\n" + df_to_md(df) + "\n"
    out_md.write_text(md, encoding="utf-8")

    # ---- Eficiências + Score ----
    df_scored = compute_efficiencies_and_score(df)
    df_scored.to_csv(reports_dir / "model_report_scored.csv", index=False)

    # ---- Gráficos ----
    try:
        plot_scatter_params_map(df_scored, plots_dir)
    except Exception as e:
        print(f"[warn] plot_scatter_params_map: {e}")

    try:
        plot_scatter_latency_map(df_scored, plots_dir)
    except Exception as e:
        print(f"[warn] plot_scatter_latency_map: {e}")

    for col, title, fname in [
        ("Params (M)",        "Parâmetros (M)",            "bars_params.png"),
        ("Size (MB)",         "Tamanho do arquivo (MB)",   "bars_size.png"),
        ("Latency (ms)",      "Latência (ms)",             "bars_latency.png"),
        ("mAP@0.5",           "mAP@0.5",                   "bars_map50.png"),
        ("mAP@0.5:0.95",      "mAP@0.5:0.95",              "bars_map5095.png"),
        ("Efficiency (mAP/Params)", "Eficiência mAP por Params", "bars_eff_map_per_params.png"),
        ("Efficiency (mAP/MB)",     "Eficiência mAP por MB",     "bars_eff_map_per_mb.png"),
        ("Score", f"Score composto (A={SCORE_WA}, B={SCORE_WB}, C={SCORE_WC})", "bars_score.png"),
    ]:
        try:
            if col in df_scored.columns:
                plot_bars(df_scored[df_scored["Kind"]=="run"], col, title, fname, plots_dir)
        except Exception as e:
            print(f"[warn] plot_bars {col}: {e}")

    print(df_to_md(df))
    print(f"\nSalvo:\n- {out_csv}\n- {out_md}\n- {reports_dir/'model_report_scored.csv'}\n- PNGs em {plots_dir}")


if __name__ == "__main__":
    main()
