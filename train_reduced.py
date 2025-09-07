import json
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

# ---------------- FIXOS ----------------
DATA_YAML   = "hardhat.yaml"
IMGSZ       = 320
EPOCHS      = 150
BATCH       = 32
WORKERS     = 2
PATIENCE    = 15
PROJECT     = "runs_train"
NAME        = "reduced_min_320_v3"
DEVICE      = "cuda:0" if torch.cuda.is_available() else "cpu"
MEASURE     = True
YAML_OUT    = Path("models/reduced_min_320_v3.yaml")
# --------------------------------------

def main():
    print(f"[env] Torch {torch.__version__} | CUDA build: {torch.version.cuda} | CUDA avail: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[env] GPU: {torch.cuda.get_device_name(0)}")

    print(f"[train] model={YAML_OUT} data={DATA_YAML} imgsz={IMGSZ} epochs={EPOCHS} batch={BATCH} device={DEVICE}")
    model = YOLO(str(YAML_OUT))
    results = model.train(
        data=DATA_YAML,
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=DEVICE,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        seed=42,
        patience=PATIENCE,
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    print(f"[done] Run dir: {run_dir}")
    print(f"[done] Best: {best.exists()} -> {best}")

    report = {}
    if MEASURE and best.exists():
        try:
            from thop import profile
            print("[measure] Calculating FLOPs/Params...")
            m = YOLO(str(best)).model.eval()
            x = torch.randn(1, 3, IMGSZ, IMGSZ, device=("cuda" if DEVICE.startswith("cuda") else "cpu"))
            if DEVICE.startswith("cuda") and torch.cuda.is_available():
                m = m.cuda()
            flops, params = profile(m, inputs=(x,), verbose=False)
            report["params_m"] = round(params/1e6, 4)
            report["flops_g"] = round(flops/1e9, 4)
            print(f"[measure] Params(M)={report['params_m']} FLOPs(G)={report['flops_g']}")
        except Exception as e:
            print(f"[measure] skipped ({e})")

    summary = {
        "stage": "reduced_arch",
        "model": str(YAML_OUT),
        "data": DATA_YAML,
        "imgsz": IMGSZ,
        "epochs": EPOCHS,
        "batch": BATCH,
        "device": DEVICE,
        "run_dir": str(run_dir),
        "best": str(best) if best.exists() else None,
        "last": str(last) if last.exists() else None,
        "report": report,
        "time": datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[summary] {run_dir/'summary.json'}")

if __name__ == "__main__":
    main()