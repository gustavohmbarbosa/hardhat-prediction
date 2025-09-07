# python test_model.py  --weights runs_train\reduced_min_320_v2\weights\best.pt  --subset 000009,000147,000164,000194            

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import yaml
import torch

try:
    import cv2
    HAVE_CV2 = True
except Exception:
    from PIL import Image, ImageDraw
    HAVE_CV2 = False

from ultralytics import YOLO

# ------------------------ utils ------------------------

def yolo_txt_to_boxes_single_class(txt_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """
    Lê YOLO .txt (cls cx cy w h) e retorna boxes xyxy (pixels) para 1 classe.
    Ignora a coluna de classe (assume '0').
    Saída: (N, 4) [x1,y1,x2,y2]
    """
    if not txt_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    lines = []
    try:
        raw = txt_path.read_text(encoding="utf-8").strip().splitlines()
    except UnicodeDecodeError:
        raw = txt_path.read_text(encoding="latin-1").strip().splitlines()
    for s in raw:
        if not s.strip():
            continue
        parts = s.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        cx *= img_w; cy *= img_h; w *= img_w; h *= img_h
        x1 = cx - w/2; y1 = cy - h/2
        x2 = cx + w/2; y2 = cy + h/2
        lines.append([x1, y1, x2, y2])
    if not lines:
        return np.zeros((0, 4), dtype=np.float32)
    a = np.array(lines, dtype=np.float32)
    a[:, 0] = np.clip(a[:, 0], 0, img_w - 1)
    a[:, 1] = np.clip(a[:, 1], 0, img_h - 1)
    a[:, 2] = np.clip(a[:, 2], 0, img_w - 1)
    a[:, 3] = np.clip(a[:, 3], 0, img_h - 1)
    return a

def iou_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """IoU entre dois conjuntos de boxes xyxy."""
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    areaA = (A[:, 2] - A[:, 0]).clip(min=0) * (A[:, 3] - A[:, 1]).clip(min=0)
    areaB = (B[:, 2] - B[:, 0]).clip(min=0) * (B[:, 3] - B[:, 1]).clip(min=0)
    iou = np.zeros((A.shape[0], B.shape[0]), dtype=np.float32)
    for i in range(A.shape[0]):
        xx1 = np.maximum(A[i, 0], B[:, 0])
        yy1 = np.maximum(A[i, 1], B[:, 1])
        xx2 = np.minimum(A[i, 2], B[:, 2])
        yy2 = np.minimum(A[i, 3], B[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        union = areaA[i] + areaB - inter + 1e-9
        iou[i, :] = inter / union
    return iou

def greedy_match_single_class(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray, iou_thr: float = 0.5):
    """Matching guloso 1-classe. Retorna (matches, fp_idx, fn_idx)."""
    Np, Ng = pred_xyxy.shape[0], gt_xyxy.shape[0]
    if Np == 0 and Ng == 0:
        return [], [], []
    used_g = set()
    matches = []
    iou_mat = iou_matrix(pred_xyxy, gt_xyxy)
    for i_p in range(Np):  # preds já ordenadas por confiança
        best_iou, best_j = -1.0, -1
        for j in range(Ng):
            if j in used_g:
                continue
            iou = iou_mat[i_p, j]
            if iou >= iou_thr and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            used_g.add(best_j)
            matches.append((i_p, best_j, float(best_iou)))
    matched_p = {i for (i, _, _) in matches}
    matched_g = {j for (_, j, _) in matches}
    fp_idx = [i for i in range(Np) if i not in matched_p]
    fn_idx = [j for j in range(Ng) if j not in matched_g]
    return matches, fp_idx, fn_idx

def draw_vis(img_path: Path, gt_xyxy: np.ndarray, pred_xyxy: np.ndarray, out_path: Path):
    """Desenha GT (verde) e Pred (vermelho) e salva."""
    if HAVE_CV2:
        img = cv2.imread(str(img_path))
        if img is None:
            return
        for x1, y1, x2, y2 in gt_xyxy:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        for x1, y1, x2, y2 in pred_xyxy:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
    else:
        im = Image.open(str(img_path)).convert("RGB")
        dr = ImageDraw.Draw(im)
        for x1, y1, x2, y2 in gt_xyxy:
            dr.rectangle([x1, y1, x2, y2], outline=(0, 220, 0), width=3)
        for x1, y1, x2, y2 in pred_xyxy:
            dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        im.save(str(out_path))

def load_yaml_paths(data_yaml: Path) -> Tuple[Path, Path]:
    """Resolve images/test e labels/test a partir do YAML Ultralytics."""
    d = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(d.get("path", "."))
    test_rel = d.get("test", "images/test")
    img_test = (root / test_rel).resolve()
    parts = list(Path(test_rel).parts)
    try:
        idx = parts.index("images")
        labels_rel = Path(*parts[:idx], "labels", *parts[idx+1:])
    except ValueError:
        labels_rel = Path("labels/test")
    lbl_test = (root / labels_rel).resolve()
    return img_test, lbl_test

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--data", type=str, default="hardhat.yaml")
    ap.add_argument("--outdir", type=str, default="eval_out")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--match_iou", type=float, default=0.5)
    ap.add_argument("--subset", type=str, default=None, help="stems separados por vírgula, ex: img1,img2,img3")
    args = ap.parse_args()    

    outdir = Path(args.outdir); vis_dir = outdir / "vis"
    outdir.mkdir(parents=True, exist_ok=True); vis_dir.mkdir(parents=True, exist_ok=True)

    img_test_dir, lbl_test_dir = load_yaml_paths(Path(args.data))
    print(f"[data] test images: {img_test_dir}")
    print(f"[data] test labels: {lbl_test_dir}")

    subset = {s.strip() for s in args.subset.split(",")} if args.subset else None
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    img_paths = sorted(p for p in img_test_dir.glob("*") if p.suffix.lower() in exts)
    if subset:
        img_paths = [p for p in img_paths if p.stem in subset]
    if not img_paths:
        raise SystemExit("Nenhuma imagem encontrada no conjunto selecionado.")

    model = YOLO(args.weights)
    print(f"[model] loaded: {args.weights}")

    rows: List[Dict] = []
    n_tp = n_fp = n_fn = 0
    ious_all: List[float] = []

    for i, img_path in enumerate(img_paths, 1):
        # dimensões + GT
        if HAVE_CV2:
            img = cv2.imread(str(img_path)); 
            if img is None: 
                print(f"[warn] falha ao abrir {img_path.name}, pulando."); 
                continue
            h, w = img.shape[:2]
        else:
            from PIL import Image
            pil = Image.open(str(img_path)).convert("RGB")
            w, h = pil.size

        gt_path = lbl_test_dir / f"{img_path.stem}.txt"
        gt_xyxy = yolo_txt_to_boxes_single_class(gt_path, w, h)

        # predição
        res = model.predict(
            source=str(img_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            verbose=False,
            save=False,
            device=0 if torch.cuda.is_available() else "cpu"
        )[0]

        if res.boxes is None or res.boxes.data.numel() == 0:
            pred_xyxy = np.zeros((0, 4), dtype=np.float32)
        else:
            b = res.boxes.cpu()
            pred_xyxy = b.xyxy.numpy().astype(np.float32)
            # ordena por confiança desc
            order = np.argsort(-b.conf.numpy())
            pred_xyxy = pred_xyxy[order]

        # matching (1 classe)
        matches, fp_idx, fn_idx = greedy_match_single_class(pred_xyxy, gt_xyxy, iou_thr=args.match_iou)
        tp, fp, fn = len(matches), len(fp_idx), len(fn_idx)
        n_tp += tp; n_fp += fp; n_fn += fn
        if matches:
            ious_all.extend([m[2] for m in matches])

        # métricas por imagem
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec  = tp / (tp + fn) if (tp + fn) else 0.0
        f1   = (2*prec*rec/(prec+rec)) if (prec+rec) else 0.0
        miou = float(np.mean([m[2] for m in matches])) if matches else 0.0

        rows.append({
            "image": img_path.name,
            "n_gt": int(gt_xyxy.shape[0]),
            "n_pred": int(pred_xyxy.shape[0]),
            "TP": tp, "FP": fp, "FN": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            f"mean_IoU@{args.match_iou}": round(miou, 4),
        })

        # visualização
        out_img = vis_dir / f"{img_path.stem}_cmp.jpg"
        draw_vis(img_path, gt_xyxy, pred_xyxy, out_img)

        if i % 20 == 0 or i == len(img_paths):
            print(f"[progress] {i}/{len(img_paths)} imagens")

    # CSV por imagem
    df = pd.DataFrame(rows)
    per_image_csv = outdir / "per_image.csv"
    df.to_csv(per_image_csv, index=False)
    print(f"[save] {per_image_csv}")

    # resumo
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0
    recall    = n_tp / (n_tp + n_fn) if (n_tp + n_fn) else 0.0
    f1        = (2*precision*recall/(precision+recall)) if (precision+recall) else 0.0
    miou_glob = float(np.mean(ious_all)) if ious_all else 0.0

    summary = {
        "weights": str(args.weights),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "num_images": len(img_paths),
        "TP": n_tp, "FP": n_fp, "FN": n_fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        f"mean_IoU@{args.match_iou}": round(miou_glob, 4),
        "outdir": str(outdir.resolve())
    }
    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\n[done] vis:   {vis_dir.resolve()}")
    print(f"[done] csv:   {per_image_csv.resolve()}")
    print(f"[done] summary: {summary_path.resolve()}")

if __name__ == "__main__":
    main()
    