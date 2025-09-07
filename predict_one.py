# python predict_one.py  --weights runs_train/reduced_min_320_v2/weights/best.pt  --image test/images.jpg
import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    from PIL import Image, ImageDraw
    HAVE_CV2 = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=str, required=True, help="modelo .pt treinado (ex: best.pt)")
    ap.add_argument("--image", type=str, required=True, help="imagem de entrada")
    ap.add_argument("--outdir", type=str, default="predict_out", help="pasta de saída")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    vis_dir = outdir / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # carrega modelo
    model = YOLO(args.weights)
    print(f"[model] carregado: {args.weights}")

    # roda predição
    res = model.predict(
        source=args.image,
        imgsz=args.imgsz,
        conf=args.conf,
        verbose=False,
        save=False,
        device=0 if torch.cuda.is_available() else "cpu"
    )[0]

    # pega boxes
    if res.boxes is None or res.boxes.data.numel() == 0:
        print("[result] nenhuma detecção")
        return

    boxes = res.boxes.cpu()
    xyxy = boxes.xyxy.numpy()
    confs = boxes.conf.numpy()
    clss  = boxes.cls.numpy().astype(int)

    # salva txt/json
    txt_path = outdir / (Path(args.image).stem + "_pred.txt")
    with txt_path.open("w", encoding="utf-8") as f:
        for c, cf, (x1, y1, x2, y2) in zip(clss, confs, xyxy):
            f.write(f"{c} {cf:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")
    print(f"[save] predições salvas em {txt_path}")

    # desenha overlay
    out_img = vis_dir / (Path(args.image).stem + "_pred.jpg")
    if HAVE_CV2:
        img = cv2.imread(args.image)
        for (x1, y1, x2, y2), cf in zip(xyxy, confs):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, f"{cf:.2f}", (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(out_img), img)
    else:
        im = Image.open(args.image).convert("RGB")
        dr = ImageDraw.Draw(im)
        for (x1, y1, x2, y2), cf in zip(xyxy, confs):
            dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            dr.text((x1, y1 - 10), f"{cf:.2f}", fill=(255, 0, 0))
        im.save(out_img)

    print(f"[save] imagem com predição em {out_img}")


if __name__ == "__main__":
    main()
