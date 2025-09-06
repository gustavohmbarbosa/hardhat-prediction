# Model Report

| Run              | Stage        | Model            |   ImgSize |   Epochs |   Batch | File (best.pt)                              |   Size (MB) |   mAP@0.5 |   mAP@0.5:0.95 |   Params (M) |   FLOPs (G) |   Latency (ms) |
|:-----------------|:-------------|:-----------------|----------:|---------:|--------:|:--------------------------------------------|------------:|----------:|---------------:|-------------:|------------:|---------------:|
| reduced_min_320  | reduced_arch | yolov8n_min.yaml |       320 |       10 |      16 | runs_train\reduced_min_320\weights\best.pt  |       3.603 |    0.7638 |         0.4493 |       1.7992 |      0.6469 |          7.316 |
| baseline_v8n_320 |              | yolov8n.pt       |       320 |       60 |      16 | runs_train\baseline_v8n_320\weights\best.pt |       5.923 |    0.8879 |         0.5803 |       3.011  |      1.0243 |          7.13  |
