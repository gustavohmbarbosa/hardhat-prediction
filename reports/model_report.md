# Model Report

| Kind   | Run                             | Stage        | Model                            |   ImgSize |   Epochs |   Batch |   Size (MB) |   mAP@0.5 |   mAP@0.5:0.95 |   Train Time (min) |   Params (M) |   FLOPs (G) |   Latency (ms) |
|:-------|:--------------------------------|:-------------|:---------------------------------|----------:|---------:|--------:|------------:|----------:|---------------:|-------------------:|-------------:|------------:|---------------:|
| run    | runs_train\reduced_min_320_tiny | reduced_arch | models\reduced_min_320_tiny.yaml |       320 |      200 |      32 |       0.735 |    0.8397 |         0.5183 |              190.3 |       0.3029 |      0.2154 |          5.58  |
| run    | runs_train\reduced_min_320_v3   | reduced_arch | models\reduced_min_320_v3.yaml   |       320 |      150 |      32 |       2.126 |    0.8651 |         0.5477 |              214.6 |       1.0207 |      0.4562 |          9.283 |
| run    | runs_train\reduced_min_320_v2   | reduced_arch | models\reduced_min_320_v2.yaml   |       320 |      100 |      32 |       3.109 |    0.861  |         0.5402 |              928.7 |       1.5387 |      0.5748 |         10.267 |
| run    | runs_train\reduced_min_320_v1   | reduced_arch | models\reduced_min_320_v1.yaml   |       320 |       10 |      16 |       3.603 |    0.7638 |         0.4493 |              997.9 |       1.7992 |      0.6469 |          9.141 |
| run    | runs_train\baseline_v8n_320     |              | models\baseline_v8n_320.yaml     |       320 |       60 |      16 |       5.923 |    0.8879 |         0.5803 |             1128.2 |       3.011  |      1.0243 |          7.516 |
