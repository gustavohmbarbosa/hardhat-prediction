# Model Report

| Kind   | Run                             | Stage        | Model                            |   ImgSize |   Epochs |   Batch | File                                            |   Size (MB) |   mAP@0.5 |   mAP@0.5:0.95 |   Train Time (min) | Params (M)   | FLOPs (G)   | Latency (ms)   |
|:-------|:--------------------------------|:-------------|:---------------------------------|----------:|---------:|--------:|:------------------------------------------------|------------:|----------:|---------------:|-------------------:|:-------------|:------------|:---------------|
| run    | runs_train\reduced_min_320_tiny | reduced_arch | models\reduced_min_320_tiny.yaml |       320 |      200 |      32 | runs_train\reduced_min_320_tiny\weights\best.pt |       0.735 |    0.8397 |         0.5183 |              190.3 | 0.3029       | 0.2154      | 4.9640         |
| run    | runs_train\reduced_min_640_tiny | reduced_arch | models/reduced_min_640_tiny.yaml |       640 |      150 |      16 | runs_train\reduced_min_640_tiny\weights\best.pt |       0.766 |    0.8787 |         0.5617 |                0   | 0.3029       | 0.2154      | 4.5850         |
| run    | runs_train\reduced_min_320_v3   | reduced_arch | models\reduced_min_320_v3.yaml   |       320 |      150 |      32 | runs_train\reduced_min_320_v3\weights\best.pt   |       2.126 |    0.8651 |         0.5477 |              214.6 | 1.0207       | 0.4562      | 5.6130         |
| run    | runs_train\reduced_min_320_v2   | reduced_arch | models\reduced_min_320_v2.yaml   |       320 |      100 |      32 | runs_train\reduced_min_320_v2\weights\best.pt   |       3.109 |    0.861  |         0.5402 |              928.7 | 1.5387       | 0.5748      | 6.2210         |
| run    | runs_train\reduced_min_320_v1   | reduced_arch | models\reduced_min_320_v1.yaml   |       320 |       10 |      16 | runs_train\reduced_min_320_v1\weights\best.pt   |       3.603 |    0.7638 |         0.4493 |              997.9 | 1.7992       | 0.6469      | 7.0910         |
| run    | runs_train\baseline_v8n_320     |              | models\baseline_v8n_320.yaml     |       320 |       60 |      16 | runs_train\baseline_v8n_320\weights\best.pt     |       5.923 |    0.8879 |         0.5803 |             1128.2 | 3.0110       | 1.0243      | 7.2300         |
| run    | runs_train\baseline_v8n_640     | reduced_arch | models\baseline_v8n_640.yaml     |       640 |      150 |      16 | runs_train\baseline_v8n_640\weights\best.pt     |       5.971 |    0.9154 |         0.6157 |                0   | 3.0110       | 1.0243      | 6.8210         |
| run    | runs_train\reduced_min_640_v2   | reduced_arch | models\reduced_min_640_v2.yaml   |       640 |      150 |      16 |       3.154 |    0.9078 |         0.6026 |              340.8 | 1.5387       | 0.5748      | 10.3850        |
| run    | runs_train\reduced_min_640_v3   | reduced_arch | models\reduced_min_640_v3.yaml   |       640 |      150 |      16 | runs_train\reduced_min_640_v3\weights\best.pt   |       2.164 |    0.903  |         0.5963 |              384.7 |       1.0207 |      0.4562 |         10.911 |

## Gráficos

![scatter_params_vs_map50](reports/scatter_params_vs_map50.png)

![scatter_params_vs_map5095](reports/scatter_params_vs_map5095.png)

![scatter_latency_vs_map50](reports/scatter_latency_vs_map50.png)

![bar_params](reports/bar_params.png)

![bar_size](reports/bar_size.png)

![bar_latency](reports/bar_latency.png)

![bar_eff_map50_per_param](reports/bar_eff_map50_per_param.png)

![bar_eff_map50_per_mb](reports/bar_eff_map50_per_mb.png)

