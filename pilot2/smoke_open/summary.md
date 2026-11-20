# Open-model smoke test (2026-11-17)

| Model / route | Pairs | Helpful: with / without | Placebo: with / without | Failed calls | Cut off at limit | Median latency (s) | Cost (USD) | Providers |
|:--|--:|:--|:--|--:|--:|--:|--:|:--|
| Claude Haiku 5.5 (study 2 calibration, Nov 16) | 10 | 5/5 / 0/5 | 0/5 / 0/5 | 0/20 | 0/20 | - | 0.0350 | - |
| Nemotron 3.5 Super VL via hf (nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16:cheapest) | 10 | 5/5 / 0/5 | 0/5 / 0/5 | 0/20 | 1/20 | 187.8 | 0.0289 | openrouter |
| Nemotron 3.5 Super VL via openrouter (nvidia/nemotron-3.5-super-vl) | 10 | 4/5 / 0/5 | 0/5 / 0/5 | 0/20 | 6/20 | 8.9 | 0.0805 | CoreWeave, DeepInfra, Venice |
| Nemotron 3.5 Super VL via nvidia (nvidia/nemotron-3.5-super-vl-120b-a12b) | 10 | 4/5 / 0/5 | 0/5 / 0/5 | 0/20 | 6/20 | 55.5 | 0.0914 | - |
