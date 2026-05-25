# WaterQuality-DCGCN-Mamba

This repository contains the implementation of the proposed water quality prediction model only.

## Files

```text
config.py
train.py
test.py
data/adj_distance.npy
data/adj_hydrological.npy
utils/data_loader.py
utils/graph_utils.py
utils/metrics.py
utils/seed.py
models/__init__.py
models/init.py
models/dc_gcn.py
models/mamba_block.py
models/bilstm_attention.py
models/fusion.py
models/water_quality_model.py
```

## Run

```bash
python train.py --device cpu --epochs 1 --hidden-dim 16 --bilstm-hidden 8 --attn-heads 2 --seq-len 8 --batch-size 8
python test.py --device cpu
```
