# DCMB-Water-Model

## Overview

WaterQuality-DCGCN-Mamba is a spatiotemporal deep learning model for multi-waterbody water quality prediction. It is designed to model both spatial dependencies among monitoring stations and temporal variations in multivariate water quality sequences.

The model integrates a Dual-Branch Cross Graph Convolutional Network (DC-GCN), a Mamba-style temporal branch, an improved BiLSTM-Attention branch, and a residual fusion structure. It can be used for water quality forecasting tasks involving multiple monitoring sites and complex spatial-temporal relationships.

## Model Structure

The model contains four main parts:

1. **Dual-graph spatial modeling**  
   Two adjacency matrices are used to describe spatial relationships:
   - `adj_distance.npy`: spatial distance graph
   - `adj_hydrological.npy`: hydrological connectivity graph

2. **DC-GCN module**  
   The DC-GCN module extracts spatial features from the distance graph and hydrological graph through two graph convolution branches. Cross-branch interaction is used to fuse complementary spatial information.

3. **Temporal feature extraction**  
   The model uses two temporal branches:
   - Mamba-style branch for long-range temporal dependencies
   - BiLSTM-Attention branch for bidirectional temporal features and local fluctuations

4. **Feature fusion and prediction**  
   Spatial features, temporal features, and residual embedding features are fused and then passed into a regression layer for water quality prediction.

## Project Structure

```text
WaterQuality-DCGCN-Mamba/
├── config.py
├── train.py
├── test.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── adj_distance.npy
│   ├── adj_hydrological.npy
│   └── sample_data.csv
│
├── models/
│   ├── __init__.py
│   ├── init.py
│   ├── dc_gcn.py
│   ├── mamba_block.py
│   ├── bilstm_attention.py
│   ├── fusion.py
│   └── water_quality_model.py
│
└── utils/
    ├── __init__.py
    ├── data_loader.py
    ├── graph_utils.py
    ├── metrics.py
    └── seed.py
```

## Main Files

- `config.py`: stores model parameters and training settings.
- `train.py`: trains the WaterQuality-DCGCN-Mamba model.
- `test.py`: evaluates the trained model on the test set.
- `models/dc_gcn.py`: implements the dual-branch cross graph convolution module.
- `models/mamba_block.py`: implements the Mamba-style temporal modeling module.
- `models/bilstm_attention.py`: implements the BiLSTM-Attention module.
- `models/fusion.py`: implements the feature fusion structure.
- `models/water_quality_model.py`: combines all modules into the complete model.
- `utils/data_loader.py`: loads data and constructs time-series samples.
- `utils/graph_utils.py`: loads and normalizes adjacency matrices.
- `utils/metrics.py`: calculates MSE, MAE, and R2.

## Data Format

The input data should be a CSV file containing monitoring site ID, time, water quality variables, and the target variable.

A simple example is:

```text
date,site_id,water_temperature,pH,dissolved_oxygen,conductivity,turbidity,permanganate_index,ammonia_nitrogen,total_phosphorus,total_nitrogen
2021-06-01,1,25.3,7.8,6.5,450,12.1,3.2,0.18,0.04,2.16
```

The two adjacency matrices should be stored in the `data/` folder:

```text
data/adj_distance.npy
data/adj_hydrological.npy
```

Both matrices should have the shape:

```text
[num_nodes, num_nodes]
```

## Installation

Create a Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run:

```bash
python train.py
```
## Notes

Before running the code, please check the settings in `config.py`, especially:

- number of monitoring sites
- number of input variables
- input sequence length
- hidden dimension
- data path
- adjacency matrix paths

The full monitoring dataset may not be included because of data availability restrictions. Users can replace `sample_data.csv` with their own dataset following the same format.
