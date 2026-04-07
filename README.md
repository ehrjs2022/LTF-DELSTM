# LTF-DELSTM
Encoder-decoder-based forecasting framework for water quality prediction using environmental, meteorological, and hydrological data, designed to incorporate future known variables and account for the characteristics of individual data sources.

This repository provides the source code to reproduce the results presented in the paper:

> A Lightweight Multi-Horizon Forecasting Framework for Operational Water Quality Management in Reservoir Systems(On the Review), Environmental Modelling & Software, 2026

## Authors
Dogeon Lee, Jihoon Shin, YoonKyung Cha
Abstract
This code repository implements the training and evaluation pipelines for the proposed LTF-EDLSTM model, along with five comparison models: LSTM, GRU, BiLSTM, Transformer, and Temporal Fusion Transformer (TFT). The repository also includes code for performance analysis and visualization based on the final optimized version of each model.

## Quick Start
### Requirements
- Python 3.8+
- Required packages: pip install -r requirements.txt
### Running the Model
1. Run: Self_supervised_learning_based_convolutional_Autoencoder.ipynb
2. Results will be saved in Result/ folder
### Key Parameters to Modify
- hidden_dims
- batch_size
- num_layer
- learning_rate
- known_feature_dim
- loss_nam
## Data
Data will be made available on request
## Project Structure
Self_supervised_learning_based_convolutional_Autoencoder.ipynb # Main execution file Data/ # Data processing scripts Result/ # Output directory
