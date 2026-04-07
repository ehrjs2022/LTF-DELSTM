# LTF-DELSTM
Encoder-decoder-based forecasting framework for water quality prediction using environmental, meteorological, and hydrological data, designed to incorporate future known variables and account for the characteristics of individual data sources.

This repository provides the source code to reproduce the results presented in the paper:

> A Lightweight Multi-Horizon Forecasting Framework for Operational Water Quality Management in Reservoir Systems(On the Review), Environmental Modelling & Software, 2026

## Authors
Bongseok Jeong, Jihoon Shin, YoonKyung Cha
Abstract
This code repository implements training and evaluation algorithms for three Self-Supervised Learning Deep Learning (SSL-DL) models along with three baseline comparison models. The repository also includes visualization code for generating all figures presented in the paper.

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
