# LTF-DELSTM
Encoder-decoder-based forecasting framework for water quality prediction using environmental, meteorological, and hydrological data, designed to incorporate future known variables and account for the characteristics of individual data sources.

This repository provides the source code to reproduce the results presented in the paper:

> A Lightweight Multi-Horizon Forecasting Framework for Operational Water Quality Management in Reservoir Systems(On the Review), Environmental Modelling & Software, 2026

## Authors
Dogeon Lee, Jihoon Shin, Taeseung Park, Jaegwan Park, Jonggyu Jung, YoonKyung Cha
Abstract
This code repository implements the training and evaluation pipelines for the proposed LTF-EDLSTM model, along with five comparison models: LSTM, GRU, BiLSTM, Transformer, and Temporal Fusion Transformer (TFT). The repository also includes code for performance analysis and visualization based on the final optimized version of each model.

## Quick Start
### Requirements
- Python 3.11+
- Required packages: pip install -r requirements.txt
### Running the Model
1. Navigate to the folder corresponding to the target prediction variable.

2. To train the proposed model, run:
   - `Model_development_LTF-EDLSTM_Turbidity.ipynb`

3. To train the comparison models, run:
   - `Model_development_Singlemodel_Turbidity.ipynb`
   - `Model_development_Transformer_Turbidity.ipynb`
   - `Model_development_TFT_Turbidity.ipynb`

4. For other prediction targets, run the corresponding notebooks in each target-specific folder, where `Turbidity` in the filename is replaced with `WTemp` or `pH`.

5. After model training, the trained model pickle files for each hyperparameter setting are saved in the `Model/` directory within each target-specific folder.

6. To evaluate the final selected models and reproduce the performance and visualization results used in the paper, run:
   - `Performance_Check_Turbidity.ipynb`

7. For other prediction targets, use the corresponding performance check notebook with `Turbidity` replaced by `WTemp` or `pH`.
### Key Parameters to Modify
```text
- hidden_dims
- batch_size
- num_layer
- learning_rate
- known_feature_dim
- loss_nam
## Data
Data will be made available on request
## Project Structure
```text
Project/
├── Turbidity/
│   ├── Model_development_LTF-EDLSTM_Turbidity.ipynb
│   ├── Model_development_Singlemodel_Turbidity.ipynb
│   ├── Model_development_Transformer_Turbidity.ipynb
│   ├── Model_development_TFT_Turbidity.ipynb
│   ├── Performance_Check_Turbidity.ipynb
│   └── Model/
├── WTemp/
│   ├── Model_development_LTF-EDLSTM_WTemp.ipynb
│   ├── Model_development_Singlemodel_WTemp.ipynb
│   ├── Model_development_Transformer_WTemp.ipynb
│   ├── Model_development_TFT_WTemp.ipynb
│   ├── Performance_Check_WTemp.ipynb
│   └── Model/
├── pH/
│   ├── Model_development_LTF-EDLSTM_pH.ipynb
│   ├── Model_development_Singlemodel_pH.ipynb
│   ├── Model_development_Transformer_pH.ipynb
│   ├── Model_development_TFT_pH.ipynb
│   ├── Performance_Check_pH.ipynb
│   └── Model/
