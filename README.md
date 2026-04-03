# LTF-DELSTM
SSL-DL
Code accompanying the paper "Development of a Self-Supervised Deep Learning Framework for Chlorophyll-a Retrieval in Data-Scarce Inland Waters" submitted to Environmental Modelling & Software.

## Authors
Bongseok Jeong, Jihoon Shin, YoonKyung Cha
Abstract
This code repository implements training and evaluation algorithms for three Self-Supervised Learning Deep Learning (SSL-DL) models along with three baseline comparison models. The repository also includes visualization code for generating all figures presented in the paper.

## Quick Start
### Requirements
Python 3.8+
Required packages: pip install -r requirements.txt
### Running the Model
Run: Self_supervised_learning_based_convolutional_Autoencoder.ipynb
Results will be saved in Result/ folder
### Key Parameters to Modify
learning_rate
batch_size
epochs
'p_weight'
'init_out_ch1'
## Project Structure
Self_supervised_learning_based_convolutional_Autoencoder.ipynb # Main execution file Data/ # Data processing scripts Result/ # Output directory

### Satellite Imagery Dataset
Due to large file sizes (>1GB), the complete satellite imagery dataset is available at:
