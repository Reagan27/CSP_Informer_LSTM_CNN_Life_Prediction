

---

# Deep Learning Models for Predictive Maintenance of Bearings

This repository contains code for implementing deep learning models with attention mechanisms for predictive maintenance of bearings in industrial machinery.

## Overview

The code provided here demonstrates the implementation of two deep learning architectures:

1. Informer Model with Correlated Single-Parameter Adjustment (CSPA) Mechanism
2. CNN-LSTM-Attention Model

These models are designed to predict the remaining useful life (RUL) of bearings based on sensor data collected from mechanical systems.

## Installation

To run the code, you need the following dependencies:

- Python 3.x
- PyTorch
- NumPy
- pandas
- Matplotlib

You can install the required packages using pip:

```bash
pip install torch numpy pandas matplotlib
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/Reagan27/CSP_Informer_LSTM_CNN_Life_Prediction.git
cd CSP_Informer_LSTM_CNN_Life_Prediction
```

2. Run the main.py file:

```bash
python main.py
```

## Data

The sensor data used for training and testing the models should be placed in the "training" directory. Ensure that the following files are available:

- Bearing1_1+1_2.csv
- Bearing2_1+2_2.csv
- Bearing3_1+3_2.csv
- acc_02764.csv

## Model Development

The code includes the following steps:

1. Data preprocessing
2. Model definition
3. Training
4. Evaluation
5. Visualization of results

## References

- Brownlee, J. (2018). Deep Learning for Time Series Forecasting. Machine Learning Mastery.
- Cho, K., van Merrienboer, B., Bahdanau, D., & Bengio, Y. (2014). On the properties of neural machine translation: Encoder-decoder approaches. arXiv preprint arXiv:1409.1259.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning (Vol. 1). MIT Press.
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation, 9(8), 1735-1780.
- Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
- Lipton, Z. C., Kale, D. C., Elkan, C., & Wetzel, R. (2016). Learning to diagnose with LSTM recurrent neural networks. arXiv preprint arXiv:1511.03677.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).

