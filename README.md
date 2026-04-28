# Wavelet Jump Classification & Probabilistic Price Forecasting

## Overview

This repository implements a wavelet-based framework for analyzing financial time series, focusing on jump detection, classification, and probabilistic forecasting.

The implementation is based on the research paper:

Aubrun, C., Morel, R., Benzaquen, M., Bouchaud, J.-P.  
"Identifying new classes of financial price jumps with wavelets"  
Proceedings of the National Academy of Sciences (PNAS)  
[https://www.pnas.org/doi/10.1073/pnas.2409156121](https://www.pnas.org/doi/10.1073/pnas.2409156121)

---

## Mathematical Foundations

### 1. Continuous Wavelet Transform (CWT)

The wavelet transform decomposes a time series into time-frequency space:

$$
W_x(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t - b}{a}\right) dt
$$

Where:
- $x(t)$ = price time series  
- $\psi$ = mother wavelet  
- $a$ = scale (frequency)  
- $b$ = translation (time)  

---

### 2. Discrete Wavelet Transform (DWT)

In practice, we use dyadic scales:

$$
W_{j,k} = \sum_{n} x[n] \psi_{j,k}[n]
$$

This allows decomposition into:

- Approximation coefficients (low frequency)
- Detail coefficients (high frequency)

---

### 3. Jump Detection

Jumps are identified as large deviations in wavelet coefficients:

$$
|W_{j,k}| > \lambda_j
$$

Where $\lambda_j$ is a scale-dependent threshold.

---

### 4. Probabilistic Forecasting

Future price distribution:

$$
P(X_{t+h} | \mathcal{F}_t)
$$

Estimated using features extracted from wavelet coefficients and jump classifications.

---

## Wavelet Decomposition (Conceptual Diagram)

```
Price Series
     │
     ▼
Wavelet Transform
     │
     ├── Low Frequency (Trend)
     │
     └── High Frequency (Details)
             │
             ├── Noise
             └── Jumps (Spikes)
```

---

## Features

- Multi-scale wavelet decomposition  
- Jump detection using thresholding  
- Unsupervised jump classification  
- Probabilistic price forecasting  
- Modular pipeline for experimentation  

---

## Project Structure

Wavelet-Jump-Classification-Probabilistic-Price-Forecasting/

- data/  
- notebooks/  
- models/  
- wavelet/  
- utils/  
- results/  
- requirements.txt  
- README.md  

---

## Installation

```bash
git clone https://github.com/RatulShashank/Wavelet-Jump-Classification-Probabilistic-Price-Forecasting.git
cd Wavelet-Jump-Classification-Probabilistic-Price-Forecasting

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

---

## Usage

```bash
jupyter notebook
```

or

```bash
python main.py
```

---

## Applications

- High-frequency trading research  
- Volatility modeling  
- Risk management  
- Crypto and forex strategy development  

---

## Credit & Acknowledgment

This repository is an independent implementation of the methodology described in:

Aubrun, C., Morel, R., Benzaquen, M., Bouchaud, J.-P.  
Identifying new classes of financial price jumps with wavelets  
PNAS

All core ideas belong to the original authors.

---

## Disclaimer

This project is for research and educational purposes only. It does not constitute financial advice.

---

## Author

Ratul Shashank
