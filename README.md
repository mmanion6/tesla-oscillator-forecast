# Tesla Price Forecast Model

An interactive dashboard that uses a **Damped Harmonic Oscillator** model to forecast $TSLA$ price action through 2040.

## How it Works
The model fits historical price data to a non-linear regression:
$$log_{10}(P) = a + b \ln(t) + A e^{-\gamma t} \sin(\omega t + \phi)$$

## Installation
1. Ensure you have a `requirements.txt` with: `streamlit`, `yfinance`, `pandas`, `scipy`, `matplotlib`.
2. Run locally: `streamlit run app.py`

## Troubleshooting
If the app fails to load, ensure your `requirements.txt` file has the `.txt` extension and that your internet connection allows access to Yahoo Finance.
