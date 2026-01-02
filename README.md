# Machine-Learned Volatility Forecasting & Time-Varying Monte Carlo Simulation

This project develops a machine-learned volatility model and integrates it into a time-varying Monte Carlo framework for option pricing. The objective is to move beyond the constant-volatility assumption of classical geometric Brownian motion (GBM) by learning volatility dynamics directly from historical market data and propagating them through simulation.

---

## Project Overview

Traditional option pricing models typically assume constant volatility, which fails to capture well-known empirical properties of financial markets such as volatility clustering and regime persistence. In this project, we:

- Train a neural network to forecast forward realized volatility using only past information  
- Evaluate predictive performance against standard rolling-window volatility benchmarks  
- Embed the trained model into a Monte Carlo simulation where volatility is re-estimated at every time step  
- Compare simulated price dynamics and option prices against constant-volatility baselines  

---

## Data & Feature Engineering

Data:
- Daily equity price data (adjusted close)

Target:
- 30-day forward realized volatility (annualized), computed as:

  sqrt(252 * mean(r_{t+1:t+30}^2))

Input Features at time t:
- 60 lagged daily log returns  
- 3 Rolling historical volatility estimates (20-, 60-, and 126-day windows)  
- 1 Log price level  

All features are constructed using only information available up to time t, ensuring no look-ahead bias.

---

## Model Architecture & Training

- Model: Multilayer Perceptron (MLP)
- Architecture:
  - 2 hidden layers
  - 64 units per layer
  - ReLU activations
  - Linear output layer (predicting volatility directly)
- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam
- Data split (chronological):
  - 70% training
  - 15% validation
  - 15% test

Model performance is evaluated out-of-sample and compared to a 126-day rolling volatility benchmark.

---

## Time-Varying Monte Carlo Simulation

The trained volatility model is embedded into a Monte Carlo GBM framework where volatility is re-predicted at each simulation step based on the evolving price history.

Key properties:
- Volatility is path-dependent and endogenous
- Each simulated path maintains its own volatility dynamics
- Simulations are initialized using recent historical returns (warm-up) to ensure continuity with observed market conditions

Simulation details:
- Number of paths: 1,000
- Time steps: 30 (daily)
- Risk-neutral drift: r - q
- Payoffs discounted using exp(-rT)
- Monte Carlo standard errors reported alongside option price estimates

Results are benchmarked against a constant-volatility GBM simulation.

---

## Results & Findings

- The MLP achieves lower out-of-sample MSE than a 126-day rolling volatility estimator
- Learned volatility forecasts exhibit regime persistence and volatility clustering
- Time-varying Monte Carlo paths display heteroskedastic dispersion absent from constant-volatility GBM
- Option prices are produced with Monte Carlo standard errors, enabling uncertainty quantification
- Simulated paths align more closely with observed historical price behavior than constant-volatility baselines

---

## Repository Structure

.
├── volatility_mlp.ipynb
├── src/
│   ├── features.py
│   ├── model.py
│   └── mc_simulation.py
├── .gitignore
└── README.md

---

## Installation & Usage

Install dependencies:

pip install -r requirements.txt

Run the notebook to:
1. Build features and targets  
2. Train and evaluate the volatility model  
3. Generate Monte Carlo simulations and visualizations  

---

## Limitations & Future Work

- The model forecasts statistical (realized) volatility rather than implied volatility
- Option prices are not calibrated to market prices
- Possible extensions include:
  - Sequence models (LSTM, Transformer)
  - Volatility-of-volatility dynamics
  - Multi-asset extensions
  - Calibration to implied volatility surfaces

---

## Takeaway

This project demonstrates how machine-learned volatility models can be meaningfully combined with Monte Carlo methods to generate realistic, path-dependent price dynamics while maintaining arbitrage-consistent pricing. It provides a flexible foundation for further research at the intersection of statistical learning and quantitative finance.
