# NASDAQ-100 Risk-Aware Prediction System

## Project Overview
This system is designed to provide probability-based insights for NASDAQ-100 stocks. It focuses on identifying potential price movements using a rule-based approach combined with machine learning (LightGBM).

## Important Disclaimer
> [!IMPORTANT]
> This software is for **educational and research purposes only**. 
> - It is **NOT** financial advice.
> - It does **NOT** guarantee any profit.
> - Financial trading involves significant risk of loss.
> - The developers are not responsible for any financial losses incurred.

## Key Safety Features
1. **Probability-Based**: The model provides likelihoods (e.g., UP: 0.63, DOWN: 0.37) rather than definitive "BUY" or "SELL" signals.
2. **NO_TRADE Signal**: If confidence is low (below 0.6), the system defaults to "NO_TRADE".
3. **Loss Awareness**: This system acknowledges that losses are a natural part of trading and focuses on risk management.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train.py`
3. Generate predictions: `python predict.py`
