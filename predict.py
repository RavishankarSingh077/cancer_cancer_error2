import numpy as np
import pandas as pd
import pickle
from data import load_data
from features import add_features

def live_predict(symbol="AAPL"):
    # 1. Load trained model
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Model file not found! Please run train.py first.")
        return

    # 2. Get latest data
    df = load_data(symbol, period="5d")
    df = add_features(df)

    # 3. Take the last available row
    feature_cols = ["return", "volume_change", "rsi", "ema", "vwap"]
    X = df[feature_cols].iloc[-1:]

    # 4. Get probabilities for each class [-1, 0, 1]
    # Classes are ordered: -1 (DOWN), 0 (NO_TRADE), 1 (UP)
    probs = model.predict_proba(X)[0]
    classes = model.classes_
    
    prob_map = dict(zip(classes, probs))
    
    up_prob = prob_map.get(1, 0)
    down_prob = prob_map.get(-1, 0)
    no_trade_prob = prob_map.get(0, 0)

    print("-" * 30)
    print(f"Prediction result for {symbol}:")
    print(f"UP probability:    {up_prob:.4f}")
    print(f"DOWN probability:  {down_prob:.4f}")
    print(f"NO TRADE prob:     {no_trade_prob:.4f}")
    print("-" * 30)

    # 5. Risk-aware Decision Making
    # Threshold 0.6 as discussed
    CONFIDENCE_THRESHOLD = 0.6

    if up_prob >= CONFIDENCE_THRESHOLD:
        decision = "BUY (UP)"
    elif down_prob >= CONFIDENCE_THRESHOLD:
        decision = "SELL (DOWN)"
    else:
        decision = "NO TRADE (Low Confidence or Neutral)"

    print(f"DECISION: {decision}")
    print("-" * 30)
    print("Disclaimer: This is for educational research only. Not financial advice.")

if __name__ == "__main__":
    live_predict("AAPL")
