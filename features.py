import ta
import pandas as pd

def add_features(df):
    """
    Adds technical indicators to the dataframe.
    """
    df = df.copy()

    # Basic returns
    df["return"] = df["Close"].pct_change()
    df["volume_change"] = df["Volume"].pct_change()

    # Technical Indicators
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    df["ema_9"] = ta.trend.EMAIndicator(df["Close"], window=9).ema_indicator()
    df["ema_20"] = ta.trend.EMAIndicator(df["Close"], window=20).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(df["Close"], window=50).ema_indicator()
    
    # EMA Crossovers
    df["ema_cross_9_20"] = df["ema_9"] - df["ema_20"]
    df["ema_cross_20_50"] = df["ema_20"] - df["ema_50"]
    
    # Distance from MAs
    df["dist_ema_9"] = (df["Close"] - df["ema_9"]) / df["ema_9"]
    df["dist_ema_50"] = (df["Close"] - df["ema_50"]) / df["ema_50"]

    # MACD
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd_diff()
    
    # ADX (Trend Strength)
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
    df["adx"] = adx.adx()
    
    # Ichimoku Cloud (Basic elements)
    ichimoku = ta.trend.IchimokuIndicator(df["High"], df["Low"])
    df["ichimoku_a"] = ichimoku.ichimoku_a()
    df["ichimoku_b"] = ichimoku.ichimoku_b()
    df["ichimoku_base"] = ichimoku.ichimoku_base_line()
    
    # Distances from Ichimoku
    df["dist_ichimoku_a"] = (df["Close"] - df["ichimoku_a"]) / df["ichimoku_a"]
    df["dist_ichimoku_base"] = (df["Close"] - df["ichimoku_base"]) / df["ichimoku_base"]
    
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["Close"])
    df["bb_high_diff"] = (bb.bollinger_hband() - df["Close"]) / df["Close"]
    df["bb_low_diff"] = (df["Close"] - bb.bollinger_lband()) / df["Close"]
    
    # Lags (Historical context)
    for i in range(1, 4):
        df[f"return_lag_{i}"] = df["return"].shift(i)
        df[f"volume_lag_{i}"] = df["volume_change"].shift(i)

    # VWAP calculation
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        volume=df["Volume"]
    ).volume_weighted_average_price()
    df["vwap_diff"] = (df["vwap"] - df["Close"]) / df["Close"]

    # Final Cleaning: Remove infinity and NaNs
    import numpy as np
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # Optional: Clip features to a reasonable range to prevent outliers from affecting RF
    # (e.g., -1 to 1 for percentage changes)
    feature_cols_to_clip = ["return", "volume_change", "vwap_diff", "bb_high_diff", "bb_low_diff", "dist_ema_9", "dist_ema_50"]
    for col in feature_cols_to_clip:
        if col in df.columns:
            df[col] = df[col].clip(-1, 1)

    return df

if __name__ == "__main__":
    from data import load_data
    df = load_data("AAPL")
    df_with_features = add_features(df)
    print(df_with_features.tail())
    print(f"Features added: {df_with_features.columns.tolist()}")
