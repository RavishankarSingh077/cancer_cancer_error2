import yfinance as yf
import pandas as pd

def load_data(symbol="AAPL", interval="5m", period="60d"):
    """
    Downloads historical stock data using yf.Ticker.history with retry logic.
    """
    import time
    print(f"Downloading data for {symbol}...")
    ticker = yf.Ticker(symbol)
    
    for attempt in range(3):
        try:
            df = ticker.history(period=period, interval=interval)
            if not df.empty:
                df.dropna(inplace=True)
                return df
        except Exception as e:
            print(f"Attempt {attempt+1} failed: {e}")
            time.sleep(2)
            
    raise ValueError(f"No data found for {symbol} after retries. Check connection/symbol.")

if __name__ == "__main__":
    df = load_data("AAPL")
    print(df.head())
    print(f"Total rows: {len(df)}")
