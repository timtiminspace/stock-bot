from pathlib import Path

import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data")

tickers = ["NVDA", "MSFT", "AAPL", "AMD", "QQQ"]

def download_price_data(tickers):
    data = yf.download(tickers, start="2020-01-01", interval="1d")
    return data

def save_data(df, file):
    df.to_csv(file)
    
def load_data(file):
    df = pd.read_csv(file)
    return df