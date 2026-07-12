from pathlib import Path

import pandas as pd
import yfinance as yf

# Ensures data is saving to files in the data folder and that they exist
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Tickers for Nvidia, Microsoft, Apple, AMD and Nasdaq-100
tickers = ["NVDA", "MSFT", "AAPL", "AMD", "QQQ"]

# Downloads daily stock info from yahoo finance since 01/01/2020
def download_price_data(tickers):
    raw_data = yf.download(
        tickers,
        start="2020-01-01",
        interval="1d",
        progress=False,
    )
    return raw_data

# Moves ticker names from the column headers into rows, labels the row-index levels as Date and ticker, then turns those index levels into normal columns.
def clean_yfinance_data(raw_data) -> pd.DataFrame:
    data = raw_data.stack(level="Ticker").rename_axis(["date", "ticker"]).reset_index()
    data.columns = data.columns.str.lower()
    return data


def save_data(df, filename):
    df.to_csv(DATA_DIR / filename, index=False)


def load_data(filename):
    df = pd.read_csv(DATA_DIR / filename, parse_dates=["date"])
    return df


if __name__ == "__main__":
    raw_data = download_price_data(tickers)
    clean_data = clean_yfinance_data(raw_data)

    save_data(clean_data, "prices.csv")

    loaded_data = load_data("prices.csv")
    print(loaded_data.head())
    print(loaded_data.info())