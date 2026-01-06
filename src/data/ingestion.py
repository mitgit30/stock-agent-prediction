import os
from datetime import datetime
from typing import Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

from src.config import Config
from src.exception import PipelineError

load_dotenv()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    """
    delta = series.diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
    """
    Calculate MACD (EMA fast - EMA slow).
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def fetch_ohlcv(
    ticker: str,
    start: str = Config().start_date,
    end: Optional[str] = None
) -> pd.DataFrame:

    config = Config()

    try:
        data = yf.download(ticker,start=start,end=end,interval="1d",progress=False
        )

        if data.empty:
            raise PipelineError(f"No data found for ticker {ticker}")

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = (
            data.reset_index().rename(columns={"Date": "date"}).loc[:, ["date", "Open", "High", "Low", "Close", "Volume"]].dropna()
        )

        # Indicators
        data["RSI14"] = rsi(data["Close"])
        data["MACD"] = macd(data["Close"])

        data = data[["date"] + config.features]
        data = data.dropna().reset_index(drop=True)

        # Validation
        required_rows = config.context_len + config.pred_len
        if len(data) < required_rows:
            raise PipelineError(
                f"Insufficient data for {ticker}: {len(data)} rows, "
                f"need at least {required_rows}"
            )

        if data[config.features].isnull().any().any():
            raise PipelineError(f"NaN values found in features for {ticker}")

        if not data[config.features].apply(pd.api.types.is_numeric_dtype).all():
            raise PipelineError(f"Non-numeric values found in features for {ticker}")

        print(f"Fetched {len(data)} rows for {ticker}")

        # integrate with Feast
        try:
            feast_data = data.copy()
            feast_data["ticker"] = ticker
            feast_data["event_timestamp"] = pd.to_datetime(feast_data["date"])
            feast_data["created_timestamp"] = datetime.now()

            folder_path = os.path.join(os.getcwd(), "feature_store")
            data_path = os.path.join(folder_path, "data", "features.parquet")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            
            import portalocker
             # this fcntl module does the work of file data locking
            # -If by mistakenly running multiple pipelines at the same time it  wont corrupt the feature data.
            # this is an important step to ensure data consistency in the production environment.
            lock_path = data_path + ".lock"

            with open(lock_path, "w") as lock_file:
                portalocker.lock(lock_file, portalocker.LOCK_EX)
                try:
                    if os.path.exists(data_path):
                        existing_df = pd.read_parquet(data_path)
                        combined_df = (pd.concat([existing_df, feast_data]).drop_duplicates(subset=["ticker", "event_timestamp"])
                        )
                        combined_df.to_parquet(data_path)
                    else:
                        feast_data.to_parquet(data_path)
                finally:
                    portalocker.lock(lock_file, portalocker.LOCK_UN)

            print(f"Saved features to {data_path}")

        except Exception as feast_error:
            print(f" Feast write failed for {ticker}: {feast_error}")

        return data

    except Exception as e:
        raise PipelineError(f"Failed to fetch data for {ticker}: {e}")
